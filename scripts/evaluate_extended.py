"""Extended evaluation — recommended config (MLP + GMM + HMM + DPCRN) on 36 samples.

36 samples = 12 per SNR level (-3, 0, +3 dB), drawn with seed=123.
Reports mean +- std for SI-SDR, PESQ, and STOI.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pesq import pesq
from pystoi import stoi

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.gmm_classifier import GenderGMM
from src.ai.dpcrn import DPCRN
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import SAMPLE_RATE

SR = SAMPLE_RATE
N_PER_SNR = 12
SNR_LEVELS = [-3.0, 0.0, 3.0]
CLIP_DURATION = 4.0
SEED = 123

CLASSIFIER_PATH = "models/classifier.joblib"
GMM_PATH        = "models/gender_gmm.joblib"
DPCRN_PATH      = "models/dpcrn.pt"


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8))


def compute_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    ref64 = reference.astype(np.float64)
    est64 = estimate.astype(np.float64)
    return {
        "SI-SDR": si_sdr(ref64, est64),
        "PESQ":   float(pesq(SR, ref64, est64, "wb")),
        "STOI":   float(stoi(ref64, est64, SR, extended=False)),
    }


def run_pipeline(mix: np.ndarray, attention: AttentionModule, dpcrn: DPCRN) -> np.ndarray:
    mask = attention.compute_mask(mix, sr=SR)
    recon = separate_nmf(mix, mask, sr=SR, mask_net=dpcrn, target_gender=0)
    return enhance(recon, sr=SR)


def stats(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    logging.disable(logging.NOTSET)
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("Loading models...")
    classifier = SpeakerClassifier.load(CLASSIFIER_PATH)
    gmm        = GenderGMM.load(GMM_PATH)
    dpcrn      = DPCRN.load(DPCRN_PATH)
    attention  = AttentionModule(classifier, gmm=gmm)

    snr_results: dict[float, dict[str, list[float]]] = {
        snr: {"SI-SDR_mix": [], "SI-SDR_out": [], "PESQ_mix": [], "PESQ_out": [], "STOI_mix": [], "STOI_out": []}
        for snr in SNR_LEVELS
    }

    total_done = 0
    total_samples = N_PER_SNR * len(SNR_LEVELS)

    for snr in SNR_LEVELS:
        log.info("--- SNR = %+.1f dB ---", snr)
        samples = make_samples(
            n_samples=N_PER_SNR,
            snr_db=snr,
            clip_duration=CLIP_DURATION,
            seed=SEED,
        )

        for i, sample in enumerate(samples):
            mix    = sample.mixture
            target = sample.target
            n = min(len(mix), len(target))
            mix, target = mix[:n], target[:n]

            m_mix = compute_metrics(target, mix)
            out   = run_pipeline(mix, attention, dpcrn)
            n2    = min(len(target), len(out))
            m_out = compute_metrics(target[:n2], out[:n2])

            r = snr_results[snr]
            r["SI-SDR_mix"].append(m_mix["SI-SDR"])
            r["SI-SDR_out"].append(m_out["SI-SDR"])
            r["PESQ_mix"].append(m_mix["PESQ"])
            r["PESQ_out"].append(m_out["PESQ"])
            r["STOI_mix"].append(m_mix["STOI"])
            r["STOI_out"].append(m_out["STOI"])

            total_done += 1
            log.info("  [%2d/%d] SNR=%+.1f dB  SI-SDR: mix=%.2f out=%.2f  delta=%.2f",
                     total_done, total_samples, snr,
                     m_mix["SI-SDR"], m_out["SI-SDR"],
                     m_out["SI-SDR"] - m_mix["SI-SDR"])

    # Aggregate across all SNR levels
    all_mix_sisdr, all_out_sisdr = [], []
    all_mix_pesq,  all_out_pesq  = [], []
    all_mix_stoi,  all_out_stoi  = [], []

    for snr in SNR_LEVELS:
        r = snr_results[snr]
        all_mix_sisdr.extend(r["SI-SDR_mix"]); all_out_sisdr.extend(r["SI-SDR_out"])
        all_mix_pesq.extend(r["PESQ_mix"]);    all_out_pesq.extend(r["PESQ_out"])
        all_mix_stoi.extend(r["STOI_mix"]);    all_out_stoi.extend(r["STOI_out"])

    W = 72
    print()
    print("=" * W)
    print("  AURALIS — EXTENDED EVALUATION (MLP + GMM + HMM + DPCRN)")
    print(f"  {total_samples} samples | SNR in {SNR_LEVELS} dB | clip={CLIP_DURATION:.1f}s | seed={SEED}")
    print("=" * W)

    # Per-SNR table
    print(f"\n{'SNR':>6}  {'SI-SDR mix':>12}  {'SI-SDR out':>12}  {'Delta':>8}  {'PESQ out':>10}  {'STOI out':>10}")
    print("-" * W)
    for snr in SNR_LEVELS:
        r = snr_results[snr]
        sm, ss = stats(r["SI-SDR_mix"])
        om, os = stats(r["SI-SDR_out"])
        pm, ps = stats(r["PESQ_out"])
        tm, ts = stats(r["STOI_out"])
        delta_m, delta_s = stats([o - i for o, i in zip(r["SI-SDR_out"], r["SI-SDR_mix"])])
        print(f"{snr:>+6.1f}  {sm:>+7.2f}+-{ss:.2f}  {om:>+7.2f}+-{os:.2f}  {delta_m:>+6.2f}+-{delta_s:.2f}  {pm:>6.3f}+-{ps:.3f}  {tm:>6.3f}+-{ts:.3f}")

    print("-" * W)

    # Overall
    sm_all, ss_all = stats(all_mix_sisdr)
    om_all, os_all = stats(all_out_sisdr)
    pm_all, ps_all = stats(all_out_pesq)
    tm_all, ts_all = stats(all_out_stoi)
    dm_all = float(np.mean([o - i for o, i in zip(all_out_sisdr, all_mix_sisdr)]))
    ds_all = float(np.std([o - i for o, i in zip(all_out_sisdr, all_mix_sisdr)]))

    print(f"{'ALL':>6}  {sm_all:>+7.2f}+-{ss_all:.2f}  {om_all:>+7.2f}+-{os_all:.2f}  {dm_all:>+6.2f}+-{ds_all:.2f}  {pm_all:>6.3f}+-{ps_all:.3f}  {tm_all:>6.3f}+-{ts_all:.3f}")
    print("=" * W)

    print(f"\n  SUMMARY (all {total_samples} samples):")
    print(f"    SI-SDR:  mix {sm_all:+.3f} dB  ->  out {om_all:+.3f} dB  (delta {dm_all:+.3f} +- {ds_all:.3f} dB)")
    print(f"    PESQ:    {pm_all:.3f} +- {ps_all:.3f}")
    print(f"    STOI:    {tm_all:.3f} +- {ts_all:.3f}")

    mix_pesq_m = float(np.mean(all_mix_pesq))
    mix_stoi_m = float(np.mean(all_mix_stoi))
    print(f"\n    Mix baseline:  PESQ {mix_pesq_m:.3f}  STOI {mix_stoi_m:.3f}")
    print(f"    System delta:  PESQ {pm_all - mix_pesq_m:+.3f}  STOI {tm_all - mix_stoi_m:+.3f}")
    print("=" * W)


if __name__ == "__main__":
    main()
