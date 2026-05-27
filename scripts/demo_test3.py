"""3-sample stress test: different speakers and SNR conditions.

Test cases:
  1. seed=10  SNR=  0 dB  equal energy (standard case)
  2. seed=25  SNR= -3 dB  female quieter than male (harder)
  3. seed=77  SNR= +3 dB  female louder than male (easier)

For each case saves:
  mix.wav        -> input mixture
  target.wav     -> ground truth female
  interferer.wav -> ground truth male
  output.wav     -> system output (MLP + GMM + HMM + DPCRN)
"""
from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

from pesq import pesq
from pystoi import stoi

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.gmm_classifier import GenderGMM
from src.ai.dpcrn import DPCRN
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import save_audio, SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SR = SAMPLE_RATE

TESTS = [
    {"seed": 10, "snr_db":  0.0, "label": "test1_snr0"},
    {"seed": 25, "snr_db": -3.0, "label": "test2_snr-3_female_quieter"},
    {"seed": 77, "snr_db": +3.0, "label": "test3_snr+3_female_louder"},
]

OUT_ROOT = Path("data/processed/test3")


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8))


def main() -> None:
    log.info("Loading models...")
    classifier = SpeakerClassifier.load("models/classifier.joblib")
    gmm        = GenderGMM.load("models/gender_gmm.joblib")
    dpcrn      = DPCRN.load("models/dpcrn.pt")
    attention  = AttentionModule(classifier, gmm=gmm)

    rows = []

    for t in TESTS:
        label   = t["label"]
        snr_db  = t["snr_db"]
        seed    = t["seed"]
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("--- %s (SNR %+.0f dB, seed=%d) ---", label, snr_db, seed)
        sample = make_samples(n_samples=1, snr_db=snr_db, clip_duration=5.0, seed=seed)[0]

        log.info("  Female speaker: %s  |  Male speaker: %s",
                 sample.target_speaker_id, sample.interferer_speaker_id)

        save_audio(out_dir / "mix.wav",        sample.mixture,    sr=SR)
        save_audio(out_dir / "target.wav",     sample.target,     sr=SR)
        save_audio(out_dir / "interferer.wav", sample.interferer, sr=SR)

        mask = attention.compute_mask(sample.mixture, sr=SR)
        recon = separate_nmf(sample.mixture, mask, sr=SR, mask_net=dpcrn, target_gender=0)
        output = enhance(recon, sr=SR)
        save_audio(out_dir / "output.wav", output, sr=SR)

        n = min(len(sample.target), len(output))
        ref = sample.target[:n].astype(np.float64)
        est = output[:n].astype(np.float64)
        mix_ref = sample.mixture[:n].astype(np.float64)

        m = {
            "label":   label,
            "snr_db":  snr_db,
            "f_id":    sample.target_speaker_id,
            "m_id":    sample.interferer_speaker_id,
            "sisdr_mix": si_sdr(ref, mix_ref),
            "sisdr_out": si_sdr(ref, est),
            "pesq_out":  float(pesq(SR, ref, est, "wb")),
            "stoi_out":  float(stoi(ref, est, SR, extended=False)),
        }
        rows.append(m)

    print()
    print("=" * 72)
    print("  3-SAMPLE STRESS TEST  —  MLP + GMM + HMM + DPCRN")
    print("=" * 72)
    hdr = f"  {'Test':<36} {'SI-SDR mix':>10} {'SI-SDR out':>11} {'delta':>7} {'PESQ':>6} {'STOI':>6}"
    print(hdr)
    print("  " + "-" * 70)
    for r in rows:
        delta = r["sisdr_out"] - r["sisdr_mix"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {r['label']:<36} {r['sisdr_mix']:>10.2f} {r['sisdr_out']:>11.2f} "
              f"{sign}{delta:>6.2f} {r['pesq_out']:>6.3f} {r['stoi_out']:>6.3f}")
    print("  " + "-" * 70)
    avg_delta = float(np.mean([r["sisdr_out"] - r["sisdr_mix"] for r in rows]))
    print(f"  {'Average delta SI-SDR':<36} {'':>10} {'':>11} {'+' if avg_delta >= 0 else ''}{avg_delta:>6.2f}")
    print("=" * 72)
    print()
    print("  Files saved to data/processed/test3/<label>/")
    print("  Listen: mix.wav -> output.wav (vs target.wav for reference)")


if __name__ == "__main__":
    main()
