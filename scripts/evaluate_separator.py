"""Extended evaluation of the separate-then-select flow (DPCRNSeparator + attention).

Same protocol as evaluate_extended.py: 36 samples (12 per SNR in {-3, 0, +3} dB,
seed=123, clip=4 s). Each mixture is separated ONCE; both streams are then
scored by the attention module and evaluated against BOTH references, so a
single run reports female-target and male-target quality plus the
stream-selection accuracy (attention choice vs SI-SDR oracle).

Metrics are reported for the raw separated stream and for the log-MMSE
enhanced stream, to verify whether the enhancement stage still helps in the
new flow.

Usage:
    python scripts/evaluate_separator.py [--separator models/separator.pt]
"""
from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pesq import pesq
from pystoi import stoi

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.dpcrn import DPCRNSeparator
from src.ai.gmm_classifier import GenderGMM
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.utils import SAMPLE_RATE

SR = SAMPLE_RATE
N_PER_SNR = 12
SNR_LEVELS = [-3.0, 0.0, 3.0]
CLIP_DURATION = 4.0
SEED = 123

CLASSIFIER_PATH = "models/classifier.joblib"
GMM_PATH        = "models/gender_gmm.joblib"
SEPARATOR_PATH  = "models/separator.pt"

GENDERS = ("F", "M")


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8))


def compute_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    n = min(len(reference), len(estimate))
    ref64 = reference[:n].astype(np.float64)
    est64 = estimate[:n].astype(np.float64)
    return {
        "SI-SDR": si_sdr(ref64, est64),
        "PESQ":   float(pesq(SR, ref64, est64, "wb")),
        "STOI":   float(stoi(ref64, est64, SR, extended=False)),
    }


def stats(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separator", default=SEPARATOR_PATH)
    args = parser.parse_args()

    logging.disable(logging.NOTSET)
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("src.utils").setLevel(logging.WARNING)

    log.info("Loading models...")
    classifier = SpeakerClassifier.load(CLASSIFIER_PATH)
    gmm        = GenderGMM.load(GMM_PATH)
    separator  = DPCRNSeparator.load(args.separator)
    attention  = AttentionModule(classifier, gmm=gmm)

    metric_keys = ("SI-SDR", "PESQ", "STOI")
    # results[gender][snr]["mix"|"raw"|"enh"][metric] -> list of values
    results = {
        g: {snr: {v: {k: [] for k in metric_keys} for v in ("mix", "raw", "enh")}
            for snr in SNR_LEVELS}
        for g in GENDERS
    }
    selection_hits = {g: 0 for g in GENDERS}

    total_done = 0
    total_samples = N_PER_SNR * len(SNR_LEVELS)

    for snr in SNR_LEVELS:
        log.info("--- SNR = %+.1f dB ---", snr)
        samples = make_samples(
            n_samples=N_PER_SNR, snr_db=snr,
            clip_duration=CLIP_DURATION, seed=SEED,
        )

        for sample in samples:
            mix = sample.mixture
            refs = {"F": sample.target, "M": sample.interferer}

            streams = separator.separate(mix)
            score_a = attention.score_female(streams[0], sr=SR)
            score_b = attention.score_female(streams[1], sr=SR)
            # Attention picks the stream with higher P(female) as F, other as M
            f_idx = 0 if score_a >= score_b else 1
            selected = {"F": streams[f_idx], "M": streams[1 - f_idx]}

            for g in GENDERS:
                ref = refs[g]
                # Oracle: which stream is actually closer to this reference?
                sdr_streams = [si_sdr(ref[: len(s)], s[: len(ref)]) for s in streams]
                oracle_idx = int(np.argmax(sdr_streams))
                chosen_idx = f_idx if g == "F" else 1 - f_idx
                selection_hits[g] += int(chosen_idx == oracle_idx)

                raw = selected[g]
                enh = enhance(raw, sr=SR)

                r = results[g][snr]
                m_mix = compute_metrics(ref, mix)
                m_raw = compute_metrics(ref, raw)
                m_enh = compute_metrics(ref, enh)
                for k in metric_keys:
                    r["mix"][k].append(m_mix[k])
                    r["raw"][k].append(m_raw[k])
                    r["enh"][k].append(m_enh[k])

            total_done += 1
            log.info(
                "  [%2d/%d] SNR=%+.1f  F: mix=%.2f raw=%.2f (d%+.2f)   M: mix=%.2f raw=%.2f (d%+.2f)",
                total_done, total_samples, snr,
                results["F"][snr]["mix"]["SI-SDR"][-1], results["F"][snr]["raw"]["SI-SDR"][-1],
                results["F"][snr]["raw"]["SI-SDR"][-1] - results["F"][snr]["mix"]["SI-SDR"][-1],
                results["M"][snr]["mix"]["SI-SDR"][-1], results["M"][snr]["raw"]["SI-SDR"][-1],
                results["M"][snr]["raw"]["SI-SDR"][-1] - results["M"][snr]["mix"]["SI-SDR"][-1],
            )

    W = 78
    print()
    print("=" * W)
    print("  AURALIS - SEPARATOR EVALUATION (DPCRNSeparator uPIT + attention selection)")
    print(f"  {total_samples} samples | SNR in {SNR_LEVELS} dB | clip={CLIP_DURATION:.1f}s | seed={SEED}")
    print("=" * W)

    for g in GENDERS:
        label = "FEMALE" if g == "F" else "MALE"
        print(f"\n  -- {label} TARGET " + "-" * (W - 18 - len(label)))
        print(f"{'SNR':>6}  {'SI-SDR mix':>12}  {'SI-SDR raw':>12}  {'Delta':>8}  "
              f"{'PESQ raw':>9}  {'STOI raw':>9}")
        print("-" * W)
        for snr in SNR_LEVELS:
            r = results[g][snr]
            sm, ss = stats(r["mix"]["SI-SDR"])
            om, os_ = stats(r["raw"]["SI-SDR"])
            dm, ds = stats([o - i for o, i in zip(r["raw"]["SI-SDR"], r["mix"]["SI-SDR"])])
            pm, _ = stats(r["raw"]["PESQ"])
            tm, _ = stats(r["raw"]["STOI"])
            print(f"{snr:>+6.1f}  {sm:>+7.2f}+-{ss:.2f}  {om:>+7.2f}+-{os_:.2f}  "
                  f"{dm:>+6.2f}+-{ds:.2f}  {pm:>9.3f}  {tm:>9.3f}")

        # Aggregates over all SNR levels
        agg = {v: {k: [] for k in metric_keys} for v in ("mix", "raw", "enh")}
        for snr in SNR_LEVELS:
            for v in ("mix", "raw", "enh"):
                for k in metric_keys:
                    agg[v][k].extend(results[g][snr][v][k])

        dm_all, ds_all = stats([o - i for o, i in zip(agg["raw"]["SI-SDR"], agg["mix"]["SI-SDR"])])
        de_all, _ = stats([o - i for o, i in zip(agg["enh"]["SI-SDR"], agg["mix"]["SI-SDR"])])
        print("-" * W)
        print(f"  ALL   SI-SDRi raw {dm_all:+.2f}+-{ds_all:.2f} dB | enh {de_all:+.2f} dB")
        print(f"        PESQ  mix {np.mean(agg['mix']['PESQ']):.3f} | raw {np.mean(agg['raw']['PESQ']):.3f} "
              f"| enh {np.mean(agg['enh']['PESQ']):.3f}")
        print(f"        STOI  mix {np.mean(agg['mix']['STOI']):.3f} | raw {np.mean(agg['raw']['STOI']):.3f} "
              f"| enh {np.mean(agg['enh']['STOI']):.3f}")
        print(f"        Stream selection accuracy: {selection_hits[g]}/{total_samples} "
              f"({100.0 * selection_hits[g] / total_samples:.1f}%)")

    print("=" * W)


if __name__ == "__main__":
    main()
