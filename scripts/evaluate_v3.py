"""v0.3.0 evaluation — compares all pipeline variants on LibriSpeech test mixes.

Variants tested:
  A) MLP + GMM + HMM          (v0.2.0 baseline)
  B) MLP + GMM + GRU          (new smoother)
  C) MLP + GMM + HMM + MaskNet  (retrained, combined loss + FiLM)
  D) MLP + GMM + HMM + DPCRN    (new architecture)

Metrics: SI-SDR (dB), PESQ (wideband), STOI.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # silence pipeline logs during evaluation

from pesq import pesq
from pystoi import stoi

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.gmm_classifier import GenderGMM
from src.ai.smoothing_gru import GRUSmoother
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import SAMPLE_RATE

SR = SAMPLE_RATE
N_SAMPLES = 10
CLIP_DURATION = 4.0
SNR_DB = 0.0
SEED = 42

CLASSIFIER_PATH   = "models/classifier.joblib"
GMM_PATH          = "models/gender_gmm.joblib"
GRU_PATH          = "models/smoothing_gru.pt"
MASKNET_PATH      = "models/mask_net.pt"
DPCRN_PATH        = "models/dpcrn.pt"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    mix: np.ndarray,
    attention: AttentionModule,
    mask_net=None,
) -> np.ndarray:
    mask = attention.compute_mask(mix, sr=SR)
    recon = separate_nmf(mix, mask, sr=SR, mask_net=mask_net, target_gender=0)
    return enhance(recon, sr=SR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.disable(logging.NOTSET)
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("Loading models...")
    classifier = SpeakerClassifier.load(CLASSIFIER_PATH)
    gmm        = GenderGMM.load(GMM_PATH)
    gru        = GRUSmoother.load(GRU_PATH)

    from src.ai.mask_net import MaskNet
    from src.ai.dpcrn import DPCRN
    masknet = MaskNet.load(MASKNET_PATH)
    dpcrn   = DPCRN.load(DPCRN_PATH)

    attn_hmm  = AttentionModule(classifier, gmm=gmm)
    attn_gru  = AttentionModule(classifier, gmm=gmm, gru_smoother=gru)

    log.info("Building test dataset (%d samples, SNR=%.1f dB)...", N_SAMPLES, SNR_DB)

    samples = make_samples(
        n_samples=N_SAMPLES,
        snr_db=SNR_DB,
        clip_duration=CLIP_DURATION,
        seed=SEED,
    )

    keys = ["A_hmm", "B_gru", "C_masknet", "D_dpcrn", "E_gru_masknet", "F_gru_dpcrn"]
    results: dict[str, list[dict]] = {k: [] for k in keys}
    mix_results: list[dict] = []

    for i, sample in enumerate(samples):
        mix    = sample.mixture
        target = sample.target
        n = min(len(mix), len(target))
        mix, target = mix[:n], target[:n]

        mix_results.append(compute_metrics(target, mix))

        outputs = {
            "A_hmm":        run_pipeline(mix, attn_hmm),
            "B_gru":        run_pipeline(mix, attn_gru),
            "C_masknet":    run_pipeline(mix, attn_hmm, mask_net=masknet),
            "D_dpcrn":      run_pipeline(mix, attn_hmm, mask_net=dpcrn),
            "E_gru_masknet": run_pipeline(mix, attn_gru, mask_net=masknet),
            "F_gru_dpcrn":  run_pipeline(mix, attn_gru, mask_net=dpcrn),
        }

        for key, out in outputs.items():
            n2 = min(len(target), len(out))
            results[key].append(compute_metrics(target[:n2], out[:n2]))

        log.info("  Sample %2d/%d done", i + 1, N_SAMPLES)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    def avg(lst: list[dict], key: str) -> float:
        return float(np.mean([d[key] for d in lst]))

    METRICS = ["SI-SDR", "PESQ", "STOI"]
    VARIANTS = {
        "Mix (input)":              mix_results,
        "A: HMM only":              results["A_hmm"],
        "B: GRU only":              results["B_gru"],
        "C: HMM + MaskNet":         results["C_masknet"],
        "D: HMM + DPCRN":           results["D_dpcrn"],
        "E: GRU + MaskNet":         results["E_gru_masknet"],
        "F: GRU + DPCRN (full)":    results["F_gru_dpcrn"],
    }

    col_w = 14
    header = f"{'Variant':<24}" + "".join(f"{m:>{col_w}}" for m in METRICS)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  AURALIS v0.3.0 — EVALUATION RESULTS")
    print(f"  {N_SAMPLES} samples | SNR={SNR_DB:+.1f} dB | clip={CLIP_DURATION:.1f}s | seed={SEED}")
    print("=" * len(header))
    print(header)
    print(sep)

    mix_vals = {m: avg(mix_results, m) for m in METRICS}
    for name, data in VARIANTS.items():
        vals = {m: avg(data, m) for m in METRICS}
        row = f"{name:<24}"
        for m in METRICS:
            v = vals[m]
            delta = v - mix_vals[m]
            sign = "+" if delta >= 0 else ""
            if name == "Mix (input)":
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{v:>{col_w-7}.3f} ({sign}{delta:.2f})"
        print(row)

    print(sep)

    # Best variant by SI-SDR
    variant_sisdr = {k: avg(v, "SI-SDR") for k, v in results.items()}
    best_key = max(variant_sisdr, key=lambda k: variant_sisdr[k])
    best_labels = {
        "A_hmm": "A (HMM)", "B_gru": "B (GRU)",
        "C_masknet": "C (HMM+MaskNet)", "D_dpcrn": "D (HMM+DPCRN)",
        "E_gru_masknet": "E (GRU+MaskNet)", "F_gru_dpcrn": "F (GRU+DPCRN)",
    }
    print(f"\n  Best variant by SI-SDR: {best_labels[best_key]}  ({variant_sisdr[best_key]:+.3f} dB)")

    v020_baseline = 3.945
    best_si_sdr = variant_sisdr[best_key] - avg(mix_results, "SI-SDR")
    delta_vs_v020 = best_si_sdr - v020_baseline
    sign = "+" if delta_vs_v020 >= 0 else ""
    print(f"  vs v0.2.0 baseline (+{v020_baseline} dB):  {sign}{delta_vs_v020:.3f} dB")

    if best_si_sdr > v020_baseline:
        print("\n  VERDICT: PASS — tag v0.3.0")
    else:
        print("\n  VERDICT: NO IMPROVEMENT over v0.2.0 — do not tag v0.3.0")
    print("=" * len(header))


if __name__ == "__main__":
    main()
