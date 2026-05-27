"""Generates one test mix and saves the output of all pipeline variants for listening."""
from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.gmm_classifier import GenderGMM
from src.ai.smoothing_gru import GRUSmoother
from src.ai.mask_net import MaskNet
from src.ai.dpcrn import DPCRN
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("data/processed/compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 99


def run(mix, attention, mask_net=None, sr=16000):
    mask = attention.compute_mask(mix, sr=sr)
    recon = separate_nmf(mix, mask, sr=sr, mask_net=mask_net, target_gender=0)
    return enhance(recon, sr=sr)


def main() -> None:
    log.info("Generating test sample (seed=%d)...", SEED)
    sample = make_samples(n_samples=1, clip_duration=5.0, seed=SEED)[0]
    sr = sample.sr

    log.info("Target (F): %s  |  Interferer (M): %s",
             sample.target_speaker_id, sample.interferer_speaker_id)

    save_audio(OUT_DIR / "mix.wav",        sample.mixture,    sr=sr)
    save_audio(OUT_DIR / "target.wav",     sample.target,     sr=sr)
    save_audio(OUT_DIR / "interferer.wav", sample.interferer, sr=sr)

    log.info("Loading models...")
    classifier = SpeakerClassifier.load("models/classifier.joblib")
    gmm        = GenderGMM.load("models/gender_gmm.joblib")
    gru        = GRUSmoother.load("models/smoothing_gru.pt")
    masknet    = MaskNet.load("models/mask_net.pt")
    dpcrn      = DPCRN.load("models/dpcrn.pt")

    attn_hmm = AttentionModule(classifier, gmm=gmm)
    attn_gru = AttentionModule(classifier, gmm=gmm, gru_smoother=gru)

    variants = [
        ("A_hmm",    attn_hmm, None),
        ("B_gru",    attn_gru, None),
        ("C_masknet", attn_hmm, masknet),
        ("D_dpcrn",  attn_hmm, dpcrn),
    ]

    for name, attn, refiner in variants:
        log.info("Running variant %s...", name)
        out = run(sample.mixture, attn, mask_net=refiner, sr=sr)
        save_audio(OUT_DIR / f"output_{name}.wav", out, sr=sr)

    print()
    print("=" * 52)
    print("  FILES SAVED TO data/processed/compare/")
    print("=" * 52)
    print("  mix.wav              -> input (F + M mixture)")
    print("  target.wav           -> ground truth (female)")
    print("  interferer.wav       -> ground truth (male)")
    print("  output_A_hmm.wav     -> MLP + GMM + HMM")
    print("  output_B_gru.wav     -> MLP + GMM + GRU")
    print("  output_C_masknet.wav -> MLP + GMM + MaskNet  ← best SI-SDR")
    print("  output_D_dpcrn.wav   -> MLP + GMM + DPCRN")
    print("=" * 52)


if __name__ == "__main__":
    main()
