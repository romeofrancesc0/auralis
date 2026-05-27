"""Quick demo: generates a male/female mix, runs the full pipeline, and saves output files.

Recommended config: MLP + GMM + HMM + DPCRN (best perceptual quality).
Falls back gracefully if optional model files are missing.

Usage:
    python demo.py

Output files in data/processed/demo/:
    - mix.wav         -> original mixture (female + male)
    - target.wav      -> ground truth female voice
    - interferer.wav  -> ground truth male voice
    - output.wav      -> system output (isolated female voice)
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH   = "models/classifier.joblib"
GMM_PATH     = "models/gender_gmm.joblib"
DPCRN_PATH   = "models/dpcrn.pt"
MASKNET_PATH = "models/mask_net.pt"
OUT_DIR      = "data/processed/demo"


def main() -> None:
    log.info("Generating a test sample from LibriSpeech...")
    samples = make_samples(n_samples=1, clip_duration=5.0, seed=7)
    sample = samples[0]
    log.info("Target (F): %s  |  Interferer (M): %s",
             sample.target_speaker_id, sample.interferer_speaker_id)

    save_audio(f"{OUT_DIR}/mix.wav",        sample.mixture,    sr=sample.sr)
    save_audio(f"{OUT_DIR}/target.wav",     sample.target,     sr=sample.sr)
    save_audio(f"{OUT_DIR}/interferer.wav", sample.interferer, sr=sample.sr)

    classifier = SpeakerClassifier.load(MODEL_PATH)

    gmm = None
    if Path(GMM_PATH).exists():
        from src.ai.gmm_classifier import GenderGMM
        gmm = GenderGMM.load(GMM_PATH)
    else:
        log.warning("GenderGMM not found at %s — running without GMM blend.", GMM_PATH)

    # DPCRN preferred; fall back to MaskNet, then no refiner
    mask_net = None
    if Path(DPCRN_PATH).exists():
        from src.ai.dpcrn import DPCRN
        mask_net = DPCRN.load(DPCRN_PATH)
        log.info("Refiner: DPCRN")
    elif Path(MASKNET_PATH).exists():
        from src.ai.mask_net import MaskNet
        mask_net = MaskNet.load(MASKNET_PATH)
        log.info("Refiner: MaskNet (DPCRN not found)")
    else:
        log.warning("No CNN refiner found — output quality will be lower.")

    attention = AttentionModule(classifier, gmm=gmm)

    log.info("Computing attention mask...")
    mask = attention.compute_mask(sample.mixture, sr=sample.sr)

    log.info("Separating (NMF + Griffin-Lim + DPCRN)...")
    reconstructed = separate_nmf(sample.mixture, mask, sr=sample.sr, mask_net=mask_net)
    output = enhance(reconstructed, sr=sample.sr)

    save_audio(f"{OUT_DIR}/output.wav", output, sr=sample.sr)

    print()
    print("=" * 55)
    print("  OUTPUT FILES SAVED TO data/processed/demo/")
    print("=" * 55)
    print("  mix.wav         -> original mixture (F + M)")
    print("  target.wav      -> female voice (ground truth)")
    print("  interferer.wav  -> male voice (ground truth)")
    print("  output.wav      -> system output (isolated female)")
    print("=" * 55)


if __name__ == "__main__":
    main()
