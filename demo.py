"""Quick demo: generates a male/female mix, runs the full pipeline, and saves output files.

Usage:
    python demo.py

Output files in data/processed/demo/:
    - mix.wav         → original mixture (female voice + male voice)
    - target.wav      → isolated female voice (ground truth)
    - interferer.wav  → male voice (ground truth)
    - output.wav      → system output (female voice extracted from the mix)
"""
from __future__ import annotations

import logging

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.dsp.dataset import make_samples
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = "models/classifier.joblib"
OUT_DIR = "data/processed/demo"


def main() -> None:
    log.info("Generating a test sample from LibriSpeech...")
    samples = make_samples(n_samples=1, clip_duration=5.0, seed=7)
    sample = samples[0]

    log.info("Target speaker (F): %s | Interferer speaker (M): %s",
             sample.target_speaker_id, sample.interferer_speaker_id)

    # Save mix and ground truth
    save_audio(f"{OUT_DIR}/mix.wav", sample.mixture, sr=sample.sr)
    save_audio(f"{OUT_DIR}/target.wav", sample.target, sr=sample.sr)
    save_audio(f"{OUT_DIR}/interferer.wav", sample.interferer, sr=sample.sr)

    # Run the pipeline
    log.info("Loading classifier from %s...", MODEL_PATH)
    classifier = SpeakerClassifier.load(MODEL_PATH)
    attention = AttentionModule(classifier)

    log.info("Computing attention mask...")
    mask = attention.compute_mask(sample.mixture, sr=sample.sr)
    log.info("Dominant gender detected in mix: %s", attention.dominant_gender(sample.mixture, sr=sample.sr))

    log.info("Separating with classifier-guided NMF...")
    reconstructed = separate_nmf(sample.mixture, mask, sr=sample.sr)
    output = enhance(reconstructed, sr=sample.sr)

    save_audio(f"{OUT_DIR}/output.wav", output, sr=sample.sr)

    print("\n" + "=" * 55)
    print("  OUTPUT FILES SAVED TO data/processed/demo/")
    print("=" * 55)
    print("  mix.wav         → original mixture (F + M)")
    print("  target.wav      → female voice (ground truth)")
    print("  interferer.wav  → male voice (ground truth)")
    print("  output.wav      → system output (extracted female voice)")
    print("=" * 55)
    print("\nListen to the files in the order above to evaluate the result.")


if __name__ == "__main__":
    main()
