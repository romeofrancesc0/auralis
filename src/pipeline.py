"""End-to-end pipeline: mixture → isolated target speaker.

Usage:
    python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
"""
from __future__ import annotations

import argparse
import logging

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.dsp.enhancement import enhance
from src.dsp.nmf_separation import separate_nmf
from src.utils import SAMPLE_RATE, load_audio, save_audio

logger = logging.getLogger(__name__)


def run(input_path: str, model_path: str, output_path: str, sr: int = SAMPLE_RATE) -> None:
    logger.info("Loading audio: %s", input_path)
    audio, sr = load_audio(input_path, sr=sr)

    logger.info("Loading classifier: %s", model_path)
    classifier = SpeakerClassifier.load(model_path)
    attention = AttentionModule(classifier)

    logger.info("Computing attention mask...")
    mask = attention.compute_mask(audio, sr=sr)

    logger.info("Separating target speaker (NMF)...")
    reconstructed = separate_nmf(audio, mask, sr=sr)

    logger.info("Enhancing reconstructed signal...")
    output = enhance(reconstructed, sr=sr)

    logger.info("Saving output: %s", output_path)
    save_audio(output_path, output, sr=sr)
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Cocktail party attention: isolate target speaker from mix.")
    parser.add_argument("--input", required=True, help="Path to input mixture WAV file.")
    parser.add_argument("--model", required=True, help="Path to trained classifier (.joblib).")
    parser.add_argument("--output", required=True, help="Path for the output WAV file.")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate (default 16000).")
    args = parser.parse_args()

    run(input_path=args.input, model_path=args.model, output_path=args.output, sr=args.sr)


if __name__ == "__main__":
    main()
