"""End-to-end pipeline: mixture → isolated target speaker.

Usage:
    python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --output out.wav
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --mask-net models/mask_net.pt --output out.wav
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


def run(
    input_path: str,
    model_path: str,
    output_path: str,
    gmm_path: str | None = None,
    mask_net_path: str | None = None,
    sr: int = SAMPLE_RATE,
) -> None:
    logger.info("Loading audio: %s", input_path)
    audio, sr = load_audio(input_path, sr=sr)

    logger.info("Loading classifier: %s", model_path)
    classifier = SpeakerClassifier.load(model_path)

    gmm = None
    if gmm_path:
        from src.ai.gmm_classifier import GenderGMM
        logger.info("Loading GenderGMM: %s", gmm_path)
        gmm = GenderGMM.load(gmm_path)

    mask_net = None
    if mask_net_path:
        from src.ai.mask_net import MaskNet
        logger.info("Loading MaskNet: %s", mask_net_path)
        mask_net = MaskNet.load(mask_net_path)

    attention = AttentionModule(classifier, gmm=gmm)

    logger.info("Computing attention mask...")
    mask = attention.compute_mask(audio, sr=sr)

    logger.info("Separating target speaker (NMF%s)...",
                " + MaskNet" if mask_net is not None else "")
    reconstructed = separate_nmf(audio, mask, sr=sr, mask_net=mask_net)

    logger.info("Enhancing reconstructed signal...")
    output = enhance(reconstructed, sr=sr)

    logger.info("Saving output: %s", output_path)
    save_audio(output_path, output, sr=sr)
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Cocktail party attention: isolate target speaker from mix."
    )
    parser.add_argument("--input", required=True, help="Path to input mixture WAV file.")
    parser.add_argument("--model", required=True, help="Path to trained classifier (.joblib).")
    parser.add_argument("--gmm", default=None,
                        help="Path to trained GenderGMM (.joblib). Optional.")
    parser.add_argument("--mask-net", default=None,
                        help="Path to trained MaskNet (.pt). Optional CNN-based IRM refinement.")
    parser.add_argument("--output", required=True, help="Path for the output WAV file.")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE,
                        help="Sample rate (default 16000).")
    args = parser.parse_args()

    run(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        gmm_path=args.gmm,
        mask_net_path=args.mask_net,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
