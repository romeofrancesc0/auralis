"""End-to-end pipeline: mixture -> isolated target speaker (separate-then-select).

The DPCRNSeparator (uPIT) segregates the scene into two streams bottom-up;
the attention module then selects the target stream top-down — mirroring the
cocktail-party model of human auditory attention.

Usage:
    python -m src.pipeline \\
        --input mix.wav \\
        --model models/classifier.joblib \\
        --gmm   models/gender_gmm.joblib \\
        --separator models/separator.pt \\
        --target female \\
        --output out.wav

Language-agnostic (pitch-based stream selection, no trained model required):
    python -m src.pipeline \\
        --input mix.wav \\
        --separator models/separator.pt \\
        --target male \\
        --stream-select pitch \\
        --output out.wav
"""
from __future__ import annotations

import argparse
import logging

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.utils import SAMPLE_RATE, load_audio, save_audio

logger = logging.getLogger(__name__)


def run(
    input_path: str,
    output_path: str,
    separator_path: str,
    model_path: str | None = None,
    gmm_path: str | None = None,
    sr: int = SAMPLE_RATE,
    target: str = "female",
    stream_select: str = "classifier",
    vad_gate: bool = False,
) -> None:
    if stream_select == "classifier" and model_path is None:
        raise ValueError(
            "--model is required when --stream-select classifier (the default). "
            "Use --stream-select pitch for language-agnostic selection without a trained model."
        )

    logger.info("Loading audio: %s", input_path)
    audio, sr = load_audio(input_path, sr=sr)

    classifier = SpeakerClassifier.load(model_path) if model_path else None

    gmm = None
    if gmm_path:
        from src.ai.gmm_classifier import GenderGMM
        logger.info("Loading GenderGMM: %s", gmm_path)
        gmm = GenderGMM.load(gmm_path)

    attention = AttentionModule(classifier, gmm=gmm)
    target_gender = 0 if target == "female" else 1

    from src.ai.dpcrn import DPCRNSeparator
    logger.info("Loading DPCRNSeparator: %s", separator_path)
    separator = DPCRNSeparator.load(separator_path)

    logger.info("Separating both sources (uPIT separator)...")
    streams = separator.separate(audio)

    logger.info("Selecting %s stream (method=%s)...", target.upper(), stream_select)
    reconstructed, confidence = attention.select_stream(
        streams, target_gender=target_gender, sr=sr, method=stream_select
    )
    logger.info("Stream selected (confidence %.3f)", confidence)

    if vad_gate:
        from src.dsp.enhancement import voice_activity_gate
        logger.info("Applying VAD gate...")
        reconstructed = voice_activity_gate(reconstructed, sr=sr)

    logger.info("Saving output: %s", output_path)
    save_audio(output_path, reconstructed, sr=sr)
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Cocktail party attention: isolate target speaker from a two-speaker mix.\n\n"
                    "Recommended: --model + --gmm + --separator (classifier stream selection)\n"
                    "Language-agnostic: --separator + --stream-select pitch (no trained model needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input",     required=True, help="Path to input mixture WAV file.")
    parser.add_argument("--output",    required=True, help="Path for the output WAV file.")
    parser.add_argument("--separator", required=True,
                        help="Path to trained DPCRNSeparator (.pt).")
    parser.add_argument("--model",  default=None,
                        help="Path to trained classifier (.joblib). "
                             "Required for --stream-select classifier (default).")
    parser.add_argument("--gmm",    default=None,
                        help="Path to trained GenderGMM (.joblib). Recommended with --model.")
    parser.add_argument("--target", choices=["female", "male"], default="female",
                        help="Speaker to isolate: 'female' (default) or 'male'.")
    parser.add_argument(
        "--stream-select",
        choices=["classifier", "pitch"],
        default="classifier",
        dest="stream_select",
        help=(
            "'classifier' (default) — uses MLP+GMM scores to pick the target stream; "
            "'pitch' — uses mean F0 comparison, language-agnostic, no trained model needed."
        ),
    )
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE,
                        help="Sample rate (default 16000).")
    parser.add_argument(
        "--vad-gate",
        action="store_true",
        default=False,
        dest="vad_gate",
        help="Apply a voice activity gate to suppress residual bleedthrough during pauses.",
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        separator_path=args.separator,
        model_path=args.model,
        gmm_path=args.gmm,
        sr=args.sr,
        target=args.target,
        stream_select=args.stream_select,
        vad_gate=args.vad_gate,
    )


if __name__ == "__main__":
    main()
