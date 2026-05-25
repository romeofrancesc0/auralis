"""End-to-end pipeline: mixture → isolated target speaker.

Usage:
    # Baseline (MLP only)
    python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav

    # With GMM blend
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --output out.wav

    # With GRU temporal smoother (replaces HMM)
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --smoothing-gru models/smoothing_gru.pt --output out.wav

    # With MaskNet CNN refinement
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --mask-net models/mask_net.pt --output out.wav

    # With DPCRN refinement (alternative to MaskNet, higher capacity)
    python -m src.pipeline --input mix.wav --model models/classifier.joblib \\
        --gmm models/gender_gmm.joblib --dpcrn models/dpcrn.pt --output out.wav
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
    dpcrn_path: str | None = None,
    smoothing_gru_path: str | None = None,
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

    gru_smoother = None
    if smoothing_gru_path:
        from src.ai.smoothing_gru import GRUSmoother
        logger.info("Loading GRUSmoother: %s", smoothing_gru_path)
        gru_smoother = GRUSmoother.load(smoothing_gru_path)

    # mask_net and dpcrn share the same interface (refine / save / load)
    # Only one is active at a time; --dpcrn takes precedence over --mask-net
    mask_net = None
    if dpcrn_path:
        from src.ai.dpcrn import DPCRN
        logger.info("Loading DPCRN: %s", dpcrn_path)
        mask_net = DPCRN.load(dpcrn_path)
    elif mask_net_path:
        from src.ai.mask_net import MaskNet
        logger.info("Loading MaskNet: %s", mask_net_path)
        mask_net = MaskNet.load(mask_net_path)

    attention = AttentionModule(classifier, gmm=gmm, gru_smoother=gru_smoother)

    logger.info("Computing attention mask...")
    mask = attention.compute_mask(audio, sr=sr)

    refiner_label = ""
    if dpcrn_path:
        refiner_label = " + DPCRN"
    elif mask_net_path:
        refiner_label = " + MaskNet"

    logger.info("Separating target speaker (NMF%s)...", refiner_label)
    reconstructed = separate_nmf(audio, mask, sr=sr, mask_net=mask_net, target_gender=0)

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
    parser.add_argument("--input",  required=True, help="Path to input mixture WAV file.")
    parser.add_argument("--model",  required=True, help="Path to trained classifier (.joblib).")
    parser.add_argument("--gmm",    default=None,
                        help="Path to trained GenderGMM (.joblib). Optional.")
    parser.add_argument("--mask-net", default=None,
                        help="Path to trained MaskNet (.pt). CNN-based IRM refinement.")
    parser.add_argument("--dpcrn", default=None,
                        help="Path to trained DPCRN (.pt). Higher-capacity IRM refinement "
                             "(takes precedence over --mask-net if both supplied).")
    parser.add_argument("--smoothing-gru", default=None,
                        help="Path to trained GRUSmoother (.pt). Replaces HMM smoothing.")
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
        dpcrn_path=args.dpcrn,
        smoothing_gru_path=args.smoothing_gru,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
