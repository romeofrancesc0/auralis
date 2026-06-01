"""End-to-end pipeline: mixture -> isolated target speaker.

Recommended usage (best perceptual quality — MLP + GMM + HMM + DPCRN):
    python -m src.pipeline --input mix.wav \\
        --model models/classifier.joblib \\
        --gmm   models/gender_gmm.joblib \\
        --dpcrn models/dpcrn.pt \\
        --output out.wav

Isolate the male speaker instead:
    python -m src.pipeline --input mix.wav \\
        --model models/classifier.joblib \\
        --gmm   models/gender_gmm.joblib \\
        --dpcrn models/dpcrn.pt \\
        --target male \\
        --output out_male.wav

Experimental alternatives (research / ablation only):
    # MaskNet instead of DPCRN (lighter, slightly lower quality)
    python -m src.pipeline --input mix.wav \\
        --model models/classifier.joblib --gmm models/gender_gmm.joblib \\
        --mask-net models/mask_net.pt --output out.wav

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
    sr: int = SAMPLE_RATE,
    target: str = "female",
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

    attention = AttentionModule(classifier, gmm=gmm)

    logger.info("Computing attention mask...")
    mask = attention.compute_mask(audio, sr=sr)

    target_gender = 0 if target == "female" else 1

    refiner_label = ""
    if dpcrn_path:
        refiner_label = " + DPCRN"
    elif mask_net_path:
        refiner_label = " + MaskNet"

    logger.info("Separating target speaker: %s (NMF%s)...", target.upper(), refiner_label)
    reconstructed = separate_nmf(audio, mask, sr=sr, mask_net=mask_net, target_gender=target_gender)

    logger.info("Enhancing reconstructed signal...")
    output = enhance(reconstructed, sr=sr)

    logger.info("Saving output: %s", output_path)
    save_audio(output_path, output, sr=sr)
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Cocktail party attention: isolate target speaker from mix.\n\n"
                    "Recommended (best quality): --model + --gmm + --mask-net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    main = parser.add_argument_group("main options")
    main.add_argument("--input",  required=True, help="Path to input mixture WAV file.")
    main.add_argument("--output", required=True, help="Path for the output WAV file.")
    main.add_argument("--model",  required=True, help="Path to trained classifier (.joblib).")
    main.add_argument("--gmm",    default=None,
                      help="Path to trained GenderGMM (.joblib). Recommended.")
    main.add_argument("--dpcrn", default=None,
                      help="Path to trained DPCRN (.pt). Recommended CNN refiner (best perceptual quality).")
    main.add_argument("--target", choices=["female", "male"], default="female",
                      help="Speaker to isolate: 'female' (default) or 'male'.")
    main.add_argument("--sr", type=int, default=SAMPLE_RATE,
                      help="Sample rate (default 16000).")

    exp = parser.add_argument_group(
        "experimental options",
        "These variants are not better than the recommended config in the current evaluation.\n"
        "Kept for research and ablation studies.",
    )
    exp.add_argument("--mask-net", default=None,
                     help="Path to trained MaskNet (.pt). Lighter alternative to DPCRN; "
                          "ignored if --dpcrn is also supplied.")
    args = parser.parse_args()

    run(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        gmm_path=args.gmm,
        mask_net_path=args.mask_net,
        dpcrn_path=args.dpcrn,
        sr=args.sr,
        target=args.target,
    )


if __name__ == "__main__":
    main()
