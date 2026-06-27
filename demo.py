"""Quick demo: generate a M+F mix from LibriSpeech, run the pipeline, save outputs.

Uses the separate-then-select flow: DPCRNSeparator (uPIT) segregates the scene
into two streams; the attention module selects female and male targets separately.

Usage:
    python demo.py

Output files in data/processed/demo/:
    mix.wav            -> original mixture (female + male)
    female_voice.wav   -> ground truth female voice
    male_voice.wav     -> ground truth male voice
    output_female.wav  -> system output (isolated female)
    output_male.wav    -> system output (isolated male)
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.ai.attention import AttentionModule
from src.ai.classifier import SpeakerClassifier
from src.ai.dpcrn import DPCRNSeparator
from src.dsp.dataset import make_samples
from src.utils import save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH     = "models/classifier.joblib"
GMM_PATH       = "models/gender_gmm.joblib"
SEPARATOR_PATH = "models/separator.pt"
OUT_DIR        = "data/processed/demo"


def main() -> None:
    log.info("Generating a test sample from LibriSpeech...")
    samples = make_samples(n_samples=1, clip_duration=5.0, seed=7)
    sample  = samples[0]
    log.info("Target (F): %s  |  Interferer (M): %s",
             sample.target_speaker_id, sample.interferer_speaker_id)

    save_audio(f"{OUT_DIR}/mix.wav",          sample.mixture,    sr=sample.sr)
    save_audio(f"{OUT_DIR}/female_voice.wav", sample.target,     sr=sample.sr)
    save_audio(f"{OUT_DIR}/male_voice.wav",   sample.interferer, sr=sample.sr)

    classifier = SpeakerClassifier.load(MODEL_PATH)

    gmm = None
    if Path(GMM_PATH).exists():
        from src.ai.gmm_classifier import GenderGMM
        gmm = GenderGMM.load(GMM_PATH)
    else:
        log.warning("GenderGMM not found at %s — running without GMM blend.", GMM_PATH)

    attention = AttentionModule(classifier, gmm=gmm)

    log.info("Loading DPCRNSeparator: %s", SEPARATOR_PATH)
    separator = DPCRNSeparator.load(SEPARATOR_PATH)

    log.info("Separating both sources (uPIT)...")
    streams = separator.separate(sample.mixture)

    log.info("Selecting female stream...")
    output_f, conf_f = attention.select_stream(streams, target_gender=0, sr=sample.sr)
    log.info("Female stream selected (confidence %.3f)", conf_f)
    save_audio(f"{OUT_DIR}/output_female.wav", output_f, sr=sample.sr)

    log.info("Selecting male stream...")
    output_m, conf_m = attention.select_stream(streams, target_gender=1, sr=sample.sr)
    log.info("Male stream selected (confidence %.3f)", conf_m)
    save_audio(f"{OUT_DIR}/output_male.wav", output_m, sr=sample.sr)

    print()
    print("=" * 55)
    print("  OUTPUT FILES SAVED TO data/processed/demo/")
    print("=" * 55)
    print("  mix.wav            -> original mixture (F + M)")
    print("  female_voice.wav   -> female voice (ground truth)")
    print("  male_voice.wav     -> male voice (ground truth)")
    print("  output_female.wav  -> system output (isolated female)")
    print("  output_male.wav    -> system output (isolated male)")
    print("=" * 55)


if __name__ == "__main__":
    main()
