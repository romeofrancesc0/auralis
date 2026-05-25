"""Train GenderGMM on clean LibriSpeech speech features.

Usage:
    python -m src.ai.train_gmm [--n-clips N] [--n-components K] [--out PATH]

The GMM is trained on CLEAN (unmixed) speech — not on mixtures — so it
models the marginal acoustic distribution of each gender independently.
At inference, the log-likelihood ratio log P(X|GMM_F) − log P(X|GMM_M)
provides a complementary signal to the MLP classifier.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from src.ai.gmm_classifier import GenderGMM
from src.dsp.dataset import LIBRISPEECH_ROOT, load_speaker_index
from src.dsp.features import extract_all
from src.utils import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)

DEFAULT_OUT = "models/gender_gmm.joblib"
DEFAULT_N_CLIPS = 50
CLIP_DURATION = 4.0


def _collect_features(
    speakers,
    n_clips: int,
    clip_duration: float,
    sr: int,
    seed: int = 42,
) -> np.ndarray:
    """Extract flat N_FEATURES-dim features from n_clips clean audio files.

    Returns:
        X: (n_frames_total, N_FEATURES)
    """
    rng = random.Random(seed)
    n_samples = int(sr * clip_duration)
    X_list: list[np.ndarray] = []

    all_files = [f for spk in speakers for f in spk.audio_files]
    if not all_files:
        raise ValueError("No audio files found for the given speakers.")

    rng.shuffle(all_files)
    chosen = all_files[:n_clips]

    for path in chosen:
        audio, _ = load_audio(path, sr=sr)
        if len(audio) >= n_samples:
            audio = audio[:n_samples]
        else:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        feat = extract_all(audio, sr=sr)   # (N_FEATURES, n_frames)
        X_list.append(feat.T)              # → (n_frames, N_FEATURES)

    return np.vstack(X_list)


def train_gmm(
    n_clips: int = DEFAULT_N_CLIPS,
    out_path: str = DEFAULT_OUT,
    subset: str = "dev-clean",
    root: Path = LIBRISPEECH_ROOT,
    sr: int = SAMPLE_RATE,
    n_components: int = 16,
    seed: int = 42,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    index = load_speaker_index(subset=subset, root=root)
    female_speakers = index["F"]
    male_speakers = index["M"]

    logger.info("Extracting features from %d clean female clips...", n_clips)
    X_female = _collect_features(female_speakers, n_clips, CLIP_DURATION, sr, seed=seed)
    logger.info("  X_female: %s frames", X_female.shape[0])

    logger.info("Extracting features from %d clean male clips...", n_clips)
    X_male = _collect_features(male_speakers, n_clips, CLIP_DURATION, sr, seed=seed + 1)
    logger.info("  X_male: %s frames", X_male.shape[0])

    logger.info("Fitting GenderGMM (n_components=%d)...", n_components)
    gmm = GenderGMM(n_components=n_components)
    gmm.fit(X_female, X_male)
    logger.info("  LLR calibration: mean=%.4f  std=%.4f", gmm._llr_mean, gmm._llr_std)

    gmm.save(out_path)
    logger.info("GenderGMM saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GenderGMM on clean LibriSpeech features."
    )
    parser.add_argument(
        "--n-clips", type=int, default=DEFAULT_N_CLIPS,
        help=f"Number of clean clips per gender (default {DEFAULT_N_CLIPS}).",
    )
    parser.add_argument(
        "--n-components", type=int, default=16,
        help="Number of GMM Gaussian components per class (default 16).",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output path for the trained GMM (default {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    train_gmm(n_clips=args.n_clips, out_path=args.out, n_components=args.n_components)


if __name__ == "__main__":
    main()
