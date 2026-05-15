"""Training script for the gender-based speaker classifier.

Usage:
    python -m src.ai.train --n-samples 400 --out models/classifier.joblib
    python -m src.ai.train --n-samples 400 --snr-db -3 0 3 --out models/classifier.joblib
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.ai.classifier import SpeakerClassifier
from src.dsp.dataset import make_ibm_dataset

logger = logging.getLogger(__name__)


def build_dataset(
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame-level IBM dataset combining multiple SNR conditions.

    Training on varied SNR values makes the classifier robust to different
    mixing conditions encountered at inference time.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for snr in snr_db_list:
        logger.info("Building dataset at SNR=%.1f dB (%d samples)...", snr, n_samples)
        X, y = make_ibm_dataset(n_samples=n_samples, snr_db=snr, clip_duration=clip_duration)
        X_parts.append(X)
        y_parts.append(y)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    logger.info(
        "Combined dataset — X: %s, y: %s (F-dominant=%d, M-dominant=%d)",
        X_all.shape, y_all.shape, (y_all == 0).sum(), (y_all == 1).sum(),
    )
    return X_all, y_all


def train(
    n_samples: int,
    snr_db_list: list[float],
    clip_duration: float,
    out: str,
    cv_folds: int,
) -> None:
    X, y = build_dataset(n_samples, snr_db_list, clip_duration)

    # Hold-out test split for unbiased accuracy estimate (Step 3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42,  # type: ignore[arg-type]
    )
    logger.info("Train frames: %d  |  Test frames: %d", len(X_train), len(X_test))

    clf = SpeakerClassifier()

    logger.info("Cross-validating on training set (%d folds)...", cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        clf.model.__class__(**clf.model.get_params()),
        clf.scaler.fit_transform(X_train), y_train,
        cv=cv, scoring="accuracy",
    )
    logger.info("CV accuracy: %.3f ± %.3f", scores.mean(), scores.std())

    # Final fit on full training split
    clf.fit(X_train, y_train)

    # Test set evaluation
    X_test_scaled = clf.scaler.transform(X_test)
    test_acc = (clf.model.predict(X_test_scaled) == y_test).mean()
    logger.info("Test accuracy: %.3f", test_acc)

    # Retrain on 100% of data before saving (standard practice after eval)
    clf.fit(X, y)
    clf.save(out)
    logger.info("Model saved to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Train the gender speaker classifier.")
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Number of mixture clips per SNR value.")
    parser.add_argument(
        "--snr-db", type=float, nargs="+", default=[-3.0, 0.0, 3.0],
        help="SNR values (dB) for data augmentation. Default: -3 0 3.",
    )
    parser.add_argument("--clip-duration", type=float, default=4.0,
                        help="Duration in seconds of each audio clip.")
    parser.add_argument("--out", type=str, default="models/classifier.joblib",
                        help="Output path for the saved model.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds.")
    args = parser.parse_args()

    train(
        n_samples=args.n_samples,
        snr_db_list=args.snr_db,
        clip_duration=args.clip_duration,
        out=args.out,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
