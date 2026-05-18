"""Gender-based speaker classifier (M/F) using features from src.dsp.features."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.dsp.features import N_FEATURES, WINDOW_SIZE, apply_window

LABEL_MAP = {"F": 0, "M": 1}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


class SpeakerClassifier:
    """MLP classifier for frame-level speaker gender detection.

    Supports an optional sliding context window: when window_size > 1 each
    training/inference sample is the concatenation of window_size consecutive
    feature frames, giving the model temporal context.

    Input to predict/predict_proba: feature matrix (N_FEATURES, n_frames).
    Input to predict_framewise: already-prepared (n_frames, n_input_features).
    """

    def __init__(self, window_size: int = 1) -> None:
        self.window_size = window_size
        self.n_input_features = N_FEATURES * window_size
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on feature matrix X of shape (n_samples, n_input_features) and int labels y."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

    def predict(self, features: np.ndarray) -> str:
        """Predict gender ('M' or 'F') from a feature matrix (N_FEATURES, n_frames).

        Predicts on every frame (with windowing if applicable) and returns
        the majority-vote gender label.
        """
        self._check_fitted()
        X = self._prepare(features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            proba = self.model.predict_proba(self.scaler.transform(X)).mean(axis=0)
        return LABEL_INV[int(np.argmax(proba))]

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        """Return mean per-class probability over all frames of a segment."""
        self._check_fitted()
        X = self._prepare(features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            proba = self.model.predict_proba(self.scaler.transform(X)).mean(axis=0)
        return {LABEL_INV[i]: float(p) for i, p in enumerate(proba)}

    def predict_framewise(self, X: np.ndarray) -> np.ndarray:
        """Return per-frame gender probabilities, shape (n_frames, n_classes).

        X must already be in sklearn convention: (n_frames, n_input_features).
        Use attention.compute_mask() for the full audio → mask flow.
        """
        self._check_fitted()
        # MLP forward pass emits IEEE-754 RuntimeWarning on subnormal BLAS inputs —
        # false positive; output is numerically correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.model.predict_proba(self.scaler.transform(X))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self.scaler, "model": self.model, "window_size": self.window_size},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SpeakerClassifier:
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.window_size = data.get("window_size", 1)
        obj.n_input_features = N_FEATURES * obj.window_size
        obj._fitted = True
        return obj

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        """Convert (N_FEATURES, n_frames) to (n_frames, n_input_features)."""
        if self.window_size > 1:
            return apply_window(features, self.window_size)
        return features.T

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SpeakerClassifier is not fitted yet. Call fit() or load() first.")
