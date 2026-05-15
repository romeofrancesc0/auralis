"""Gender-based speaker classifier (M/F) using features from src.dsp.features."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

LABEL_MAP = {"F": 0, "M": 1}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


class SpeakerClassifier:
    """Wraps a Random Forest that classifies speaker gender from per-frame features.

    Input to predict: feature matrix of shape (n_features, n_frames).
    Features are averaged across frames (segment-level prediction).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on feature matrix X of shape (n_samples, n_features) and int labels y."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

    def predict(self, features: np.ndarray) -> str:
        """Predict gender ('M' or 'F') from a feature matrix (n_features, n_frames).

        Aggregates frame-level features into a single segment vector via mean.
        """
        self._check_fitted()
        x = self._aggregate(features)
        label_int = self.model.predict(x)[0]
        return LABEL_INV[label_int]

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        """Return per-class probability for a single segment."""
        self._check_fitted()
        x = self._aggregate(features)
        proba = self.model.predict_proba(x)[0]
        return {LABEL_INV[i]: float(p) for i, p in enumerate(proba)}

    def predict_framewise(self, features: np.ndarray) -> np.ndarray:
        """Return per-frame gender probabilities, shape (n_frames, n_classes).

        Each frame is classified independently — useful for the attention mask.
        """
        self._check_fitted()
        # features: (n_features, n_frames) → transpose to (n_frames, n_features)
        X = self.scaler.transform(features.T)
        return self.model.predict_proba(X)  # (n_frames, 2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> SpeakerClassifier:
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj._fitted = True
        return obj

    def _aggregate(self, features: np.ndarray) -> np.ndarray:
        """Mean-pool (n_features, n_frames) → (1, n_features)."""
        return features.mean(axis=1, keepdims=True).T

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SpeakerClassifier is not fitted yet. Call fit() or load() first.")
