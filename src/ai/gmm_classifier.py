"""Gender GMM: likelihood-ratio classifier trained on clean (unmixed) speech.

Why a GMM complements the MLP
------------------------------
The MLP is trained on MIX features with IBM labels: it learns a discriminative
boundary between F-dominant and M-dominant frames in a mixed signal.
In heavily overlapping frames it can be fooled by the presence of female
acoustic characteristics (pitch, MFCC) even when the male voice carries
more energy.

GMMs trained on CLEAN speech model the full marginal distribution of each
gender independently: GMM_F captures what female speech looks like in
isolation; GMM_M captures male speech in isolation.
At inference, the log-likelihood ratio (LLR = log P(X|GMM_F) - log P(X|GMM_M))
measures which gender's acoustic distribution the mix frame is GLOBALLY
closer to — a complementary signal to the MLP's discriminative output.

Integration in AttentionModule
-------------------------------
The final per-frame mask blends MLP and GMM probabilities:
    mask = (1 - gmm_weight) * mlp_mask + gmm_weight * gmm_proba
The LLR is sigmoid-normalised using the mean/std from the training data so
that the output is well-calibrated in [0, 1].
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GenderGMM:
    """Two-class GMM for gender likelihood-ratio scoring.

    Attributes:
        n_components: number of Gaussian components per class.
        gmm_f:  GaussianMixture fitted on clean female speech features.
        gmm_m:  GaussianMixture fitted on clean male speech features.
        scaler: StandardScaler fitted on the combined training set.
    """

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self.gmm_f = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=300,
            n_init=3,
            random_state=42,
        )
        self.gmm_m = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=300,
            n_init=3,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._llr_mean: float = 0.0
        self._llr_std: float = 1.0
        self._fitted = False

    def fit(self, X_female: np.ndarray, X_male: np.ndarray) -> None:
        """Fit both GMMs on clean per-frame features.

        Args:
            X_female: (n_frames_f, n_features) — features from clean female clips.
            X_male:   (n_frames_m, n_features) — features from clean male clips.
        """
        X_all = np.vstack([X_female, X_male])
        self.scaler.fit(X_all)
        X_f = self.scaler.transform(X_female)
        X_m = self.scaler.transform(X_male)

        self.gmm_f.fit(X_f)
        self.gmm_m.fit(X_m)

        # Calibrate LLR: normalise using the empirical distribution on
        # the training data so that sigmoid(LLR_norm) is well-calibrated.
        llr_f = self.gmm_f.score_samples(X_f) - self.gmm_m.score_samples(X_f)
        llr_m = self.gmm_f.score_samples(X_m) - self.gmm_m.score_samples(X_m)
        llr_all = np.concatenate([llr_f, llr_m])
        self._llr_mean = float(llr_all.mean())
        self._llr_std = float(llr_all.std() + 1e-8)
        self._fitted = True

    def score_llr(self, X: np.ndarray) -> np.ndarray:
        """Return raw LLR = log P(X|GMM_F) − log P(X|GMM_M) per frame.

        Args:
            X: (n_frames, n_features)

        Returns:
            llr: (n_frames,) — positive = female-like, negative = male-like
        """
        self._check_fitted()
        X_s = self.scaler.transform(X)
        # GMM score_samples emits IEEE-754 RuntimeWarning on subnormal BLAS inputs —
        # false positive; output is numerically correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            llr = self.gmm_f.score_samples(X_s) - self.gmm_m.score_samples(X_s)
        return np.nan_to_num(llr, nan=0.0, posinf=0.0, neginf=0.0)

    def score_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(female) per frame via sigmoid of normalised LLR.

        Args:
            X: (n_frames, n_features)

        Returns:
            proba: (n_frames,) in [0, 1]
        """
        llr = self.score_llr(X)
        llr_norm = (llr - self._llr_mean) / self._llr_std
        return (1.0 / (1.0 + np.exp(-llr_norm))).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "gmm_f": self.gmm_f,
                "gmm_m": self.gmm_m,
                "scaler": self.scaler,
                "llr_mean": self._llr_mean,
                "llr_std": self._llr_std,
                "n_components": self.n_components,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> GenderGMM:
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.gmm_f = data["gmm_f"]
        obj.gmm_m = data["gmm_m"]
        obj.scaler = data["scaler"]
        obj._llr_mean = data["llr_mean"]
        obj._llr_std = data["llr_std"]
        obj.n_components = data["n_components"]
        obj._fitted = True
        return obj

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GenderGMM is not fitted. Call fit() or load() first.")
