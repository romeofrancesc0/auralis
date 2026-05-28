"""Attention module: given a mix, decide which frames belong to the target (F)."""
from __future__ import annotations

import numpy as np

from src.ai.classifier import SpeakerClassifier
from src.dsp.features import extract_all, extract_windowed
from src.utils import SAMPLE_RATE

TARGET_GENDER = "F"
TARGET_CLASS_IDX = 0  # index of 'F' in LABEL_MAP

GMM_WEIGHT = 0.4  # weight of GMM probabilities in the blended mask


class AttentionModule:
    """Selective attention: produces a per-frame soft mask for the female speaker.

    The mask values are in [0, 1] where 1 = high confidence target frame.
    Automatically uses windowed feature extraction when the loaded classifier
    was trained with window_size > 1.

    When a GenderGMM is provided, the final mask blends MLP and GMM probabilities:
        mask = (1 - gmm_weight) * mlp_mask + gmm_weight * gmm_proba
    The GMM is trained on CLEAN speech and captures the global acoustic
    distribution of each gender — complementary to the MLP's discriminative output
    which is trained on mixed frames with IBM labels.

    Smoothing (smooth=True, default): 2-state HMM forward-backward enforces
    temporal coherence and reduces mask choppiness.
    """

    def __init__(
        self,
        classifier: SpeakerClassifier,
        gmm=None,
        gmm_weight: float = GMM_WEIGHT,
    ) -> None:
        self.classifier = classifier
        self.gmm        = gmm
        self.gmm_weight = gmm_weight

    def compute_mask(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        smooth: bool = True,
    ) -> np.ndarray:
        """Return per-frame attention weights for the target (F) speaker.

        Args:
            audio:  mono waveform of the mixture, shape (n_samples,)
            sr:     sample rate
            smooth: if True, apply HMM temporal smoothing

        Returns:
            mask: shape (n_frames,), values in [0, 1]
        """
        if self.classifier.window_size > 1:
            X = extract_windowed(audio, sr=sr, window_size=self.classifier.window_size)
        else:
            X = extract_all(audio, sr=sr).T      # (n_frames, N_FEATURES)

        proba = self.classifier.predict_framewise(X)   # (n_frames, 2)
        mask = proba[:, TARGET_CLASS_IDX]              # (n_frames,)

        if self.gmm is not None:
            # GMM uses flat N_FEATURES-dim features, not the windowed representation.
            # Re-extracting avoids passing the windowed MLP input to the GMM.
            X_flat = extract_all(audio, sr=sr).T       # (n_frames, N_FEATURES)
            gmm_proba = self.gmm.score_proba(X_flat)   # (n_frames,)
            n = min(len(mask), len(gmm_proba))
            mask = (1.0 - self.gmm_weight) * mask[:n] + self.gmm_weight * gmm_proba[:n]

        if smooth:
            from src.ai.smoothing import hmm_smooth
            mask = hmm_smooth(mask)

        return mask

    def dominant_gender(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Return the dominant gender in the audio clip ('M' or 'F')."""
        features = extract_all(audio, sr=sr)
        return self.classifier.predict(features)
