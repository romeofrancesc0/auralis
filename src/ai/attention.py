"""Attention module: given a mix, decide which frames belong to the target (F)."""
from __future__ import annotations

import numpy as np

from src.ai.classifier import SpeakerClassifier
from src.dsp.features import extract_all
from src.utils import SAMPLE_RATE

TARGET_GENDER = "F"
TARGET_CLASS_IDX = 0  # index of 'F' in LABEL_MAP


class AttentionModule:
    """Selective attention: produces a per-frame soft mask for the female speaker.

    The mask values are in [0, 1] where 1 = high confidence target frame.
    """

    def __init__(self, classifier: SpeakerClassifier) -> None:
        self.classifier = classifier

    def compute_mask(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Return per-frame attention weights for the target (F) speaker.

        Args:
            audio: mono waveform of the mixture, shape (n_samples,)
            sr: sample rate

        Returns:
            mask: shape (n_frames,), values in [0, 1]
        """
        features = extract_all(audio, sr=sr)                    # (17, n_frames)
        proba = self.classifier.predict_framewise(features)      # (n_frames, 2)
        return proba[:, TARGET_CLASS_IDX]                        # (n_frames,)

    def dominant_gender(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Return the dominant gender in the audio clip ('M' or 'F')."""
        features = extract_all(audio, sr=sr)
        return self.classifier.predict(features)
