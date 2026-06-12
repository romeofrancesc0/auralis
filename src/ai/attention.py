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
        target_gender: int = 0,
    ) -> np.ndarray:
        """Return per-frame attention weights for the target speaker.

        Args:
            audio:         mono waveform of the mixture, shape (n_samples,)
            sr:            sample rate
            smooth:        if True, apply HMM temporal smoothing
            target_gender: 0=Female (default), 1=Male. When 1, the mask is
                           inverted before HMM smoothing so the HMM temporal
                           bias operates in the correct direction for male targets.

        Returns:
            mask: shape (n_frames,), values in [0, 1] — P(target frame)
        """
        if self.classifier.window_size > 1:
            X = extract_windowed(audio, sr=sr, window_size=self.classifier.window_size)
        else:
            X = extract_all(audio, sr=sr).T      # (n_frames, N_FEATURES)

        proba = self.classifier.predict_framewise(X)   # (n_frames, 2)
        mask = proba[:, TARGET_CLASS_IDX]              # (n_frames,) — P(female)

        if self.gmm is not None:
            # GMM uses flat N_FEATURES-dim features, not the windowed representation.
            # Re-extracting avoids passing the windowed MLP input to the GMM.
            X_flat = extract_all(audio, sr=sr).T       # (n_frames, N_FEATURES)
            gmm_proba = self.gmm.score_proba(X_flat)   # (n_frames,)
            n = min(len(mask), len(gmm_proba))
            mask = (1.0 - self.gmm_weight) * mask[:n] + self.gmm_weight * gmm_proba[:n]

        # Invert before HMM so temporal smoothing biases toward the correct target.
        # hmm_smooth(1-P(female)) gives P(male) with male-biased inertia,
        # which is symmetric to hmm_smooth(P(female)) for the female case.
        if target_gender == 1:
            mask = 1.0 - mask

        if smooth:
            from src.ai.smoothing import hmm_smooth
            mask = hmm_smooth(mask)

        return mask

    def dominant_gender(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Return the dominant gender in the audio clip ('M' or 'F')."""
        features = extract_all(audio, sr=sr)
        return self.classifier.predict(features)

    def score_female(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """Mean P(female) over the clip — used to rank separated streams.

        Frames with negligible energy are excluded: a separated stream is
        mostly silence wherever the other speaker was active, and silent
        frames would drag the score toward the classifier's 0.5 prior.
        """
        mask = self.compute_mask(audio, sr=sr, smooth=False, target_gender=0)

        frame_length = 512
        hop = 128
        n_frames = len(mask)
        rms = np.array([
            np.sqrt(np.mean(audio[t * hop : t * hop + frame_length] ** 2))
            for t in range(n_frames)
        ])
        active = rms > max(rms.max() * 0.1, 1e-6)
        if active.sum() < 3:
            return float(mask.mean())
        return float(mask[active].mean())

    def select_stream(
        self,
        streams: tuple[np.ndarray, np.ndarray],
        target_gender: int,
        sr: int = SAMPLE_RATE,
    ) -> tuple[np.ndarray, float]:
        """Pick the separated stream matching the target gender.

        This is the top-down attentional selection step of the cocktail-party
        model: the separator segregates the auditory scene bottom-up, and the
        classifier-based attention decides which stream to attend to.

        Args:
            streams:       two separated waveforms (arbitrary permutation)
            target_gender: 0=Female, 1=Male
            sr:            sample rate

        Returns:
            (selected_stream, confidence) — confidence is |score_a − score_b|,
            useful for diagnostics (near 0 = ambiguous selection).
        """
        score_a = self.score_female(streams[0], sr=sr)
        score_b = self.score_female(streams[1], sr=sr)
        a_is_target = score_a >= score_b if target_gender == 0 else score_a < score_b
        selected = streams[0] if a_is_target else streams[1]
        return selected, abs(score_a - score_b)
