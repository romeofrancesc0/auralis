"""Attention module: top-down stream selection for the separate-then-select pipeline."""
from __future__ import annotations

import librosa
import numpy as np

from src.ai.classifier import SpeakerClassifier
from src.dsp.features import extract_all, extract_windowed
from src.dsp.stft import HOP_LENGTH
from src.utils import SAMPLE_RATE

TARGET_CLASS_IDX = 0  # index of P(female) in classifier output

GMM_WEIGHT = 0.4  # weight of GMM probabilities in the blended score

# Full vocal range for pitch-based stream selection (language-agnostic)
_PITCH_FMIN = 70.0    # bottom of male F0 range
_PITCH_FMAX = 400.0   # top of female F0 range


def _mean_active_pitch(audio: np.ndarray, sr: int) -> float:
    """Mean F0 over voiced frames in the full vocal range (70–400 Hz).

    Language-agnostic: relies only on acoustic F0, not on a trained model.
    Returns 0.0 if no voiced frames are detected.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=_PITCH_FMIN,
        fmax=_PITCH_FMAX,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    voiced = f0[voiced_flag]
    return float(voiced.mean()) if len(voiced) > 0 else 0.0


class AttentionModule:
    """Top-down attentional selection: given two separated streams, picks the target.

    Used in the separate-then-select flow: the DPCRNSeparator segregates the
    auditory scene bottom-up; this module decides which stream to attend to,
    mirroring the selective attention mechanism of the cocktail-party model.

    When a GenderGMM is provided, the female probability blends MLP and GMM:
        score = (1 - gmm_weight) * mlp_score + gmm_weight * gmm_score
    """

    def __init__(
        self,
        classifier: SpeakerClassifier | None,
        gmm=None,
        gmm_weight: float = GMM_WEIGHT,
    ) -> None:
        self.classifier = classifier
        self.gmm        = gmm
        self.gmm_weight = gmm_weight

    def score_female(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """Mean P(female) over active frames — used to rank separated streams.

        Active frames are those whose RMS exceeds 10 % of the clip peak, so
        that silent bins (where the other speaker was dominant) do not drag
        the score toward the classifier's 0.5 prior.
        """
        if self.classifier is None:
            raise ValueError("score_female() requires a trained classifier. "
                             "Use stream_select='pitch' instead.")

        if self.classifier.window_size > 1:
            X = extract_windowed(audio, sr=sr, window_size=self.classifier.window_size)
        else:
            X = extract_all(audio, sr=sr).T       # (n_frames, N_FEATURES)

        proba = self.classifier.predict_framewise(X)
        mask  = proba[:, TARGET_CLASS_IDX]         # P(female), shape (n_frames,)

        if self.gmm is not None:
            X_flat    = extract_all(audio, sr=sr).T
            gmm_proba = self.gmm.score_proba(X_flat)
            n = min(len(mask), len(gmm_proba))
            mask = (1.0 - self.gmm_weight) * mask[:n] + self.gmm_weight * gmm_proba[:n]

        frame_length = 512
        hop          = 128
        n_frames     = len(mask)
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
        method: str = "classifier",
    ) -> tuple[np.ndarray, float]:
        """Pick the separated stream matching the target gender.

        This is the top-down attentional selection step: the separator
        segregates the auditory scene bottom-up; this method decides which
        stream to attend to.

        Args:
            streams:       two separated waveforms (arbitrary permutation)
            target_gender: 0=Female, 1=Male
            sr:            sample rate
            method:        "classifier" — MLP+GMM score (default);
                           "pitch"      — mean F0 comparison (language-agnostic).

        Returns:
            (selected_stream, confidence) — confidence is the absolute score
            difference between the two streams; near 0 = ambiguous selection.
        """
        if method == "pitch":
            pitch_a = _mean_active_pitch(streams[0], sr)
            pitch_b = _mean_active_pitch(streams[1], sr)
            a_is_target = pitch_a >= pitch_b if target_gender == 0 else pitch_a < pitch_b
            selected = streams[0] if a_is_target else streams[1]
            return selected, abs(pitch_a - pitch_b)

        score_a = self.score_female(streams[0], sr=sr)
        score_b = self.score_female(streams[1], sr=sr)
        a_is_target = score_a >= score_b if target_gender == 0 else score_a < score_b
        selected = streams[0] if a_is_target else streams[1]
        return selected, abs(score_a - score_b)
