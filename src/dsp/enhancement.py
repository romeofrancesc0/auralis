"""Post-processing: noise reduction and normalization after separation."""
from __future__ import annotations

import noisereduce as nr
import numpy as np

from src.utils import SAMPLE_RATE


def reduce_noise(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    stationary: bool = False,
    prop_decrease: float = 0.6,
) -> np.ndarray:
    """Apply spectral noise reduction to the reconstructed signal.

    Args:
        audio: waveform to clean, shape (n_samples,)
        sr: sample rate
        stationary: if True, assumes stationary noise (faster but less adaptive)
        prop_decrease: proportion by which to reduce the noise (0–1)

    Returns:
        denoised waveform, same shape
    """
    return nr.reduce_noise(y=audio, sr=sr, stationary=stationary, prop_decrease=prop_decrease)


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude 1.0."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def enhance(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full enhancement chain: noise reduction → peak normalization."""
    denoised = reduce_noise(audio, sr=sr)
    return peak_normalize(denoised)
