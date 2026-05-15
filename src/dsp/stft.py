from __future__ import annotations

import librosa
import numpy as np

# Centralised STFT config — all modules must import from here.
# Changing these values affects both analysis and synthesis (STFT/ISTFT must match).
N_FFT = 512
HOP_LENGTH = 128
WINDOW = "hann"


def compute_stft(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
) -> np.ndarray:
    """Return complex STFT matrix of shape (1 + n_fft/2, n_frames)."""
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)


def compute_istft(
    stft_matrix: np.ndarray,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct waveform from complex STFT matrix."""
    return librosa.istft(stft_matrix, hop_length=hop_length, window=window, length=length)
