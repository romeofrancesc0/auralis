from __future__ import annotations

import librosa
import numpy as np

from src.dsp.stft import HOP_LENGTH, N_FFT
from src.utils import SAMPLE_RATE

N_MFCC = 13
# Restricted to female vocal range — makes pitch a strong gender discriminant:
# F-dominant frames: F0 detected (150-310 Hz); M-dominant frames: pyin returns NaN → 0.
PITCH_FMIN = 150.0
PITCH_FMAX = 310.0
N_LPC = 12  # LPC filter order — models the vocal tract as a 12th-order all-pole filter

N_FEATURES = 56  # 13 MFCC + 13 delta + 13 delta² + pitch + rms + centroid + rolloff + ZCR + 12 LPC
WINDOW_SIZE = 11  # default sliding-window width for contextual feature extraction


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return MFCC matrix of shape (n_mfcc, n_frames)."""
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)


def extract_mfcc_delta(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return MFCC delta and delta-delta stacked, shape (2 * n_mfcc, n_frames).

    Temporal derivatives capture vocal dynamics — significantly more discriminant
    than static MFCC for gender classification on mixed signals.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([delta1, delta2])


def extract_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return F0 per frame restricted to female vocal range, shape (1, n_frames).

    F-dominant frames → pitch in [PITCH_FMIN, PITCH_FMAX].
    M-dominant frames → pyin finds nothing in range → 0.
    Unvoiced frames are set to 0 (NaN replaced).
    """
    f0, _, _ = librosa.pyin(
        audio,
        fmin=PITCH_FMIN,
        fmax=PITCH_FMAX,
        sr=sr,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0[np.newaxis, :]


def extract_rms(
    audio: np.ndarray,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return RMS energy per frame, shape (1, n_frames)."""
    return librosa.feature.rms(y=audio, hop_length=hop_length)


def extract_spectral(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return spectral centroid and rolloff stacked, shape (2, n_frames)."""
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
    return np.vstack([centroid, rolloff])


def extract_zcr(
    audio: np.ndarray,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return zero crossing rate per frame, shape (1, n_frames)."""
    return librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)


def extract_lpc(
    audio: np.ndarray,
    hop_length: int = HOP_LENGTH,
    order: int = N_LPC,
) -> np.ndarray:
    """Return LPC coefficients per frame, shape (order, n_frames).

    LPC models the vocal tract as an all-pole filter of the given order.
    The coefficients complement MFCC by capturing formant structure via the
    autocorrelation method (Levinson-Durbin recursion) rather than the cepstrum.
    Each frame is windowed with a Hann window before LPC estimation.

    Returns zeros for audio shorter than the analysis frame (N_FFT samples).
    """
    if len(audio) < N_FFT:
        return np.zeros((order, 0))

    frames = librosa.util.frame(audio, frame_length=N_FFT, hop_length=hop_length)
    # frames: (N_FFT, n_frames)
    window = np.hanning(N_FFT)
    coeffs = np.array([
        np.nan_to_num(
            librosa.lpc(frm * window, order=order)[1:],  # drop a[0] = 1
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        for frm in frames.T
    ])  # (n_frames, order)
    return coeffs.T  # (order, n_frames)


def extract_all(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract and concatenate all features, shape (N_FEATURES, n_frames).

    Feature layout:
        [0:13]   MFCC (13 coefficients)
        [13:26]  MFCC delta
        [26:39]  MFCC delta-delta
        [39]     Pitch / F0 (female range 150-310 Hz)
        [40]     RMS energy
        [41]     Spectral centroid
        [42]     Spectral rolloff
        [43]     Zero crossing rate
        [44:56]  LPC coefficients (order 12)
    """
    mfcc = extract_mfcc(audio, sr)              # (13, n_frames)
    mfcc_delta = extract_mfcc_delta(audio, sr)  # (26, n_frames)
    pitch = extract_pitch(audio, sr)            # (1,  n_frames)
    rms = extract_rms(audio)                    # (1,  n_frames)
    spectral = extract_spectral(audio, sr)      # (2,  n_frames)
    zcr = extract_zcr(audio)                    # (1,  n_frames)
    lpc = extract_lpc(audio)                    # (12, n_frames)

    n_frames = min(
        mfcc.shape[1], mfcc_delta.shape[1],
        pitch.shape[1], rms.shape[1],
        spectral.shape[1], zcr.shape[1],
        lpc.shape[1],
    )
    return np.vstack([
        mfcc[:, :n_frames],
        mfcc_delta[:, :n_frames],
        pitch[:, :n_frames],
        rms[:, :n_frames],
        spectral[:, :n_frames],
        zcr[:, :n_frames],
        lpc[:, :n_frames],
    ])


def apply_window(features: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Apply a sliding context window to a feature matrix.

    Args:
        features: (N_FEATURES, n_frames) — per-frame feature matrix.
        window_size: number of consecutive frames to concatenate per sample.
            Must be odd so the window is symmetric around the centre frame.

    Returns:
        (n_frames, N_FEATURES * window_size) — one row per frame, each row
        contains the flattened features of the surrounding window.
        Edge frames are replicated (not zero-padded) to avoid boundary artefacts.
    """
    n_features, n_frames = features.shape
    half = window_size // 2
    padded = np.concatenate([
        np.repeat(features[:, :1], half, axis=1),
        features,
        np.repeat(features[:, -1:], half, axis=1),
    ], axis=1)  # (n_features, n_frames + window_size - 1)
    return np.stack([padded[:, t:t + window_size].ravel() for t in range(n_frames)])


def extract_windowed(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Extract per-frame features and apply a sliding context window.

    Returns:
        (n_frames, N_FEATURES * window_size)
    """
    return apply_window(extract_all(audio, sr=sr), window_size)
