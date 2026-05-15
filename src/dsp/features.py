from __future__ import annotations

import librosa
import numpy as np

from src.dsp.stft import HOP_LENGTH
from src.utils import SAMPLE_RATE

N_MFCC = 13
# Restricted to female vocal range — makes pitch a strong gender discriminant:
# F-dominant frames: F0 detected (150-310 Hz); M-dominant frames: pyin returns NaN → 0.
PITCH_FMIN = 150.0
PITCH_FMAX = 310.0

N_FEATURES = 44  # 13 MFCC + 13 delta + 13 delta² + pitch + rms + centroid + rolloff + ZCR


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
    """
    mfcc = extract_mfcc(audio, sr)           # (13, n_frames)
    mfcc_delta = extract_mfcc_delta(audio, sr)  # (26, n_frames)
    pitch = extract_pitch(audio, sr)          # (1,  n_frames)
    rms = extract_rms(audio)                 # (1,  n_frames)
    spectral = extract_spectral(audio, sr)    # (2,  n_frames)
    zcr = extract_zcr(audio)                 # (1,  n_frames)

    n_frames = min(
        mfcc.shape[1], mfcc_delta.shape[1],
        pitch.shape[1], rms.shape[1],
        spectral.shape[1], zcr.shape[1],
    )
    return np.vstack([
        mfcc[:, :n_frames],
        mfcc_delta[:, :n_frames],
        pitch[:, :n_frames],
        rms[:, :n_frames],
        spectral[:, :n_frames],
        zcr[:, :n_frames],
    ])
