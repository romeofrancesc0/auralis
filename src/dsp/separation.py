"""Time-frequency masking and signal reconstruction."""
from __future__ import annotations

import librosa
import numpy as np

from src.dsp.stft import HOP_LENGTH, N_FFT, compute_istft, compute_stft
from src.utils import SAMPLE_RATE

# Pitch ranges (Hz) — kept non-overlapping to avoid cross-detection
FEMALE_PITCH_FMIN = 150.0
FEMALE_PITCH_FMAX = 310.0
MALE_PITCH_FMIN = 80.0
MALE_PITCH_FMAX = 145.0  # stays below female range to prevent false positives

HARMONIC_FLOOR = 0.85    # confirmed female harmonic bins always pass at ≥ this weight
MALE_SUPPRESSION = 0.08  # confirmed male harmonic bins always suppressed to ≤ this weight
UNCERTAIN_CAP = 0.25     # broadband ceiling in frames where classifier is near 0.5


def sharpen_mask(mask: np.ndarray, power: float = 2.5) -> np.ndarray:
    """Apply sigmoid sharpening centered at 0.5.

    Values > 0.5 are pushed toward 1.0; values < 0.5 are pushed toward 0.0.
    Higher power = steeper sigmoid = more binary-like mask.
    """
    k = power * 4.0
    return 1.0 / (1.0 + np.exp(-k * (mask - 0.5)))


def compute_pitch_mask(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_harmonics: int = 8,
    bin_radius: int = 1,
    voiced_nonharmonic_penalty: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a harmonic mask that highlights female F0 and its overtones.

    Unvoiced frames → 1.0 everywhere (neutral: let the ratio mask decide).
    Voiced female frames → 1.0 on harmonic bins, voiced_nonharmonic_penalty
    on all other bins (attenuate male-frequency content in those frames).

    Args:
        audio:                     mixture waveform
        sr:                        sample rate
        n_fft:                     FFT size (must match STFT used elsewhere)
        hop_length:                hop length (must match STFT used elsewhere)
        n_harmonics:               how many harmonics to mark per voiced frame
        bin_radius:                number of bins on each side of a harmonic
        voiced_nonharmonic_penalty: weight for non-harmonic bins in voiced frames

    Returns:
        mask:          shape (n_freqs, n_frames), values in [0.1, 1.0]
        harmonic_bins: shape (n_freqs, n_frames), True only on confirmed female
                       harmonic bins (voiced frames, not unvoiced neutrals)
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=FEMALE_PITCH_FMIN,
        fmax=FEMALE_PITCH_FMAX,
        sr=sr,
        hop_length=hop_length,
    )

    n_freqs = n_fft // 2 + 1
    n_frames = len(f0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # (n_freqs,)

    # Neutral (1.0) by default — unvoiced frames pass through the ratio mask unchanged
    mask = np.ones((n_freqs, n_frames), dtype=np.float32)
    # harmonic_bins tracks ONLY confirmed voiced harmonic bins (excludes unvoiced 1.0 neutrals)
    harmonic_bins = np.zeros((n_freqs, n_frames), dtype=bool)

    for t in range(n_frames):
        if not voiced_flag[t] or np.isnan(f0[t]) or f0[t] <= 0:
            continue
        # Voiced female frame: attenuate non-harmonic bins, then restore harmonics
        mask[:, t] = voiced_nonharmonic_penalty
        f_fund = f0[t]
        for h in range(1, n_harmonics + 1):
            harmonic_freq = f_fund * h
            if harmonic_freq >= sr / 2:
                break
            bin_idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
            lo = max(0, bin_idx - bin_radius)
            hi = min(n_freqs, bin_idx + bin_radius + 1)
            mask[lo:hi, t] = 1.0
            harmonic_bins[lo:hi, t] = True

    return mask, harmonic_bins


def compute_male_suppression_mask(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_harmonics: int = 6,
    bin_radius: int = 1,
    suppression: float = 0.1,
) -> np.ndarray:
    """Build a mask that suppresses detected male harmonic bins.

    When male F0 is detected in the mix, the corresponding harmonic bins are
    attenuated to `suppression`. Frames where no male pitch is found are left
    at 1.0 (neutral) so unvoiced or female-only frames are untouched.

    Args:
        audio:       mixture waveform
        sr:          sample rate
        n_fft:       FFT size (must match STFT used elsewhere)
        hop_length:  hop length (must match STFT used elsewhere)
        n_harmonics: number of harmonics to suppress per voiced male frame
        bin_radius:  bins on each side of each harmonic to suppress
        suppression: residual weight for suppressed bins (0 = silence, 0.1 default)

    Returns:
        mask: shape (n_freqs, n_frames), values in [suppression, 1.0]
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=MALE_PITCH_FMIN,
        fmax=MALE_PITCH_FMAX,
        sr=sr,
        hop_length=hop_length,
    )

    n_freqs = n_fft // 2 + 1
    n_frames = len(f0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mask = np.ones((n_freqs, n_frames), dtype=np.float32)

    for t in range(n_frames):
        if not voiced_flag[t] or np.isnan(f0[t]) or f0[t] <= 0:
            continue
        f_fund = f0[t]
        for h in range(1, n_harmonics + 1):
            harmonic_freq = f_fund * h
            if harmonic_freq >= sr / 2:
                break
            bin_idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
            lo = max(0, bin_idx - bin_radius)
            hi = min(n_freqs, bin_idx + bin_radius + 1)
            mask[lo:hi, t] = suppression

    return mask


def compute_ratio_mask(attention_weights: np.ndarray, n_freqs: int) -> np.ndarray:
    """Broadcast per-frame attention weights to shape (n_freqs, n_frames)."""
    return np.tile(attention_weights, (n_freqs, 1))


def apply_mask(stft_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a real-valued mask to a complex STFT matrix."""
    n_frames = stft_matrix.shape[1]
    return stft_matrix * mask[:, :n_frames]


def separate(
    audio: np.ndarray,
    attention_weights: np.ndarray,
    sr: int = SAMPLE_RATE,
    mask_power: float = 4.0,
    pitch_blend: float = 0.7,
) -> np.ndarray:
    """Reconstruct the target (female) speaker from the mixture.

    Pipeline:
        1. Sharpen attention weights via steep sigmoid (frame-level temporal mask)
        2. Broadcast to ratio mask over all frequency bins
        3. Compute female pitch mask — preserves F harmonics, attenuates others
        4. Blend ratio and female pitch masks
        5. Compute male suppression mask — directly attenuates detected M harmonics
        6. Apply combined mask to STFT and reconstruct via ISTFT

    Args:
        audio:             mixture waveform, shape (n_samples,)
        attention_weights: per-frame F-probability from AttentionModule (n_frames,)
        sr:                sample rate
        mask_power:        sigmoid steepness (higher = more binary separation)
        pitch_blend:       weight of the female pitch mask (0=off, 1=full)

    Returns:
        reconstructed waveform, shape (n_samples,)
    """
    stft = compute_stft(audio)                           # (n_freqs, n_frames_stft)
    n_freqs, n_frames_stft = stft.shape

    sharp_weights = sharpen_mask(attention_weights, power=mask_power)
    n_frames = min(len(sharp_weights), n_frames_stft)
    sharp_weights = sharp_weights[:n_frames]
    raw_weights = attention_weights[:n_frames]

    ratio_mask = compute_ratio_mask(sharp_weights, n_freqs)

    pitch_mask, harmonic_bins = compute_pitch_mask(audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    pitch_mask = pitch_mask[:, :n_frames]
    harmonic_bins = harmonic_bins[:, :n_frames]

    male_mask = compute_male_suppression_mask(audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    male_mask = male_mask[:, :n_frames]
    male_harmonic_bins = male_mask < 0.5

    # Step 1 — base: ratio mask blended with female pitch mask
    combined = ratio_mask * ((1.0 - pitch_blend) + pitch_blend * pitch_mask)

    # Step 2 — cap broadband bins in uncertain frames.
    # When the classifier is near 0.5, both voices are present; capping at UNCERTAIN_CAP
    # halves the male broadband leakage without touching bins that pitch will rescue.
    uncertain_frames = (raw_weights >= 0.4) & (raw_weights <= 0.6)
    combined[:, uncertain_frames] = np.minimum(
        combined[:, uncertain_frames], UNCERTAIN_CAP
    )

    # Step 3 — female harmonic floor: confirmed F harmonic bins are raised to at least
    # HARMONIC_FLOOR regardless of classifier uncertainty. Pitch evidence is more
    # reliable than frame-level classification in concurrent-speech frames.
    combined = np.where(harmonic_bins, np.maximum(combined, HARMONIC_FLOOR), combined)

    # Step 4 — male harmonic suppression: applied last so it overrides the female floor
    # in the rare case where both pitch tracks claim the same bin.
    combined = np.where(male_harmonic_bins, np.minimum(combined, MALE_SUPPRESSION), combined)

    masked_stft = apply_mask(stft, combined)
    return compute_istft(masked_stft, length=len(audio))
