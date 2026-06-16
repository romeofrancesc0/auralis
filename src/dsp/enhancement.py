"""Post-processing: log-MMSE speech enhancement and normalization.

Implements the log-MMSE spectral amplitude estimator (Ephraim & Malah 1985)
with decision-directed a priori SNR estimation (Ephraim & Malah 1984) and
minimum-statistics noise PSD tracking (Martin 2001).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.special import expn

from src.dsp.stft import HOP_LENGTH, compute_istft, compute_stft
from src.utils import SAMPLE_RATE

# VAD gate parameters
GATE_THRESHOLD_DB: float = -40.0  # dB below peak frame RMS → frame treated as silent
GATE_HOLD_MS:      float = 200.0  # ms to stay open after energy drops (protects word endings)
GATE_ATTACK_MS:    float = 10.0   # ms to fully open the gate
GATE_RELEASE_MS:   float = 60.0   # ms to fully close the gate

# Decision-directed smoothing factor α (Ephraim & Malah 1984; typical range 0.92–0.99)
DD_ALPHA: float = 0.98
# Hard floor on the gain — prevents complete spectral suppression of residual signal
GAIN_FLOOR: float = 0.01
# Bias correction for the minimum-statistics noise estimator (Martin 2001)
NOISE_BIAS: float = 1.5
# Sliding window length (frames) for minimum-statistics noise tracking
NOISE_WINDOW: int = 15
# Minimum a posteriori SNR — prevents over-suppression of stationary target signals.
# Without this floor, a stationary target (power ≈ noise estimate) yields γ < 1 → ξ → 0
# → gain → GAIN_FLOOR, effectively suppressing the signal rather than the noise.
GAMMA_MIN: float = 2.0


def _estimate_noise_psd(power: np.ndarray) -> np.ndarray:
    """Minimum-statistics noise PSD estimate (Martin 2001, simplified).

    Computes the sliding minimum of the power spectrogram along the time axis
    and applies a bias correction factor to compensate for the expected
    underestimation of the true noise floor.  Non-causal (offline) computation.

    Args:
        power: (n_freq, n_frames) instantaneous power spectrogram.

    Returns:
        (n_freq, n_frames) estimated noise power spectrum, lower-bounded at 1e-12.
    """
    min_power = minimum_filter1d(power, size=NOISE_WINDOW, axis=1, mode="nearest")
    return np.maximum(min_power * NOISE_BIAS, 1e-12)


def _log_mmse_gain(xi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Log-MMSE spectral amplitude gain (Ephraim & Malah 1985, Eq. 14).

    G(ξ, γ) = ξ/(1+ξ) · exp(½ · E₁(ν)),   ν = ξγ/(1+ξ)

    where E₁ is the exponential integral of order 1 (scipy.special.expn(1, ·)).
    Clipped to [GAIN_FLOOR, 1.0] for numerical stability.
    """
    xi = np.maximum(xi, 1e-10)
    nu = np.clip(xi * gamma / (1.0 + xi), 1e-10, 500.0)
    gain = (xi / (1.0 + xi)) * np.exp(0.5 * expn(1, nu))
    return np.clip(gain, GAIN_FLOOR, 1.0)


def mmse_stsa_enhance(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    alpha: float = DD_ALPHA,
) -> np.ndarray:
    """Log-MMSE speech enhancement with decision-directed a priori SNR estimation.

    Pipeline:
        1. STFT analysis → power spectrogram.
        2. Noise PSD estimation via minimum statistics.
        3. A posteriori SNR: γ[t] = |Y[t]|² / σ_n[t].
        4. Decision-directed a priori SNR (Ephraim & Malah 1984, Eq. 22):
               ξ[t] = α · G[t-1]² · γ[t-1]  +  (1-α) · max(γ[t]-1, 0)
        5. Log-MMSE gain G[t] applied per T-F bin.
        6. ISTFT synthesis with original phase (phase-unchanged estimator).

    Args:
        audio: input waveform, shape (n_samples,).
        sr: sample rate (unused — kept for API consistency with enhance()).
        alpha: decision-directed smoothing factor.

    Returns:
        Enhanced waveform, same length as input.
    """
    if audio.size == 0:
        return audio

    stft = compute_stft(audio)                        # (n_freq, n_frames), complex
    magnitude = np.abs(stft)
    power = magnitude ** 2

    sigma_n = _estimate_noise_psd(power)              # (n_freq, n_frames)
    # Clip gamma to GAMMA_MIN: when sigma_n overestimates the noise (e.g. stationary
    # target signal where min-stats tracks the signal itself), raw gamma < 1 would
    # push xi → 0 → gain → GAIN_FLOOR, suppressing the target instead of the noise.
    gamma = np.maximum(power / sigma_n, GAMMA_MIN)   # a posteriori SNR

    n_freq, n_frames = power.shape
    gain = np.empty_like(magnitude)

    # t = 0: bootstrap a priori SNR with the maximum-likelihood estimate
    xi = np.maximum(gamma[:, 0] - 1.0, 0.0)
    gain[:, 0] = _log_mmse_gain(xi, gamma[:, 0])

    for t in range(1, n_frames):
        # Decision-directed update — tracks a priori SNR across frames
        xi = (
            alpha * (gain[:, t - 1] ** 2) * gamma[:, t - 1]
            + (1.0 - alpha) * np.maximum(gamma[:, t] - 1.0, 0.0)
        )
        gain[:, t] = _log_mmse_gain(xi, gamma[:, t])

    enhanced_stft = gain * magnitude * np.exp(1j * np.angle(stft))
    return compute_istft(enhanced_stft, length=len(audio))


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude 1.0."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def voice_activity_gate(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold_db: float = GATE_THRESHOLD_DB,
    hold_ms: float = GATE_HOLD_MS,
    attack_ms: float = GATE_ATTACK_MS,
    release_ms: float = GATE_RELEASE_MS,
) -> np.ndarray:
    """Suppress bleedthrough during target speaker pauses.

    Applies a smooth amplitude gate keyed on per-frame RMS energy.
    Frames whose RMS falls more than threshold_db below the clip's peak
    RMS are considered silent and are attenuated — preventing residual
    interferer voice from emerging during target speaker inter-word gaps.

    Attack, release, and hold times avoid audible clicks and protect
    brief inter-phoneme gaps from being incorrectly muted.

    Args:
        audio:        mono waveform, shape (n_samples,)
        sr:           sample rate
        threshold_db: silence threshold in dB below peak frame RMS
        hold_ms:      keep gate open this long after energy drops (ms)
        attack_ms:    gate opening time (ms)
        release_ms:   gate closing time (ms)

    Returns:
        Gated waveform, same shape and dtype as input.
    """
    if audio.size == 0:
        return audio

    # Per-frame RMS from STFT magnitude — frame grid consistent with the pipeline
    stft = compute_stft(audio)
    rms = np.sqrt((np.abs(stft) ** 2).mean(axis=0) + 1e-12)  # (n_frames,)

    peak_rms = rms.max()
    if peak_rms < 1e-8:
        return audio

    rms_db = 20.0 * np.log10(rms / peak_rms)
    active = rms_db > threshold_db                            # (n_frames,) bool

    # Hold: after energy drops, keep gate open for hold_frames more frames
    # so that word endings (low-energy consonants) are not cut off.
    hold_frames = max(1, round(hold_ms * sr / (HOP_LENGTH * 1000.0)))
    held = np.convolve(active.astype(float), np.ones(hold_frames), mode="full")[: len(active)]
    active_held = held > 0.0

    # Smoothed gain: separate attack and release slopes to avoid clicks
    attack_step  = 1.0 / max(1, round(attack_ms  * sr / (HOP_LENGTH * 1000.0)))
    release_step = 1.0 / max(1, round(release_ms * sr / (HOP_LENGTH * 1000.0)))

    gain_frames = np.empty(len(active_held), dtype=np.float64)
    g = 0.0
    for i, act in enumerate(active_held):
        g = min(1.0, g + attack_step) if act else max(0.0, g - release_step)
        gain_frames[i] = g

    # Interpolate frame-level gain to sample-level for smooth transitions
    frame_centers = np.arange(len(gain_frames)) * HOP_LENGTH + HOP_LENGTH // 2
    gain_samples = np.interp(
        np.arange(len(audio)), frame_centers, gain_frames,
        left=gain_frames[0], right=gain_frames[-1],
    )

    return (audio * gain_samples).astype(audio.dtype)


def enhance(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full enhancement chain: log-MMSE noise reduction → peak normalization."""
    denoised = mmse_stsa_enhance(audio, sr=sr)
    return peak_normalize(denoised)
