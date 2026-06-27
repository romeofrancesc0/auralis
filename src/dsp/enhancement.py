"""Post-processing: voice activity gate and peak normalization."""
from __future__ import annotations

import numpy as np

from src.dsp.stft import HOP_LENGTH, compute_stft
from src.utils import SAMPLE_RATE

# VAD gate parameters
GATE_THRESHOLD_DB: float = -40.0  # dB below peak frame RMS → frame treated as silent
GATE_HOLD_MS:      float = 200.0  # ms to stay open after energy drops (protects word endings)
GATE_ATTACK_MS:    float = 10.0   # ms to fully open the gate
GATE_RELEASE_MS:   float = 60.0   # ms to fully close the gate


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
    RMS are treated as silent and attenuated — preventing residual
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

    stft    = compute_stft(audio)
    rms     = np.sqrt((np.abs(stft) ** 2).mean(axis=0) + 1e-12)  # (n_frames,)
    peak_rms = rms.max()
    if peak_rms < 1e-8:
        return audio

    rms_db = 20.0 * np.log10(rms / peak_rms)
    active = rms_db > threshold_db

    hold_frames = max(1, round(hold_ms * sr / (HOP_LENGTH * 1000.0)))
    held        = np.convolve(active.astype(float), np.ones(hold_frames), mode="full")[: len(active)]
    active_held = held > 0.0

    attack_step  = 1.0 / max(1, round(attack_ms  * sr / (HOP_LENGTH * 1000.0)))
    release_step = 1.0 / max(1, round(release_ms * sr / (HOP_LENGTH * 1000.0)))

    gain_frames = np.empty(len(active_held), dtype=np.float64)
    g = 0.0
    for i, act in enumerate(active_held):
        g = min(1.0, g + attack_step) if act else max(0.0, g - release_step)
        gain_frames[i] = g

    frame_centers = np.arange(len(gain_frames)) * HOP_LENGTH + HOP_LENGTH // 2
    gain_samples  = np.interp(
        np.arange(len(audio)), frame_centers, gain_frames,
        left=gain_frames[0], right=gain_frames[-1],
    )
    return (audio * gain_samples).astype(audio.dtype)
