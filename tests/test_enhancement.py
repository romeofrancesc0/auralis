"""Tests for the voice activity gate.

Kept from the removed v0.4.0 pipeline tests: the gate itself stays, earmarked in
docs/research/unknown-n-state-of-the-art.md as a reusable asset for the
termination-criterion work (G2), where a VAD is one of the candidate stop signals
for recursive extraction.
"""
from __future__ import annotations

import numpy as np

from src.dsp.enhancement import voice_activity_gate
from src.utils import SAMPLE_RATE


def _sine(freq: float, duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _speech_with_bleedthrough_pause(sr: int = SAMPLE_RATE) -> np.ndarray:
    """0.5s speech (250 Hz) + 0.8s near-silence (120 Hz at −60 dB) + 0.5s speech."""
    speech       = _sine(250.0, duration=0.5, sr=sr)
    bleedthrough = _sine(120.0, duration=0.8, sr=sr) * 1e-3
    speech2      = _sine(250.0, duration=0.5, sr=sr)
    return np.concatenate([speech, bleedthrough, speech2]).astype(np.float32)


def test_voice_activity_gate_attenuates_bleedthrough() -> None:
    """Gate suppresses residual energy in the inter-word pause by at least 20 dB."""
    audio = _speech_with_bleedthrough_pause()
    gated = voice_activity_gate(audio, sr=SAMPLE_RATE)

    check_start = int(0.85 * SAMPLE_RATE)
    check_end   = int(1.10 * SAMPLE_RATE)
    rms_before  = np.sqrt(np.mean(audio[check_start:check_end] ** 2))
    rms_after   = np.sqrt(np.mean(gated[check_start:check_end] ** 2))
    assert rms_after < rms_before * 0.1


def test_voice_activity_gate_preserves_sustained_speech() -> None:
    """Gate does not attenuate sustained active speech after the attack phase."""
    speech = _sine(250.0, duration=1.0)
    gated  = voice_activity_gate(speech, sr=SAMPLE_RATE)

    onset = int(0.05 * SAMPLE_RATE)
    assert np.allclose(gated[onset:], speech[onset:], atol=1e-3)


def test_voice_activity_gate_empty_returns_empty() -> None:
    """Gate returns an empty array unchanged."""
    result = voice_activity_gate(np.zeros(0, dtype=np.float32), sr=SAMPLE_RATE)
    assert len(result) == 0
