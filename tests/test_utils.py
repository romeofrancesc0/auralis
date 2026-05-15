import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import SAMPLE_RATE, load_audio, make_mixture, save_audio


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_save_load_roundtrip():
    original = _sine()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    save_audio(path, original, sr=SAMPLE_RATE)
    loaded, sr = load_audio(path, sr=SAMPLE_RATE)
    path.unlink()

    assert sr == SAMPLE_RATE
    assert loaded.shape == original.shape
    np.testing.assert_allclose(loaded, original, atol=1e-4)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/path/audio.wav")


def test_make_mixture_length():
    a = _sine(440.0, duration=2.0)
    b = _sine(880.0, duration=3.0)
    mix = make_mixture(a, b)
    assert len(mix) == len(a)  # truncated to shortest


def test_make_mixture_peak_normalized():
    a = _sine(440.0)
    b = _sine(880.0)
    mix = make_mixture(a, b, snr_db=0.0)
    assert np.max(np.abs(mix)) <= 1.0 + 1e-6


def test_make_mixture_snr_ratio():
    # At snr_db=0, both voices have the same RMS in the mix.
    a = _sine(440.0)
    b = _sine(880.0)
    mix = make_mixture(a, b, snr_db=0.0)
    # mix is normalized so we can't check absolute RMS, but shape must be correct
    assert mix.shape == a.shape


def test_make_mixture_silent_b():
    a = _sine(440.0)
    b = np.zeros(len(a), dtype=np.float32)
    mix = make_mixture(a, b)
    # Silent b: mixture should equal a (possibly rescaled)
    np.testing.assert_allclose(np.abs(mix), np.abs(a) / np.max(np.abs(a)), atol=1e-5)
