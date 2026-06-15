import numpy as np
import pytest

from src.dsp.augment import (
    apply_rir,
    load_rir_index,
    reverberate_pair,
    split_rirs,
)
from src.utils import SAMPLE_RATE, make_mixture_with_sources


def _sine(freq: float = 220.0, duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _decaying_rir(length: int = 800, seed: int = 0) -> np.ndarray:
    """A unit-direct-path RIR with an exponentially decaying random tail."""
    rng = np.random.default_rng(seed)
    rir = rng.standard_normal(length).astype(np.float32)
    rir *= np.exp(-np.arange(length) / (length / 4.0)).astype(np.float32)
    rir[0] = 1.0  # direct path at index 0
    return rir


def test_apply_rir_preserves_length():
    sig = _sine()
    rir = _decaying_rir()
    wet = apply_rir(sig, rir)
    assert wet.shape == sig.shape


def test_apply_rir_unit_impulse_is_identity():
    # A single-sample unit impulse is an anechoic "room" — output equals input.
    sig = _sine()
    wet = apply_rir(sig, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(wet, sig, atol=1e-5)


def test_apply_rir_empty_returns_signal():
    sig = _sine()
    wet = apply_rir(sig, np.array([], dtype=np.float32))
    np.testing.assert_allclose(wet, sig)


def test_reverberant_mixture_additivity():
    # uPIT SI-SDR training relies on mix == src_f + src_m even after reverb.
    f = apply_rir(_sine(220.0), _decaying_rir(seed=1))
    m = apply_rir(_sine(330.0), _decaying_rir(seed=2))
    mix, src_f, src_m = make_mixture_with_sources(f, m, snr_db=0.0)
    np.testing.assert_allclose(mix, src_f + src_m, atol=1e-6)


def test_load_rir_index_missing_dir_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert load_rir_index(missing) == []


def test_load_rir_index_empty_dir_returns_empty(tmp_path):
    assert load_rir_index(tmp_path) == []


def test_split_rirs_deterministic_and_held_out():
    rirs = [tmp for tmp in map(lambda i: f"rir_{i:03d}.wav", range(10))]
    from pathlib import Path
    paths = [Path(p) for p in rirs]
    train, val = split_rirs(paths, fraction=0.8)
    assert len(train) == 8 and len(val) == 2
    # No room appears in both splits.
    assert set(train).isdisjoint(set(val))
    # Deterministic across calls.
    assert split_rirs(paths, fraction=0.8) == (train, val)


def test_reverberate_pair_preserves_lengths(tmp_path):
    import random
    import soundfile as sf

    rir_path = tmp_path / "room.wav"
    sf.write(rir_path, _decaying_rir(), SAMPLE_RATE)

    f, m = _sine(220.0), _sine(330.0)
    rev_f, rev_m = reverberate_pair(f, m, [rir_path], random.Random(0))
    assert rev_f.shape == f.shape
    assert rev_m.shape == m.shape
