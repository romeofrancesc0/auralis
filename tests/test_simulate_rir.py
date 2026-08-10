"""Tests for src.dsp.simulate_rir.

The benchmark's reverberation axis is only meaningful if a requested RT60 is the
RT60 actually delivered, so the calibration guarantee is the main thing pinned
here — uncalibrated image-source simulation overshoots by up to 58 %.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dsp.augment import apply_rir
from src.dsp.simulate_rir import (
    MIN_SOURCE_SEPARATION_M,
    RT60_TOLERANCE,
    measure_rt60,
    simulate_room_rirs,
)

SR = 8000


def test_anechoic_condition_returns_unit_impulses():
    response = simulate_room_rirs(0.0, 3, sr=SR, rng=np.random.default_rng(0))

    assert response.is_anechoic
    assert response.measured_rt60_s == 0.0
    assert len(response.rirs) == 3
    for rir in response.rirs:
        assert rir[0] == 1.0
        assert np.count_nonzero(rir) == 1


def test_applying_an_anechoic_response_leaves_the_signal_unchanged():
    signal = np.random.default_rng(1).normal(size=2000).astype(np.float32)
    response = simulate_room_rirs(0.0, 1, sr=SR, rng=np.random.default_rng(0))

    assert np.allclose(apply_rir(signal, response.rirs[0]), signal)


@pytest.mark.parametrize("target_rt60", [0.2, 0.5])
def test_measured_rt60_matches_the_target_within_tolerance(target_rt60):
    response = simulate_room_rirs(target_rt60, 2, sr=SR, rng=np.random.default_rng(3))

    relative_error = abs(response.measured_rt60_s - target_rt60) / target_rt60
    assert relative_error <= RT60_TOLERANCE
    assert response.target_rt60_s == target_rt60


def test_reported_rt60_is_the_measured_one_not_the_requested_one():
    response = simulate_room_rirs(0.6, 1, sr=SR, rng=np.random.default_rng(7))

    remeasured = measure_rt60(response.rirs[0], sr=SR)
    assert response.measured_rt60_s == pytest.approx(remeasured, rel=1e-6)


def test_same_seed_reproduces_the_same_room():
    first = simulate_room_rirs(0.4, 2, sr=SR, rng=np.random.default_rng(11))
    second = simulate_room_rirs(0.4, 2, sr=SR, rng=np.random.default_rng(11))

    assert first.room_dim_m == second.room_dim_m
    assert first.measured_rt60_s == second.measured_rt60_s
    for left, right in zip(first.rirs, second.rirs):
        assert np.array_equal(left, right)


def test_different_seeds_draw_different_rooms():
    first = simulate_room_rirs(0.4, 1, sr=SR, rng=np.random.default_rng(11))
    second = simulate_room_rirs(0.4, 1, sr=SR, rng=np.random.default_rng(12))

    assert first.room_dim_m != second.room_dim_m


def test_sources_share_a_room_but_not_a_position():
    response = simulate_room_rirs(0.4, 3, sr=SR, rng=np.random.default_rng(5))

    assert len(response.rirs) == 3
    # Lengths differ by the propagation delay of each source-microphone pair,
    # but a shared room keeps them within the same decay envelope — and
    # distinct positions must give distinct responses.
    lengths = [rir.size for rir in response.rirs]
    assert min(lengths) > 0
    assert max(lengths) - min(lengths) < 0.1 * SR
    assert not np.array_equal(response.rirs[0], response.rirs[1])
    assert not np.array_equal(response.rirs[1], response.rirs[2])


def test_rejects_a_room_without_sources():
    with pytest.raises(ValueError, match="at least one source"):
        simulate_room_rirs(0.4, 0, sr=SR)


def test_rejects_an_unreachable_reverberation_time():
    # No shoebox of 4-8 m sides can decay this fast, even fully absorbent.
    with pytest.raises(ValueError, match="not reachable"):
        simulate_room_rirs(0.01, 1, sr=SR, rng=np.random.default_rng(0))


def test_simulated_response_is_usable_by_the_augmentation_pipeline():
    signal = np.random.default_rng(2).normal(size=4000).astype(np.float32)
    response = simulate_room_rirs(0.4, 1, sr=SR, rng=np.random.default_rng(4))

    reverberant = apply_rir(signal, response.rirs[0])

    assert reverberant.shape == signal.shape
    assert np.isfinite(reverberant).all()
    assert not np.allclose(reverberant, signal)


def test_source_separation_constant_is_respected_by_construction():
    # Guards the constant itself: a zero or negative minimum would silently
    # allow two speakers to occupy the same point.
    assert MIN_SOURCE_SEPARATION_M > 0.0
