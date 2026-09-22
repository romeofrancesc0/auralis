"""Tests for the separation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import best_permutation, pesq_score, si_sdr, stoi_score

SR = 16_000


def _tone(freq: float, n: int = SR, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_si_sdr_perfect_reconstruction_is_high():
    ref = _tone(220.0)
    assert si_sdr(ref, ref) > 100.0


def test_si_sdr_is_scale_invariant():
    """A constant gain on the estimate must not change the score."""
    rng = np.random.default_rng(0)
    ref = _tone(220.0)
    estimate = ref + 0.1 * rng.standard_normal(len(ref)).astype(np.float32)
    assert si_sdr(estimate * 7.3, ref) == pytest.approx(si_sdr(estimate, ref), abs=1e-4)


def test_si_sdr_penalises_wrong_source():
    ref = _tone(220.0)
    assert si_sdr(_tone(3000.0), ref) < 3.0


def test_si_sdr_truncates_to_shorter_signal():
    ref = _tone(220.0)
    assert si_sdr(ref[:1000], ref) > 100.0


def test_best_permutation_recovers_swapped_order():
    a, b = _tone(220.0), _tone(3000.0)
    assignment, score = best_permutation([b, a], [a, b])
    assert assignment == (1, 0)
    assert score > 100.0


def test_best_permutation_keeps_natural_order_when_correct():
    a, b = _tone(220.0), _tone(3000.0)
    assignment, _ = best_permutation([a, b], [a, b])
    assert assignment == (0, 1)


def test_best_permutation_handles_over_separation():
    a, b = _tone(220.0), _tone(3000.0)
    spurious = np.zeros_like(a) + 1e-3
    assignment, _ = best_permutation([spurious, b, a], [a, b])
    assert assignment == (2, 1)           # both references matched, extra estimate dropped


def test_best_permutation_handles_under_separation():
    a, b = _tone(220.0), _tone(3000.0)
    assignment, _ = best_permutation([b], [a, b])
    assert assignment == (-1, 0)          # first reference left unmatched


def test_best_permutation_rejects_empty_input():
    with pytest.raises(ValueError):
        best_permutation([], [_tone(220.0)])


def test_perceptual_scores_are_optional_but_never_raise():
    ref = _tone(220.0)
    for score in (pesq_score(ref, ref, SR), stoi_score(ref, ref, SR)):
        assert score is None or isinstance(score, float)
