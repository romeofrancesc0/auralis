"""Tests for src.eval.metrics.

Everything here runs on synthetic signals: the point is to pin down the scoring
conventions (permutation, count errors, the floor) independently of any model or
dataset, so a later reproduction failure can be blamed on the pipeline rather
than on the metric.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    CEILING_DB,
    DEFAULT_FLOOR_DB,
    pesq_score,
    score_mixture,
    si_sdr,
    si_sdr_improvement,
    stoi_score,
)

SR = 8000
SAMPLES = 4000


def _tone(freq_hz: float, samples: int = SAMPLES) -> np.ndarray:
    t = np.arange(samples) / SR
    return np.sin(2 * np.pi * freq_hz * t)


@pytest.fixture
def sources() -> tuple[np.ndarray, np.ndarray]:
    return _tone(140.0), _tone(330.0)


def test_si_sdr_is_infinite_for_exact_reconstruction():
    reference = _tone(200.0)
    assert si_sdr(reference, reference) == float("inf")


def test_si_sdr_is_scale_invariant():
    reference = _tone(200.0)
    estimate = reference + 0.1 * _tone(700.0)
    assert si_sdr(estimate, reference) == pytest.approx(si_sdr(7.3 * estimate, reference))


def test_si_sdr_is_negative_infinity_for_silent_estimate():
    reference = _tone(200.0)
    assert si_sdr(np.zeros(SAMPLES), reference) == float("-inf")


def test_si_sdr_rejects_silent_reference():
    with pytest.raises(ValueError, match="all-zero reference"):
        si_sdr(_tone(200.0), np.zeros(SAMPLES))


def test_si_sdr_degrades_as_interference_grows():
    reference = _tone(200.0)
    interferer = _tone(700.0)
    clean = si_sdr(reference + 0.05 * interferer, reference)
    dirty = si_sdr(reference + 0.5 * interferer, reference)
    assert clean > dirty


def test_si_sdr_improvement_is_positive_when_estimate_beats_mixture(sources):
    source_a, source_b = sources
    mixture = source_a + source_b
    estimate = source_a + 0.1 * source_b
    assert si_sdr_improvement(estimate, source_a, mixture) > 0.0


def test_si_sdr_truncates_to_common_length():
    reference = _tone(200.0)
    estimate = np.concatenate([reference, np.zeros(7)])
    assert si_sdr(estimate, reference) == float("inf")


def test_score_mixture_recovers_swapped_permutation(sources):
    source_a, source_b = sources
    mixture = source_a + source_b
    # Estimates arrive in the opposite order to the references.
    score = score_mixture(np.stack([source_b, source_a]), np.stack([source_a, source_b]), mixture)

    assert score.assignment == (1, 0)
    assert score.count_correct
    assert score.si_sdr_db == (CEILING_DB, CEILING_DB)


def test_score_mixture_prefers_the_exact_match_over_a_near_one(sources):
    """An exact reconstruction (+inf, capped) must not tie with a near one."""
    source_a, source_b = sources
    mixture = source_a + source_b
    near_b = source_b + 0.01 * source_a
    score = score_mixture([near_b, source_a], [source_a, source_b], mixture)

    assert score.assignment == (1, 0)
    assert score.si_sdr_db[0] == CEILING_DB
    assert score.si_sdr_db[1] < CEILING_DB


def test_score_mixture_accepts_sequences_of_unequal_length(sources):
    """Separators return lists, and framing leaves them a few samples off."""
    source_a, source_b = sources
    mixture = source_a + source_b
    estimates = [np.concatenate([source_b, np.zeros(5)]), source_a[:-3]]

    score = score_mixture(estimates, [source_a, source_b], mixture)

    assert score.assignment == (1, 0)


def test_score_mixture_over_estimation_picks_best_subset(sources):
    source_a, source_b = sources
    mixture = source_a + source_b
    noise = np.random.default_rng(0).normal(scale=0.3, size=SAMPLES)

    score = score_mixture(np.stack([noise, source_b, source_a]), np.stack([source_a, source_b]), mixture)

    assert score.n_estimated == 3
    assert not score.count_correct
    assert score.assignment == (2, 1)  # the spurious stream is left unmatched


def test_score_mixture_under_estimation_floors_the_missing_source(sources):
    source_a, source_b = sources
    mixture = source_a + source_b

    score = score_mixture(source_a[np.newaxis, :], np.stack([source_a, source_b]), mixture)

    assert score.n_estimated == 1
    assert not score.count_correct
    assert score.assignment == (0, None)
    assert score.si_sdr_db[1] == DEFAULT_FLOOR_DB


def test_a_missed_speaker_is_never_an_improvement():
    """With N=3 the mixture scores below 0 dB against each source. Scoring the
    missed speaker's silence at 0 dB — what an epsilon in the denominator does —
    would turn under-counting into a gain; the floor must keep it a loss."""
    references = np.stack([_tone(140.0), _tone(330.0), _tone(710.0)])
    mixture = references.sum(axis=0)

    score = score_mixture(references[:2], references, mixture)

    assert score.si_sdr_mixture_db[2] < 0.0
    assert score.si_sdri_db[2] < 0.0


def test_score_mixture_honours_a_custom_floor(sources):
    source_a, source_b = sources
    mixture = source_a + source_b

    score = score_mixture(source_a[np.newaxis, :], np.stack([source_a, source_b]), mixture, floor_db=-12.5)

    assert score.si_sdr_db[1] == -12.5
    assert score.floor_db == -12.5


def test_score_mixture_handles_zero_estimates(sources):
    source_a, source_b = sources
    mixture = source_a + source_b

    score = score_mixture([], np.stack([source_a, source_b]), mixture)

    assert score.n_estimated == 0
    assert score.assignment == (None, None)
    assert score.si_sdr_db == (DEFAULT_FLOOR_DB, DEFAULT_FLOOR_DB)


def test_score_mixture_rejects_empty_references(sources):
    source_a, _ = sources
    with pytest.raises(ValueError, match="at least one reference"):
        score_mixture(source_a[np.newaxis, :], np.zeros((0, SAMPLES)), source_a)


def test_si_sdri_is_the_gain_over_the_mixture(sources):
    source_a, source_b = sources
    mixture = source_a + source_b
    estimate = source_a + 0.05 * source_b

    score = score_mixture([estimate, source_b + 0.05 * source_a], [source_a, source_b], mixture)

    assert score.si_sdri_db[0] == pytest.approx(si_sdr_improvement(estimate, source_a, mixture))


def test_perceptual_scores_are_optional_but_never_raise():
    reference = _tone(220.0)
    for value in (pesq_score(reference, reference, SR), stoi_score(reference, reference, SR)):
        assert value is None or isinstance(value, float)
