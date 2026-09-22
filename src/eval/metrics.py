"""Separation metrics for a known or unknown number of speakers.

Pure numpy/scipy, so the evaluation harness runs without torch. PESQ and STOI are
optional: when the packages are missing the corresponding scores are ``None``
rather than an error, so a quick SI-SDR-only run needs no extra install.

Scoring a separator is only ambiguous in two places, and both are settled here
so every experiment in the project answers them the same way.

**Permutation.** A separator has no notion of which output is which speaker, so
estimates are matched to references under the assignment that maximises the
total SI-SDR (permutation-invariant scoring).

**Speaker-count errors.** When the system estimates N̂ instead of being told N,
the assignment stops being a permutation. The convention used by SepEDA, SepTDA
and SepNetEDCI — and therefore the one that keeps our numbers comparable with
theirs — is: if N̂ > N, score the subset of estimates that best matches the
references; if N̂ < N, treat each unmatched reference as having been estimated by
a silent signal.

A silent estimate scores −∞ against a live reference, which would make any
average meaningless. Rather than hide an epsilon in a denominator — which quietly
turns silence into 0 dB, and a missed speaker into an *improvement* whenever the
mixture scores below 0 dB, as it does for N ≥ 3 — per-reference scores are
clipped at an explicit floor (``floor_db``) that is reported alongside every
result. The symmetric case — an exact reconstruction, +∞ — is clipped at
``CEILING_DB`` so aggregates and the JSON records stay finite; only an oracle
reaches it. See docs/research/g1-benchmark-design.md.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# A silent (or hopelessly wrong) estimate would score -inf. This floor is far
# below any meaningful separation result — the mixture itself scores around
# 0 dB for 2 speakers and -6 dB for 5 — so it never binds on real estimates.
DEFAULT_FLOOR_DB = -30.0

# Real separators land in the 5-30 dB range; 100 dB is numerically exact
# reconstruction, reachable only by handing back the references themselves.
CEILING_DB = 100.0

Sources = np.ndarray | Sequence[np.ndarray]


@dataclass(frozen=True)
class SeparationScore:
    """Result of scoring one mixture's estimates against its references."""

    n_reference: int
    n_estimated: int
    assignment: tuple[int | None, ...]
    si_sdr_db: tuple[float, ...]
    si_sdr_mixture_db: tuple[float, ...]
    floor_db: float

    @property
    def count_correct(self) -> bool:
        return self.n_estimated == self.n_reference

    @property
    def si_sdri_db(self) -> tuple[float, ...]:
        return tuple(
            achieved - baseline
            for achieved, baseline in zip(self.si_sdr_db, self.si_sdr_mixture_db)
        )


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio, in dB (Le Roux et al. 2019).

    Both signals are mean-removed first, per the standard definition. Returns
    ``+inf`` for an exact reconstruction and ``-inf`` for a silent estimate;
    ``score_mixture`` clips both before anything is averaged.

    Raises:
        ValueError: if the reference is all zeros, which leaves the projection
            undefined.
    """
    estimate, reference = _align(estimate, reference)
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    reference_energy = float(reference @ reference)
    if reference_energy == 0.0:
        raise ValueError("SI-SDR is undefined for an all-zero reference")

    scale = float(estimate @ reference) / reference_energy
    target = scale * reference
    distortion = estimate - target

    target_energy = float(target @ target)
    distortion_energy = float(distortion @ distortion)
    # Order matters: a silent estimate has zero target *and* zero distortion,
    # and must score -inf rather than the +inf of a perfect reconstruction.
    if target_energy == 0.0:
        return float("-inf")
    if distortion_energy == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(target_energy / distortion_energy))


def si_sdr_improvement(
    estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray
) -> float:
    """SI-SDR of the estimate minus SI-SDR of the unprocessed mixture, in dB.

    Reporting the improvement rather than the absolute SI-SDR is what makes a
    number comparable across datasets: the mixture baseline differs with the
    number of speakers and their relative levels.
    """
    return si_sdr(estimate, reference) - si_sdr(mixture, reference)


def score_mixture(
    estimates: Sources,
    references: Sources,
    mixture: np.ndarray,
    *,
    floor_db: float = DEFAULT_FLOOR_DB,
) -> SeparationScore:
    """Score one mixture, allowing the estimate count to differ from the truth.

    Args:
        estimates: separated signals, a (n_estimated, samples) array or a
            sequence of 1-D signals. Zero estimates is legitimate: a recursive
            extractor may stop before extracting anyone.
        references: ground-truth sources, same layout.
        mixture: the unprocessed input, shape (samples,).
        floor_db: value substituted for per-reference scores at or below it,
            including the −∞ of a missing source. Scores are also capped at
            ``CEILING_DB``.

    Returns:
        A SeparationScore whose ``assignment[i]`` is the index of the estimate
        matched to reference ``i``, or None when the system produced too few
        estimates to cover it.
    """
    estimate_rows = _as_signal_list(estimates)
    reference_rows = _as_signal_list(references)
    if not reference_rows:
        raise ValueError("at least one reference source is required")

    # Separators routinely return a few samples more or fewer than the input
    # because of framing; everything is truncated to the common length.
    mixture = np.asarray(mixture, dtype=np.float64).ravel()
    length = min(signal.size for signal in [*estimate_rows, *reference_rows, mixture])
    if length == 0:
        raise ValueError("cannot score empty signals")
    estimate_rows = [signal[:length] for signal in estimate_rows]
    reference_rows = [signal[:length] for signal in reference_rows]
    mixture = mixture[:length]

    n_estimated = len(estimate_rows)
    n_reference = len(reference_rows)

    # Padding with silence turns an under-estimating system into a square
    # assignment problem: the padded columns score -inf and are floored below,
    # which is exactly the convention for a source the system never produced.
    padding = [np.zeros(length)] * max(0, n_reference - n_estimated)
    candidates = estimate_rows + padding

    scores = np.array([
        [_clipped(si_sdr(candidate, reference), floor_db) for candidate in candidates]
        for reference in reference_rows
    ])                                                   # (n_reference, n_candidates)

    ref_idx, cand_idx = linear_sum_assignment(scores, maximize=True)
    matched = dict(zip(ref_idx.tolist(), cand_idx.tolist()))

    return SeparationScore(
        n_reference=n_reference,
        n_estimated=n_estimated,
        assignment=tuple(
            matched[i] if matched[i] < n_estimated else None for i in range(n_reference)
        ),
        si_sdr_db=tuple(float(scores[i, matched[i]]) for i in range(n_reference)),
        si_sdr_mixture_db=tuple(si_sdr(mixture, reference) for reference in reference_rows),
        floor_db=floor_db,
    )


def pesq_score(estimate: np.ndarray, reference: np.ndarray, sr: int) -> float | None:
    """PESQ (wideband at 16 kHz, narrowband at 8 kHz), or None if unavailable."""
    try:
        from pesq import pesq as _pesq
    except ImportError:
        return None
    if sr not in (8_000, 16_000):
        logger.debug("PESQ undefined at %d Hz — skipped", sr)
        return None
    estimate, reference = _align(estimate, reference)
    try:
        return float(_pesq(sr, reference, estimate, "wb" if sr == 16_000 else "nb"))
    except Exception as exc:                       # pesq raises on silent/degenerate frames
        logger.debug("PESQ failed on one item: %s", exc)
        return None


def stoi_score(estimate: np.ndarray, reference: np.ndarray, sr: int) -> float | None:
    """STOI intelligibility, or None if the optional `pystoi` package is unavailable."""
    try:
        from pystoi import stoi as _stoi
    except ImportError:
        return None
    estimate, reference = _align(estimate, reference)
    try:
        return float(_stoi(reference, estimate, sr, extended=False))
    except Exception as exc:
        logger.debug("STOI failed on one item: %s", exc)
        return None


def _as_signal_list(sources: Sources) -> list[np.ndarray]:
    """Coerce a source collection to a list of 1-D float64 signals.

    A single 1-D array is one signal; a 2-D array is one signal per row; an
    empty collection yields no signals.
    """
    if isinstance(sources, np.ndarray):
        if sources.ndim == 1:
            return [sources.astype(np.float64)]
        if sources.ndim != 2:
            raise ValueError(f"sources must be 1-D or 2-D, got shape {sources.shape}")
        return [row.astype(np.float64) for row in sources]
    return [np.asarray(signal, dtype=np.float64).ravel() for signal in sources]


def _align(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both signals to their common length.

    At a handful of framing samples out of tens of thousands, truncating is
    standard practice and does not move the metric.
    """
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    length = min(estimate.size, reference.size)
    if length == 0:
        raise ValueError("cannot score empty signals")
    return estimate[:length], reference[:length]


def _clipped(value: float, floor_db: float) -> float:
    return min(max(value, floor_db), CEILING_DB)
