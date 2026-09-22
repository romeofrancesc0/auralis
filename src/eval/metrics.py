"""Separation metrics — SI-SDR, permutation matching, PESQ and STOI.

Pure numpy, so the evaluation harness runs without torch. PESQ and STOI are
optional: when the packages are missing the corresponding scores are ``None``
rather than an error, so a quick SI-SDR-only run needs no extra install.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

EPS = 1e-8


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019).

    The estimate is compared against its own optimal rescaling of the
    reference, so a constant gain difference costs nothing — which is what
    makes the metric comparable across systems that do not preserve level.
    """
    estimate, reference = _align(estimate, reference)
    reference = reference - reference.mean()
    estimate  = estimate - estimate.mean()

    ref_energy = np.sum(reference ** 2) + EPS
    alpha      = np.sum(estimate * reference) / ref_energy
    target     = alpha * reference
    noise      = estimate - target
    return float(10 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(noise ** 2) + EPS)))


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both signals to the shorter length."""
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)


def best_permutation(
    estimates: list[np.ndarray],
    references: list[np.ndarray],
) -> tuple[tuple[int, ...], float]:
    """Assign estimates to references by maximising mean SI-SDR.

    Handles a count mismatch, which is the normal case once the separator
    estimates the number of speakers itself: the assignment covers
    ``min(n_est, n_ref)`` pairs and the surplus on either side is left
    unmatched (penalised by the count metrics, not by SI-SDR).

    Returns:
        (assignment, mean_si_sdr) where ``assignment[k]`` is the index of the
        estimate matched to ``references[k]``, or -1 if unmatched.
    """
    if not estimates or not references:
        raise ValueError("best_permutation() needs at least one estimate and one reference")

    scores = np.array([
        [si_sdr(est, ref) for est in estimates]
        for ref in references
    ])                                              # (n_ref, n_est)

    # Optimal (not greedy) rectangular assignment — exact for any count mismatch.
    ref_idx, est_idx = linear_sum_assignment(scores, maximize=True)

    assignment = [-1] * len(references)
    for r, e in zip(ref_idx, est_idx):
        assignment[r] = int(e)
    return tuple(assignment), float(scores[ref_idx, est_idx].mean())


def pesq_score(estimate: np.ndarray, reference: np.ndarray, sr: int) -> float | None:
    """Wideband PESQ, or None if the optional `pesq` package is unavailable."""
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
