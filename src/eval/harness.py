"""Evaluation harness — one protocol, every experiment comparable.

A separator is passed in as a plain callable, so the harness stays free of
torch and of any particular architecture:

    def separate(mixture: np.ndarray, sr: int) -> list[np.ndarray]: ...

Reported per mixture, then aggregated: SI-SDR improvement over the unprocessed
mixture (the only number comparable with published baselines), PESQ, STOI, and
— for models that decide how many speakers are present — the speaker-count
error. Metrics are computed under the optimal estimate-to-reference assignment,
never under a fixed output order, with the count-error conventions of
src.eval.metrics (a missed speaker is scored as silence, clipped at ``floor_db``).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

import numpy as np

from src.data.librimix import MixtureItem
from src.eval.metrics import DEFAULT_FLOOR_DB, pesq_score, score_mixture, stoi_score

logger = logging.getLogger(__name__)

SeparateFn = Callable[[np.ndarray, int], list[np.ndarray]]


@dataclass
class ItemResult:
    """Metrics for a single mixture."""

    mixture_id: str
    n_ref: int
    n_est: int
    si_sdr_in: list[float]          # mixture vs each reference
    si_sdr_out: list[float]         # matched estimate vs each reference
    si_sdri: list[float]
    pesq: list[float] = field(default_factory=list)
    stoi: list[float] = field(default_factory=list)

    @property
    def count_correct(self) -> bool:
        return self.n_est == self.n_ref


@dataclass
class EvaluationResult:
    """Aggregate over a dataset, plus the per-item records behind it."""

    dataset: str
    n_items: int
    items: list[ItemResult]
    elapsed_s: float
    floor_db: float = DEFAULT_FLOOR_DB
    config: dict[str, Any] = field(default_factory=dict)

    def count_confusion(self) -> dict[str, int]:
        """How many mixtures of N references drew N_hat estimates, as "N->N_hat".

        A single accuracy figure hides whether count errors are off-by-one or
        catastrophic; reproduction-targets.md asks for the breakdown whenever a
        sweep spans more than two speaker counts.
        """
        confusion: dict[str, int] = {}
        for it in self.items:
            key = f"{it.n_ref}->{it.n_est}"
            confusion[key] = confusion.get(key, 0) + 1
        return dict(sorted(confusion.items()))

    def summary(self) -> dict[str, Any]:
        """Flat dict of headline numbers — what gets written to results/."""
        si_sdri = [v for it in self.items for v in it.si_sdri]
        # Separation quality on the mixtures the system counted correctly, so it
        # is not dragged down by counting failures; reported next to the overall
        # figure because either alone misleads — a system that always answers
        # N=1 can look clean on the first and hopeless on the second.
        si_sdri_counted = [v for it in self.items if it.count_correct for v in it.si_sdri]
        si_sdr_in = [v for it in self.items for v in it.si_sdr_in]
        si_sdr_out = [v for it in self.items for v in it.si_sdr_out]
        pesq = [v for it in self.items for v in it.pesq]
        stoi = [v for it in self.items for v in it.stoi]

        return {
            "dataset":            self.dataset,
            "n_items":            self.n_items,
            "n_pairs":            len(si_sdri),
            "si_sdri_mean":       _mean(si_sdri),
            "si_sdri_std":        _stdev(si_sdri),
            "si_sdri_mean_count_correct": _mean(si_sdri_counted),
            "si_sdr_in_mean":     _mean(si_sdr_in),
            "si_sdr_out_mean":    _mean(si_sdr_out),
            "pesq_mean":          _mean(pesq),
            "stoi_mean":          _mean(stoi),
            "count_accuracy":     _mean([float(it.count_correct) for it in self.items]),
            "count_confusion":    self.count_confusion(),
            "floor_db":           self.floor_db,
            "elapsed_s":          round(self.elapsed_s, 1),
        }


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _stdev(values: list[float]) -> float | None:
    return round(stdev(values), 4) if len(values) > 1 else None


def evaluate(
    separate_fn: SeparateFn,
    dataset: Iterable[MixtureItem],
    dataset_name: str = "?",
    perceptual: bool = True,
    config: dict[str, Any] | None = None,
    log_every: int = 25,
    floor_db: float = DEFAULT_FLOOR_DB,
) -> EvaluationResult:
    """Run `separate_fn` over every mixture and collect metrics.

    Args:
        separate_fn:  mixture, sr -> list of estimated source waveforms
        dataset:      iterable of MixtureItem (see src.data.librimix)
        dataset_name: identifier stored with the results
        perceptual:   also compute PESQ and STOI (slower; needs pesq/pystoi)
        config:       experiment configuration, stored verbatim with the results
        log_every:    progress logging interval, in mixtures
        floor_db:     SI-SDR assigned to a missed speaker (see src.eval.metrics);
                      stored with the results because it moves unknown-N numbers
    """
    items: list[ItemResult] = []
    started = time.perf_counter()

    for i, item in enumerate(dataset, start=1):
        # An empty list is a legitimate answer (N_hat = 0) for a system that
        # decides the count itself; it is scored, not rejected.
        estimates = list(separate_fn(item.mixture, item.sr))
        score = score_mixture(estimates, item.sources, item.mixture, floor_db=floor_db)

        pesq_vals, stoi_vals = [], []
        # Perceptual scores on a substituted silence are meaningless, not merely bad.
        if perceptual:
            for ref_idx, est_idx in enumerate(score.assignment):
                if est_idx is None:
                    continue
                estimate, reference = estimates[est_idx], item.sources[ref_idx]
                if (p := pesq_score(estimate, reference, item.sr)) is not None:
                    pesq_vals.append(p)
                if (s := stoi_score(estimate, reference, item.sr)) is not None:
                    stoi_vals.append(s)

        items.append(ItemResult(
            mixture_id=item.mixture_id,
            n_ref=score.n_reference,
            n_est=score.n_estimated,
            si_sdr_in=list(score.si_sdr_mixture_db),
            si_sdr_out=list(score.si_sdr_db),
            si_sdri=list(score.si_sdri_db),
            pesq=pesq_vals,
            stoi=stoi_vals,
        ))

        if log_every and i % log_every == 0:
            running = _mean([v for it in items for v in it.si_sdri])
            logger.info("  %d mixtures — running SI-SDRi %s dB", i, running)

    elapsed = time.perf_counter() - started
    result = EvaluationResult(
        dataset=dataset_name,
        n_items=len(items),
        items=items,
        elapsed_s=elapsed,
        floor_db=floor_db,
        config=config or {},
    )
    logger.info("Evaluation done: %s", result.summary())
    return result
