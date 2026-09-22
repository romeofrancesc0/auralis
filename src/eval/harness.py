"""Evaluation harness — one protocol, every experiment comparable.

A separator is passed in as a plain callable, so the harness stays free of
torch and of any particular architecture:

    def separate(mixture: np.ndarray, sr: int) -> list[np.ndarray]: ...

Reported per mixture, then aggregated: SI-SDR improvement over the unprocessed
mixture (the only number comparable with published baselines), PESQ, STOI, and
— for models that decide how many speakers are present — the speaker-count
error. Metrics are computed under the optimal estimate-to-reference assignment,
never under a fixed output order.
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
from src.eval.metrics import best_permutation, pesq_score, si_sdr, stoi_score

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
            "si_sdr_in_mean":     _mean(si_sdr_in),
            "si_sdr_out_mean":    _mean(si_sdr_out),
            "pesq_mean":          _mean(pesq),
            "stoi_mean":          _mean(stoi),
            "count_accuracy":     _mean([float(it.count_correct) for it in self.items]),
            "count_confusion":    self.count_confusion(),
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
) -> EvaluationResult:
    """Run `separate_fn` over every mixture and collect metrics.

    Args:
        separate_fn:  mixture, sr -> list of estimated source waveforms
        dataset:      iterable of MixtureItem (see src.data.librimix)
        dataset_name: identifier stored with the results
        perceptual:   also compute PESQ and STOI (slower; needs pesq/pystoi)
        config:       experiment configuration, stored verbatim with the results
        log_every:    progress logging interval, in mixtures
    """
    items: list[ItemResult] = []
    started = time.perf_counter()

    for i, item in enumerate(dataset, start=1):
        estimates = list(separate_fn(item.mixture, item.sr))
        if not estimates:
            raise ValueError(f"separate_fn returned no estimate for {item.mixture_id!r}")

        assignment, _ = best_permutation(estimates, item.sources)

        si_in, si_out, si_imp, pesq_vals, stoi_vals = [], [], [], [], []
        for ref_idx, est_idx in enumerate(assignment):
            reference = item.sources[ref_idx]
            matched   = est_idx >= 0

            # Under-estimation (N_hat < N): the unmatched reference is scored against a
            # silent estimate, the convention used by SepEDA / SepTDA / SepNetEDCI and
            # pinned in docs/research/reproduction-targets.md. Skipping it instead would
            # quietly reward a model for emitting fewer sources than it should.
            # Silence carries error energy equal to the reference, hence 0 dB SI-SDR, so
            # the missed speaker contributes an improvement of -SI-SDR(mixture) — a real
            # penalty against the matched sources' double-digit gains.
            estimate = estimates[est_idx] if matched else np.zeros_like(reference)

            baseline = si_sdr(item.mixture, reference)
            achieved = si_sdr(estimate, reference)
            si_in.append(baseline)
            si_out.append(achieved)
            si_imp.append(achieved - baseline)

            # Perceptual scores on a substituted silence are meaningless, not merely bad.
            if perceptual and matched:
                if (p := pesq_score(estimate, reference, item.sr)) is not None:
                    pesq_vals.append(p)
                if (s := stoi_score(estimate, reference, item.sr)) is not None:
                    stoi_vals.append(s)

        items.append(ItemResult(
            mixture_id=item.mixture_id,
            n_ref=item.n_sources,
            n_est=len(estimates),
            si_sdr_in=si_in,
            si_sdr_out=si_out,
            si_sdri=si_imp,
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
        config=config or {},
    )
    logger.info("Evaluation done: %s", result.summary())
    return result
