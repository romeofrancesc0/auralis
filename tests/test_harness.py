"""Tests for the evaluation harness and the experiment registry."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.librimix import MixtureItem
from src.eval.harness import evaluate
from src.eval.registry import append_log_row, save_results, slugify

SR = 16_000


def _tone(freq: float, n: int = SR // 2) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def dataset():
    """Two mixtures of two well-separated tones each."""
    items = []
    for i, (f_a, f_b) in enumerate([(220.0, 3000.0), (180.0, 2500.0)]):
        sources = [_tone(f_a), _tone(f_b)]
        items.append(MixtureItem(f"mix-{i}", sources[0] + sources[1], sources, SR))
    return items


def _oracle(item_lookup):
    return lambda mixture, sr: list(item_lookup[float(mixture.sum())])


def test_control_separator_scores_about_zero_improvement(dataset):
    """Handing back the mixture must measure as no improvement at all."""
    result = evaluate(lambda mixture, sr: [mixture.copy(), mixture.copy()],
                      dataset, "control", perceptual=False)
    assert result.summary()["si_sdri_mean"] == pytest.approx(0.0, abs=1e-6)
    assert result.summary()["count_accuracy"] == 1.0


def test_oracle_separator_scores_a_large_improvement(dataset):
    lookup = {float(it.mixture.sum()): it.sources for it in dataset}
    result = evaluate(_oracle(lookup), dataset, "oracle", perceptual=False)
    assert result.summary()["si_sdri_mean"] > 50.0


def test_metrics_are_invariant_to_output_order(dataset):
    """Swapping the outputs must not change the score — assignment is optimal."""
    lookup = {float(it.mixture.sum()): it.sources for it in dataset}
    straight = evaluate(_oracle(lookup), dataset, "oracle", perceptual=False)
    swapped = evaluate(lambda m, sr: list(reversed(lookup[float(m.sum())])),
                       dataset, "oracle-swapped", perceptual=False)
    assert straight.summary()["si_sdri_mean"] == pytest.approx(
        swapped.summary()["si_sdri_mean"], abs=1e-6)


def test_count_accuracy_reflects_over_separation(dataset):
    result = evaluate(lambda m, sr: [m.copy(), m.copy(), m.copy()],
                      dataset, "over", perceptual=False)
    summary = result.summary()
    assert summary["count_accuracy"] == 0.0
    assert summary["n_pairs"] == 4               # 2 references matched per mixture


def test_under_separation_penalises_the_missed_reference(dataset):
    """A missed speaker is scored against silence, not skipped (see
    docs/research/reproduction-targets.md): every reference must be accounted for.

    A separator that perfectly extracts one of two speakers must not score like a
    perfect two-speaker system — the reference it never emitted contributes the
    silence score (0 dB SI-SDR, hence no improvement over the mixture).
    """
    lookup = {float(it.mixture.sum()): it.sources for it in dataset}
    result = evaluate(lambda m, sr: [lookup[float(m.sum())][0]],
                      dataset, "under", perceptual=False)
    summary = result.summary()

    assert summary["count_accuracy"] == 0.0
    assert summary["n_pairs"] == 4               # both references scored, one against silence
    assert summary["count_confusion"] == {"2->1": 2}

    extracted, missed = max(result.items[0].si_sdri), min(result.items[0].si_sdri)
    assert extracted > 50.0                      # the speaker the model did emit
    assert missed == pytest.approx(0.0, abs=0.5) # the one it did not

    full = evaluate(lambda m, sr: list(lookup[float(m.sum())]),
                    dataset, "oracle", perceptual=False)
    assert summary["si_sdri_mean"] < full.summary()["si_sdri_mean"]


def test_count_confusion_breaks_down_the_errors(dataset):
    result = evaluate(lambda m, sr: [m.copy(), m.copy(), m.copy()],
                      dataset, "over", perceptual=False)
    assert result.summary()["count_confusion"] == {"2->3": 2}


def test_empty_output_is_rejected(dataset):
    with pytest.raises(ValueError, match="no estimate"):
        evaluate(lambda m, sr: [], dataset, "broken", perceptual=False)


def test_save_results_writes_json_and_log_row(tmp_path, dataset):
    result = evaluate(lambda m, sr: [m.copy(), m.copy()], dataset, "control",
                      perceptual=False, config={"separator": "none"})
    log = tmp_path / "EXPERIMENTS.md"
    path = save_results(result, "ctrl-run", notes="control",
                        results_dir=tmp_path / "results", log_path=log)

    record = json.loads(path.read_text())
    assert record["experiment_id"] == "ctrl-run"
    assert record["config"] == {"separator": "none"}
    assert len(record["items"]) == 2
    assert record["summary"]["si_sdri_mean"] == pytest.approx(0.0, abs=1e-6)

    assert "| ctrl-run |".replace("ctrl-run", "`ctrl-run`") in log.read_text()


def test_log_rows_accumulate_newest_last(tmp_path):
    log = tmp_path / "EXPERIMENTS.md"
    for name in ("first", "second"):
        append_log_row(log, {
            "timestamp": "2026-09-19T00:00:00", "experiment_id": name,
            "git_revision": "abc1234", "notes": "",
            "summary": {"dataset": "d", "si_sdri_mean": 1.0, "pesq_mean": None,
                        "stoi_mean": None, "count_accuracy": 1.0},
        })
    content = log.read_text()
    assert content.index("`first`") < content.index("`second`")
    assert content.count("| Date | Experiment |") == 1


def test_slugify_builds_safe_identifiers():
    assert slugify("DPCRN uPIT v0.4.0 / Libri2Mix") == "dpcrn-upit-v0-4-0-libri2mix"
