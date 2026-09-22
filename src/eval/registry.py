"""Experiment registry — every run leaves a versioned, comparable trace.

Two artefacts per run, both committed to git on purpose:

* ``results/<experiment_id>.json`` — summary, full configuration and per-item
  metrics, enough to re-derive any aggregate without re-running the model.
* a row in ``EXPERIMENTS.md`` — the human-readable log, newest last.

Results are research output, not scratch: keeping them in the repository is
what makes "did this change help?" answerable six months later.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.harness import EvaluationResult

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
EXPERIMENTS_LOG = Path("EXPERIMENTS.md")

_ROW_MARKER = "<!-- experiments:rows -->"


def git_revision() -> str:
    """Short commit hash, with a ``-dirty`` suffix on uncommitted changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def save_results(
    result: EvaluationResult,
    experiment_id: str,
    notes: str = "",
    results_dir: Path = RESULTS_DIR,
    log_path: Path | None = EXPERIMENTS_LOG,
) -> Path:
    """Write the JSON record and append a row to the experiment log."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary()

    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision":  git_revision(),
        "python":        platform.python_version(),
        "notes":         notes,
        "config":        result.config,
        "summary":       summary,
        "items": [
            {
                "mixture_id": it.mixture_id,
                "n_ref":      it.n_ref,
                "n_est":      it.n_est,
                "si_sdri":    [round(v, 3) for v in it.si_sdri],
                "pesq":       [round(v, 3) for v in it.pesq],
                "stoi":       [round(v, 3) for v in it.stoi],
            }
            for it in result.items
        ],
    }

    out_path = results_dir / f"{experiment_id}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    logger.info("Results written → %s", out_path)

    if log_path is not None:
        append_log_row(log_path, record)
    return out_path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def append_log_row(log_path: Path, record: dict[str, Any]) -> None:
    """Append one row to the EXPERIMENTS.md table, creating the file if needed."""
    summary = record["summary"]
    row = " | ".join([
        "",
        record["timestamp"][:10],
        f"`{record['experiment_id']}`",
        summary["dataset"],
        _fmt(summary["si_sdri_mean"]),
        _fmt(summary["pesq_mean"]),
        _fmt(summary["stoi_mean"]),
        _fmt(summary["count_accuracy"]),
        f"`{record['git_revision']}`",
        record["notes"].replace("|", "/") or "—",
        "",
    ]).strip()

    if not log_path.exists():
        log_path.write_text(_LOG_TEMPLATE)

    content = log_path.read_text()
    if _ROW_MARKER not in content:
        content = content.rstrip("\n") + "\n\n" + _ROW_MARKER + "\n"
    content = content.replace(_ROW_MARKER, f"{row}\n{_ROW_MARKER}")
    log_path.write_text(content)
    logger.info("Experiment log updated → %s", log_path)


def slugify(text: str) -> str:
    """Filesystem-safe experiment id fragment."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


_LOG_TEMPLATE = """# Experiments

Running log of every evaluated run, newest last. One row per run; the matching
`results/<id>.json` holds the full configuration and per-mixture metrics.

**Rules of the log**

1. Change one variable per experiment — otherwise a gain cannot be attributed.
2. Always report on a frozen benchmark split, never on a set tuned along the way.
3. A row is added by `src.eval.registry.save_results`, not by hand.

| Date | Experiment | Dataset | SI-SDRi (dB) | PESQ | STOI | Count acc. | Commit | Notes |
|---|---|---|---|---|---|---|---|---|
<!-- experiments:rows -->
"""
