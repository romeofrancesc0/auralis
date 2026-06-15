"""Room Impulse Response (RIR) augmentation for reverberant separation training.

Real-world recordings always carry reverberation: the room's reflections
convolved with the dry voice. A separator trained only on anechoic LibriSpeech
mixtures suffers a synthetic-to-real gap — it has never seen the reflections a
microphone actually captures. Convolving the clean voices with measured or
simulated RIRs lets the model learn to segregate sources *in reverb*, which is
the condition a human listener faces in a real cocktail-party room.

Design choice — reverberant targets, not dereverberation:
both the mixture input and the uPIT targets are the reverberant sources. The
model only has to separate (not also remove the reverb), which keeps the SI-SDR
additivity invariant ``mix == src_f + src_m`` intact and the training stable.

The module degrades gracefully: if the RIR directory is missing or empty,
``load_rir_index`` returns an empty list and the caller simply trains on clean
mixtures as before.
"""
from __future__ import annotations

import logging
import random
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import fftconvolve

from src.utils import SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_RIR_ROOT = Path("data/raw/rir")
RIR_EXTENSIONS = (".wav", ".flac")
RIR_TRAIN_FRACTION = 0.8   # remaining 20% are held-out rooms for validation


def load_rir_index(root: Path | str = DEFAULT_RIR_ROOT) -> list[Path]:
    """Index every RIR file under ``root`` (recursively).

    Returns an empty list (with a warning) if the directory is missing or holds
    no RIR files, so reverberant augmentation is simply skipped rather than
    crashing the training run.
    """
    root = Path(root)
    if not root.exists():
        logger.warning(
            "RIR directory not found: %s — training without reverb augmentation",
            root,
        )
        return []

    rirs = sorted(
        p for ext in RIR_EXTENSIONS for p in root.rglob(f"*{ext}")
    )
    if not rirs:
        logger.warning(
            "No RIR files (%s) under %s — training without reverb augmentation",
            "/".join(RIR_EXTENSIONS), root,
        )
    else:
        logger.info("Indexed %d RIR files under %s", len(rirs), root)
    return rirs


def split_rirs(
    rirs: list[Path],
    fraction: float = RIR_TRAIN_FRACTION,
) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val split over rooms (sorted by path, held-out tail).

    Validation RIRs are unseen rooms, so reverberant validation measures
    generalisation to room acoustics the model never trained on.
    """
    ordered = sorted(rirs)
    n_train = int(len(ordered) * fraction)
    return ordered[:n_train], ordered[n_train:]


@lru_cache(maxsize=256)
def _load_rir_cached(path_str: str, sr: int) -> np.ndarray:
    """Load + resample one RIR, normalised so the direct-path peak is unit gain.

    Cached per process: RIRs are short and reused across many mixtures, so
    re-reading them from disk on every sample would dominate the dataloader.
    """
    rir, _ = librosa.load(path_str, sr=sr, mono=True)
    peak = float(np.max(np.abs(rir)))
    if peak > 0:
        rir = rir / peak
    return rir.astype(np.float32)


def load_rir(path: Path | str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a normalised RIR at the project sample rate (cached)."""
    return _load_rir_cached(str(path), sr)


def apply_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve ``signal`` with ``rir``, preserving length and onset alignment.

    The convolution output is sliced from the RIR's direct-path index so the
    reverberant signal stays time-aligned with the dry input — the early
    reflections trail *after* the original onset rather than shifting the whole
    waveform. Keeping the onset fixed is what lets the reverberant signal still
    serve as a valid SI-SDR target.
    """
    if rir.size == 0:
        return signal.astype(np.float32, copy=True)
    direct = int(np.argmax(np.abs(rir)))
    wet = fftconvolve(signal, rir)[direct : direct + len(signal)]
    if len(wet) < len(signal):
        wet = np.pad(wet, (0, len(signal) - len(wet)))
    return wet.astype(np.float32)


def reverberate_pair(
    src_f: np.ndarray,
    src_m: np.ndarray,
    rirs: list[Path],
    rng: random.Random,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an independent RIR to each source (two speaker positions).

    Each source is convolved with its own randomly drawn RIR, approximating two
    speakers at different positions. RIRs are drawn independently rather than
    constrained to a single room — a deliberate simplification, since the
    indexed sets are not grouped by room.
    """
    rir_f = load_rir(rng.choice(rirs), sr)
    rir_m = load_rir(rng.choice(rirs), sr)
    return apply_rir(src_f, rir_f), apply_rir(src_m, rir_m)
