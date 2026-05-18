"""Dataset utilities for LibriSpeech: speaker loading and mixture generation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import SAMPLE_RATE, load_audio, make_mixture

LIBRISPEECH_ROOT = Path("data/raw/librispeech")
SPEAKERS_FILE = LIBRISPEECH_ROOT / "SPEAKERS.TXT"


@dataclass
class Speaker:
    id: str
    gender: str  # 'M' or 'F'
    audio_files: list[Path]


def load_speaker_index(
    subset: str = "dev-clean",
    root: Path = LIBRISPEECH_ROOT,
) -> dict[str, list[Speaker]]:
    """Parse SPEAKERS.TXT and return {'M': [...], 'F': [...]} for the given subset."""
    speakers_file = root / "SPEAKERS.TXT"
    speakers: dict[str, list[Speaker]] = {"M": [], "F": []}

    with open(speakers_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            speaker_id, gender, speaker_subset = parts[0], parts[1], parts[2]
            if speaker_subset != subset:
                continue

            speaker_dir = root / subset / speaker_id
            if not speaker_dir.exists():
                continue

            audio_files = sorted(speaker_dir.rglob("*.flac"))
            if not audio_files:
                continue

            speakers[gender].append(Speaker(id=speaker_id, gender=gender, audio_files=audio_files))

    return speakers


@dataclass
class MixtureSample:
    mixture: np.ndarray
    target: np.ndarray       # the voice we want to isolate (female by convention)
    interferer: np.ndarray   # the other voice
    target_gender: str       # 'F'
    interferer_gender: str   # 'M'
    target_speaker_id: str
    interferer_speaker_id: str
    sr: int


def make_samples(
    n_samples: int,
    subset: str = "dev-clean",
    root: Path = LIBRISPEECH_ROOT,
    snr_db: float = 0.0,
    clip_duration: float = 3.0,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
) -> list[MixtureSample]:
    """Generate n_samples mixtures, each pairing one female + one male speaker.

    Target is always the female speaker (F). The interferer is male (M).
    Clips are trimmed / zero-padded to clip_duration seconds.
    """
    rng = random.Random(seed)
    index = load_speaker_index(subset=subset, root=root)

    female_speakers = index["F"]
    male_speakers = index["M"]

    if not female_speakers or not male_speakers:
        raise ValueError(f"Not enough speakers in subset '{subset}' under {root}")

    n_clip = int(sr * clip_duration)
    samples: list[MixtureSample] = []

    for _ in range(n_samples):
        female = rng.choice(female_speakers)
        male = rng.choice(male_speakers)

        f_audio = _load_clip(rng.choice(female.audio_files), n_clip, sr)
        m_audio = _load_clip(rng.choice(male.audio_files), n_clip, sr)

        mixture = make_mixture(f_audio, m_audio, snr_db=snr_db)

        samples.append(MixtureSample(
            mixture=mixture,
            target=f_audio,
            interferer=m_audio,
            target_gender="F",
            interferer_gender="M",
            target_speaker_id=female.id,
            interferer_speaker_id=male.id,
            sr=sr,
        ))

    return samples


def make_ibm_dataset(
    n_samples: int,
    subset: str = "dev-clean",
    root: Path = LIBRISPEECH_ROOT,
    snr_db: float = 0.0,
    clip_duration: float = 3.0,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
    window_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a frame-level dataset for IBM (Ideal Binary Mask) training.

    Features are extracted from the MIXTURE (not isolated voices), so the
    classifier learns to distinguish F-dominant frames from M-dominant ones
    in a real mixed signal — eliminating the train/inference domain mismatch.

    When window_size > 1, each sample is a concatenation of window_size
    consecutive feature frames, giving the model temporal context.

    IBM label per frame: 0 = F-dominant, 1 = M-dominant (matches LABEL_MAP).

    Returns:
        X: (n_frames_total, n_features * window_size)
        y: (n_frames_total,) — 0 = F-dominant frame, 1 = M-dominant frame
    """
    from src.dsp.features import apply_window, extract_all
    from src.dsp.stft import compute_stft

    samples = make_samples(
        n_samples=n_samples, subset=subset, root=root,
        snr_db=snr_db, clip_duration=clip_duration, sr=sr, seed=seed,
    )

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for sample in samples:
        # Feature matrix from the MIXTURE: (n_features, n_frames)
        feat = extract_all(sample.mixture, sr=sr)

        # Per-frame energy for each isolated source
        stft_f = compute_stft(sample.target)
        stft_m = compute_stft(sample.interferer)
        energy_f = (np.abs(stft_f) ** 2).mean(axis=0)
        energy_m = (np.abs(stft_m) ** 2).mean(axis=0)

        # Align lengths across feature extractor and STFT
        n_frames = min(feat.shape[1], len(energy_f), len(energy_m))
        feat = feat[:, :n_frames]
        energy_f = energy_f[:n_frames]
        energy_m = energy_m[:n_frames]

        # IBM label: 0 = F-dominant, 1 = M-dominant
        ibm_labels = (energy_f <= energy_m).astype(int)

        # Apply sliding window if requested
        X_frames = apply_window(feat, window_size) if window_size > 1 else feat.T
        X_list.append(X_frames)    # (n_frames, n_features * window_size)
        y_list.append(ibm_labels)

    return np.vstack(X_list), np.concatenate(y_list)


def _load_clip(path: Path, n_samples: int, sr: int) -> np.ndarray:
    """Load audio from path, trim or zero-pad to exactly n_samples."""
    audio, _ = load_audio(path, sr=sr)
    if len(audio) >= n_samples:
        return audio[:n_samples]
    # zero-pad if shorter than requested
    return np.pad(audio, (0, n_samples - len(audio)))
