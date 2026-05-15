from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, _ = librosa.load(path, sr=sr, mono=True)
    logger.info("Loaded %s — %d samples @ %d Hz", path.name, len(audio), sr)
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
    logger.info("Saved %s — %d samples @ %d Hz", path.name, len(audio), sr)


def make_mixture(
    voice_a: np.ndarray,
    voice_b: np.ndarray,
    snr_db: float = 0.0,
) -> np.ndarray:
    """Mix two mono signals at a given SNR (voice_a is the reference).

    voice_b is scaled so that SNR = 20*log10(rms_a / rms_b_scaled) = snr_db.
    The result is peak-normalized to [-1, 1] to prevent clipping.
    """
    min_len = min(len(voice_a), len(voice_b))
    a = voice_a[:min_len].copy()
    b = voice_b[:min_len].copy()

    rms_a = np.sqrt(np.mean(a ** 2))
    rms_b = np.sqrt(np.mean(b ** 2))

    if rms_b > 0 and rms_a > 0:
        scale = rms_a / (rms_b * (10 ** (snr_db / 20.0)))
        b *= scale

    mixture = a + b
    peak = np.max(np.abs(mixture))
    if peak > 0:
        mixture /= peak
    return mixture
