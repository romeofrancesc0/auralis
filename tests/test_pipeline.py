"""End-to-end integration tests for src.pipeline.run()."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.ai.classifier import SpeakerClassifier
from src.dsp.features import extract_all
from src.pipeline import run
from src.utils import SAMPLE_RATE, make_mixture, save_audio

_DURATION = 1.5  # seconds — short enough for fast tests, long enough for stable features


def _sine(freq: float, duration: float = _DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    dot = np.dot(ref, est)
    projection = (dot / (np.dot(ref, ref) + 1e-8)) * ref
    noise = est - projection
    return 10.0 * np.log10((np.dot(projection, projection) + 1e-8) / (np.dot(noise, noise) + 1e-8))


@pytest.fixture(scope="module")
def classifier_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a minimal SpeakerClassifier on sine waves and return the model path.

    250 Hz sits in the female pitch range (150–310 Hz); 120 Hz is below it.
    This gives the pitch feature enough signal to discriminate M/F.
    """
    female = _sine(250.0)
    male = _sine(120.0)

    feat_f = extract_all(female, sr=SAMPLE_RATE)  # (44, n_frames)
    feat_m = extract_all(male, sr=SAMPLE_RATE)    # (44, n_frames)

    n_f, n_m = feat_f.shape[1], feat_m.shape[1]
    X = np.hstack([feat_f, feat_m]).T             # (n_frames_total, 44)
    y = np.array([0] * n_f + [1] * n_m)           # 0 = F, 1 = M

    clf = SpeakerClassifier()
    clf.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "classifier.joblib"
    clf.save(path)
    return str(path)


@pytest.fixture(scope="module")
def mix_and_target(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, np.ndarray]:
    """Save a synthetic mix to disk and return (mix_path, target_signal)."""
    target = _sine(250.0)
    interferer = _sine(120.0)
    mix = make_mixture(target, interferer, snr_db=0.0)

    mix_path = tmp_path_factory.mktemp("audio") / "mix.wav"
    save_audio(mix_path, mix, sr=SAMPLE_RATE)
    return str(mix_path), target


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_pipeline_output_created(classifier_path: str, mix_and_target: tuple, tmp_path: Path) -> None:
    """Pipeline runs end-to-end without raising and creates the output file."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path)

    assert Path(output_path).exists()


def test_pipeline_output_is_valid_wav(classifier_path: str, mix_and_target: tuple, tmp_path: Path) -> None:
    """Output is a readable WAV file at the expected sample rate."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path)

    audio, sr = sf.read(output_path)
    assert sr == SAMPLE_RATE
    assert len(audio) > 0


def test_pipeline_output_length_matches_input(
    classifier_path: str, mix_and_target: tuple, tmp_path: Path
) -> None:
    """Output sample count is within 5 % of the input (STFT boundary tolerance)."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path)

    mix_samples = int(_DURATION * SAMPLE_RATE)
    output, _ = sf.read(output_path)
    assert abs(len(output) - mix_samples) / mix_samples < 0.05


# ---------------------------------------------------------------------------
# Quality test
# ---------------------------------------------------------------------------

def test_pipeline_si_sdr_above_threshold(
    classifier_path: str, mix_and_target: tuple, tmp_path: Path
) -> None:
    """SI-SDR of the output vs. the target speaker must exceed -5 dB.

    -5 dB is a deliberately lenient threshold: it validates that the output
    retains meaningful correlation with the target, not that separation is
    perfect. Raises the bar once evaluation metrics are established.
    """
    mix_path, target = mix_and_target
    output_path = str(tmp_path / "output.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path)

    output, _ = sf.read(output_path)
    min_len = min(len(output), len(target))
    score = _si_sdr(target[:min_len].astype(np.float64), output[:min_len].astype(np.float64))
    assert score > -5.0, f"SI-SDR too low: {score:.2f} dB"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_pipeline_missing_model_raises(mix_and_target: tuple, tmp_path: Path) -> None:
    """run() raises an error when the model file does not exist."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output.wav")

    with pytest.raises(Exception):
        run(
            input_path=mix_path,
            model_path="/nonexistent/path/classifier.joblib",
            output_path=output_path,
        )


def test_pipeline_missing_input_raises(classifier_path: str, tmp_path: Path) -> None:
    """run() raises FileNotFoundError when the input audio file does not exist."""
    output_path = str(tmp_path / "output.wav")

    with pytest.raises(FileNotFoundError):
        run(
            input_path="/nonexistent/path/mix.wav",
            model_path=classifier_path,
            output_path=output_path,
        )
