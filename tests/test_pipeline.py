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
    """SI-SDR of the output vs. the target speaker must exceed -10 dB.

    Threshold is deliberately lenient: validates that the output retains
    meaningful correlation with the target, not that separation is perfect.
    Pure-tone test signals are not representative of real speech and interact
    poorly with Griffin-Lim phase reconstruction (designed for natural speech).
    """
    mix_path, target = mix_and_target
    output_path = str(tmp_path / "output.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path)

    output, _ = sf.read(output_path)
    min_len = min(len(output), len(target))
    score = _si_sdr(target[:min_len].astype(np.float64), output[:min_len].astype(np.float64))
    assert score > -10.0, f"SI-SDR too low: {score:.2f} dB"


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


# ---------------------------------------------------------------------------
# Male target
# ---------------------------------------------------------------------------

def test_pipeline_male_target_output_created(
    classifier_path: str, mix_and_target: tuple, tmp_path: Path
) -> None:
    """Pipeline runs with target='male' without raising and creates the output file."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output_male.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path, target="male")

    assert Path(output_path).exists()


def test_pipeline_male_target_output_length(
    classifier_path: str, mix_and_target: tuple, tmp_path: Path
) -> None:
    """Male-target output length is within 5 % of the input length."""
    mix_path, _ = mix_and_target
    output_path = str(tmp_path / "output_male.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=output_path, target="male")

    mix_samples = int(_DURATION * SAMPLE_RATE)
    output, _ = sf.read(output_path)
    assert abs(len(output) - mix_samples) / mix_samples < 0.05


def test_pipeline_male_female_outputs_differ(
    classifier_path: str, mix_and_target: tuple, tmp_path: Path
) -> None:
    """Female and male target outputs are distinct signals (different masking applied)."""
    mix_path, _ = mix_and_target
    out_female = str(tmp_path / "out_female.wav")
    out_male   = str(tmp_path / "out_male.wav")

    run(input_path=mix_path, model_path=classifier_path, output_path=out_female, target="female")
    run(input_path=mix_path, model_path=classifier_path, output_path=out_male,   target="male")

    audio_f, _ = sf.read(out_female)
    audio_m, _ = sf.read(out_male)
    min_len = min(len(audio_f), len(audio_m))
    # Outputs must not be identical — different masks produce different waveforms
    assert not np.allclose(audio_f[:min_len], audio_m[:min_len])


# ---------------------------------------------------------------------------
# Pitch-based stream selection
# ---------------------------------------------------------------------------

def test_mean_active_pitch_female_higher_than_male() -> None:
    """A 250 Hz sine (female range) reports a higher mean pitch than 120 Hz (male range)."""
    from src.ai.attention import _mean_active_pitch

    female_stream = _sine(250.0)
    male_stream   = _sine(120.0)
    assert _mean_active_pitch(female_stream, SAMPLE_RATE) > _mean_active_pitch(male_stream, SAMPLE_RATE)


def test_select_stream_pitch_selects_female(classifier_path: str) -> None:
    """Pitch method selects the higher-F0 stream when target is female."""
    from src.ai.attention import AttentionModule
    from src.ai.classifier import SpeakerClassifier

    attn = AttentionModule(SpeakerClassifier.load(classifier_path))
    female_stream = _sine(250.0)
    male_stream   = _sine(120.0)

    selected, conf = attn.select_stream(
        (female_stream, male_stream), target_gender=0, sr=SAMPLE_RATE, method="pitch"
    )
    assert np.allclose(selected, female_stream)
    assert conf > 0.0


def test_select_stream_pitch_selects_male(classifier_path: str) -> None:
    """Pitch method selects the lower-F0 stream when target is male."""
    from src.ai.attention import AttentionModule
    from src.ai.classifier import SpeakerClassifier

    attn = AttentionModule(SpeakerClassifier.load(classifier_path))
    female_stream = _sine(250.0)
    male_stream   = _sine(120.0)

    selected, conf = attn.select_stream(
        (female_stream, male_stream), target_gender=1, sr=SAMPLE_RATE, method="pitch"
    )
    assert np.allclose(selected, male_stream)
    assert conf > 0.0


# ---------------------------------------------------------------------------
# VAD gate
# ---------------------------------------------------------------------------

def _speech_with_bleedthrough_pause(sr: int = SAMPLE_RATE) -> np.ndarray:
    """0.5s speech (250 Hz) + 0.8s near-silence (120 Hz at −60 dB) + 0.5s speech."""
    speech      = _sine(250.0, duration=0.5, sr=sr)
    bleedthrough = _sine(120.0, duration=0.8, sr=sr) * 1e-3   # −60 dB relative to speech
    speech2     = _sine(250.0, duration=0.5, sr=sr)
    return np.concatenate([speech, bleedthrough, speech2]).astype(np.float32)


def test_voice_activity_gate_attenuates_bleedthrough() -> None:
    """Gate suppresses residual energy in the inter-word pause by at least 20 dB."""
    from src.dsp.enhancement import voice_activity_gate

    audio = _speech_with_bleedthrough_pause()
    gated = voice_activity_gate(audio, sr=SAMPLE_RATE)

    # Centre of silence region: 0.5s speech + hold(0.2s) + release(0.06s) → safe from ~0.8s
    check_start = int(0.85 * SAMPLE_RATE)
    check_end   = int(1.10 * SAMPLE_RATE)
    rms_before = np.sqrt(np.mean(audio[check_start:check_end] ** 2))
    rms_after  = np.sqrt(np.mean(gated[check_start:check_end] ** 2))
    assert rms_after < rms_before * 0.1   # > 20 dB attenuation


def test_voice_activity_gate_preserves_sustained_speech() -> None:
    """Gate does not attenuate sustained active speech after the attack phase."""
    from src.dsp.enhancement import voice_activity_gate

    speech = _sine(250.0, duration=1.0)
    gated  = voice_activity_gate(speech, sr=SAMPLE_RATE)

    onset = int(0.05 * SAMPLE_RATE)   # skip first 50 ms (attack ramp)
    assert np.allclose(gated[onset:], speech[onset:], atol=1e-3)


def test_voice_activity_gate_empty_returns_empty() -> None:
    """Gate returns an empty array unchanged."""
    from src.dsp.enhancement import voice_activity_gate

    result = voice_activity_gate(np.zeros(0, dtype=np.float32), sr=SAMPLE_RATE)
    assert len(result) == 0
