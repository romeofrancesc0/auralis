"""Tests for the separate-then-select pipeline components."""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.classifier import SpeakerClassifier
from src.dsp.features import extract_all
from src.utils import SAMPLE_RATE

_DURATION = 1.5  # seconds


def _sine(freq: float, duration: float = _DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture(scope="module")
def classifier_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Minimal SpeakerClassifier trained on sine waves (250 Hz = F, 120 Hz = M)."""
    female = _sine(250.0)
    male   = _sine(120.0)

    feat_f = extract_all(female, sr=SAMPLE_RATE)
    feat_m = extract_all(male,   sr=SAMPLE_RATE)

    n_f, n_m = feat_f.shape[1], feat_m.shape[1]
    X = np.hstack([feat_f, feat_m]).T
    y = np.array([0] * n_f + [1] * n_m)

    clf = SpeakerClassifier()
    clf.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "classifier.joblib"
    clf.save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Pipeline interface
# ---------------------------------------------------------------------------

def test_run_classifier_mode_requires_model() -> None:
    """run() raises ValueError when classifier mode is selected without a model."""
    from src.pipeline import run

    with pytest.raises(ValueError, match="--model is required"):
        run(
            input_path="dummy.wav",
            output_path="dummy_out.wav",
            separator_path="dummy_sep.pt",
            model_path=None,
            stream_select="classifier",
        )


# ---------------------------------------------------------------------------
# Pitch-based stream selection
# ---------------------------------------------------------------------------

def test_mean_active_pitch_female_higher_than_male() -> None:
    """A 250 Hz sine (female range) reports a higher mean pitch than 120 Hz (male range)."""
    from src.ai.attention import _mean_active_pitch

    assert _mean_active_pitch(_sine(250.0), SAMPLE_RATE) > _mean_active_pitch(_sine(120.0), SAMPLE_RATE)


def test_select_stream_pitch_selects_female(classifier_path: str) -> None:
    """Pitch method selects the higher-F0 stream when target is female."""
    from src.ai.attention import AttentionModule

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
    speech       = _sine(250.0, duration=0.5, sr=sr)
    bleedthrough = _sine(120.0, duration=0.8, sr=sr) * 1e-3
    speech2      = _sine(250.0, duration=0.5, sr=sr)
    return np.concatenate([speech, bleedthrough, speech2]).astype(np.float32)


def test_voice_activity_gate_attenuates_bleedthrough() -> None:
    """Gate suppresses residual energy in the inter-word pause by at least 20 dB."""
    from src.dsp.enhancement import voice_activity_gate

    audio = _speech_with_bleedthrough_pause()
    gated = voice_activity_gate(audio, sr=SAMPLE_RATE)

    check_start = int(0.85 * SAMPLE_RATE)
    check_end   = int(1.10 * SAMPLE_RATE)
    rms_before  = np.sqrt(np.mean(audio[check_start:check_end] ** 2))
    rms_after   = np.sqrt(np.mean(gated[check_start:check_end] ** 2))
    assert rms_after < rms_before * 0.1


def test_voice_activity_gate_preserves_sustained_speech() -> None:
    """Gate does not attenuate sustained active speech after the attack phase."""
    from src.dsp.enhancement import voice_activity_gate

    speech = _sine(250.0, duration=1.0)
    gated  = voice_activity_gate(speech, sr=SAMPLE_RATE)

    onset = int(0.05 * SAMPLE_RATE)
    assert np.allclose(gated[onset:], speech[onset:], atol=1e-3)


def test_voice_activity_gate_empty_returns_empty() -> None:
    """Gate returns an empty array unchanged."""
    from src.dsp.enhancement import voice_activity_gate

    result = voice_activity_gate(np.zeros(0, dtype=np.float32), sr=SAMPLE_RATE)
    assert len(result) == 0
