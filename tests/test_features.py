import numpy as np
import pytest

from src.dsp.features import (
    N_FEATURES,
    N_MFCC,
    extract_all,
    extract_mfcc,
    extract_mfcc_delta,
    extract_pitch,
    extract_rms,
    extract_spectral,
    extract_zcr,
)
from src.utils import SAMPLE_RATE

DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def _sine(freq: float, duration: float = DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _silence(duration: float = DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


# --- MFCC ---

def test_mfcc_shape():
    audio = _sine(440.0)
    mfcc = extract_mfcc(audio)
    assert mfcc.shape[0] == N_MFCC
    assert mfcc.shape[1] > 0


# --- Pitch ---

def test_pitch_shape():
    audio = _sine(200.0)
    pitch = extract_pitch(audio)
    assert pitch.shape[0] == 1
    assert pitch.shape[1] > 0


def test_pitch_no_nan():
    audio = _sine(200.0)
    pitch = extract_pitch(audio)
    assert not np.any(np.isnan(pitch))


def test_pitch_silence_is_zero():
    audio = _silence()
    pitch = extract_pitch(audio)
    # Silence has no voiced frames — all F0 values should be 0
    assert np.all(pitch == 0.0)


# --- RMS ---

def test_rms_shape():
    audio = _sine(440.0)
    rms = extract_rms(audio)
    assert rms.shape[0] == 1
    assert rms.shape[1] > 0


def test_rms_silence_near_zero():
    audio = _silence()
    rms = extract_rms(audio)
    assert np.all(rms < 1e-6)


def test_rms_louder_signal_higher_energy():
    quiet = _sine(440.0) * 0.1
    loud = _sine(440.0) * 0.9
    assert np.mean(extract_rms(loud)) > np.mean(extract_rms(quiet))


# --- Spectral ---

def test_spectral_shape():
    audio = _sine(440.0)
    spec = extract_spectral(audio)
    assert spec.shape[0] == 2
    assert spec.shape[1] > 0


# --- MFCC delta ---

def test_mfcc_delta_shape():
    audio = _sine(440.0)
    delta = extract_mfcc_delta(audio)
    assert delta.shape[0] == 2 * N_MFCC
    assert delta.shape[1] > 0


# --- ZCR ---

def test_zcr_shape():
    audio = _sine(440.0)
    zcr = extract_zcr(audio)
    assert zcr.shape[0] == 1
    assert zcr.shape[1] > 0


def test_zcr_silence_near_zero():
    audio = _silence()
    zcr = extract_zcr(audio)
    assert np.all(zcr < 1e-6)


# --- extract_all ---

def test_extract_all_shape():
    audio = _sine(440.0)
    features = extract_all(audio)
    assert features.shape[0] == N_FEATURES  # 13 MFCC + 13Δ + 13Δ² + pitch + rms + centroid + rolloff + ZCR
    assert features.shape[1] > 0


def test_extract_all_no_nan():
    audio = _sine(440.0)
    features = extract_all(audio)
    assert not np.any(np.isnan(features))


def test_extract_all_silence_no_nan():
    audio = _silence()
    features = extract_all(audio)
    assert not np.any(np.isnan(features))
