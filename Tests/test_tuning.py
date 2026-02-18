"""Tests for tuning estimation."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=1.0):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_estimate_tuning_returns_float():
    y = _make_sine(440.0)
    result = metalmom.estimate_tuning(y=y, sr=22050)
    assert isinstance(result, float)


def test_estimate_tuning_440_near_zero():
    y = _make_sine(440.0, sr=22050, duration=1.0)
    tuning = metalmom.estimate_tuning(y=y, sr=22050, n_fft=2048)
    assert abs(tuning) < 0.15, f"440 Hz tuning should be near 0, got {tuning}"


def test_estimate_tuning_442_positive():
    y = _make_sine(442.0, sr=22050, duration=1.0)
    tuning = metalmom.estimate_tuning(y=y, sr=22050, n_fft=2048)
    assert tuning > 0, f"442 Hz should produce positive tuning, got {tuning}"


def test_estimate_tuning_438_negative():
    y = _make_sine(438.0, sr=22050, duration=1.0)
    tuning = metalmom.estimate_tuning(y=y, sr=22050, n_fft=2048)
    assert tuning < 0, f"438 Hz should produce negative tuning, got {tuning}"


def test_estimate_tuning_in_range():
    y = _make_sine(440.0)
    tuning = metalmom.estimate_tuning(y=y, sr=22050)
    assert -0.5 <= tuning <= 0.5, f"Tuning {tuning} outside [-0.5, 0.5]"


def test_estimate_tuning_silence():
    y = np.zeros(22050, dtype=np.float32)
    tuning = metalmom.estimate_tuning(y=y, sr=22050)
    assert tuning == 0.0, f"Silence should give tuning=0.0, got {tuning}"


def test_estimate_tuning_compat_shim():
    from metalmom.compat.librosa.pitch import estimate_tuning as compat_estimate_tuning
    y = _make_sine(440.0)
    tuning = compat_estimate_tuning(y=y, sr=22050)
    assert isinstance(tuning, float)
    assert -0.5 <= tuning <= 0.5
