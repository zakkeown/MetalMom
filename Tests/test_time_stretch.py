"""Tests for time_stretch (phase vocoder time stretching)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_time_stretch_returns_1d():
    y = _make_sine()
    result = metalmom.time_stretch(y, rate=1.0, sr=22050)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_time_stretch_speed_up():
    """rate=2.0 should produce shorter output."""
    y = _make_sine(duration=0.5)
    result = metalmom.time_stretch(y, rate=2.0, sr=22050)
    # Output should be roughly half the input length
    assert len(result) < len(y), (
        f"rate=2.0 output ({len(result)}) should be shorter than input ({len(y)})"
    )


def test_time_stretch_slow_down():
    """rate=0.5 should produce longer output."""
    y = _make_sine(duration=0.5)
    result = metalmom.time_stretch(y, rate=0.5, sr=22050)
    # Output should be roughly double the input length
    assert len(result) > len(y), (
        f"rate=0.5 output ({len(result)}) should be longer than input ({len(y)})"
    )


def test_time_stretch_identity():
    """rate=1.0 should approximately preserve length."""
    y = _make_sine(duration=0.5)
    result = metalmom.time_stretch(y, rate=1.0, sr=22050)
    ratio = len(result) / len(y)
    assert 0.8 < ratio < 1.2, (
        f"rate=1.0 length ratio {ratio:.3f} should be ~1.0"
    )


def test_time_stretch_finite_values():
    """Output should contain only finite values."""
    y = _make_sine(duration=0.25)
    result = metalmom.time_stretch(y, rate=1.5, sr=22050)
    assert np.all(np.isfinite(result)), "Output should contain only finite values"


def test_time_stretch_different_rates():
    """Faster rate should produce shorter output than slower rate."""
    y = _make_sine(duration=0.5)
    fast = metalmom.time_stretch(y, rate=2.0, sr=22050)
    slow = metalmom.time_stretch(y, rate=0.5, sr=22050)
    assert len(slow) > len(fast), (
        f"rate=0.5 ({len(slow)}) should be longer than rate=2.0 ({len(fast)})"
    )


def test_time_stretch_custom_params():
    """Verify time_stretch works with custom n_fft and hop_length."""
    y = _make_sine(duration=0.25)
    result = metalmom.time_stretch(y, rate=1.5, sr=22050, n_fft=1024, hop_length=128)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) > 0


def test_compat_shim():
    """Test the librosa compat shim."""
    from metalmom.compat.librosa.effects import time_stretch as compat_ts

    y = _make_sine(duration=0.25)
    result = compat_ts(y, rate=1.5, sr=22050)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa
    y = _make_sine(duration=0.25)
    result = librosa.effects.time_stretch(y, rate=1.5, sr=22050)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
