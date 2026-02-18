"""Tests for YIN pitch estimation."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=1.0):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_yin_returns_1d():
    y = _make_sine(440.0)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert result.ndim == 1
    assert len(result) > 0


def test_yin_sine_440():
    y = _make_sine(440.0)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050,
                          frame_length=2048, trough_threshold=0.1)
    # Skip edge frames
    interior = result[2:-2]
    voiced = interior[interior > 0]
    assert len(voiced) > len(interior) * 0.8, "Most interior frames should be voiced"
    avg_error = np.mean(np.abs(voiced - 440.0))
    assert avg_error < 5.0, f"Average F0 error should be < 5 Hz, got {avg_error}"


def test_yin_sine_220():
    y = _make_sine(220.0)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050,
                          frame_length=2048, trough_threshold=0.1)
    interior = result[2:-2]
    voiced = interior[interior > 0]
    assert len(voiced) > 0, "Should detect voiced frames for 220 Hz sine"
    avg_error = np.mean(np.abs(voiced - 220.0))
    assert avg_error < 5.0, f"Average F0 error should be < 5 Hz for 220 Hz, got {avg_error}"


def test_yin_silence():
    y = np.zeros(22050, dtype=np.float32)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050)
    np.testing.assert_array_equal(result, 0, err_msg="Silence should yield all zeros (unvoiced)")


def test_yin_non_negative():
    y = _make_sine(440.0)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert np.all(result >= 0), "YIN F0 should be non-negative"


def test_yin_finite():
    y = _make_sine(440.0)
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert np.all(np.isfinite(result)), "YIN F0 should be finite"


def test_yin_output_shape_center():
    y = _make_sine(440.0, sr=22050, duration=1.0)
    frame_length = 2048
    hop_length = 512
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050,
                          frame_length=frame_length, hop_length=hop_length,
                          center=True)
    padded = len(y) + frame_length
    expected = 1 + (padded - frame_length) // hop_length
    assert len(result) == expected, f"Expected {expected} frames, got {len(result)}"


def test_yin_output_shape_no_center():
    y = _make_sine(440.0, sr=22050, duration=1.0)
    frame_length = 2048
    hop_length = 512
    result = metalmom.yin(y, fmin=65.0, fmax=2093.0, sr=22050,
                          frame_length=frame_length, hop_length=hop_length,
                          center=False)
    expected = 1 + (len(y) - frame_length) // hop_length
    assert len(result) == expected, f"Expected {expected} frames, got {len(result)}"


def test_yin_compat_shim():
    from metalmom.compat.librosa.pitch import yin as compat_yin
    y = _make_sine(440.0)
    result = compat_yin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert result.ndim == 1
    assert len(result) > 0
