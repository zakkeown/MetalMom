"""Tests for preemphasis and deemphasis filters."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.1):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


# -- Preemphasis tests --------------------------------------------------------

def test_preemphasis_returns_1d():
    """preemphasis() should return a 1-D array."""
    y = _make_sine()
    result = metalmom.preemphasis(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_preemphasis_same_length():
    """Output length should match input length."""
    y = _make_sine()
    result = metalmom.preemphasis(y)
    assert len(result) == len(y)


def test_preemphasis_coef_zero_identity():
    """With coef=0, preemphasis should be identity."""
    y = _make_sine()
    result = metalmom.preemphasis(y, coef=0.0)
    np.testing.assert_allclose(result, y, atol=1e-6)


def test_preemphasis_first_sample():
    """First sample should equal input x[0]."""
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = metalmom.preemphasis(y, coef=0.97)
    assert abs(result[0] - 1.0) < 1e-6


def test_preemphasis_manual_values():
    """Check against manually computed values."""
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    coef = 0.97
    result = metalmom.preemphasis(y, coef=coef)
    expected = np.array([
        1.0,
        2.0 - coef * 1.0,
        3.0 - coef * 2.0,
        4.0 - coef * 3.0,
    ], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# -- Deemphasis tests ---------------------------------------------------------

def test_deemphasis_returns_1d():
    """deemphasis() should return a 1-D array."""
    y = _make_sine()
    result = metalmom.deemphasis(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_deemphasis_same_length():
    """Output length should match input length."""
    y = _make_sine()
    result = metalmom.deemphasis(y)
    assert len(result) == len(y)


def test_deemphasis_coef_zero_identity():
    """With coef=0, deemphasis should be identity."""
    y = _make_sine()
    result = metalmom.deemphasis(y, coef=0.0)
    np.testing.assert_allclose(result, y, atol=1e-6)


def test_deemphasis_manual_values():
    """Check against manually computed values."""
    y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    coef = 0.5
    result = metalmom.deemphasis(y, coef=coef)
    expected = np.array([1.0, 1.5, 1.75, 1.875], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# -- Round-trip tests ----------------------------------------------------------

def test_roundtrip_preemphasis_then_deemphasis():
    """deemphasis(preemphasis(x)) should approximately recover x."""
    y = _make_sine(duration=0.05)
    pre = metalmom.preemphasis(y, coef=0.97)
    roundtrip = metalmom.deemphasis(pre, coef=0.97)
    np.testing.assert_allclose(roundtrip, y, atol=1e-4)


def test_roundtrip_deemphasis_then_preemphasis():
    """preemphasis(deemphasis(x)) should approximately recover x."""
    y = _make_sine(duration=0.05)
    de = metalmom.deemphasis(y, coef=0.97)
    roundtrip = metalmom.preemphasis(de, coef=0.97)
    np.testing.assert_allclose(roundtrip, y, atol=1e-4)


# -- Compat shim tests --------------------------------------------------------

def test_preemphasis_compat_shim():
    """Test the librosa compat shim for preemphasis."""
    from metalmom.compat.librosa.effects import preemphasis as compat_preemphasis

    y = _make_sine()
    result = compat_preemphasis(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(y)


def test_deemphasis_compat_shim():
    """Test the librosa compat shim for deemphasis."""
    from metalmom.compat.librosa.effects import deemphasis as compat_deemphasis

    y = _make_sine()
    result = compat_deemphasis(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(y)


def test_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa

    y = _make_sine()
    pre = librosa.effects.preemphasis(y)
    de = librosa.effects.deemphasis(y)
    assert isinstance(pre, np.ndarray)
    assert isinstance(de, np.ndarray)
