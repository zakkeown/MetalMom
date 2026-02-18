"""Tests for pitch_shift (pitch shifting via time stretch + resample)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_pitch_shift_returns_1d():
    y = _make_sine()
    result = metalmom.pitch_shift(y, sr=22050, n_steps=3)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_pitch_shift_same_length():
    """Output length should match input length."""
    y = _make_sine(duration=0.5)
    result = metalmom.pitch_shift(y, sr=22050, n_steps=5)
    assert len(result) == len(y), (
        f"Output length ({len(result)}) should match input ({len(y)})"
    )


def test_pitch_shift_zero_steps():
    """n_steps=0 should return same-length output."""
    y = _make_sine(duration=0.5)
    result = metalmom.pitch_shift(y, sr=22050, n_steps=0)
    assert len(result) == len(y)


def test_pitch_shift_positive_negative_differ():
    """Shifting up vs down should produce different results."""
    y = _make_sine(duration=0.25)
    up = metalmom.pitch_shift(y, sr=22050, n_steps=5)
    down = metalmom.pitch_shift(y, sr=22050, n_steps=-5)
    # They should differ
    assert not np.allclose(up, down, atol=1e-4), (
        "Positive and negative pitch shifts should produce different results"
    )


def test_pitch_shift_finite_values():
    """Output should contain only finite values."""
    y = _make_sine(duration=0.25)
    result = metalmom.pitch_shift(y, sr=22050, n_steps=7)
    assert np.all(np.isfinite(result)), "Output should contain only finite values"


def test_pitch_shift_custom_params():
    """Verify pitch_shift works with custom n_fft, hop_length, bins_per_octave."""
    y = _make_sine(duration=0.25)
    result = metalmom.pitch_shift(
        y, sr=22050, n_steps=3,
        bins_per_octave=24, n_fft=1024, hop_length=128,
    )
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(y)


def test_pitch_shift_negative():
    """Test negative pitch shift (shift down)."""
    y = _make_sine(duration=0.25)
    result = metalmom.pitch_shift(y, sr=22050, n_steps=-3)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(y)
    assert np.all(np.isfinite(result))


def test_pitch_shift_octave_up():
    """Test shifting up one octave (12 semitones)."""
    y = _make_sine(duration=0.25)
    result = metalmom.pitch_shift(y, sr=22050, n_steps=12)
    assert len(result) == len(y)
    assert np.all(np.isfinite(result))


def test_compat_shim():
    """Test the librosa compat shim."""
    from metalmom.compat.librosa.effects import pitch_shift as compat_ps

    y = _make_sine(duration=0.25)
    result = compat_ps(y, sr=22050, n_steps=3)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(y)


def test_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa
    y = _make_sine(duration=0.25)
    result = librosa.effects.pitch_shift(y, sr=22050, n_steps=3)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
