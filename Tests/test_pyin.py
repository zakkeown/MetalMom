"""Tests for pYIN probabilistic pitch estimation."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=1.0):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_pyin_returns_tuple_of_3():
    y = _make_sine(440.0)
    result = metalmom.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert isinstance(result, tuple), "pyin should return a tuple"
    assert len(result) == 3, "pyin should return 3 arrays"
    f0, voiced_flag, voiced_probs = result
    assert f0.ndim == 1
    assert voiced_flag.ndim == 1
    assert voiced_probs.ndim == 1
    assert len(f0) == len(voiced_flag) == len(voiced_probs)
    assert len(f0) > 0


def test_pyin_sine_440():
    y = _make_sine(440.0)
    f0, voiced_flag, voiced_probs = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050,
        frame_length=2048,
    )
    # Skip edge frames
    interior_f0 = f0[2:-2]
    interior_flag = voiced_flag[2:-2]
    voiced = interior_f0[interior_flag]
    assert len(voiced) > len(interior_f0) * 0.7, (
        f"At least 70% of interior frames should be voiced, got {len(voiced)}/{len(interior_f0)}"
    )
    # Remove NaN for error computation
    voiced_valid = voiced[np.isfinite(voiced)]
    if len(voiced_valid) > 0:
        avg_error = np.mean(np.abs(voiced_valid - 440.0))
        assert avg_error < 10.0, f"Average F0 error should be < 10 Hz, got {avg_error}"


def test_pyin_silence():
    y = np.zeros(22050, dtype=np.float32)
    f0, voiced_flag, voiced_probs = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050
    )
    # Most frames should be unvoiced for silence
    unvoiced_ratio = np.sum(~voiced_flag) / len(voiced_flag)
    assert unvoiced_ratio > 0.7, (
        f"At least 70% of frames should be unvoiced for silence, got {unvoiced_ratio}"
    )


def test_pyin_fill_na():
    y = np.zeros(22050, dtype=np.float32)
    f0, voiced_flag, _ = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050, fill_na=np.nan
    )
    # Unvoiced frames should have NaN
    unvoiced_f0 = f0[~voiced_flag]
    if len(unvoiced_f0) > 0:
        assert np.all(np.isnan(unvoiced_f0)), "Unvoiced frames should be NaN with fill_na=np.nan"


def test_pyin_fill_na_zero():
    y = np.zeros(22050, dtype=np.float32)
    f0, voiced_flag, _ = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050, fill_na=0.0
    )
    # Unvoiced frames should have 0
    unvoiced_f0 = f0[~voiced_flag]
    if len(unvoiced_f0) > 0:
        assert np.all(unvoiced_f0 == 0), "Unvoiced frames should be 0 with fill_na=0"


def test_pyin_fill_na_none():
    y = np.zeros(22050, dtype=np.float32)
    f0, voiced_flag, _ = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050, fill_na=None
    )
    # Unvoiced frames should have 0 (raw output)
    unvoiced_f0 = f0[~voiced_flag]
    if len(unvoiced_f0) > 0:
        assert np.all(unvoiced_f0 == 0), "Unvoiced frames should be 0 with fill_na=None"


def test_pyin_voiced_probs_in_range():
    y = _make_sine(440.0)
    _, _, voiced_probs = metalmom.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert np.all(voiced_probs >= 0), "Voiced probs should be >= 0"
    assert np.all(voiced_probs <= 1), "Voiced probs should be <= 1"


def test_pyin_voiced_flag_boolean():
    y = _make_sine(440.0)
    _, voiced_flag, _ = metalmom.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert voiced_flag.dtype == bool, f"Voiced flag should be bool, got {voiced_flag.dtype}"


def test_pyin_f0_in_range():
    y = _make_sine(440.0)
    f0, voiced_flag, _ = metalmom.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        assert np.all(voiced_f0 >= 65.0), "Voiced f0 should be >= fmin"
        assert np.all(voiced_f0 <= 2093.0), "Voiced f0 should be <= fmax"


def test_pyin_compat_shim():
    from metalmom.compat.librosa.pitch import pyin as compat_pyin
    y = _make_sine(440.0)
    result = compat_pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_pyin_output_shape_center():
    y = _make_sine(440.0, sr=22050, duration=1.0)
    frame_length = 2048
    hop_length = 512
    f0, _, _ = metalmom.pyin(
        y, fmin=65.0, fmax=2093.0, sr=22050,
        frame_length=frame_length, hop_length=hop_length, center=True
    )
    padded = len(y) + frame_length
    expected = 1 + (padded - frame_length) // hop_length
    assert len(f0) == expected, f"Expected {expected} frames, got {len(f0)}"
