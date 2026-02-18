"""Tests for piptrack (parabolic interpolation pitch tracking)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=1.0):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def test_piptrack_returns_tuple_of_two():
    y = _make_sine(440.0)
    result = metalmom.piptrack(y=y, sr=22050)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_piptrack_shapes_match():
    y = _make_sine(440.0)
    n_fft = 2048
    pitches, magnitudes = metalmom.piptrack(y=y, sr=22050, n_fft=n_fft)
    n_freqs = n_fft // 2 + 1
    assert pitches.shape[0] == n_freqs
    assert magnitudes.shape[0] == n_freqs
    assert pitches.shape[1] == magnitudes.shape[1]
    assert pitches.shape[1] > 0


def test_piptrack_sine_440_has_peak():
    y = _make_sine(440.0, sr=22050, duration=1.0)
    pitches, magnitudes = metalmom.piptrack(y=y, sr=22050, n_fft=2048,
                                             fmin=100.0, fmax=8000.0,
                                             threshold=0.1)
    # For each interior frame, find strongest pitch
    n_frames = pitches.shape[1]
    found_440 = 0
    start = 2
    end = max(start, n_frames - 2)

    for frame in range(start, end):
        col_mags = magnitudes[:, frame]
        col_pitches = pitches[:, frame]
        if col_mags.max() > 0:
            best_pitch = col_pitches[col_mags.argmax()]
            if best_pitch > 0 and abs(best_pitch - 440.0) < 20.0:
                found_440 += 1

    interior_count = end - start
    assert interior_count > 0
    ratio = found_440 / interior_count
    assert ratio > 0.7, f"Expected >70% frames near 440 Hz, got {ratio:.2%}"


def test_piptrack_pitches_in_range():
    y = _make_sine(440.0)
    fmin, fmax = 150.0, 4000.0
    pitches, _ = metalmom.piptrack(y=y, sr=22050, fmin=fmin, fmax=fmax)
    nonzero = pitches[pitches > 0]
    if len(nonzero) > 0:
        assert nonzero.min() >= fmin, f"Min pitch {nonzero.min()} < fmin {fmin}"
        assert nonzero.max() <= fmax, f"Max pitch {nonzero.max()} > fmax {fmax}"


def test_piptrack_magnitudes_non_negative():
    y = _make_sine(440.0)
    _, magnitudes = metalmom.piptrack(y=y, sr=22050)
    assert np.all(magnitudes >= 0), "Magnitudes should be non-negative"


def test_piptrack_silence_all_zeros():
    y = np.zeros(22050, dtype=np.float32)
    pitches, magnitudes = metalmom.piptrack(y=y, sr=22050)
    np.testing.assert_array_equal(pitches, 0, err_msg="Silence pitches should be all zeros")
    np.testing.assert_array_equal(magnitudes, 0, err_msg="Silence magnitudes should be all zeros")


def test_piptrack_compat_shim():
    from metalmom.compat.librosa.pitch import piptrack as compat_piptrack
    y = _make_sine(440.0)
    pitches, magnitudes = compat_piptrack(y=y, sr=22050)
    assert pitches.ndim == 2
    assert magnitudes.ndim == 2
    assert pitches.shape == magnitudes.shape
