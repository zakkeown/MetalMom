"""Tests for tempogram and fourier_tempogram."""

import numpy as np
import metalmom


def _make_clicks_signal(bpm=120.0, sr=22050, duration=5.0):
    """Generate a signal with short sine bursts at the given BPM."""
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    interval = 60.0 / bpm
    t = 0.0
    while t < duration:
        idx = int(t * sr)
        burst_len = min(256, n - idx)
        if burst_len > 0:
            burst = np.sin(np.arange(burst_len) * 1000.0 * 2 * np.pi / sr).astype(np.float32)
            y[idx:idx + burst_len] = burst * 0.9
        t += interval
    return y


# ---------- Autocorrelation tempogram ----------

def test_tempogram_returns_2d():
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = metalmom.tempogram(y=y, sr=22050)
    assert isinstance(result, np.ndarray), "tempogram should return numpy array"
    assert result.ndim == 2, f"tempogram should be 2D, got {result.ndim}D"


def test_tempogram_correct_shape():
    win_length = 384
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = metalmom.tempogram(y=y, sr=22050, win_length=win_length)
    assert result.shape[0] == win_length, (
        f"First dim should be win_length={win_length}, got {result.shape[0]}"
    )
    assert result.shape[1] > 0, "Should have at least one frame"


def test_tempogram_custom_win_length():
    win_length = 128
    y = _make_clicks_signal(bpm=120.0, duration=2.0)
    result = metalmom.tempogram(y=y, sr=22050, win_length=win_length)
    assert result.shape[0] == win_length, (
        f"First dim should be win_length={win_length}, got {result.shape[0]}"
    )


def test_tempogram_values_finite():
    y = _make_clicks_signal(bpm=120.0, duration=2.0)
    result = metalmom.tempogram(y=y, sr=22050, win_length=128)
    assert np.all(np.isfinite(result)), "All tempogram values should be finite"


def test_tempogram_silence():
    y = np.zeros(22050 * 2, dtype=np.float32)
    result = metalmom.tempogram(y=y, sr=22050, win_length=64)
    assert result.shape[0] == 64
    assert result.shape[1] > 0
    # Silence: all values should be zero (or very close)
    assert np.allclose(result, 0, atol=1e-6), "Tempogram of silence should be all zeros"


def test_tempogram_raises_on_none():
    try:
        metalmom.tempogram(y=None, sr=22050)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------- Fourier tempogram ----------

def test_fourier_tempogram_returns_2d():
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = metalmom.fourier_tempogram(y=y, sr=22050)
    assert isinstance(result, np.ndarray), "fourier_tempogram should return numpy array"
    assert result.ndim == 2, f"fourier_tempogram should be 2D, got {result.ndim}D"


def test_fourier_tempogram_correct_shape():
    win_length = 384
    expected_freqs = win_length // 2 + 1
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = metalmom.fourier_tempogram(y=y, sr=22050, win_length=win_length)
    assert result.shape[0] == expected_freqs, (
        f"First dim should be win_length//2+1={expected_freqs}, got {result.shape[0]}"
    )
    assert result.shape[1] > 0, "Should have at least one frame"


def test_fourier_tempogram_custom_win_length():
    win_length = 128
    expected_freqs = win_length // 2 + 1
    y = _make_clicks_signal(bpm=120.0, duration=2.0)
    result = metalmom.fourier_tempogram(y=y, sr=22050, win_length=win_length)
    assert result.shape[0] == expected_freqs, (
        f"First dim should be {expected_freqs}, got {result.shape[0]}"
    )


def test_fourier_tempogram_non_negative():
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = metalmom.fourier_tempogram(y=y, sr=22050, win_length=128)
    assert np.all(result >= 0), "Fourier tempogram magnitudes should be non-negative"


def test_fourier_tempogram_values_finite():
    y = _make_clicks_signal(bpm=120.0, duration=2.0)
    result = metalmom.fourier_tempogram(y=y, sr=22050, win_length=128)
    assert np.all(np.isfinite(result)), "All Fourier tempogram values should be finite"


def test_fourier_tempogram_silence():
    y = np.zeros(22050 * 2, dtype=np.float32)
    result = metalmom.fourier_tempogram(y=y, sr=22050, win_length=64)
    expected_freqs = 64 // 2 + 1
    assert result.shape[0] == expected_freqs
    assert result.shape[1] > 0
    assert np.allclose(result, 0, atol=1e-6), "Fourier tempogram of silence should be all zeros"


def test_fourier_tempogram_raises_on_none():
    try:
        metalmom.fourier_tempogram(y=None, sr=22050)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------- Cross-variant consistency ----------

def test_frame_counts_match():
    """Both tempogram variants should have the same number of output frames."""
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    win_length = 128
    acf = metalmom.tempogram(y=y, sr=22050, win_length=win_length)
    ft = metalmom.fourier_tempogram(y=y, sr=22050, win_length=win_length)
    assert acf.shape[1] == ft.shape[1], (
        f"ACF frames ({acf.shape[1]}) should match Fourier frames ({ft.shape[1]})"
    )


# ---------- Compat shim ----------

def test_tempogram_compat_shim():
    from metalmom.compat.librosa.feature import tempogram as compat_tempogram
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = compat_tempogram(y=y, sr=22050, win_length=128)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[0] == 128


def test_fourier_tempogram_compat_shim():
    from metalmom.compat.librosa.feature import fourier_tempogram as compat_ft
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    result = compat_ft(y=y, sr=22050, win_length=128)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[0] == 128 // 2 + 1
