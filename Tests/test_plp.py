"""Tests for Predominant Local Pulse (PLP)."""

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


def test_plp_returns_1d_array():
    y = _make_clicks_signal(bpm=120.0)
    pulse = metalmom.plp(y=y, sr=22050)
    assert isinstance(pulse, np.ndarray)
    assert pulse.ndim == 1


def test_plp_non_negative():
    y = _make_clicks_signal(bpm=120.0)
    pulse = metalmom.plp(y=y, sr=22050)
    assert np.all(pulse >= 0), "PLP values should be non-negative (half-wave rectified)"


def test_plp_reasonable_length():
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    pulse = metalmom.plp(y=y, sr=22050)
    # Should have a reasonable number of frames
    assert len(pulse) > 0, "PLP should return non-empty array"
    # Roughly: n_frames ~ signal_length / hop_length
    expected_approx = len(y) // 512
    assert len(pulse) > expected_approx * 0.5, "PLP length should be reasonable"
    assert len(pulse) < expected_approx * 2, "PLP length should be reasonable"


def test_plp_periodic_signal_has_peaks():
    y = _make_clicks_signal(bpm=120.0, duration=5.0)
    pulse = metalmom.plp(y=y, sr=22050)
    assert np.max(pulse) > 0, "PLP of periodic signal should have non-zero peaks"


def test_plp_silence_returns_zeros():
    y = np.zeros(22050 * 2, dtype=np.float32)
    pulse = metalmom.plp(y=y, sr=22050)
    assert isinstance(pulse, np.ndarray)
    assert np.max(pulse) < 1e-6, "PLP of silence should be near-zero"


def test_plp_normalized_to_unit_range():
    y = _make_clicks_signal(bpm=120.0, duration=3.0)
    pulse = metalmom.plp(y=y, sr=22050)
    assert np.all(pulse <= 1.0 + 1e-6), "PLP values should be <= 1.0"


def test_plp_compat_shim():
    from metalmom.compat.librosa.beat import plp as compat_plp
    y = _make_clicks_signal(bpm=120.0)
    pulse = compat_plp(y=y, sr=22050)
    assert isinstance(pulse, np.ndarray)
    assert pulse.ndim == 1
    assert np.all(pulse >= 0)


def test_plp_custom_params():
    y = _make_clicks_signal(bpm=120.0, duration=4.0)
    pulse = metalmom.plp(
        y=y, sr=22050, hop_length=256, win_length=192,
        tempo_min=60.0, tempo_max=200.0,
    )
    assert isinstance(pulse, np.ndarray)
    assert pulse.ndim == 1
    assert len(pulse) > 0
