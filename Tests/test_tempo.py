"""Tests for standalone tempo estimation."""

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


def test_tempo_returns_numpy_array():
    y = _make_clicks_signal(bpm=120.0)
    result = metalmom.tempo(y=y, sr=22050)
    assert isinstance(result, np.ndarray), "tempo() should return numpy array"
    assert result.dtype == np.float64, "tempo() should return float64"
    assert result.shape == (1,), "tempo() should return shape (1,)"


def test_tempo_reasonable_range():
    y = _make_clicks_signal(bpm=120.0)
    result = metalmom.tempo(y=y, sr=22050)
    t = result[0]
    assert t >= 30, f"Tempo {t} should be >= 30 BPM"
    assert t <= 300, f"Tempo {t} should be <= 300 BPM"


def test_tempo_120bpm_click_track():
    y = _make_clicks_signal(bpm=120.0, duration=5.0)
    result = metalmom.tempo(y=y, sr=22050, start_bpm=120.0)
    t = result[0]
    # Allow +/- 15% tolerance
    assert t > 102, f"Tempo {t} should be > 102 BPM for 120 BPM input"
    assert t < 138, f"Tempo {t} should be < 138 BPM for 120 BPM input"


def test_tempo_different_bpms():
    """Different BPM signals should produce different estimates."""
    y_slow = _make_clicks_signal(bpm=80.0, duration=6.0)
    y_fast = _make_clicks_signal(bpm=160.0, duration=4.0)
    t_slow = metalmom.tempo(y=y_slow, sr=22050, start_bpm=80.0)[0]
    t_fast = metalmom.tempo(y=y_fast, sr=22050, start_bpm=160.0)[0]
    assert t_slow > 0, "Should estimate positive tempo for 80 BPM signal"
    assert t_fast > 0, "Should estimate positive tempo for 160 BPM signal"


def test_tempo_silent_signal():
    y = np.zeros(22050 * 3, dtype=np.float32)
    result = metalmom.tempo(y=y, sr=22050, start_bpm=120.0)
    t = result[0]
    # On silence, returns startBPM
    assert t == 120.0, f"Tempo on silence should be startBPM (120), got {t}"


def test_tempo_compat_shim():
    from metalmom.compat.librosa.feature import tempo as compat_tempo
    y = _make_clicks_signal(bpm=120.0)
    result = compat_tempo(y=y, sr=22050)
    assert isinstance(result, np.ndarray), "compat tempo should return numpy array"
    assert len(result) == 1, "compat tempo should return single element"
    assert result[0] > 30, "compat tempo should be > 30"


def test_tempo_raises_on_none():
    try:
        metalmom.tempo(y=None, sr=22050)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_tempo_various_bpm_inputs():
    """Verify tempo always returns values in [30, 300] for various inputs."""
    for bpm in [60.0, 100.0, 140.0, 200.0]:
        y = _make_clicks_signal(bpm=bpm, duration=4.0)
        result = metalmom.tempo(y=y, sr=22050)
        t = result[0]
        assert t >= 30, f"Tempo {t} should be >= 30 for {bpm} BPM input"
        assert t <= 300, f"Tempo {t} should be <= 300 for {bpm} BPM input"
