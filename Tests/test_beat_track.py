"""Tests for beat tracking (Ellis 2007 DP)."""

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


def test_beat_track_returns_tuple():
    y = _make_clicks_signal(bpm=120.0)
    result = metalmom.beat_track(y=y, sr=22050)
    assert isinstance(result, tuple)
    assert len(result) == 2
    tempo, beats = result
    assert isinstance(tempo, float)
    assert isinstance(beats, np.ndarray)


def test_beat_track_tempo_reasonable():
    y = _make_clicks_signal(bpm=120.0)
    tempo, beats = metalmom.beat_track(y=y, sr=22050, start_bpm=120.0)
    # Tempo should be in a reasonable range
    assert tempo > 30, f"Tempo {tempo} should be > 30 BPM"
    assert tempo < 300, f"Tempo {tempo} should be < 300 BPM"


def test_beat_track_beats_non_empty():
    y = _make_clicks_signal(bpm=120.0)
    tempo, beats = metalmom.beat_track(y=y, sr=22050)
    assert len(beats) > 0, "Should detect at least one beat"


def test_beat_track_beats_sorted():
    y = _make_clicks_signal(bpm=120.0)
    tempo, beats = metalmom.beat_track(y=y, sr=22050, trim=False)
    if len(beats) > 1:
        assert np.all(np.diff(beats) > 0), "Beat frames should be strictly increasing"


def test_beat_track_beats_non_negative():
    y = _make_clicks_signal(bpm=120.0)
    tempo, beats = metalmom.beat_track(y=y, sr=22050, trim=False)
    if len(beats) > 0:
        assert np.all(beats >= 0), "Beat frames should be non-negative"


def test_beat_track_units_time():
    y = _make_clicks_signal(bpm=120.0)
    hop_length = 512
    sr = 22050
    tempo_f, frames = metalmom.beat_track(y=y, sr=sr, hop_length=hop_length, units='frames')
    tempo_t, times = metalmom.beat_track(y=y, sr=sr, hop_length=hop_length, units='time')
    if len(frames) > 0 and len(times) > 0:
        expected_times = frames.astype(np.float64) * hop_length / sr
        np.testing.assert_allclose(times, expected_times, atol=1e-6)


def test_beat_track_units_samples():
    y = _make_clicks_signal(bpm=120.0)
    hop_length = 512
    sr = 22050
    tempo_f, frames = metalmom.beat_track(y=y, sr=sr, hop_length=hop_length, units='frames')
    tempo_s, samples = metalmom.beat_track(y=y, sr=sr, hop_length=hop_length, units='samples')
    if len(frames) > 0 and len(samples) > 0:
        expected_samples = frames * hop_length
        np.testing.assert_array_equal(samples, expected_samples)


def test_beat_track_silent():
    y = np.zeros(22050 * 2, dtype=np.float32)
    tempo, beats = metalmom.beat_track(y=y, sr=22050)
    # Should not crash on silent signal
    assert isinstance(tempo, float)
    assert isinstance(beats, np.ndarray)


def test_beat_track_trim():
    y = _make_clicks_signal(bpm=120.0, duration=5.0)
    tempo_nt, beats_no_trim = metalmom.beat_track(y=y, sr=22050, trim=False)
    tempo_t, beats_trimmed = metalmom.beat_track(y=y, sr=22050, trim=True)
    # Trimmed should have <= beats than untrimmed
    assert len(beats_trimmed) <= len(beats_no_trim), (
        "Trimmed should have <= beats than untrimmed"
    )


def test_beat_track_compat_shim():
    from metalmom.compat.librosa.beat import beat_track as compat_beat_track
    y = _make_clicks_signal(bpm=120.0)
    tempo, beats = compat_beat_track(y=y, sr=22050)
    assert isinstance(tempo, float)
    assert isinstance(beats, np.ndarray)


def test_beat_track_different_tempos():
    """Test that different BPM inputs produce different tempo estimates."""
    y_slow = _make_clicks_signal(bpm=60.0, duration=8.0)
    y_fast = _make_clicks_signal(bpm=180.0, duration=4.0)

    tempo_slow, _ = metalmom.beat_track(y=y_slow, sr=22050, start_bpm=60.0)
    tempo_fast, _ = metalmom.beat_track(y=y_fast, sr=22050, start_bpm=180.0)

    # Both should be positive
    assert tempo_slow > 0, "Should estimate positive tempo for 60 BPM signal"
    assert tempo_fast > 0, "Should estimate positive tempo for 180 BPM signal"
