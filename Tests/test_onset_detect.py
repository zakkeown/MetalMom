"""Tests for onset detection (peak picking)."""

import numpy as np
import metalmom


def _make_clicks_signal(click_times, sr=22050, duration=2.0):
    """Generate a signal with short sine bursts at given times."""
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    for t in click_times:
        idx = int(t * sr)
        burst_len = min(512, n - idx)
        if burst_len > 0:
            burst = np.sin(np.arange(burst_len) * 1000.0 * 2 * np.pi / sr).astype(np.float32)
            y[idx:idx + burst_len] = burst * 0.9
    return y


def test_onset_detect_returns_array():
    y = _make_clicks_signal([0.25, 0.75, 1.25, 1.75])
    frames = metalmom.onset_detect(y=y, sr=22050, delta=0.05, wait=5)
    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 1


def test_onset_detect_finds_onsets():
    y = _make_clicks_signal([0.25, 0.75, 1.25, 1.75])
    frames = metalmom.onset_detect(y=y, sr=22050, delta=0.05, wait=5)
    # Should detect some onsets from clicks
    assert len(frames) > 0, "Should detect at least one onset from clicks"


def test_onset_detect_silent():
    y = np.zeros(22050, dtype=np.float32)
    frames = metalmom.onset_detect(y=y, sr=22050)
    assert len(frames) == 0, "Silent signal should produce no onsets"


def test_onset_detect_frames_non_negative():
    y = _make_clicks_signal([0.5, 1.0, 1.5])
    frames = metalmom.onset_detect(y=y, sr=22050, delta=0.05, wait=5)
    if len(frames) > 0:
        assert np.all(frames >= 0), "Frame indices should be non-negative"


def test_onset_detect_units_time():
    y = _make_clicks_signal([0.5, 1.0, 1.5])
    hop_length = 512
    sr = 22050
    frames = metalmom.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                    delta=0.05, wait=5, units='frames')
    times = metalmom.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                   delta=0.05, wait=5, units='time')
    if len(frames) > 0 and len(times) > 0:
        # times should be frames * hop_length / sr
        expected_times = frames.astype(np.float64) * hop_length / sr
        np.testing.assert_allclose(times, expected_times, atol=1e-6)


def test_onset_detect_units_samples():
    y = _make_clicks_signal([0.5, 1.0, 1.5])
    hop_length = 512
    sr = 22050
    frames = metalmom.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                    delta=0.05, wait=5, units='frames')
    samples = metalmom.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                     delta=0.05, wait=5, units='samples')
    if len(frames) > 0 and len(samples) > 0:
        expected_samples = frames * hop_length
        np.testing.assert_array_equal(samples, expected_samples)


def test_onset_detect_backtrack():
    y = _make_clicks_signal([0.5, 1.5])
    frames_no_bt = metalmom.onset_detect(y=y, sr=22050, delta=0.05,
                                          wait=5, backtrack=False)
    frames_bt = metalmom.onset_detect(y=y, sr=22050, delta=0.05,
                                       wait=5, backtrack=True)
    if len(frames_no_bt) > 0 and len(frames_bt) > 0:
        # Backtracked onsets should be at or before the peak-based onsets
        for f_bt, f_no in zip(frames_bt, frames_no_bt):
            assert f_bt <= f_no, (
                f"Backtracked onset {f_bt} should be <= peak onset {f_no}"
            )


def test_onset_detect_compat_shim():
    from metalmom.compat.librosa.onset import onset_detect as compat_detect
    y = _make_clicks_signal([0.5, 1.0])
    frames = compat_detect(y=y, sr=22050, delta=0.05, wait=5)
    assert isinstance(frames, np.ndarray)


def test_onset_detect_wait_parameter():
    y = _make_clicks_signal([0.3, 0.35, 0.4, 1.0])
    # With large wait, close-together clicks should be merged
    frames_large_wait = metalmom.onset_detect(y=y, sr=22050, delta=0.01,
                                               wait=50)
    frames_small_wait = metalmom.onset_detect(y=y, sr=22050, delta=0.01,
                                               wait=1)
    # Larger wait should produce fewer or equal onsets
    assert len(frames_large_wait) <= len(frames_small_wait)
