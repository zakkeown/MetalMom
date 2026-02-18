"""Tests for split (non-silent interval detection)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def _make_gapped_signal(tone_duration=0.2, silence_duration=0.3,
                         tone_count=3, freq=440.0, sr=22050):
    """Generate a signal with tone bursts separated by silence gaps."""
    tone_samples = int(tone_duration * sr)
    silence_samples = int(silence_duration * sr)
    total = tone_count * tone_samples + (tone_count - 1) * silence_samples

    y = np.zeros(total, dtype=np.float32)
    t = np.arange(tone_samples, dtype=np.float32)
    for i in range(tone_count):
        offset = i * (tone_samples + silence_samples)
        y[offset:offset + tone_samples] = np.sin(
            2 * np.pi * freq * t / sr
        ).astype(np.float32)
    return y


def test_split_returns_2d_array():
    """split() should return a 2D array with shape (n, 2)."""
    y = _make_gapped_signal()
    result = metalmom.split(y)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_split_continuous_signal():
    """Continuous signal should produce 1 interval."""
    y = _make_sine(duration=0.5)
    result = metalmom.split(y)

    assert result.shape[0] == 1, (
        f"Continuous signal should give 1 interval, got {result.shape[0]}"
    )
    # The interval should cover most of the signal
    start, end = int(result[0, 0]), int(result[0, 1])
    assert start <= 512, f"Start {start} should be near beginning"
    assert end > len(y) - 2048, f"End {end} should be near signal end"


def test_split_with_silence_gaps():
    """Signal with silence gaps should produce multiple intervals."""
    y = _make_gapped_signal(tone_duration=0.2, silence_duration=0.3, tone_count=3)
    result = metalmom.split(y, top_db=40)

    assert result.shape[0] >= 2, (
        f"Should detect at least 2 intervals, got {result.shape[0]}"
    )

    # Each interval should have valid bounds
    for i in range(result.shape[0]):
        start, end = int(result[i, 0]), int(result[i, 1])
        assert start >= 0, f"Interval {i} start should be >= 0"
        assert end <= len(y), f"Interval {i} end should be <= signal length"
        assert start < end, f"Interval {i} start should be < end"

    # Intervals should be non-overlapping and in order
    for i in range(1, result.shape[0]):
        assert result[i, 0] >= result[i - 1, 1], (
            f"Interval {i} should start after interval {i-1} ends"
        )


def test_split_all_silence():
    """An all-silence signal should return empty array with shape (0, 2)."""
    y = np.zeros(22050, dtype=np.float32)
    result = metalmom.split(y)

    assert result.shape == (0, 2), (
        f"All-silence should return shape (0, 2), got {result.shape}"
    )


def test_split_custom_top_db():
    """Different top_db values should affect detection sensitivity."""
    y = _make_gapped_signal(tone_duration=0.2, silence_duration=0.3, tone_count=3)

    result_strict = metalmom.split(y, top_db=20)
    result_loose = metalmom.split(y, top_db=80)

    # With a looser threshold, we may detect more intervals or wider ones
    # Both should return valid results
    assert result_strict.ndim == 2
    assert result_loose.ndim == 2
    assert result_strict.shape[1] == 2
    assert result_loose.shape[1] == 2


def test_split_compat_shim():
    """Test the librosa compat shim for split."""
    from metalmom.compat.librosa.effects import split as compat_split

    y = _make_gapped_signal()
    result = compat_split(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_split_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa

    y = _make_gapped_signal()
    result = librosa.effects.split(y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2
