"""Tests for trim (silence trimming)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def _make_silence_padded(silence_start=0.2, tone_duration=0.3,
                         silence_end=0.2, freq=440.0, sr=22050):
    """Generate a signal with silence at start and end surrounding a sine burst."""
    start_samples = int(silence_start * sr)
    tone_samples = int(tone_duration * sr)
    end_samples = int(silence_end * sr)
    total = start_samples + tone_samples + end_samples

    y = np.zeros(total, dtype=np.float32)
    t = np.arange(tone_samples, dtype=np.float32)
    y[start_samples:start_samples + tone_samples] = np.sin(
        2 * np.pi * freq * t / sr
    ).astype(np.float32)
    return y


def test_trim_returns_tuple():
    """trim() should return (array, (start, end))."""
    y = _make_silence_padded()
    result = metalmom.trim(y)
    assert isinstance(result, tuple)
    assert len(result) == 2

    y_trimmed, index = result
    assert isinstance(y_trimmed, np.ndarray)
    assert isinstance(index, tuple)
    assert len(index) == 2


def test_trim_removes_silence():
    """Trimmed signal should be shorter than original when silence is present."""
    y = _make_silence_padded(silence_start=0.3, tone_duration=0.3, silence_end=0.3)
    y_trimmed, (start, end) = metalmom.trim(y)

    assert len(y_trimmed) < len(y), (
        f"Trimmed ({len(y_trimmed)}) should be shorter than original ({len(y)})"
    )
    assert len(y_trimmed) > 0, "Trimmed should not be empty"


def test_trim_indices_make_sense():
    """Start and end indices should bound the non-silent region."""
    y = _make_silence_padded(silence_start=0.2, tone_duration=0.3, silence_end=0.2)
    y_trimmed, (start, end) = metalmom.trim(y)

    # Start should be past leading silence
    assert start > 0, f"Start index {start} should be > 0"
    # End should be before total length
    assert end < len(y), f"End index {end} should be < {len(y)}"
    # Trimmed length should match index range
    assert len(y_trimmed) == end - start, (
        f"Trimmed len {len(y_trimmed)} != end-start {end - start}"
    )


def test_trim_no_silence():
    """A signal with no silence should return approximately the same signal."""
    y = _make_sine(duration=0.5)
    y_trimmed, (start, end) = metalmom.trim(y)

    assert len(y_trimmed) > 0
    # Start should be near the beginning
    assert start <= 512, f"Start {start} should be near beginning"


def test_trim_all_silence():
    """An all-silence signal should return empty or near-empty."""
    y = np.zeros(22050, dtype=np.float32)
    y_trimmed, (start, end) = metalmom.trim(y)

    assert len(y_trimmed) == 0, f"All-silence should return empty, got {len(y_trimmed)}"


def test_trim_custom_top_db():
    """Different top_db values should affect trim aggressiveness."""
    y = _make_silence_padded(silence_start=0.2, tone_duration=0.3, silence_end=0.2)

    y_strict, _ = metalmom.trim(y, top_db=20)
    y_loose, _ = metalmom.trim(y, top_db=80)

    # Looser threshold should keep at least as much signal
    assert len(y_loose) >= len(y_strict), (
        f"Loose ({len(y_loose)}) should be >= strict ({len(y_strict)})"
    )


def test_trim_1d_output():
    """Trimmed output should be 1D."""
    y = _make_silence_padded()
    y_trimmed, _ = metalmom.trim(y)
    assert y_trimmed.ndim == 1, f"Expected 1D output, got {y_trimmed.ndim}D"


def test_trim_compat_shim():
    """Test the librosa compat shim for trim."""
    from metalmom.compat.librosa.effects import trim as compat_trim

    y = _make_silence_padded()
    y_trimmed, (start, end) = compat_trim(y)
    assert isinstance(y_trimmed, np.ndarray)
    assert isinstance(start, int)
    assert isinstance(end, int)


def test_trim_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa

    y = _make_silence_padded()
    y_trimmed, (start, end) = librosa.effects.trim(y)
    assert isinstance(y_trimmed, np.ndarray)
    assert len(y_trimmed) < len(y)
