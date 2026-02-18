"""Memory stress tests for MetalMom Python bindings.

Verifies that repeated STFT calls do not leak memory (buffer_to_numpy properly
frees C-side buffers) and that large signals can be processed without crashing.
"""

import tracemalloc

import numpy as np
import pytest


def test_repeated_stft_no_leak():
    """Run metalmom.stft 500x on a 1-second signal. Verify Python-side memory
    growth stays below 50 MB.

    This exercises the full Python->C->Swift->C->Python round-trip including:
    - cffi buffer allocation in mm_stft
    - buffer_to_numpy copy into NumPy array
    - mm_buffer_free deallocation of C-side data
    """
    from metalmom import stft

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Warm up — let any one-time init allocations settle
    _ = stft(signal, n_fft=2048, hop_length=512)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for _ in range(500):
        result = stft(signal, n_fft=2048, hop_length=512)
        assert result.shape[0] == 1025
        assert result.shape[1] > 0

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compare top stats to measure growth
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    total_growth_mb = total_growth_bytes / (1024 * 1024)

    assert total_growth_mb < 50.0, (
        f"Python memory grew by {total_growth_mb:.1f} MB over 500 STFT calls — "
        f"possible leak in buffer_to_numpy / mm_buffer_free cycle"
    )


def test_large_signal_stft():
    """Create a 5-minute signal (22050 * 300 = 6.6M samples). Run metalmom.stft
    once. Verify it completes with a valid output shape.

    This tests that the library can handle large contiguous allocations without
    crashing or producing corrupt output.
    """
    from metalmom import stft

    sr = 22050
    duration = 300  # 5 minutes
    num_samples = sr * duration  # 6,615,000

    t = np.arange(num_samples, dtype=np.float32) / sr
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    n_fft = 2048
    hop_length = 512

    result = stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Verify output shape
    expected_n_freqs = n_fft // 2 + 1  # 1025
    assert result.shape[0] == expected_n_freqs, (
        f"Expected {expected_n_freqs} frequency bins, got {result.shape[0]}"
    )

    # With center=True (default), padded length = num_samples + n_fft
    padded_length = num_samples + n_fft
    expected_frames = 1 + (padded_length - n_fft) // hop_length
    assert result.shape[1] == expected_frames, (
        f"Expected {expected_frames} frames, got {result.shape[1]}"
    )

    assert result.dtype == np.float32, (
        f"Expected float32 output, got {result.dtype}"
    )

    # Verify data is non-trivial
    assert np.max(result) > 0.0, "5-minute STFT should have non-zero energy"
    assert np.all(np.isfinite(result)), "STFT output should contain only finite values"
