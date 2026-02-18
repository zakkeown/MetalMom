"""Tests for Python cffi bindings and buffer interop."""

import pytest
import numpy as np


def test_native_load():
    """Test that the native library loads successfully."""
    from metalmom._native import ffi, lib
    assert lib is not None
    assert ffi is not None


def test_context_lifecycle():
    """Test creating and destroying a context."""
    from metalmom._native import lib
    ctx = lib.mm_init()
    assert ctx is not None
    lib.mm_destroy(ctx)


def test_stft_via_bridge():
    """Test STFT through the C bridge."""
    from metalmom._native import ffi, lib
    from metalmom._buffer import buffer_to_numpy

    # Create a simple sine wave
    sr = 22050
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Create context
    ctx = lib.mm_init()

    # Set up STFT params
    params = ffi.new("MMSTFTParams*")
    params.n_fft = 2048
    params.hop_length = 512
    params.win_length = 2048
    params.center = 1

    # Allocate output buffer
    out = ffi.new("MMBuffer*")

    # Run STFT
    signal_ptr = ffi.cast("const float*", signal.ctypes.data)
    status = lib.mm_stft(ctx, signal_ptr, len(signal), sr, params, out)
    assert status == 0  # MM_OK

    # Convert to numpy
    result = buffer_to_numpy(out)

    # Verify shape: [n_fft/2 + 1, n_frames]
    assert result.shape[0] == 1025  # n_fft/2 + 1
    assert result.shape[1] > 0
    assert result.dtype == np.float32

    # Check that there's energy (not all zeros)
    assert result.max() > 0

    # Clean up
    lib.mm_destroy(ctx)
