"""Tests for bridge error handling and buffer lifecycle safety.

Covers:
  - C bridge error codes for invalid inputs
  - Double-destroy safety (context registry fix)
  - buffer_to_numpy exception safety (try/finally fix)
  - Edge cases: empty arrays, NaN, short signals, wrong dtypes
"""

import numpy as np
import pytest
import metalmom
from metalmom._native import ffi, lib
from metalmom._buffer import buffer_to_numpy


# ---------------------------------------------------------------------------
# High-level API edge cases
# ---------------------------------------------------------------------------

def test_empty_array_stft():
    """Empty signal should raise a clean RuntimeError, not crash."""
    with pytest.raises(RuntimeError, match="failed with status"):
        metalmom.stft(y=np.array([], dtype=np.float32))


def test_nan_input_stft():
    """NaN signal should not crash.  May return NaN values in output."""
    y = np.full(4096, np.nan, dtype=np.float32)
    result = metalmom.stft(y=y)
    # Should complete without crash; output shape should be valid
    assert result.ndim == 2
    assert result.shape[0] == 1025  # n_fft/2+1 with default n_fft=2048


def test_very_short_signal():
    """A 10-sample signal should survive STFT (center-padded)."""
    y = np.sin(np.arange(10, dtype=np.float32))
    result = metalmom.stft(y=y)
    assert result.ndim == 2
    assert result.shape[0] == 1025
    assert result.shape[1] >= 1  # at least one frame with center padding


def test_int32_input_stft():
    """Non-float input should be auto-converted to float32 by the wrapper."""
    y = np.ones(2048, dtype=np.int32)
    result = metalmom.stft(y=y)
    assert result.dtype == np.float32
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# Direct bridge error codes
# ---------------------------------------------------------------------------

def test_context_double_destroy():
    """Double mm_destroy should not crash (context registry fix)."""
    ctx = lib.mm_init()
    assert ctx != ffi.NULL
    lib.mm_destroy(ctx)
    lib.mm_destroy(ctx)  # second destroy is a no-op -- should not crash


def test_bridge_error_null_signal():
    """mm_stft with NULL signal should return MM_ERR_INVALID_INPUT (-1)."""
    ctx = lib.mm_init()
    try:
        params = ffi.new("MMSTFTParams*")
        params.n_fft = 2048
        params.hop_length = 512
        params.win_length = 2048
        params.center = 1
        out = ffi.new("MMBuffer*")

        status = lib.mm_stft(ctx, ffi.NULL, 22050, 22050, params, out)
        assert status == -1, f"Expected MM_ERR_INVALID_INPUT (-1), got {status}"
    finally:
        lib.mm_destroy(ctx)


def test_bridge_non_power_of_two():
    """mm_stft with n_fft=1000 (not power of 2) should return error."""
    ctx = lib.mm_init()
    try:
        signal = np.sin(np.arange(22050) * 2 * np.pi * 440 / 22050).astype(np.float32)
        ptr = ffi.cast("const float*", signal.ctypes.data)

        params = ffi.new("MMSTFTParams*")
        params.n_fft = 1000
        params.hop_length = 512
        params.win_length = 1000
        params.center = 1
        out = ffi.new("MMBuffer*")

        status = lib.mm_stft(ctx, ptr, len(signal), 22050, params, out)
        assert status == -1, f"Expected MM_ERR_INVALID_INPUT (-1), got {status}"

        # Buffer should not have been allocated
        if out.data != ffi.NULL:
            lib.mm_buffer_free(out)
    finally:
        lib.mm_destroy(ctx)


def test_buffer_to_numpy_exception_safety():
    """buffer_to_numpy must free the C buffer even if reshape raises.

    The bug fix wrapped the copy+reshape in try/finally so the C buffer is
    always freed.  We verify this by creating a valid buffer but giving it
    an impossible shape, then checking that the free still happens.
    """
    ctx = lib.mm_init()
    try:
        # Produce a real STFT output buffer
        signal = np.sin(np.arange(4096, dtype=np.float32))
        ptr = ffi.cast("const float*", signal.ctypes.data)

        params = ffi.new("MMSTFTParams*")
        params.n_fft = 2048
        params.hop_length = 512
        params.win_length = 2048
        params.center = 1
        out = ffi.new("MMBuffer*")

        status = lib.mm_stft(ctx, ptr, len(signal), 22050, params, out)
        assert status == 0

        # Corrupt the shape to make reshape fail (product != count)
        original_count = out.count
        assert original_count > 0
        out.shape[0] = 9999
        out.shape[1] = 9999
        out.ndim = 2

        with pytest.raises(ValueError):
            buffer_to_numpy(out)

        # After the exception, data should have been freed (set to NULL by mm_buffer_free)
        assert out.data == ffi.NULL, "C buffer should be freed even when reshape fails"
    finally:
        lib.mm_destroy(ctx)


def test_griffinlim_error():
    """mm_griffinlim with invalid input should return error code."""
    ctx = lib.mm_init()
    try:
        out = ffi.new("MMBuffer*")
        # Pass NULL data, zero counts -- should return error
        status = lib.mm_griffinlim(
            ctx,
            ffi.NULL,  # mag_data
            0,         # mag_count
            0,         # n_freqs
            0,         # n_frames
            22050,     # sample_rate
            32,        # n_iter
            512,       # hop_length
            2048,      # win_length
            1,         # center
            0,         # output_length
            out,
        )
        assert status == -1, f"Expected MM_ERR_INVALID_INPUT (-1), got {status}"
    finally:
        lib.mm_destroy(ctx)


def test_neural_beat_decode_error():
    """mm_neural_beat_decode with NULL activations should return error code."""
    ctx = lib.mm_init()
    try:
        out_tempo = ffi.new("float*")
        out_beats = ffi.new("MMBuffer*")

        status = lib.mm_neural_beat_decode(
            ctx,
            ffi.NULL,   # activations (NULL)
            0,          # n_frames
            100.0,      # fps
            55.0,       # min_bpm
            215.0,      # max_bpm
            100.0,      # transition_lambda
            0.5,        # threshold
            1,          # trim
            out_tempo,
            out_beats,
        )
        assert status == -1, f"Expected MM_ERR_INVALID_INPUT (-1), got {status}"
    finally:
        lib.mm_destroy(ctx)
