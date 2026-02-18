"""Minimal-copy MMBuffer <-> NumPy interop."""

import numpy as np
from ._native import ffi, lib


def buffer_to_numpy(buf):
    """Convert an MMBuffer to a NumPy array, then free the C-side buffer.

    This is a minimal-copy operation: data is copied from C memory into
    a NumPy array, then the C buffer is freed.
    """
    if buf.data == ffi.NULL or buf.count == 0:
        shape = tuple(buf.shape[i] for i in range(buf.ndim))
        lib.mm_buffer_free(buf)
        return np.empty(shape, dtype=np.float32)

    count = buf.count
    # Copy data from C buffer into NumPy
    data = np.frombuffer(
        ffi.buffer(buf.data, count * 4),  # 4 bytes per float32
        dtype=np.float32,
    ).copy()  # .copy() makes a Python-owned copy

    # Reshape according to MMBuffer shape
    shape = tuple(buf.shape[i] for i in range(buf.ndim))
    data = data.reshape(shape)

    # Free the C-side buffer
    lib.mm_buffer_free(buf)

    return data


def numpy_to_float_ptr(arr):
    """Convert a NumPy array to a cffi float pointer.

    The array must be contiguous float32. Returns (pointer, length).
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ptr = ffi.cast("const float*", arr.ctypes.data)
    return ptr, len(arr)
