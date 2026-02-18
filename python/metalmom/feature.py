"""Feature extraction functions (mel, mfcc, dB scaling, etc.)."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    """Convert an amplitude (magnitude) spectrogram to dB-scaled spectrogram.

    Parameters
    ----------
    S : np.ndarray
        Input amplitude spectrogram (non-negative).
    ref : float or callable
        Reference value. If callable, `ref(S)` is used (e.g., `np.max`).
        Default: 1.0.
    amin : float
        Minimum amplitude threshold to avoid log(0). Default: 1e-5.
    top_db : float or None
        If not None, clip output to be no more than `top_db` below the peak.
        Default: 80.0.

    Returns
    -------
    np.ndarray
        dB-scaled spectrogram, same shape as input.
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    original_shape = S.shape

    # Handle callable ref (e.g., np.max)
    if callable(ref):
        ref = float(ref(S))

    # Flatten for C bridge
    flat = S.ravel()
    count = len(flat)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", flat.ctypes.data)

        # top_db <= 0 signals "no clipping" to the C bridge
        c_top_db = float(top_db) if top_db is not None else 0.0

        status = lib.mm_amplitude_to_db(ctx, data_ptr, count, ref, amin, c_top_db, out)
        if status != 0:
            raise RuntimeError(f"mm_amplitude_to_db failed with status {status}")

        result = buffer_to_numpy(out)
        return result.reshape(original_shape)
    finally:
        lib.mm_destroy(ctx)


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram to dB-scaled spectrogram.

    Parameters
    ----------
    S : np.ndarray
        Input power spectrogram (non-negative).
    ref : float or callable
        Reference value. If callable, `ref(S)` is used (e.g., `np.max`).
        Default: 1.0.
    amin : float
        Minimum power threshold to avoid log(0). Default: 1e-10.
    top_db : float or None
        If not None, clip output to be no more than `top_db` below the peak.
        Default: 80.0.

    Returns
    -------
    np.ndarray
        dB-scaled spectrogram, same shape as input.
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    original_shape = S.shape

    if callable(ref):
        ref = float(ref(S))

    flat = S.ravel()
    count = len(flat)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", flat.ctypes.data)

        c_top_db = float(top_db) if top_db is not None else 0.0

        status = lib.mm_power_to_db(ctx, data_ptr, count, ref, amin, c_top_db, out)
        if status != 0:
            raise RuntimeError(f"mm_power_to_db failed with status {status}")

        result = buffer_to_numpy(out)
        return result.reshape(original_shape)
    finally:
        lib.mm_destroy(ctx)
