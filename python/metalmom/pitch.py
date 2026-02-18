"""Pitch estimation functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def yin(y, fmin, fmax, sr=22050, frame_length=2048, hop_length=None,
        trough_threshold=0.1, center=True, **kwargs):
    """Estimate fundamental frequency using YIN.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    sr : int
        Sample rate. Default: 22050.
    frame_length : int
        Analysis frame length. Default: 2048.
    hop_length : int or None
        Hop length. Default: frame_length // 4.
    trough_threshold : float
        Threshold for CMNDF. Default: 0.1.
    center : bool
        Center-pad signal. Default: True.

    Returns
    -------
    np.ndarray
        F0 estimates in Hz, shape (n_frames,). Unvoiced frames = 0.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_hop = int(hop_length) if hop_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_yin(
            ctx, signal_ptr, len(y),
            sr, float(fmin), float(fmax),
            frame_length, c_hop,
            trough_threshold,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_yin failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)
