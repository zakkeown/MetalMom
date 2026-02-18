"""Core audio analysis functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def stft(y, n_fft=2048, hop_length=None, win_length=None, center=True):
    """Compute the Short-Time Fourier Transform (magnitude).

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32 or float64 array).
    n_fft : int
        FFT window size (default: 2048).
    hop_length : int or None
        Hop length (default: n_fft // 4).
    win_length : int or None
        Window length (default: n_fft).
    center : bool
        Center the signal (default: True).

    Returns
    -------
    np.ndarray
        Magnitude spectrogram, shape (n_fft // 2 + 1, n_frames).
    """
    # Ensure float32 contiguous
    y = np.ascontiguousarray(y, dtype=np.float32)

    hop = hop_length if hop_length is not None else n_fft // 4
    win = win_length if win_length is not None else n_fft

    # Create context
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        # Set up params
        params = ffi.new("MMSTFTParams*")
        params.n_fft = n_fft
        params.hop_length = hop
        params.win_length = win
        params.center = 1 if center else 0

        # Allocate output
        out = ffi.new("MMBuffer*")

        # Call bridge
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_stft(ctx, signal_ptr, len(y), 22050, params, out)

        if status != 0:
            raise RuntimeError(f"mm_stft failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
