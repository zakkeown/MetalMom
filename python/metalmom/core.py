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


def istft(stft_matrix, hop_length=None, win_length=None, center=True, length=None):
    """Inverse STFT: reconstruct time-domain signal from complex spectrogram.

    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex spectrogram, shape (n_fft//2 + 1, n_frames).
        Can be complex64/complex128 or real-valued (treated as real+0j).
    hop_length : int or None
        Hop length (default: n_fft // 4).
    win_length : int or None
        Window length (default: n_fft).
    center : bool
        If True, assumes the STFT was computed with center=True and
        trims the padding (default: True).
    length : int or None
        If specified, the output signal is truncated or zero-padded to
        this exact length.

    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal, 1D float32 array.
    """
    # Convert to complex64 if needed
    if not np.iscomplexobj(stft_matrix):
        stft_matrix = stft_matrix.astype(np.complex64)
    else:
        stft_matrix = np.asarray(stft_matrix, dtype=np.complex64)

    n_freqs, n_frames = stft_matrix.shape
    n_fft = (n_freqs - 1) * 2
    hop = hop_length if hop_length is not None else n_fft // 4
    win = win_length if win_length is not None else n_fft

    # Convert complex64 to interleaved float32 for the C bridge.
    # stft_matrix is row-major [n_freqs, n_frames] complex64.
    # np.complex64 stores real,imag as two consecutive float32 values,
    # so viewing as float32 gives interleaved [r0,i0, r1,i1, ...] in row-major order.
    stft_interleaved = np.ascontiguousarray(stft_matrix).view(np.float32)
    stft_count = stft_interleaved.size  # 2 * n_freqs * n_frames

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")

        stft_ptr = ffi.cast("const float*", stft_interleaved.ctypes.data)
        output_length = length if length is not None else 0

        status = lib.mm_istft(
            ctx, stft_ptr, stft_count,
            n_freqs, n_frames, 22050,
            hop, win,
            1 if center else 0,
            output_length, out
        )

        if status != 0:
            raise RuntimeError(f"mm_istft failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
