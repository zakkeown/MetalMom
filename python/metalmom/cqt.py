"""Constant-Q Transform, Variable-Q Transform, and Hybrid CQT."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def cqt(y, sr=22050, hop_length=None, fmin=32.70, fmax=None,
        bins_per_octave=12, n_fft=0, center=True, **kwargs):
    """Compute the Constant-Q Transform magnitude spectrogram.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32 or float64 array).
    sr : int
        Sample rate. Default: 22050.
    hop_length : int or None
        Hop length in samples. Default: auto-selected.
    fmin : float
        Lowest CQT frequency in Hz. Default: 32.70 (C1).
    fmax : float or None
        Highest CQT frequency in Hz. Default: sr / 2.
    bins_per_octave : int
        Number of frequency bins per octave. Default: 12.
    n_fft : int
        FFT size for the underlying STFT. 0 = auto-select. Default: 0.
    center : bool
        Center the signal before STFT. Default: True.

    Returns
    -------
    np.ndarray
        CQT magnitude spectrogram, shape (n_bins, n_frames).
    """
    y = np.ascontiguousarray(y, dtype=np.float32)

    hop = int(hop_length) if hop_length is not None else 0
    c_fmax = float(fmax) if fmax is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_cqt(
            ctx, signal_ptr, len(y),
            sr, hop,
            float(fmin), c_fmax, bins_per_octave,
            int(n_fft), 1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_cqt failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def vqt(y, sr=22050, hop_length=None, fmin=32.70, fmax=None,
        bins_per_octave=12, gamma=0.0, n_fft=0, center=True, **kwargs):
    """Compute the Variable-Q Transform magnitude spectrogram.

    Like CQT but with a variable quality factor controlled by gamma.
    Higher gamma values give wider bandwidth (better time resolution)
    at low frequencies.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32 or float64 array).
    sr : int
        Sample rate. Default: 22050.
    hop_length : int or None
        Hop length in samples. Default: auto-selected.
    fmin : float
        Lowest frequency in Hz. Default: 32.70 (C1).
    fmax : float or None
        Highest frequency in Hz. Default: sr / 2.
    bins_per_octave : int
        Number of frequency bins per octave. Default: 12.
    gamma : float
        VQT gamma parameter. 0 = standard CQT. Default: 0.0.
    n_fft : int
        FFT size. 0 = auto-select. Default: 0.
    center : bool
        Center the signal. Default: True.

    Returns
    -------
    np.ndarray
        VQT magnitude spectrogram, shape (n_bins, n_frames).
    """
    y = np.ascontiguousarray(y, dtype=np.float32)

    hop = int(hop_length) if hop_length is not None else 0
    c_fmax = float(fmax) if fmax is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_vqt(
            ctx, signal_ptr, len(y),
            sr, hop,
            float(fmin), c_fmax, bins_per_octave,
            float(gamma), int(n_fft), 1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_vqt failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def hybrid_cqt(y, sr=22050, hop_length=None, fmin=32.70, fmax=None,
               bins_per_octave=12, n_fft=0, center=True, **kwargs):
    """Compute the Hybrid CQT magnitude spectrogram.

    Uses CQT filterbank for bins below a transition frequency and
    linear STFT bins above it.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32 or float64 array).
    sr : int
        Sample rate. Default: 22050.
    hop_length : int or None
        Hop length in samples. Default: auto-selected.
    fmin : float
        Lowest CQT frequency in Hz. Default: 32.70 (C1).
    fmax : float or None
        Highest frequency in Hz. Default: sr / 2.
    bins_per_octave : int
        Number of frequency bins per octave. Default: 12.
    n_fft : int
        FFT size. 0 = auto-select. Default: 0.
    center : bool
        Center the signal. Default: True.

    Returns
    -------
    np.ndarray
        Hybrid CQT magnitude spectrogram, shape (n_bins, n_frames).
    """
    y = np.ascontiguousarray(y, dtype=np.float32)

    hop = int(hop_length) if hop_length is not None else 0
    c_fmax = float(fmax) if fmax is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_hybrid_cqt(
            ctx, signal_ptr, len(y),
            sr, hop,
            float(fmin), c_fmax, bins_per_octave,
            int(n_fft), 1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_hybrid_cqt failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
