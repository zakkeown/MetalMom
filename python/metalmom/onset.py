"""Onset detection functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def onset_strength(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   n_mels=128, fmin=0.0, fmax=None, center=True,
                   aggregate=True, **kwargs):
    """Compute onset strength envelope.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal.
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed spectrogram (not supported yet).
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int
        Hop length. Default: 512.
    n_mels : int
        Number of mel bands. Default: 128.
    fmin : float
        Minimum frequency. Default: 0.0.
    fmax : float or None
        Maximum frequency. Default: None (sr/2).
    center : bool
        Center-pad signal. Default: True.
    aggregate : bool
        Average across frequency bands. Default: True.

    Returns
    -------
    np.ndarray
        Onset strength envelope, shape (n_frames,) if aggregate else (n_mels, n_frames).
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_fmax = float(fmax) if fmax is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_onset_strength(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop_length,
            n_mels, fmin, c_fmax,
            1 if center else 0,
            1 if aggregate else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_onset_strength failed with status {status}")

        result = buffer_to_numpy(out)
        if aggregate:
            return result.ravel()
        return result
    finally:
        lib.mm_destroy(ctx)
