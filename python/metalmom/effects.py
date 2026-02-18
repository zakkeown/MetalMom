"""Audio effects: HPSS and related functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


__all__ = ["hpss", "harmonic", "percussive", "time_stretch", "pitch_shift"]


def hpss(y, kernel_size=31, power=2.0, margin=1.0,
         sr=22050, n_fft=2048, hop_length=None, win_length=None,
         center=True, **kwargs):
    """Harmonic-percussive source separation.

    Decomposes an audio signal into its harmonic and percussive
    components using median filtering on the magnitude spectrogram.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    kernel_size : int
        Median filter kernel size. Default: 31.
    power : float
        Exponent for soft masks. Default: 2.0.
    margin : float
        Margin for mask separation. Default: 1.0.
    sr : int
        Sample rate. Default: 22050.
    n_fft : int
        FFT size. Default: 2048.
    hop_length : int or None
        Hop length. Default: n_fft // 4.
    win_length : int or None
        Window length. Default: n_fft.
    center : bool
        Center-pad signal. Default: True.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (harmonic, percussive) components as 1-D arrays.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_hop = int(hop_length) if hop_length is not None else 0
    c_win = int(win_length) if win_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_hpss(
            ctx, signal_ptr, len(y),
            sr, n_fft, c_hop, c_win,
            1 if center else 0,
            kernel_size, float(power), float(margin),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_hpss failed with status {status}")

        result = buffer_to_numpy(out)
        # result has shape [2, signal_length]
        h = result[0].copy()
        p = result[1].copy()
        return h, p
    finally:
        lib.mm_destroy(ctx)


def harmonic(y, **kwargs):
    """Return the harmonic component of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    **kwargs
        Additional keyword arguments passed to hpss().

    Returns
    -------
    np.ndarray
        Harmonic component as a 1-D array.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    sr = kwargs.pop("sr", 22050)
    n_fft = kwargs.pop("n_fft", 2048)
    hop_length = kwargs.pop("hop_length", None)
    win_length = kwargs.pop("win_length", None)
    center = kwargs.pop("center", True)
    kernel_size = kwargs.pop("kernel_size", 31)
    power = kwargs.pop("power", 2.0)
    margin = kwargs.pop("margin", 1.0)

    c_hop = int(hop_length) if hop_length is not None else 0
    c_win = int(win_length) if win_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_harmonic(
            ctx, signal_ptr, len(y),
            sr, n_fft, c_hop, c_win,
            1 if center else 0,
            kernel_size, float(power), float(margin),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_harmonic failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def percussive(y, **kwargs):
    """Return the percussive component of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    **kwargs
        Additional keyword arguments passed to hpss().

    Returns
    -------
    np.ndarray
        Percussive component as a 1-D array.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    sr = kwargs.pop("sr", 22050)
    n_fft = kwargs.pop("n_fft", 2048)
    hop_length = kwargs.pop("hop_length", None)
    win_length = kwargs.pop("win_length", None)
    center = kwargs.pop("center", True)
    kernel_size = kwargs.pop("kernel_size", 31)
    power = kwargs.pop("power", 2.0)
    margin = kwargs.pop("margin", 1.0)

    c_hop = int(hop_length) if hop_length is not None else 0
    c_win = int(win_length) if win_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_percussive(
            ctx, signal_ptr, len(y),
            sr, n_fft, c_hop, c_win,
            1 if center else 0,
            kernel_size, float(power), float(margin),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_percussive failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def time_stretch(y, rate, sr=22050, n_fft=2048, hop_length=None, **kwargs):
    """Time-stretch audio by the given rate.

    Uses a phase vocoder to change the duration of the signal without
    changing its pitch.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    rate : float
        Stretch rate. rate > 1 speeds up (shorter output),
        rate < 1 slows down (longer output).
    sr : int
        Sample rate. Default: 22050.
    n_fft : int
        FFT size. Default: 2048.
    hop_length : int or None
        Hop length. Default: n_fft // 4.

    Returns
    -------
    np.ndarray
        Time-stretched audio signal (1-D).
    """
    if y is None:
        raise ValueError("y must be provided")
    if rate <= 0:
        raise ValueError("rate must be positive")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_hop = int(hop_length) if hop_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_time_stretch(
            ctx, signal_ptr, len(y),
            sr, float(rate),
            n_fft, c_hop,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_time_stretch failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def pitch_shift(y, sr=22050, n_steps=0, bins_per_octave=12,
                n_fft=2048, hop_length=None, **kwargs):
    """Shift the pitch of an audio signal by n_steps semitones.

    Changes the pitch without changing the duration, using time
    stretching followed by resampling.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    sr : int
        Sample rate. Default: 22050.
    n_steps : float
        Number of steps to shift. Positive shifts pitch up,
        negative shifts pitch down. Default: 0.
    bins_per_octave : int
        Number of steps per octave. Default: 12 (semitones).
    n_fft : int
        FFT size for the phase vocoder. Default: 2048.
    hop_length : int or None
        Hop length for the phase vocoder. Default: n_fft // 4.

    Returns
    -------
    np.ndarray
        Pitch-shifted audio signal (1-D), same length as input.
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

        status = lib.mm_pitch_shift(
            ctx, signal_ptr, len(y),
            sr, float(n_steps),
            bins_per_octave, n_fft, c_hop,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_pitch_shift failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)
