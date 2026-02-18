"""Audio effects: HPSS and related functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


__all__ = [
    "hpss", "harmonic", "percussive", "time_stretch", "pitch_shift",
    "trim", "split", "preemphasis", "deemphasis",
    "phase_vocoder", "griffinlim",
]


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


def trim(y, top_db=60, ref=None, frame_length=2048, hop_length=512, **kwargs):
    """Trim leading and trailing silence from an audio signal.

    Removes leading and trailing regions whose RMS energy is below
    a threshold (in dB) relative to the peak RMS frame.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    top_db : float
        Threshold in dB below the peak RMS. Frames with energy
        below ``peak - top_db`` are considered silence. Default: 60.
    ref : ignored
        Accepted for librosa API compatibility but not used.
    frame_length : int
        Length of each analysis frame. Default: 2048.
    hop_length : int
        Number of samples between successive frames. Default: 512.

    Returns
    -------
    y_trimmed : np.ndarray
        Trimmed signal.
    index : tuple of (int, int)
        Start and end sample indices of the non-silent region.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        out_start = ffi.new("int64_t*")
        out_end = ffi.new("int64_t*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_trim(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used in algorithm, but required by bridge)
            float(top_db),
            int(frame_length), int(hop_length),
            out, out_start, out_end,
        )
        if status != 0:
            raise RuntimeError(f"mm_trim failed with status {status}")

        start_idx = int(out_start[0])
        end_idx = int(out_end[0])

        result = buffer_to_numpy(out)
        return result.ravel(), (start_idx, end_idx)
    finally:
        lib.mm_destroy(ctx)


def split(y, top_db=60, ref=None, frame_length=2048, hop_length=512, **kwargs):
    """Split audio into non-silent intervals.

    Detects contiguous non-silent regions in an audio signal based on
    a dB threshold relative to the peak RMS energy, matching the
    behavior of ``librosa.effects.split``.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    top_db : float
        Threshold in dB below the peak RMS. Frames with energy
        below ``peak - top_db`` are considered silence. Default: 60.
    ref : ignored
        Accepted for librosa API compatibility but not used.
    frame_length : int
        Length of each analysis frame. Default: 2048.
    hop_length : int
        Number of samples between successive frames. Default: 512.

    Returns
    -------
    np.ndarray, shape (n_intervals, 2)
        Each row is [start_sample, end_sample].
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_split(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used in algorithm, but required by bridge)
            float(top_db),
            int(frame_length), int(hop_length),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_split failed with status {status}")

        result = buffer_to_numpy(out)
        if result.size == 0:
            return np.empty((0, 2), dtype=result.dtype)
        return result.reshape(-1, 2)
    finally:
        lib.mm_destroy(ctx)


def preemphasis(y, coef=0.97, zi=None, return_zf=False, **kwargs):
    """Apply a preemphasis filter to an audio signal.

    Implements the first-order high-pass filter
    ``y[n] = x[n] - coef * x[n-1]``, matching the behavior of
    ``librosa.effects.preemphasis``.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    coef : float
        Filter coefficient. Default: 0.97.
    zi : ignored
        Accepted for librosa API compatibility but not used.
    return_zf : ignored
        Accepted for librosa API compatibility but not used.

    Returns
    -------
    np.ndarray
        Filtered signal (1-D), same length as input.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_preemphasis(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used in algorithm, but required by bridge)
            float(coef),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_preemphasis failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def deemphasis(y, coef=0.97, zi=None, return_zf=False, **kwargs):
    """Apply a deemphasis filter to an audio signal.

    Implements the first-order low-pass filter
    ``y[n] = x[n] + coef * y[n-1]``, which is the inverse of
    preemphasis: ``deemphasis(preemphasis(x)) == x``.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D).
    coef : float
        Filter coefficient. Default: 0.97.
    zi : ignored
        Accepted for librosa API compatibility but not used.
    return_zf : ignored
        Accepted for librosa API compatibility but not used.

    Returns
    -------
    np.ndarray
        Filtered signal (1-D), same length as input.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_deemphasis(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used in algorithm, but required by bridge)
            float(coef),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_deemphasis failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def phase_vocoder(D, rate, hop_length=None, **kwargs):
    """Phase vocoder: time-stretch a complex STFT by a rate factor.

    Parameters
    ----------
    D : np.ndarray
        Complex STFT matrix, shape (n_freqs, n_frames).
        Can be complex64 or complex128.
    rate : float
        Stretch rate. rate > 1 speeds up (fewer frames),
        rate < 1 slows down (more frames).
    hop_length : int or None
        Hop length used to produce D. Default: (n_freqs - 1) * 2 // 4.

    Returns
    -------
    np.ndarray
        Time-stretched complex STFT, shape (n_freqs, new_n_frames).
        Returned as complex64.
    """
    if D is None:
        raise ValueError("D must be provided")
    if rate <= 0:
        raise ValueError("rate must be positive")

    # Convert to complex64
    if not np.iscomplexobj(D):
        D = D.astype(np.complex64)
    else:
        D = np.asarray(D, dtype=np.complex64)

    n_freqs, n_frames = D.shape
    c_hop = int(hop_length) if hop_length is not None else 0

    # Convert to interleaved float32 for C bridge
    stft_interleaved = np.ascontiguousarray(D).view(np.float32)
    stft_count = stft_interleaved.size

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        stft_ptr = ffi.cast("const float*", stft_interleaved.ctypes.data)

        status = lib.mm_phase_vocoder(
            ctx, stft_ptr, stft_count,
            n_freqs, n_frames, 22050,
            float(rate), c_hop,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_phase_vocoder failed with status {status}")

        # Result is interleaved float32, reshape to complex64
        result = buffer_to_numpy(out)
        # result shape is [n_freqs, new_n_frames * 2] (interleaved real/imag)
        # Actually fillBuffer returns raw floats; we need to view as complex
        return result.view(np.complex64)
    finally:
        lib.mm_destroy(ctx)


def griffinlim(S, n_iter=32, hop_length=None, win_length=None, center=True,
               length=None, **kwargs):
    """Reconstruct audio from a magnitude spectrogram using Griffin-Lim.

    Parameters
    ----------
    S : np.ndarray
        Magnitude spectrogram, shape (n_freqs, n_frames).
    n_iter : int
        Number of iterations. Default: 32.
    hop_length : int or None
        Hop length. Default: n_fft // 4.
    win_length : int or None
        Window length. Default: n_fft.
    center : bool
        Assumes centered STFT. Default: True.
    length : int or None
        Desired output length. Default: None.

    Returns
    -------
    np.ndarray
        Reconstructed audio signal, 1-D float32 array.
    """
    if S is None:
        raise ValueError("S must be provided")

    S = np.ascontiguousarray(S, dtype=np.float32)
    n_freqs, n_frames = S.shape

    c_hop = int(hop_length) if hop_length is not None else 0
    c_win = int(win_length) if win_length is not None else 0
    c_length = int(length) if length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        mag_ptr = ffi.cast("const float*", S.ctypes.data)

        status = lib.mm_griffinlim(
            ctx, mag_ptr, S.size,
            n_freqs, n_frames, 22050,
            n_iter, c_hop, c_win,
            1 if center else 0,
            c_length,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_griffinlim failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)
