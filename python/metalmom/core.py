"""Core audio analysis functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def load(path, sr=22050, mono=True, offset=0.0, duration=None, **kwargs):
    """Load an audio file.

    Parameters
    ----------
    path : str
        Path to the audio file.
    sr : int or None
        Target sample rate. If None, uses native sample rate. Default: 22050.
    mono : bool
        Convert to mono. Default: True.
    offset : float
        Start reading at this time (seconds). Default: 0.0.
    duration : float or None
        Read only this many seconds. Default: None (entire file).

    Returns
    -------
    tuple of (np.ndarray, int)
        Audio data (1D float32) and sample rate.
    """
    path_bytes = path.encode('utf-8')
    c_sr = int(sr) if sr is not None else 0
    c_dur = float(duration) if duration is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        out_sr = ffi.new("int32_t*")

        status = lib.mm_load(
            ctx,
            path_bytes,
            c_sr,
            1 if mono else 0,
            float(offset),
            c_dur,
            out,
            out_sr,
        )
        if status != 0:
            raise RuntimeError(f"mm_load failed with status {status}")

        audio = buffer_to_numpy(out)
        actual_sr = int(out_sr[0])
        return audio, actual_sr
    finally:
        lib.mm_destroy(ctx)


def resample(y, orig_sr, target_sr, **kwargs):
    """Resample audio from one sample rate to another.

    Uses high-quality sinc interpolation with a Kaiser-windowed filter.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32 array).
    orig_sr : int
        Original sample rate.
    target_sr : int
        Target sample rate.

    Returns
    -------
    np.ndarray
        Resampled audio signal (1D float32 array).
    """
    if orig_sr == target_sr:
        return np.array(y, dtype=np.float32, copy=False)

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_resample(
            ctx, signal_ptr, len(y),
            orig_sr, target_sr,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_resample failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def db_to_amplitude(S_db, ref=1.0):
    """Convert dB-scaled values back to amplitude (magnitude).

    Parameters
    ----------
    S_db : np.ndarray
        Input dB-scaled spectrogram.
    ref : float
        Reference amplitude. Default: 1.0.

    Returns
    -------
    np.ndarray
        Amplitude values: ``ref * 10^(S_db / 20)``.
    """
    S_db = np.asarray(S_db, dtype=np.float32)
    return ref * np.power(10.0, S_db / 20.0)


def db_to_power(S_db, ref=1.0):
    """Convert dB-scaled values back to power.

    Parameters
    ----------
    S_db : np.ndarray
        Input dB-scaled spectrogram.
    ref : float
        Reference power. Default: 1.0.

    Returns
    -------
    np.ndarray
        Power values: ``ref * 10^(S_db / 10)``.
    """
    S_db = np.asarray(S_db, dtype=np.float32)
    return ref * np.power(10.0, S_db / 10.0)


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


def tone(frequency, sr=22050, length=None, duration=None, phi=0.0, **kwargs):
    """Generate a pure sine tone.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    sr : int
        Sample rate. Default: 22050.
    length : int or None
        Number of samples. If None, computed from duration (default: 1 second).
    duration : float or None
        Duration in seconds. Overrides length if both given.
    phi : float
        Phase offset in radians. Default: 0.0.

    Returns
    -------
    np.ndarray
        1D float32 array containing the sine tone.
    """
    if length is None:
        length = int(sr * (duration if duration is not None else 1.0))

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        status = lib.mm_tone(ctx, float(frequency), sr, length, float(phi), out)
        if status != 0:
            raise RuntimeError(f"mm_tone failed with status {status}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def chirp(fmin, fmax, sr=22050, length=None, duration=None, linear=True, **kwargs):
    """Generate a frequency sweep (chirp).

    Parameters
    ----------
    fmin : float
        Start frequency in Hz.
    fmax : float
        End frequency in Hz.
    sr : int
        Sample rate. Default: 22050.
    length : int or None
        Number of samples. If None, computed from duration (default: 1 second).
    duration : float or None
        Duration in seconds. Overrides length if both given.
    linear : bool
        If True, linear sweep. If False, logarithmic sweep. Default: True.

    Returns
    -------
    np.ndarray
        1D float32 array containing the chirp signal.
    """
    if length is None:
        length = int(sr * (duration if duration is not None else 1.0))

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        status = lib.mm_chirp(ctx, float(fmin), float(fmax), sr, length,
                               1 if linear else 0, out)
        if status != 0:
            raise RuntimeError(f"mm_chirp failed with status {status}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def clicks(times=None, sr=22050, length=None, click_freq=1000.0,
           click_duration=0.1, **kwargs):
    """Generate a click track.

    Parameters
    ----------
    times : array-like or None
        Click times in seconds. If None, generates default click pattern.
    sr : int
        Sample rate. Default: 22050.
    length : int or None
        Total length in samples. If None, auto-sized.
    click_freq : float
        Click frequency in Hz. Default: 1000.0.
    click_duration : float
        Click duration in seconds. Default: 0.1.

    Returns
    -------
    np.ndarray
        1D float32 array containing the click track.
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        if times is not None:
            times_arr = np.ascontiguousarray(times, dtype=np.float32)
            times_ptr = ffi.cast("const float*", times_arr.ctypes.data)
            n_times = len(times_arr)
        else:
            times_ptr = ffi.NULL
            n_times = 0

        c_length = int(length) if length is not None else 0
        status = lib.mm_clicks(ctx, times_ptr, n_times, sr, c_length,
                                float(click_freq), float(click_duration), out)
        if status != 0:
            raise RuntimeError(f"mm_clicks failed with status {status}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
