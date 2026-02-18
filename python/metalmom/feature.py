"""Feature extraction functions (mel spectrogram, mfcc, dB scaling, etc.)."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                   win_length=None, center=True, power=2.0, n_mels=128,
                   fmin=0.0, fmax=None, **kwargs):
    """Compute a mel-scaled spectrogram.

    If a pre-computed (log-)power spectrogram ``S`` is provided, it is
    used directly.  Otherwise, ``S`` is computed from audio ``y`` via
    ``STFT magnitude -> power -> mel filterbank``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).  Ignored if ``S`` is
        provided.
    sr : int
        Sample rate of ``y``.  Default: 22050.
    S : np.ndarray or None
        Pre-computed (power) spectrogram.  If provided, ``y`` is ignored
        and the mel filterbank is applied directly to ``S``.
    n_fft : int
        FFT window size.  Default: 2048.
    hop_length : int or None
        Hop length.  Default: ``n_fft // 4``.
    win_length : int or None
        Window length.  Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT.  Default: True.
    power : float
        Exponent for the magnitude spectrogram (1.0 = amplitude,
        2.0 = power).  Default: 2.0.
    n_mels : int
        Number of mel bands.  Default: 128.
    fmin : float
        Lowest frequency (Hz) for the mel filterbank.  Default: 0.0.
    fmax : float or None
        Highest frequency (Hz).  If None, uses ``sr / 2``.

    Returns
    -------
    np.ndarray
        Mel spectrogram, shape ``(n_mels, n_frames)``.
    """
    if S is not None:
        # Pre-computed spectrogram path: apply mel filterbank in Python
        # (matches librosa behaviour when S is given)
        S = np.ascontiguousarray(S, dtype=np.float32)
        # Build mel filterbank using our own algorithm (matches Swift FilterBank.mel)
        from ._mel_fb import _mel_filterbank
        mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        return mel_fb @ S

    if y is None:
        raise ValueError("Either y or S must be provided")

    # Native path: compute mel spectrogram entirely in Swift/Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Match librosa convention: default hop_length=512 (NOT n_fft//4)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft
    c_fmax = float(fmax) if fmax is not None else 0.0  # 0.0 signals "use sr/2"

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_mel_spectrogram(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            power, n_mels,
            fmin, c_fmax,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_mel_spectrogram failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def mfcc(y=None, sr=22050, S=None, n_mfcc=20, n_fft=2048, hop_length=None,
         win_length=None, n_mels=128, fmin=0.0, fmax=None, center=True, **kwargs):
    """Compute Mel-frequency cepstral coefficients (MFCCs).

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).  Ignored if ``S`` is
        provided.
    sr : int
        Sample rate of ``y``.  Default: 22050.
    S : np.ndarray or None
        Pre-computed log-power mel spectrogram.  If provided, ``y`` is
        ignored and the DCT is applied directly to ``S``.
    n_mfcc : int
        Number of MFCC coefficients to return.  Default: 20.
    n_fft : int
        FFT window size.  Default: 2048.
    hop_length : int or None
        Hop length.  Default: ``n_fft // 4``.
    win_length : int or None
        Window length.  Default: ``n_fft``.
    n_mels : int
        Number of mel bands.  Default: 128.
    fmin : float
        Lowest frequency (Hz) for the mel filterbank.  Default: 0.0.
    fmax : float or None
        Highest frequency (Hz).  If None, uses ``sr / 2``.
    center : bool
        Centre-pad signal before STFT.  Default: True.

    Returns
    -------
    np.ndarray
        MFCCs, shape ``(n_mfcc, n_frames)``.
    """
    if S is not None:
        # Pre-computed log-mel spectrogram: apply DCT-II in Python
        from scipy.fftpack import dct
        S = np.ascontiguousarray(S, dtype=np.float32)
        return dct(S, axis=0, type=2, norm='ortho')[:n_mfcc]

    if y is None:
        raise ValueError("Either y or S must be provided")

    # Native path: compute MFCCs entirely in Swift/Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Match librosa convention: default hop_length=512 (NOT n_fft//4)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft
    c_fmax = float(fmax) if fmax is not None else 0.0  # 0.0 signals "use sr/2"

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_mfcc(
            ctx, signal_ptr, len(y),
            sr, n_mfcc, n_fft, hop, win,
            n_mels, fmin, c_fmax,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_mfcc failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


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
