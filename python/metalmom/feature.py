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


def chroma_stft(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                win_length=None, n_chroma=12, center=True, norm=None,
                tuning=0.0, **kwargs):
    """Compute STFT-based chroma features.

    Chroma features represent pitch content independent of octave by
    mapping the power spectrogram onto ``n_chroma`` pitch-class bins
    (C, C#, D, ..., B).

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).  Ignored if ``S`` is
        provided.
    sr : int
        Sample rate of ``y``.  Default: 22050.
    S : np.ndarray or None
        Pre-computed power spectrogram.  If provided, ``y`` is ignored
        and the chroma filterbank is applied directly to ``S``.
    n_fft : int
        FFT window size.  Default: 2048.
    hop_length : int or None
        Hop length.  Default: ``n_fft // 4``.
    win_length : int or None
        Window length.  Default: ``n_fft``.
    n_chroma : int
        Number of chroma bins.  Default: 12.
    center : bool
        Centre-pad signal before STFT.  Default: True.
    norm : float or None
        Normalization order per frame.  None = no normalization,
        2.0 = L2 normalization.  Default: None.
    tuning : float
        Tuning deviation from A440 in cents.  Default: 0.0.

    Returns
    -------
    np.ndarray
        Chroma features, shape ``(n_chroma, n_frames)``.
    """
    if S is not None:
        # Pre-computed power spectrogram path: apply chroma filterbank in Python
        S = np.ascontiguousarray(S, dtype=np.float32)
        n_freqs = S.shape[0]
        inferred_n_fft = (n_freqs - 1) * 2
        fb = _chroma_filterbank(sr=sr, n_fft=inferred_n_fft,
                                n_chroma=n_chroma, tuning=tuning)
        chroma = fb @ S
        if norm is not None:
            chroma = _normalize_frames(chroma, norm=norm)
        return chroma

    if y is None:
        raise ValueError("Either y or S must be provided")

    # Native path: compute chroma entirely in Swift/Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Match librosa convention: default hop_length=512 (NOT n_fft//4)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_chroma_stft(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            n_chroma,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_chroma_stft failed with status {status}")

        result = buffer_to_numpy(out)

        # Apply normalization in Python if requested (Swift bridge doesn't pass norm)
        if norm is not None:
            result = _normalize_frames(result, norm=norm)

        return result
    finally:
        lib.mm_destroy(ctx)


def _chroma_filterbank(sr, n_fft, n_chroma=12, tuning=0.0,
                       ctroct=5.0, octwidth=2.0, fb_norm=2.0, base_c=True):
    """Build a chroma filterbank matrix in Python (for the S= path).

    Matches librosa's ``librosa.filters.chroma()`` algorithm: Gaussian-windowed
    pitch class profiles with octave weighting.

    Returns shape ``(n_chroma, n_fft // 2 + 1)``.
    """
    n_freqs = n_fft // 2 + 1

    # FFT bin frequencies (excluding DC)
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    # Convert to fractional chroma bins
    a440 = 440.0 * 2.0 ** (tuning / n_chroma)
    frqbins = n_chroma * np.log2(frequencies / (a440 / 16.0))

    # DC placeholder: 1.5 octaves below bin 1
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    # Bin widths
    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    # Distance matrix D[c, k] = frqbins[k] - c, wrapped to [-nChroma/2, nChroma/2)
    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T
    n_chroma2 = np.round(float(n_chroma) / 2)
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # L2 normalize each column
    if fb_norm == 2.0:
        col_norms = np.sqrt(np.sum(wts ** 2, axis=0))
        col_norms[col_norms == 0] = 1.0
        wts /= col_norms
    elif fb_norm == 1.0:
        col_norms = np.sum(np.abs(wts), axis=0)
        col_norms[col_norms == 0] = 1.0
        wts /= col_norms

    # Octave weighting
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2)),
            (n_chroma, 1),
        )

    # Roll to start at C
    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    return np.ascontiguousarray(wts[:, :n_freqs], dtype=np.float32)


def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                      win_length=None, center=True, freq=None, **kwargs):
    """Compute spectral centroid.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed magnitude spectrogram. If provided, ``y`` is ignored.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: ``n_fft // 4``.
    win_length : int or None
        Window length. Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT. Default: True.

    Returns
    -------
    np.ndarray
        Spectral centroid, shape ``(1, n_frames)``.
    """
    if S is not None:
        S = np.ascontiguousarray(S, dtype=np.float32)
        n_freqs = S.shape[0]
        inferred_n_fft = (n_freqs - 1) * 2
        freqs = np.arange(n_freqs, dtype=np.float32) * sr / inferred_n_fft
        total = S.sum(axis=0, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return (freqs[:, None] * S).sum(axis=0, keepdims=True) / total

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_spectral_centroid(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_spectral_centroid failed with status {status}")
        result = buffer_to_numpy(out)
        return result.reshape(1, -1)
    finally:
        lib.mm_destroy(ctx)


def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                        win_length=None, center=True, freq=None, p=2, **kwargs):
    """Compute spectral bandwidth.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed magnitude spectrogram.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: ``n_fft // 4``.
    win_length : int or None
        Window length. Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT. Default: True.
    p : float
        Power for p-th moment. Default: 2.

    Returns
    -------
    np.ndarray
        Spectral bandwidth, shape ``(1, n_frames)``.
    """
    if S is not None:
        S = np.ascontiguousarray(S, dtype=np.float32)
        n_freqs = S.shape[0]
        inferred_n_fft = (n_freqs - 1) * 2
        freqs = np.arange(n_freqs, dtype=np.float32) * sr / inferred_n_fft
        centroid = spectral_centroid(S=S, sr=sr, n_fft=inferred_n_fft)
        total = S.sum(axis=0, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        deviation = np.abs(freqs[:, None] - centroid)
        bw = ((S * deviation ** p).sum(axis=0, keepdims=True) / total) ** (1.0 / p)
        return bw

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_spectral_bandwidth(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            float(p),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_spectral_bandwidth failed with status {status}")
        result = buffer_to_numpy(out)
        return result.reshape(1, -1)
    finally:
        lib.mm_destroy(ctx)


def spectral_contrast(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                       win_length=None, center=True, n_bands=6, fmin=200.0,
                       quantile=0.02, **kwargs):
    """Compute spectral contrast.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed magnitude spectrogram.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: ``n_fft // 4``.
    win_length : int or None
        Window length. Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT. Default: True.
    n_bands : int
        Number of octave sub-bands. Default: 6.
    fmin : float
        Low frequency bound. Default: 200.0.
    quantile : float
        Fraction for peak/valley. Default: 0.02.

    Returns
    -------
    np.ndarray
        Spectral contrast, shape ``(n_bands + 1, n_frames)``.
    """
    if S is not None:
        raise NotImplementedError("spectral_contrast with pre-computed S not yet supported")

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_spectral_contrast(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            n_bands, float(fmin),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_spectral_contrast failed with status {status}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                      win_length=None, center=True, roll_percent=0.85, **kwargs):
    """Compute spectral rolloff frequency.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed magnitude spectrogram.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: ``n_fft // 4``.
    win_length : int or None
        Window length. Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT. Default: True.
    roll_percent : float
        Energy fraction threshold. Default: 0.85.

    Returns
    -------
    np.ndarray
        Spectral rolloff, shape ``(1, n_frames)``.
    """
    if S is not None:
        S = np.ascontiguousarray(S, dtype=np.float32)
        n_freqs = S.shape[0]
        inferred_n_fft = (n_freqs - 1) * 2
        freqs = np.arange(n_freqs, dtype=np.float32) * sr / inferred_n_fft
        total_energy = S.sum(axis=0, keepdims=True)
        threshold = roll_percent * total_energy
        cumulative = np.cumsum(S, axis=0)
        result = np.zeros((1, S.shape[1]), dtype=np.float32)
        for f in range(S.shape[1]):
            if total_energy[0, f] <= 0:
                result[0, f] = 0
            else:
                idx = np.searchsorted(cumulative[:, f], threshold[0, f])
                idx = min(idx, n_freqs - 1)
                result[0, f] = freqs[idx]
        return result

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_spectral_rolloff(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            float(roll_percent),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_spectral_rolloff failed with status {status}")
        result = buffer_to_numpy(out)
        return result.reshape(1, -1)
    finally:
        lib.mm_destroy(ctx)


def spectral_flatness(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
                       win_length=None, center=True, power=2.0, amin=1e-10, **kwargs):
    """Compute spectral flatness (Wiener entropy).

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    S : np.ndarray or None
        Pre-computed magnitude spectrogram.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: ``n_fft // 4``.
    win_length : int or None
        Window length. Default: ``n_fft``.
    center : bool
        Centre-pad signal before STFT. Default: True.
    power : float
        Exponent for magnitude spectrogram. Default: 2.0.
    amin : float
        Minimum amplitude. Default: 1e-10.

    Returns
    -------
    np.ndarray
        Spectral flatness, shape ``(1, n_frames)``.
    """
    if S is not None:
        S = np.ascontiguousarray(S, dtype=np.float32)
        S_floored = np.maximum(S, amin)
        raw_sum = S.sum(axis=0)
        geo_mean = np.exp(np.mean(np.log(S_floored), axis=0))
        arith_mean = np.mean(S_floored, axis=0)
        result = np.where(raw_sum > amin, geo_mean / arith_mean, 0.0)
        return result.reshape(1, -1).astype(np.float32)

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)
        status = lib.mm_spectral_flatness(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_spectral_flatness failed with status {status}")
        result = buffer_to_numpy(out)
        return result.reshape(1, -1)
    finally:
        lib.mm_destroy(ctx)


def rms(y=None, S=None, frame_length=2048, hop_length=512, center=True, **kwargs):
    """Compute root-mean-square (RMS) energy.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    S : np.ndarray or None
        Pre-computed spectrogram.  If provided, RMS is computed from the
        spectrogram columns: ``RMS[t] = sqrt(mean(S[:, t]**2))``.
    frame_length : int
        Length of each analysis frame. Default: 2048.
    hop_length : int
        Hop length. Default: 512.
    center : bool
        Centre-pad signal before framing. Default: True.

    Returns
    -------
    np.ndarray
        RMS energy, shape ``(1, n_frames)``.
    """
    if S is not None:
        # Pre-computed spectrogram path: RMS from columns
        S = np.ascontiguousarray(S, dtype=np.float32)
        rms_vals = np.sqrt(np.mean(S ** 2, axis=0, keepdims=True))
        return rms_vals

    if y is None:
        raise ValueError("Either y or S must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_rms(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used by RMS, but required by bridge)
            frame_length, hop_length,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_rms failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True, **kwargs):
    """Compute zero-crossing rate.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D float32/float64 array).
    frame_length : int
        Length of each analysis frame. Default: 2048.
    hop_length : int
        Hop length. Default: 512.
    center : bool
        Centre-pad signal before framing. Default: True.

    Returns
    -------
    np.ndarray
        Zero-crossing rate, shape ``(1, n_frames)``.
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

        status = lib.mm_zero_crossing_rate(
            ctx, signal_ptr, len(y),
            22050,  # sample_rate (not used by ZCR, but required by bridge)
            frame_length, hop_length,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_zero_crossing_rate failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def tonnetz(y=None, sr=22050, chroma=None, n_fft=2048, hop_length=None,
            win_length=None, n_chroma=12, center=True, **kwargs):
    """Compute tonnetz (tonal centroid) features.

    Tonnetz features are 6-dimensional representations of tonal content
    based on projections of chroma features onto harmonic interval circles
    (fifths, minor thirds, major thirds) on the Tonnetz lattice.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).  Ignored if ``chroma``
        is provided.
    sr : int
        Sample rate of ``y``.  Default: 22050.
    chroma : np.ndarray or None
        Pre-computed chroma features, shape ``(n_chroma, n_frames)``.
        If provided, ``y`` is ignored and tonnetz is computed from this
        chroma directly.
    n_fft : int
        FFT window size.  Default: 2048.
    hop_length : int or None
        Hop length.  Default: ``n_fft // 4``.
    win_length : int or None
        Window length.  Default: ``n_fft``.
    n_chroma : int
        Number of chroma bins.  Default: 12.
    center : bool
        Centre-pad signal before STFT.  Default: True.

    Returns
    -------
    np.ndarray
        Tonnetz features, shape ``(6, n_frames)``.
    """
    if chroma is not None:
        # Pre-computed chroma path: compute tonnetz in Python
        chroma = np.ascontiguousarray(chroma, dtype=np.float32)
        n_c = chroma.shape[0]

        # L1-normalize per frame
        col_sums = np.sum(np.abs(chroma), axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-10, col_sums, 1.0)
        chroma_norm = chroma / col_sums

        # Angular basis vectors (matches librosa exactly)
        # Even dims are sin, odd dims are cos
        r1, r2, r3 = 1.0, 1.0, 0.5
        c_idx = np.arange(n_c, dtype=np.float64)
        phi1 = c_idx * 7.0 * np.pi / 6.0   # fifths
        phi2 = c_idx * 3.0 * np.pi / 2.0   # minor thirds
        phi3 = c_idx * 2.0 * np.pi / 3.0   # major thirds

        basis = np.array([
            r1 * np.sin(phi1),
            r1 * np.cos(phi1),
            r2 * np.sin(phi2),
            r2 * np.cos(phi2),
            r3 * np.sin(phi3),
            r3 * np.cos(phi3),
        ], dtype=np.float32)  # shape (6, n_chroma)

        return (basis @ chroma_norm).astype(np.float32)

    if y is None:
        raise ValueError("Either y or chroma must be provided")

    # Native path: compute tonnetz entirely in Swift/Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Match librosa convention: default hop_length=512 (NOT n_fft//4)
    hop = hop_length if hop_length is not None else 512
    win = win_length if win_length is not None else n_fft

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_tonnetz(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop, win,
            n_chroma,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_tonnetz failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def _normalize_frames(chroma, norm=2.0):
    """Normalize each frame (column) of a chroma matrix.

    Parameters
    ----------
    chroma : np.ndarray, shape (n_chroma, n_frames)
    norm : float
        Normalization order (1.0 = L1, 2.0 = L2, np.inf = Linf).

    Returns
    -------
    np.ndarray
        Normalized chroma matrix.
    """
    chroma = chroma.copy()
    for f in range(chroma.shape[1]):
        col = chroma[:, f]
        if norm == 2.0:
            n = np.sqrt(np.sum(col ** 2))
        elif norm == 1.0:
            n = np.sum(np.abs(col))
        elif norm == np.inf:
            n = np.max(np.abs(col))
        else:
            n = np.linalg.norm(col, ord=norm)
        if n > 1e-10:
            chroma[:, f] /= n
    return chroma


def tempo(y=None, sr=22050, onset_envelope=None, hop_length=512,
          n_fft=2048, n_mels=128, fmin=0.0, fmax=None,
          start_bpm=120.0, center=True, **kwargs):
    """Estimate tempo (beats per minute).

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal (1D float32/float64 array).
    sr : int
        Sample rate. Default: 22050.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope (not used by native backend;
        accepted for API compatibility but ignored -- the native code
        computes its own envelope from ``y``).
    hop_length : int
        Hop length. Default: 512.
    n_fft : int
        FFT window size. Default: 2048.
    n_mels : int
        Number of mel bands. Default: 128.
    fmin : float
        Minimum frequency. Default: 0.0.
    fmax : float or None
        Maximum frequency. Default: None (sr/2).
    start_bpm : float
        Initial tempo estimate (BPM). Default: 120.0.
    center : bool
        Centre-pad signal before STFT. Default: True.

    Returns
    -------
    np.ndarray
        Array of estimated tempos (single element).
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_fmax = float(fmax) if fmax is not None else 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_tempo = ffi.new("float*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_tempo(
            ctx, signal_ptr, len(y),
            sr, hop_length, n_fft,
            n_mels, fmin, c_fmax,
            start_bpm, 1 if center else 0,
            out_tempo,
        )
        if status != 0:
            raise RuntimeError(f"mm_tempo failed with status {status}")

        return np.array([float(out_tempo[0])], dtype=np.float64)
    finally:
        lib.mm_destroy(ctx)


def poly_features(S=None, y=None, sr=22050, n_fft=2048, hop_length=None,
                   win_length=None, center=True, order=1, **kwargs):
    """Compute polynomial features from a spectrogram.

    For each frame, fits a polynomial of given order to the frequency axis
    and returns the coefficients in descending power order.

    Parameters
    ----------
    S : np.ndarray or None
        Pre-computed spectrogram, shape (n_freqs, n_frames).
    y : np.ndarray or None
        Audio signal. If S is None, computes STFT magnitude first.
    sr : int
        Sample rate. Default: 22050.
    n_fft : int
        FFT window size. Default: 2048.
    hop_length : int or None
        Hop length. Default: 512.
    win_length : int or None
        Window length. Default: n_fft.
    center : bool
        Centre-pad signal. Default: True.
    order : int
        Polynomial order. Default: 1.

    Returns
    -------
    np.ndarray
        Polynomial coefficients, shape (order+1, n_frames).
    """
    if S is None:
        if y is None:
            raise ValueError("Either y or S must be provided")
        from .core import stft
        hop = hop_length if hop_length is not None else 512
        win = win_length if win_length is not None else n_fft
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop, win_length=win, center=center))

    S = np.ascontiguousarray(S, dtype=np.float32)
    n_features, n_frames = S.shape
    flat = S.ravel()

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", flat.ctypes.data)

        # Infer n_fft from spectrogram shape: n_freqs = n_fft // 2 + 1
        inferred_n_fft = (n_features - 1) * 2

        status = lib.mm_poly_features(
            ctx, data_ptr, len(flat),
            n_features, n_frames,
            order, sr, inferred_n_fft,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_poly_features failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def delta(data, width=9, order=1, axis=-1, **kwargs):
    """Compute delta (derivative) features.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix, shape (n_features, n_frames).
    width : int
        Half-width of the Savitzky-Golay filter window. Default: 9.
    order : int
        Order of the derivative (1=delta, 2=delta-delta). Default: 1.
    axis : int
        Axis along which to compute delta. Default: -1 (last axis, i.e., time).

    Returns
    -------
    np.ndarray
        Delta features, same shape as input.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_features, n_frames = data.shape
    flat = data.ravel()

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", flat.ctypes.data)

        status = lib.mm_delta(
            ctx, data_ptr, len(flat),
            n_features, n_frames,
            width, order,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_delta failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def stack_memory(data, n_steps=2, delay=1, **kwargs):
    """Stack consecutive frames of a feature matrix.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix, shape (n_features, n_frames).
    n_steps : int
        Number of time steps to stack. Default: 2.
    delay : int
        Number of frames to delay per step. Default: 1.

    Returns
    -------
    np.ndarray
        Stacked features, shape (n_features * n_steps, n_frames).
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_features, n_frames = data.shape
    flat = data.ravel()

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", flat.ctypes.data)

        status = lib.mm_stack_memory(
            ctx, data_ptr, len(flat),
            n_features, n_frames,
            n_steps, delay,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_stack_memory failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
