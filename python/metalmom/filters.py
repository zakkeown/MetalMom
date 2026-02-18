"""Filter functions: mel, chroma, constant-Q, and semitone filterbanks."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy
from ._mel_fb import _mel_filterbank, _hz_to_mel, _mel_to_hz


def semitone_filterbank(y, sr=22050, midi_low=24, midi_high=119, order=4):
    """Apply a semitone bandpass filterbank to an audio signal.

    Filters the input signal through bandpass filters centered at each
    semitone in the specified MIDI range. Each filter has a constant-Q
    bandwidth of one semitone.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal (1D, float32).
    sr : int
        Sample rate in Hz. Default: 22050.
    midi_low : int
        Lowest MIDI note number. Default: 24 (C1, ~32.7 Hz).
    midi_high : int
        Highest MIDI note number. Default: 119 (B8, ~7902 Hz).
    order : int
        Filter order (number of cascaded biquad pairs). Default: 4.

    Returns
    -------
    np.ndarray
        Filtered signal, shape ``(n_semitones, n_samples)``.
    """
    y = np.ascontiguousarray(y, dtype=np.float32).ravel()

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_semitone_filterbank(
            ctx,
            data_ptr,
            len(y),
            int(sr),
            int(midi_low),
            int(midi_high),
            int(order),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_semitone_filterbank failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def semitone_frequencies(midi_low=24, midi_high=119):
    """Compute center frequencies for semitone bands.

    Parameters
    ----------
    midi_low : int
        Lowest MIDI note. Default: 24 (C1).
    midi_high : int
        Highest MIDI note. Default: 119 (B8).

    Returns
    -------
    np.ndarray
        Array of center frequencies in Hz.
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        status = lib.mm_semitone_frequencies(
            ctx,
            int(midi_low),
            int(midi_high),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_semitone_frequencies failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------

def mel(sr=22050, n_fft=2048, n_mels=128, fmin=0.0, fmax=None):
    """Create a mel filterbank matrix.

    Builds a matrix of triangular mel-scaled filters suitable for
    converting a spectrogram to a mel spectrogram.

    Parameters
    ----------
    sr : int
        Sample rate of the audio. Default: 22050.
    n_fft : int
        FFT window size. Default: 2048.
    n_mels : int
        Number of mel bands. Default: 128.
    fmin : float
        Lowest filter frequency in Hz. Default: 0.0.
    fmax : float or None
        Highest filter frequency in Hz. Defaults to ``sr / 2``.

    Returns
    -------
    np.ndarray
        Mel filterbank matrix, shape ``(n_mels, n_fft // 2 + 1)``,
        dtype float32.
    """
    return _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)


# ---------------------------------------------------------------------------
# Chroma filterbank
# ---------------------------------------------------------------------------

def chroma(sr=22050, n_fft=2048, n_chroma=12, tuning=0.0, ctroct=5.0,
           octwidth=2, base_c=True):
    """Create a chroma filterbank matrix.

    Maps FFT bins to chroma (pitch-class) bins using log-frequency
    mapping with optional Gaussian octave weighting.

    Parameters
    ----------
    sr : int
        Sample rate. Default: 22050.
    n_fft : int
        FFT window size. Default: 2048.
    n_chroma : int
        Number of chroma bins. Default: 12.
    tuning : float
        Tuning deviation in fractions of a chroma bin. Default: 0.0.
    ctroct : float
        Center octave for the Gaussian octave weight. Default: 5.0.
    octwidth : float
        Width (standard deviation) of the Gaussian octave weight.
        Set to 0 or negative for uniform octave weighting. Default: 2.
    base_c : bool
        If True, chroma bin 0 corresponds to C. If False, bin 0
        corresponds to A. Default: True.

    Returns
    -------
    np.ndarray
        Chroma filterbank matrix, shape ``(n_chroma, n_fft // 2 + 1)``,
        dtype float32.
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freqs[0] = 1e-6  # avoid log(0) for DC bin

    # Convert frequencies to fractional chroma bin positions.
    # A4 = 440 Hz. In C-based chroma, A is pitch class 9.
    A440_chroma = 9.0 if base_c else 0.0
    fft_chroma = n_chroma * np.log2(freqs / 440.0) + A440_chroma + tuning
    fft_chroma = fft_chroma % n_chroma

    # Build filterbank: Gaussian around each chroma bin
    wts = np.zeros((n_chroma, n_freqs), dtype=np.float64)
    for i in range(n_chroma):
        # Circular distance to chroma bin i
        diff = (fft_chroma - i + n_chroma / 2) % n_chroma - n_chroma / 2
        wts[i] = np.exp(-0.5 * diff ** 2)

    # Gaussian octave weighting centred at ctroct
    fft_octave = np.log2(freqs / (440.0 / 16))  # octave number
    if octwidth > 0:
        oct_weight = np.exp(-0.5 * ((fft_octave - ctroct) / octwidth) ** 2)
    else:
        oct_weight = np.ones_like(fft_octave)

    wts *= oct_weight[np.newaxis, :]

    # Normalize each column to unit L2 norm
    col_norms = np.sqrt(np.sum(wts ** 2, axis=0, keepdims=True))
    col_norms[col_norms == 0] = 1
    wts /= col_norms

    return wts.astype(np.float32)


# ---------------------------------------------------------------------------
# Constant-Q filterbank
# ---------------------------------------------------------------------------

def constant_q(sr=22050, fmin=None, n_bins=84, bins_per_octave=12,
               tuning=0.0, norm=1, pad_fft=True):
    """Create a constant-Q filterbank (complex-valued).

    Each row is the FFT of a windowed complex sinusoid at the
    corresponding CQ frequency. This is a time-domain filterbank
    represented in the frequency domain.

    Parameters
    ----------
    sr : int
        Sample rate. Default: 22050.
    fmin : float or None
        Minimum frequency. Defaults to C1 (~32.703 Hz), adjusted
        for *tuning*.
    n_bins : int
        Number of CQ frequency bins. Default: 84 (7 octaves).
    bins_per_octave : int
        Number of bins per octave. Default: 12.
    tuning : float
        Tuning offset in fractions of a bin. Default: 0.0.
    norm : {1, 2} or None
        Normalization type for each filter.
        ``1`` normalizes by L1 norm, ``2`` by L2 norm, ``None`` skips.
        Default: 1.
    pad_fft : bool
        If True, zero-pad filters to the next power of two. Default: True.

    Returns
    -------
    np.ndarray
        Complex-valued CQ filterbank, shape ``(n_bins, fft_len)``,
        dtype complex64.
    """
    if fmin is None:
        fmin = 32.703 * 2.0 ** (tuning / bins_per_octave)  # C1 adjusted
    else:
        fmin = float(fmin) * 2.0 ** (tuning / bins_per_octave)

    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1)

    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)

    # Filter lengths: ceil(Q * sr / freq)
    lengths = np.ceil(Q * sr / freqs).astype(int)

    max_len = int(max(lengths))
    if pad_fft:
        fft_len = 1
        while fft_len < max_len:
            fft_len *= 2
    else:
        fft_len = max_len

    # Build windowed sinusoids and take their FFT
    filters = np.zeros((n_bins, fft_len), dtype=np.complex128)
    for i in range(n_bins):
        N = int(lengths[i])
        t = np.arange(N)
        # Hamming window, normalised sinusoid
        window = np.hamming(N)
        sig = window * np.exp(2j * np.pi * freqs[i] * t / sr) / N

        if norm == 1:
            sig /= np.sum(np.abs(sig))
        elif norm == 2:
            sig /= np.sqrt(np.sum(np.abs(sig) ** 2))

        filters[i, :N] = sig

    # FFT each filter row
    filters = np.fft.fft(filters, n=fft_len, axis=1)

    return filters.astype(np.complex64)


# ---------------------------------------------------------------------------
# Frequency helpers (pure-Python, no native library required)
# ---------------------------------------------------------------------------

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """Array of mel-spaced frequencies.

    Pure-Python implementation using the Slaney mel scale.

    Parameters
    ----------
    n_mels : int
        Number of mel frequencies. Default: 128.
    fmin : float
        Minimum frequency in Hz. Default: 0.0.
    fmax : float
        Maximum frequency in Hz. Default: 11025.0.

    Returns
    -------
    np.ndarray
        Mel-spaced frequencies in Hz, shape ``(n_mels,)``, dtype float32.
    """
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = np.array([_mel_to_hz(m) for m in mels], dtype=np.float32)
    # Return n_mels interior points (excluding endpoints)
    return freqs[1:-1]


def fft_frequencies(sr=22050, n_fft=2048):
    """Array of FFT bin center frequencies.

    Parameters
    ----------
    sr : int
        Sample rate in Hz. Default: 22050.
    n_fft : int
        FFT window size. Default: 2048.

    Returns
    -------
    np.ndarray
        Center frequencies for each FFT bin, shape ``(n_fft // 2 + 1,)``,
        dtype float32.
    """
    return np.linspace(0, sr / 2.0, n_fft // 2 + 1, dtype=np.float32)
