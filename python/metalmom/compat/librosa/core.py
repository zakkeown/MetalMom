"""librosa.core compatibility shim."""

import numpy as np

from metalmom.core import load, resample, stft, istft, db_to_amplitude, db_to_power, tone, chirp, clicks, get_duration, get_samplerate, stream, reassigned_spectrogram
from metalmom.effects import phase_vocoder, griffinlim, griffinlim_cqt
from metalmom.feature import amplitude_to_db, power_to_db, pcen
from metalmom.pitch import yin, pyin, piptrack, estimate_tuning
from metalmom.convert import (
    hz_to_midi, midi_to_hz, hz_to_note, note_to_hz,
    midi_to_note, note_to_midi,
    fft_frequencies, mel_frequencies,
    frames_to_time, samples_to_time,
    frames_to_samples, samples_to_frames,
)
from metalmom.cqt import cqt, vqt, hybrid_cqt


# ---------------------------------------------------------------------------
# Pure-Python helpers that librosa exposes at the top level
# ---------------------------------------------------------------------------

def magphase(D, power=1):
    """Separate a complex spectrogram into magnitude and phase.

    Parameters
    ----------
    D : np.ndarray
        Complex spectrogram (e.g. from STFT).
    power : float
        Exponent for the magnitude. Default: 1.

    Returns
    -------
    magnitude : np.ndarray
        Magnitude raised to ``power``.
    phase : np.ndarray
        Unit-phase complex array (angle preserved, magnitude 1).
    """
    D = np.asarray(D)
    mag = np.abs(D)
    if power != 1:
        mag = mag ** power
    # Unit-phase: exp(1j * angle)
    phase = np.exp(1j * np.angle(D))
    return mag.astype(np.float32), phase.astype(np.complex64)


def to_mono(y):
    """Convert multi-channel audio to mono by averaging channels.

    Parameters
    ----------
    y : np.ndarray
        Audio signal. If 1D, returned as-is. If 2D, shape (n_channels, n_samples),
        averaged across channels.

    Returns
    -------
    np.ndarray
        Mono audio signal (1D float32).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0).astype(np.float32)


def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True,
                   zero_pos=True, axis=-1, **kwargs):
    """Find zero-crossings in a signal.

    Parameters
    ----------
    y : np.ndarray
        Signal array.
    threshold : float
        Threshold for considering a crossing. Default: 1e-10.
    ref_magnitude : float or callable or None
        Reference magnitude for thresholding. Default: None.
    pad : bool
        If True, pad the first sample to match length. Default: True.
    zero_pos : bool
        If True, 0-to-positive counts as a crossing. Default: True.
    axis : int
        Axis along which to find crossings. Default: -1.

    Returns
    -------
    np.ndarray
        Boolean array of zero-crossing locations.
    """
    y = np.asarray(y)

    if ref_magnitude is not None:
        if callable(ref_magnitude):
            threshold = threshold * ref_magnitude(np.abs(y))
        else:
            threshold = threshold * ref_magnitude

    if zero_pos:
        y = y.copy()
        y[y == 0] = 1e-20

    # Compare signs of consecutive samples
    sign_changes = np.diff(np.sign(y), axis=axis)
    crossings = np.abs(sign_changes) > 0

    if pad:
        # Pad the first position along the specified axis
        pad_width = [(0, 0)] * y.ndim
        pad_width[axis] = (1, 0)
        crossings = np.pad(crossings, pad_width, mode='constant',
                           constant_values=False)

    return crossings


def autocorrelate(y, max_size=None, axis=-1, **kwargs):
    """Bounded auto-correlation of a signal.

    Parameters
    ----------
    y : np.ndarray
        Signal array.
    max_size : int or None
        Maximum correlation lag. Default: None (full).
    axis : int
        Axis along which to correlate. Default: -1.

    Returns
    -------
    np.ndarray
        Auto-correlation array.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    Y = np.fft.rfft(y, n=fft_size, axis=axis)
    acf = np.fft.irfft(Y * np.conj(Y), n=fft_size, axis=axis)

    # Trim to original length
    slices = [slice(None)] * y.ndim
    if max_size is not None:
        slices[axis] = slice(0, min(max_size, n))
    else:
        slices[axis] = slice(0, n)
    return acf[tuple(slices)].astype(np.float32)


def lpc(y, order, axis=-1, **kwargs):
    """Linear prediction coefficients via Burg's method.

    Parameters
    ----------
    y : np.ndarray
        Signal array.
    order : int
        LPC order.
    axis : int
        Axis along which to compute. Default: -1.

    Returns
    -------
    np.ndarray
        LPC coefficients, shape ``(..., order + 1)``.
    """
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        return _lpc_burg_1d(y, order)

    # Multi-dimensional: apply along axis
    return np.apply_along_axis(_lpc_burg_1d, axis, y, order)


def _lpc_burg_1d(y, order):
    """Burg's method for a 1D signal."""
    n = len(y)
    if n <= order:
        raise ValueError(f"Signal length {n} must exceed order {order}")

    # Initialize
    d = np.zeros(order + 1, dtype=np.float64)
    d[0] = 1.0

    ef = np.array(y, dtype=np.float64)
    eb = np.array(y, dtype=np.float64)

    for i in range(order):
        # Reflection coefficient
        efp = ef[i + 1:]
        ebp = eb[i:-1]
        num = -2.0 * np.dot(efp, ebp)
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        if den == 0:
            k = 0.0
        else:
            k = num / den

        # Update forward/backward errors
        ef_new = efp + k * ebp
        eb_new = ebp + k * efp
        ef = np.concatenate([ef[:i + 1], ef_new])
        eb = np.concatenate([eb[:i + 1], eb_new])

        # Update coefficients
        d_new = np.zeros(order + 1, dtype=np.float64)
        for j in range(i + 2):
            d_new[j] = d[j] + k * d[i + 1 - j]
        d = d_new

    return d.astype(np.float32)


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0, **kwargs):
    """Compute center frequencies for CQT bins.

    Parameters
    ----------
    n_bins : int
        Number of CQ bins.
    fmin : float
        Minimum frequency in Hz.
    bins_per_octave : int
        Number of bins per octave. Default: 12.
    tuning : float
        Tuning offset in fractions of a bin. Default: 0.0.

    Returns
    -------
    np.ndarray
        Center frequencies in Hz, shape ``(n_bins,)``.
    """
    correction = 2.0 ** (tuning / bins_per_octave)
    return (fmin * correction *
            2.0 ** (np.arange(n_bins) / bins_per_octave)).astype(np.float32)


def tempo_frequencies(n_bins, hop_length=512, sr=22050, **kwargs):
    """Compute tempo (BPM) axis for an autocorrelation tempogram.

    Parameters
    ----------
    n_bins : int
        Number of lag bins.
    hop_length : int
        Hop length. Default: 512.
    sr : float
        Sample rate. Default: 22050.

    Returns
    -------
    np.ndarray
        Tempo values in BPM, shape ``(n_bins,)``.
    """
    bin_freqs = np.zeros(n_bins, dtype=np.float64)
    bin_freqs[0] = np.inf  # lag-0 -> infinite tempo
    bin_freqs[1:] = 60.0 * sr / (hop_length * np.arange(1, n_bins))
    return bin_freqs.astype(np.float32)


def fourier_tempo_frequencies(sr=22050, win_length=384, hop_length=512, **kwargs):
    """Compute tempo (BPM) axis for a Fourier tempogram.

    Parameters
    ----------
    sr : float
        Sample rate. Default: 22050.
    win_length : int
        Window length of the Fourier tempogram. Default: 384.
    hop_length : int
        Hop length. Default: 512.

    Returns
    -------
    np.ndarray
        Tempo values in BPM, shape ``(win_length // 2 + 1,)``.
    """
    n_bins = win_length // 2 + 1
    bin_freqs = np.arange(n_bins, dtype=np.float64) * sr / (hop_length * win_length)
    return (bin_freqs * 60.0).astype(np.float32)


def times_like(X, sr=22050, hop_length=512, n_fft=None, axis=-1, **kwargs):
    """Return time values matching the frames of a feature array.

    Parameters
    ----------
    X : np.ndarray or float
        Feature array (the number of frames is inferred from ``X.shape[axis]``).
    sr : float
        Sample rate. Default: 22050.
    hop_length : int
        Hop length. Default: 512.
    n_fft : int or None
        FFT size (unused, for compat). Default: None.
    axis : int
        Axis from which to infer number of frames. Default: -1.

    Returns
    -------
    np.ndarray
        Time values in seconds.
    """
    if np.isscalar(X):
        n_frames = int(X)
    else:
        X = np.asarray(X)
        n_frames = X.shape[axis]
    return (np.arange(n_frames) * hop_length / sr).astype(np.float32)


def samples_like(X, hop_length=512, n_fft=None, axis=-1, **kwargs):
    """Return sample indices matching the frames of a feature array.

    Parameters
    ----------
    X : np.ndarray or float
        Feature array (the number of frames is inferred from ``X.shape[axis]``).
    hop_length : int
        Hop length. Default: 512.
    n_fft : int or None
        FFT size (unused, for compat). Default: None.
    axis : int
        Axis from which to infer number of frames. Default: -1.

    Returns
    -------
    np.ndarray
        Sample indices.
    """
    if np.isscalar(X):
        n_frames = int(X)
    else:
        X = np.asarray(X)
        n_frames = X.shape[axis]
    return (np.arange(n_frames) * hop_length).astype(np.intp)


def time_to_frames(times, sr=22050, hop_length=512, n_fft=None, **kwargs):
    """Alias for ``times_to_frames`` (librosa uses both spellings)."""
    from metalmom.convert import times_to_frames
    return times_to_frames(times, sr=sr, hop_length=hop_length, n_fft=n_fft)


def time_to_samples(times, sr=22050, **kwargs):
    """Alias for ``times_to_samples`` (librosa uses both spellings)."""
    from metalmom.convert import times_to_samples
    return times_to_samples(times, sr=sr)


def hz_to_mel(frequencies, htk=False, **kwargs):
    """Convert Hz to mel scale.

    Parameters
    ----------
    frequencies : float or np.ndarray
        Frequency/frequencies in Hz.
    htk : bool
        Use HTK formula. Default: False (Slaney/O'Shaughnessy).

    Returns
    -------
    float or np.ndarray
        Mel value(s).
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if htk:
        return (2595.0 * np.log10(1.0 + frequencies / 700.0)).astype(np.float32)

    # Slaney formula: linear below 1000 Hz, log above
    f_0 = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_0) / f_sp
    logstep = np.log(6.4) / 27.0

    result = np.where(
        frequencies < min_log_hz,
        (frequencies - f_0) / f_sp,
        min_log_mel + np.log(frequencies / min_log_hz) / logstep,
    )
    scalar = frequencies.ndim == 0
    return float(result) if scalar else result.astype(np.float32)


def mel_to_hz(mels, htk=False, **kwargs):
    """Convert mel scale to Hz.

    Parameters
    ----------
    mels : float or np.ndarray
        Mel value(s).
    htk : bool
        Use HTK formula. Default: False (Slaney/O'Shaughnessy).

    Returns
    -------
    float or np.ndarray
        Frequency/frequencies in Hz.
    """
    mels = np.asarray(mels, dtype=np.float64)
    if htk:
        return (700.0 * (10.0 ** (mels / 2595.0) - 1.0)).astype(np.float32)

    # Slaney formula inverse
    f_0 = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_0) / f_sp
    logstep = np.log(6.4) / 27.0

    result = np.where(
        mels < min_log_mel,
        f_0 + f_sp * mels,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
    )
    scalar = mels.ndim == 0
    return float(result) if scalar else result.astype(np.float32)


def hz_to_octs(frequencies, tuning=0.0, bins_per_octave=12, **kwargs):
    """Convert Hz to octave number.

    Parameters
    ----------
    frequencies : float or np.ndarray
        Frequency/frequencies in Hz.
    tuning : float
        Tuning offset in fractions of a bin. Default: 0.0.
    bins_per_octave : int
        Number of bins per octave. Default: 12.

    Returns
    -------
    float or np.ndarray
        Octave number(s).
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    result = np.log2(frequencies / (A440 / 16.0))
    scalar = frequencies.ndim == 0
    return float(result) if scalar else result.astype(np.float32)


def octs_to_hz(octs, tuning=0.0, bins_per_octave=12, **kwargs):
    """Convert octave number to Hz.

    Parameters
    ----------
    octs : float or np.ndarray
        Octave number(s).
    tuning : float
        Tuning offset in fractions of a bin. Default: 0.0.
    bins_per_octave : int
        Number of bins per octave. Default: 12.

    Returns
    -------
    float or np.ndarray
        Frequency/frequencies in Hz.
    """
    octs = np.asarray(octs, dtype=np.float64)
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    result = (A440 / 16.0) * 2.0 ** octs
    scalar = octs.ndim == 0
    return float(result) if scalar else result.astype(np.float32)


def salience(S, freqs, harmonics, weights=None, aggregate=None,
             filter_peaks=True, fill_value=np.nan, kind='linear',
             axis=-2, **kwargs):
    """Harmonic salience function.

    Parameters
    ----------
    S : np.ndarray
        Spectrogram or STFT magnitude.
    freqs : np.ndarray
        Center frequencies for each bin of S.
    harmonics : array-like
        Harmonic ratios to sum over.
    weights : array-like or None
        Weights for each harmonic. Default: equal weights.
    aggregate : callable or None
        Aggregation function. Default: np.average.
    filter_peaks : bool
        Only consider peaks. Default: True.
    fill_value : float
        Fill value for out-of-range interpolation. Default: NaN.
    kind : str
        Interpolation kind. Default: 'linear'.
    axis : int
        Frequency axis. Default: -2.

    Returns
    -------
    np.ndarray
        Salience matrix with same shape as S.
    """
    from scipy.interpolate import interp1d

    S = np.asarray(S, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    harmonics = np.asarray(harmonics, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(harmonics)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if aggregate is None:
        aggregate = np.average

    n_harmonics = len(harmonics)

    # Build interpolation along frequency axis
    if S.ndim == 1:
        S = S[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n_freqs, n_frames = S.shape
    result = np.full_like(S, fill_value, dtype=np.float64)

    for t in range(n_frames):
        interp_fn = interp1d(freqs, S[:, t], kind=kind,
                             bounds_error=False, fill_value=fill_value)
        sal_components = np.zeros((n_harmonics, n_freqs), dtype=np.float64)
        for h_idx, h_ratio in enumerate(harmonics):
            sal_components[h_idx] = interp_fn(freqs * h_ratio) * weights[h_idx]

        result[:, t] = aggregate(sal_components, axis=0)

    if squeeze:
        result = result.ravel()

    return result.astype(np.float32)


def interp_harmonics(x, freqs, harmonics, kind='linear', fill_value=0,
                     axis=-2, **kwargs):
    """Interpolate a spectrogram at harmonic frequencies.

    Parameters
    ----------
    x : np.ndarray
        Spectrogram.
    freqs : np.ndarray
        Frequency array for each bin.
    harmonics : array-like
        Harmonic ratios.
    kind : str
        Interpolation kind. Default: 'linear'.
    fill_value : float
        Fill value for out-of-range. Default: 0.
    axis : int
        Frequency axis. Default: -2.

    Returns
    -------
    np.ndarray
        Interpolated harmonic spectrograms stacked along a new first axis.
    """
    from scipy.interpolate import interp1d

    x = np.asarray(x, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    harmonics = np.asarray(harmonics, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    n_freqs, n_frames = x.shape
    n_harmonics = len(harmonics)

    result = np.full((n_harmonics, n_freqs, n_frames), fill_value, dtype=np.float64)

    for t in range(n_frames):
        interp_fn = interp1d(freqs, x[:, t], kind=kind,
                             bounds_error=False, fill_value=fill_value)
        for h_idx, h_ratio in enumerate(harmonics):
            result[h_idx, :, t] = interp_fn(freqs * h_ratio)

    return result.astype(np.float32)


def f0_harmonics(x, f0, freqs, harmonics, kind='linear', fill_value=0,
                 axis=-2, **kwargs):
    """Extract harmonics from a spectrogram relative to an F0 estimate.

    Parameters
    ----------
    x : np.ndarray
        Spectrogram.
    f0 : np.ndarray
        Fundamental frequency per frame.
    freqs : np.ndarray
        Frequency array for each bin.
    harmonics : array-like
        Harmonic indices (e.g. [1, 2, 3, ...]).
    kind : str
        Interpolation kind. Default: 'linear'.
    fill_value : float
        Fill value for out-of-range. Default: 0.
    axis : int
        Frequency axis. Default: -2.

    Returns
    -------
    np.ndarray
        Harmonic energies, shape (n_harmonics, n_frames).
    """
    from scipy.interpolate import interp1d

    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64)
    harmonics = np.asarray(harmonics, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    n_freqs, n_frames = x.shape
    n_harmonics = len(harmonics)

    result = np.full((n_harmonics, n_frames), fill_value, dtype=np.float64)

    for t in range(n_frames):
        if np.isnan(f0[t]) or f0[t] <= 0:
            continue
        interp_fn = interp1d(freqs, x[:, t], kind=kind,
                             bounds_error=False, fill_value=fill_value)
        for h_idx, h in enumerate(harmonics):
            result[h_idx, t] = interp_fn(f0[t] * h)

    return result.astype(np.float32)


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12, **kwargs):
    """Estimate tuning from a set of frequencies.

    Parameters
    ----------
    frequencies : array-like
        Pitch frequencies in Hz.
    resolution : float
        Histogram resolution in fractions of a bin. Default: 0.01.
    bins_per_octave : int
        Number of bins per octave. Default: 12.

    Returns
    -------
    float
        Estimated tuning offset in fractions of a bin.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
    frequencies = frequencies[frequencies > 0]
    frequencies = frequencies[~np.isnan(frequencies)]

    if len(frequencies) == 0:
        return 0.0

    # Compute residual in bins
    residual = bins_per_octave * np.log2(frequencies / 440.0)
    residual = residual - np.round(residual)

    # Histogram
    bins = np.arange(-0.5, 0.5 + resolution, resolution)
    counts, edges = np.histogram(residual, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return float(centers[np.argmax(counts)])
