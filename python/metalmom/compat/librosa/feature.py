"""librosa.feature compatibility shim."""

import numpy as np

from metalmom.feature import (
    melspectrogram, mfcc, chroma_stft,
    chroma_cqt, chroma_cens, chroma_vqt,
    spectral_centroid, spectral_bandwidth, spectral_contrast,
    spectral_rolloff, spectral_flatness,
    rms, zero_crossing_rate, tonnetz,
    delta, stack_memory, poly_features,
    tempo, tempogram, fourier_tempogram,
    pcen, mel_to_audio, mfcc_to_mel, mfcc_to_audio,
)


def tempogram_ratio(y=None, sr=22050, onset_envelope=None,
                    tg=None, bpm=None, hop_length=512, win_length=384,
                    start_bpm=120.0, std_bpm=1.0, max_tempo=320.0,
                    freqs=None, factors=None, aggregate=None,
                    prior=None, center=True, window='hann',
                    kind='linear', fill_value=0, norm=np.inf, **kwargs):
    """Compute tempogram ratio features.

    Computes the ratio of energy at harmonically-related tempo periods
    in the tempogram (e.g., 2:1 for half/double tempo relationships).

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal.
    sr : float
        Sample rate. Default: 22050.
    onset_envelope : np.ndarray or None
        Pre-computed onset envelope.
    tg : np.ndarray or None
        Pre-computed tempogram. If None, computed from y.
    bpm : np.ndarray or None
        Tempo axis for the tempogram.
    hop_length : int
        Hop length. Default: 512.
    win_length : int
        Window length. Default: 384.
    start_bpm : float
        Starting BPM estimate. Default: 120.0.
    std_bpm : float
        BPM standard deviation. Default: 1.0.
    max_tempo : float or None
        Maximum tempo. Default: 320.0.
    freqs : np.ndarray or None
        Frequency axis.
    factors : np.ndarray or None
        Tempo ratio factors to compute. Default: [1, 2, 3, 4].
    aggregate : callable or None
        Aggregation function. Default: None.
    prior : object or None
        Tempo prior distribution. Default: None.
    center : bool
        Center the signal. Default: True.
    window : str
        Window type. Default: 'hann'.
    kind : str
        Interpolation kind. Default: 'linear'.
    fill_value : float
        Fill value. Default: 0.
    norm : float or None
        Normalization order. Default: np.inf.

    Returns
    -------
    np.ndarray
        Tempogram ratio features.
    """
    if factors is None:
        factors = np.array([1, 2, 3, 4], dtype=np.float64)
    else:
        factors = np.asarray(factors, dtype=np.float64)

    # Compute tempogram if not provided
    if tg is None:
        if y is None:
            raise ValueError("Either y or tg must be provided")
        tg = tempogram(y=y, sr=sr, hop_length=hop_length,
                       win_length=win_length, center=center)

    tg = np.asarray(tg, dtype=np.float64)
    n_bins, n_frames = tg.shape

    # Build BPM axis if not provided
    if bpm is None:
        bpm = np.zeros(n_bins, dtype=np.float64)
        bpm[0] = np.inf  # lag-0
        bpm[1:] = 60.0 * sr / (hop_length * np.arange(1, n_bins))

    bpm = np.asarray(bpm, dtype=np.float64)

    from scipy.interpolate import interp1d

    n_factors = len(factors)
    ratio = np.zeros((n_factors, n_frames), dtype=np.float64)

    for t in range(n_frames):
        # Interpolation function over BPM axis
        # Need to handle non-monotonic BPM (it's decreasing for autocorrelation tempogram)
        valid = np.isfinite(bpm) & (bpm > 0)
        if valid.sum() < 2:
            continue

        valid_bpm = bpm[valid]
        valid_tg = tg[valid, t]

        # Sort by BPM for interpolation
        sort_idx = np.argsort(valid_bpm)
        sorted_bpm = valid_bpm[sort_idx]
        sorted_tg = valid_tg[sort_idx]

        interp_fn = interp1d(sorted_bpm, sorted_tg, kind=kind,
                             bounds_error=False, fill_value=fill_value)

        for f_idx, factor in enumerate(factors):
            ratio[f_idx, t] = interp_fn(start_bpm * factor)

    # Normalize
    if norm is not None:
        for t in range(n_frames):
            col = ratio[:, t]
            if norm == np.inf:
                n_val = np.max(np.abs(col))
            elif norm == 1:
                n_val = np.sum(np.abs(col))
            elif norm == 2:
                n_val = np.sqrt(np.sum(col ** 2))
            else:
                n_val = np.linalg.norm(col, ord=norm)
            if n_val > 1e-10:
                ratio[:, t] /= n_val

    return ratio.astype(np.float32)
