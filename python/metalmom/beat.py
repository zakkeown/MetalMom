"""Beat tracking functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def plp(y=None, sr=22050, onset_envelope=None, hop_length=512,
        n_fft=2048, n_mels=128, fmin=0.0, fmax=None,
        center=True, win_length=384,
        tempo_min=30.0, tempo_max=300.0, **kwargs):
    """Compute Predominant Local Pulse (Grosche & Mueller 2011).

    Estimates a local pulse (periodicity) curve from the tempogram.
    Peaks of this curve correspond to beat positions.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal.
    sr : int
        Sample rate. Default: 22050.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope (accepted for API
        compatibility but ignored -- the native code computes its
        own envelope from ``y``).
    hop_length : int
        Hop length. Default: 512.
    n_fft : int
        FFT window size for onset envelope. Default: 2048.
    n_mels : int
        Number of mel bands. Default: 128.
    fmin : float
        Minimum frequency for mel filterbank. Default: 0.0.
    fmax : float or None
        Maximum frequency. Default: None (sr/2).
    center : bool
        Center-pad onset windows. Default: True.
    win_length : int
        Window length for local tempogram analysis. Default: 384.
    tempo_min : float
        Minimum tempo in BPM. Default: 30.0.
    tempo_max : float
        Maximum tempo in BPM. Default: 300.0.

    Returns
    -------
    np.ndarray
        Pulse curve, shape ``(n_frames,)``.
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

        status = lib.mm_plp(
            ctx, signal_ptr, len(y),
            sr, hop_length, n_fft,
            n_mels, fmin, c_fmax,
            1 if center else 0, win_length,
            tempo_min, tempo_max,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_plp failed with status {status}")

        result = buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)

    return result.ravel()


def beat_track(y=None, sr=22050, onset_envelope=None, hop_length=512,
               n_fft=2048, n_mels=128, fmin=0.0, fmax=None,
               start_bpm=120.0, trim=True, units='frames', **kwargs):
    """Track beats using the Ellis 2007 dynamic programming algorithm.

    Parameters
    ----------
    y : np.ndarray or None
        Audio signal.
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
    trim : bool
        Trim first and last beats. Default: True.
    units : str
        ``'frames'`` (default) returns frame indices;
        ``'time'`` returns beat times in seconds;
        ``'samples'`` returns sample indices.

    Returns
    -------
    tempo : float
        Estimated tempo in BPM.
    beats : np.ndarray
        Beat locations as frame indices, times, or sample indices.
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
        out_beats = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_beat_track(
            ctx, signal_ptr, len(y),
            sr, hop_length, n_fft,
            n_mels, fmin, c_fmax,
            start_bpm, 1 if trim else 0,
            out_tempo, out_beats,
        )
        if status != 0:
            raise RuntimeError(f"mm_beat_track failed with status {status}")

        tempo = float(out_tempo[0])
        result = buffer_to_numpy(out_beats)
        frames = result.ravel().astype(int)
    finally:
        lib.mm_destroy(ctx)

    if units == 'frames':
        return tempo, frames
    elif units == 'time':
        return tempo, frames.astype(np.float64) * hop_length / sr
    elif units == 'samples':
        return tempo, frames * hop_length
    else:
        raise ValueError(f"Unknown units: {units!r}. Must be 'frames', 'time', or 'samples'.")


def neural_beat_track(activations, fps=100.0, min_bpm=55.0, max_bpm=215.0,
                      transition_lambda=100.0, threshold=0.05, trim=True,
                      units='frames', hop_length=441, sr=44100, **kwargs):
    """Decode beat positions from pre-computed neural activation probabilities.

    Uses dynamic programming to find the optimal beat sequence given
    activation probabilities (e.g., from an RNN ensemble as in madmom's
    DBNBeatTrackingProcessor).

    Parameters
    ----------
    activations : np.ndarray
        Beat activation probabilities, shape ``(n_frames,)``, values in [0, 1].
    fps : float
        Frames per second of the activation signal. Default: 100.0.
    min_bpm : float
        Minimum tempo in BPM. Default: 55.0.
    max_bpm : float
        Maximum tempo in BPM. Default: 215.0.
    transition_lambda : float
        Penalty for tempo deviations. Higher = smoother. Default: 100.0.
    threshold : float
        Minimum activation to consider as a beat. Default: 0.05.
    trim : bool
        Trim first and last beats. Default: True.
    units : str
        ``'frames'`` (default) returns frame indices;
        ``'time'`` returns beat times in seconds;
        ``'samples'`` returns sample indices.
    hop_length : int
        Hop length (used for time/sample conversion). Default: 441.
    sr : int
        Sample rate (used for time/sample conversion). Default: 44100.

    Returns
    -------
    tempo : float
        Estimated tempo in BPM.
    beats : np.ndarray
        Beat locations as frame indices, times, or sample indices.
    """
    activations = np.ascontiguousarray(activations, dtype=np.float32)
    n_frames = len(activations)
    if n_frames == 0:
        return 0.0, np.array([], dtype=int)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_tempo = ffi.new("float*")
        out_beats = ffi.new("MMBuffer*")
        act_ptr = ffi.cast("const float*", activations.ctypes.data)

        status = lib.mm_neural_beat_decode(
            ctx, act_ptr, n_frames,
            float(fps), float(min_bpm), float(max_bpm),
            float(transition_lambda), float(threshold),
            1 if trim else 0,
            out_tempo, out_beats,
        )
        if status != 0:
            raise RuntimeError(f"mm_neural_beat_decode failed with status {status}")

        tempo = float(out_tempo[0])
        result = buffer_to_numpy(out_beats)
        frames = result.ravel().astype(int)
    finally:
        lib.mm_destroy(ctx)

    if units == 'frames':
        return tempo, frames
    elif units == 'time':
        return tempo, frames.astype(np.float64) / fps
    elif units == 'samples':
        return tempo, frames * hop_length
    else:
        raise ValueError(f"Unknown units: {units!r}. Must be 'frames', 'time', or 'samples'.")
