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


def onset_detect(y=None, sr=22050, onset_envelope=None, hop_length=512,
                 n_fft=2048, n_mels=128, fmin=0.0, fmax=None,
                 center=True, backtrack=False,
                 pre_max=3, post_max=3, pre_avg=3, post_avg=3,
                 delta=0.07, wait=30, units='frames', **kwargs):
    """Detect onset events in an audio signal.

    Computes the onset strength envelope, picks peaks using local max / mean
    threshold / wait constraints, and optionally backtracks each onset to the
    nearest preceding local energy minimum.  Matches librosa's ``onset_detect()``
    behavior.

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
    center : bool
        Center-pad signal. Default: True.
    backtrack : bool
        Snap onsets to preceding local energy minimum. Default: False.
    pre_max : int
        Samples before n for local max check. Default: 3.
    post_max : int
        Samples after n for local max check. Default: 3.
    pre_avg : int
        Samples before n for mean threshold. Default: 3.
    post_avg : int
        Samples after n for mean threshold. Default: 3.
    delta : float
        Threshold offset above mean. Default: 0.07.
    wait : int
        Minimum samples between peaks. Default: 30.
    units : str
        ``'frames'`` (default) returns frame indices;
        ``'time'`` returns onset times in seconds;
        ``'samples'`` returns sample indices.

    Returns
    -------
    np.ndarray
        Onset locations as frame indices, times, or sample indices.
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

        status = lib.mm_onset_detect(
            ctx, signal_ptr, len(y),
            sr, n_fft, hop_length,
            n_mels, fmin, c_fmax,
            1 if center else 0,
            pre_max, post_max,
            pre_avg, post_avg,
            delta, wait,
            1 if backtrack else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_onset_detect failed with status {status}")

        result = buffer_to_numpy(out)
        frames = result.ravel().astype(int)
    finally:
        lib.mm_destroy(ctx)

    if units == 'frames':
        return frames
    elif units == 'time':
        return frames.astype(np.float64) * hop_length / sr
    elif units == 'samples':
        return frames * hop_length
    else:
        raise ValueError(f"Unknown units: {units!r}. Must be 'frames', 'time', or 'samples'.")


def neural_onset_detect(activations, fps=100.0, threshold=0.3,
                        pre_max=1, post_max=1, pre_avg=3, post_avg=3,
                        combine='adaptive', wait=1, units='frames',
                        hop_length=None, sr=None, **kwargs):
    """Detect onsets from pre-computed neural network activation probabilities.

    Takes onset activation probabilities (e.g. from a CNN/RNN model) and
    picks peaks using local-max / threshold / wait constraints.

    Parameters
    ----------
    activations : np.ndarray
        Onset activation probabilities, shape (n_frames,), values in [0, 1].
    fps : float
        Frames per second of the activation function. Default: 100.0.
    threshold : float
        Threshold for peak detection. Default: 0.3.
    pre_max : int
        Frames before n for local max check. Default: 1.
    post_max : int
        Frames after n for local max check. Default: 1.
    pre_avg : int
        Frames before n for moving average threshold. Default: 3.
    post_avg : int
        Frames after n for moving average threshold. Default: 3.
    combine : str
        Threshold combination method: ``'fixed'``, ``'adaptive'`` (default),
        or ``'combined'``.
    wait : int
        Minimum frames between detected onsets. Default: 1.
    units : str
        ``'frames'`` (default), ``'time'`` (seconds), or ``'samples'``.
    hop_length : int or None
        Required when ``units='samples'``.
    sr : int or None
        Sample rate. Required when ``units='time'``.

    Returns
    -------
    np.ndarray
        Onset locations as frame indices, times, or sample indices.
    """
    activations = np.ascontiguousarray(activations, dtype=np.float32)

    combine_map = {'fixed': 0, 'adaptive': 1, 'combined': 2}
    c_combine = combine_map.get(combine, 1)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        act_ptr = ffi.cast("const float*", activations.ctypes.data)

        status = lib.mm_neural_onset_detect(
            ctx, act_ptr, len(activations),
            float(fps), float(threshold),
            pre_max, post_max,
            pre_avg, post_avg,
            c_combine, wait,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_neural_onset_detect failed with status {status}")

        result = buffer_to_numpy(out)
        frames = result.ravel().astype(int)
    finally:
        lib.mm_destroy(ctx)

    if units == 'frames':
        return frames
    elif units == 'time':
        return frames.astype(np.float64) / fps
    elif units == 'samples':
        if hop_length is None:
            raise ValueError("hop_length is required when units='samples'")
        return frames * hop_length
    else:
        raise ValueError(f"Unknown units: {units!r}. Must be 'frames', 'time', or 'samples'.")
