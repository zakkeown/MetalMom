"""Beat tracking functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


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
