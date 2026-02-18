"""librosa.segment compatibility shim."""

import numpy as np

from metalmom.segment import recurrence_matrix, cross_similarity, agglomerative, dtw


def recurrence_to_lag(rec, pad=True, axis=-1, **kwargs):
    """Convert a recurrence matrix to a lag matrix.

    Parameters
    ----------
    rec : np.ndarray
        Recurrence matrix, shape (N, N).
    pad : bool
        Zero-pad to maintain shape. Default: True.
    axis : int
        Axis to lag along. Default: -1.

    Returns
    -------
    np.ndarray
        Lag matrix.
    """
    rec = np.asarray(rec, dtype=np.float32)
    n = rec.shape[0]

    if pad:
        lag = np.zeros((n, n), dtype=np.float32)
    else:
        lag = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            k = j - i
            if pad:
                # Map to lag index (centered)
                lag_idx = k % n if k >= 0 else (k + n) % n
                lag[lag_idx, i] = rec[j, i] if axis == -1 else rec[i, j]
            else:
                if 0 <= k < n:
                    lag[k, i] = rec[j, i]

    return lag


def lag_to_recurrence(lag, axis=-1, **kwargs):
    """Convert a lag matrix to a recurrence matrix.

    Parameters
    ----------
    lag : np.ndarray
        Lag matrix, shape (N, N).
    axis : int
        Axis to convert. Default: -1.

    Returns
    -------
    np.ndarray
        Recurrence matrix.
    """
    lag = np.asarray(lag, dtype=np.float32)
    n = lag.shape[0]

    rec = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for k in range(n):
            j = (i + k) % n
            rec[j, i] = lag[k, i]

    return rec


def path_enhance(R, n, window='hann', max_ratio=2.0, min_ratio=None,
                 n_filters=7, zero_mean=False, clip=True, **kwargs):
    """Enhance path-like structure in a self-similarity matrix.

    Applies diagonal median filtering and optional smoothing.

    Parameters
    ----------
    R : np.ndarray
        Self-similarity or recurrence matrix.
    n : int
        Filter width along the diagonal.
    window : str
        Window type. Default: 'hann'.
    max_ratio : float
        Maximum ratio for bandwidth. Default: 2.0.
    min_ratio : float or None
        Minimum ratio. Default: None.
    n_filters : int
        Number of diagonal filters. Default: 7.
    zero_mean : bool
        Subtract mean from each filter output. Default: False.
    clip : bool
        Clip negative values. Default: True.

    Returns
    -------
    np.ndarray
        Enhanced matrix.
    """
    from scipy.ndimage import median_filter as _median_filter

    R = np.asarray(R, dtype=np.float64)
    result = np.zeros_like(R)

    # Apply diagonal median filters at various angles
    for i in range(n_filters):
        angle = np.pi * i / n_filters
        dx = int(np.round(n * np.cos(angle)))
        dy = int(np.round(n * np.sin(angle)))

        # Use a filter kernel oriented along the diagonal
        kx = max(abs(dx), 1)
        ky = max(abs(dy), 1)
        filtered = _median_filter(R, size=(ky, kx))
        result += filtered

    result /= n_filters

    if zero_mean:
        result -= result.mean()

    if clip:
        result = np.maximum(result, 0)

    return result.astype(np.float32)


def subsegment(data, frames, n_segments=4, axis=-1, **kwargs):
    """Sub-divide segment intervals into smaller pieces.

    Parameters
    ----------
    data : np.ndarray
        Feature data.
    frames : np.ndarray
        Segment boundary frames.
    n_segments : int
        Number of sub-segments per segment. Default: 4.
    axis : int
        Time axis. Default: -1.

    Returns
    -------
    np.ndarray
        Refined boundary frames.
    """
    frames = np.asarray(frames, dtype=np.intp).ravel()
    n_frames = data.shape[axis]

    # Add start and end if not present
    boundaries = list(frames)
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != n_frames:
        boundaries.append(n_frames)

    new_boundaries = [boundaries[0]]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment_len = end - start
        if segment_len <= 1 or n_segments <= 1:
            new_boundaries.append(end)
            continue

        # Uniform sub-division
        sub_bounds = np.linspace(start, end, n_segments + 1, dtype=np.intp)
        for sb in sub_bounds[1:]:
            new_boundaries.append(int(sb))

    return np.unique(np.array(new_boundaries, dtype=np.intp))


def timelag_filter(function, pad=True, index=0):
    """Decorator that converts a recurrence filter to work in lag space.

    Parameters
    ----------
    function : callable
        Filter function.
    pad : bool
        Pad the lag matrix. Default: True.
    index : int
        Argument index of the matrix. Default: 0.

    Returns
    -------
    callable
        Wrapped function that operates in lag space.
    """
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        args = list(args)
        rec = args[index]
        lag = recurrence_to_lag(rec, pad=pad)
        args[index] = lag
        result = function(*args, **kwargs)
        return lag_to_recurrence(result)

    return wrapper
