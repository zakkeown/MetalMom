"""Decomposition: NMF, Nearest-Neighbor Filter."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy, numpy_to_float_ptr


def nmf(V, n_components=8, n_iter=200, objective="euclidean", sr=22050):
    """Non-negative Matrix Factorization: decompose V ~ W * H.

    Parameters
    ----------
    V : np.ndarray
        Non-negative input matrix, shape (n_features, n_samples).
        Typically a magnitude spectrogram.
    n_components : int
        Number of components (rank). Default 8.
    n_iter : int
        Number of multiplicative update iterations. Default 200.
    objective : str
        Distance metric: "euclidean" (Frobenius) or "kl" (KL divergence).
        Default "euclidean".
    sr : int
        Sample rate (metadata only). Default 22050.

    Returns
    -------
    W : np.ndarray
        Basis matrix, shape (n_features, n_components).
    H : np.ndarray
        Activation matrix, shape (n_components, n_samples).
    """
    if V is None:
        raise ValueError("V must be provided")

    V = np.ascontiguousarray(V, dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got {V.ndim}D")

    n_features, n_samples = V.shape
    if n_features == 0 or n_samples == 0:
        raise ValueError("V dimensions must be positive")

    obj_code = 1 if objective == "kl" else 0

    data_ptr = ffi.cast("const float*", V.ctypes.data)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_w = ffi.new("MMBuffer*")
        out_h = ffi.new("MMBuffer*")
        rc = lib.mm_nmf(
            ctx, data_ptr,
            int(n_features), int(n_samples), int(sr),
            int(n_components), int(n_iter), obj_code,
            out_w, out_h,
        )
        if rc != 0:
            raise RuntimeError(f"mm_nmf failed with code {rc}")

        W = buffer_to_numpy(out_w)
        H = buffer_to_numpy(out_h)
        return W, H
    finally:
        lib.mm_destroy(ctx)


def nn_filter(S, k=10, metric="cosine", aggregate="mean", exclude_self=True, sr=22050):
    """Nearest-neighbor filter for spectrograms.

    Replaces each frame with the aggregation of its k nearest neighbors,
    preserving repeating structure and smoothing transient events.
    Matches ``librosa.decompose.nn_filter``.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram, shape (n_features, n_frames).
    k : int
        Number of nearest neighbors. Default 10.
    metric : str
        Distance metric: "cosine" or "euclidean". Default "cosine".
    aggregate : str
        Aggregation method: "mean" or "median". Default "mean".
    exclude_self : bool
        Exclude the frame itself from its neighbor set. Default True.
    sr : int
        Sample rate (metadata only). Default 22050.

    Returns
    -------
    np.ndarray
        Filtered spectrogram, same shape as input.
    """
    if S is None:
        raise ValueError("S must be provided")

    S = np.ascontiguousarray(S, dtype=np.float32)
    if S.ndim != 2:
        raise ValueError(f"S must be 2D, got {S.ndim}D")

    n_features, n_frames = S.shape
    if n_features == 0 or n_frames == 0:
        raise ValueError("S dimensions must be positive")

    metric_code = 1 if metric == "euclidean" else 0  # 0=cosine, 1=euclidean
    agg_code = 1 if aggregate == "median" else 0     # 0=mean, 1=median
    exclude_code = 1 if exclude_self else 0

    data_ptr = ffi.cast("const float*", S.ctypes.data)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_nn_filter(
            ctx, data_ptr,
            int(n_features), int(n_frames), int(sr),
            int(k), metric_code, agg_code, exclude_code,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_nn_filter failed with code {rc}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
