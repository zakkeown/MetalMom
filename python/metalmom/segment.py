"""Segmentation: recurrence matrix, cross-similarity, RQA, and DTW."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def recurrence_matrix(features, mode="knn", k=5, threshold=None,
                      metric="euclidean", symmetric=False):
    """Compute a self-recurrence matrix from a feature sequence.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix, shape (n_features, n_frames).
        Each column is a feature vector for one frame.
    mode : str
        Recurrence mode: "knn" (k-nearest neighbors), "threshold",
        or "soft" (raw distances). Default "knn".
    k : int
        Number of nearest neighbors (used when mode="knn"). Default 5.
    threshold : float or None
        Distance threshold (used when mode="threshold"). Required if
        mode="threshold".
    metric : str
        Distance metric: "euclidean" or "cosine". Default "euclidean".
    symmetric : bool
        If True, use mutual nearest neighbors. Default False.

    Returns
    -------
    np.ndarray
        Recurrence matrix, shape (n_frames, n_frames).
        Binary (0/1) for knn/threshold modes, distances for soft mode.
    """
    if features is None:
        raise ValueError("features must be provided")

    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")

    n_features, n_frames = features.shape
    if n_features == 0 or n_frames == 0:
        raise ValueError("features dimensions must be positive")

    # Encode mode
    if mode == "knn":
        mode_code = 0
        mode_param = float(k)
    elif mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided when mode='threshold'")
        mode_code = 1
        mode_param = float(threshold)
    elif mode == "soft":
        mode_code = 2
        mode_param = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metric_code = 1 if metric == "cosine" else 0
    sym_code = 1 if symmetric else 0

    data_ptr = ffi.cast("const float*", features.ctypes.data)
    count = int(n_features * n_frames)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_recurrence_matrix(
            ctx, data_ptr, count,
            int(n_features), int(n_frames),
            mode_code, mode_param,
            metric_code, sym_code,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_recurrence_matrix failed with code {rc}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def cross_similarity(features_a, features_b, metric="euclidean"):
    """Compute cross-similarity matrix between two feature sequences.

    Parameters
    ----------
    features_a : np.ndarray
        Feature matrix A, shape (n_features, n_frames_a).
    features_b : np.ndarray
        Feature matrix B, shape (n_features, n_frames_b).
    metric : str
        Distance metric: "euclidean" or "cosine". Default "euclidean".

    Returns
    -------
    np.ndarray
        Cross-similarity matrix, shape (n_frames_a, n_frames_b).
    """
    if features_a is None or features_b is None:
        raise ValueError("features_a and features_b must be provided")

    features_a = np.ascontiguousarray(features_a, dtype=np.float32)
    features_b = np.ascontiguousarray(features_b, dtype=np.float32)

    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("features must be 2D")

    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature dimensions must match: {features_a.shape[0]} vs {features_b.shape[0]}"
        )

    n_features = features_a.shape[0]
    n_frames_a = features_a.shape[1]
    n_frames_b = features_b.shape[1]

    metric_code = 1 if metric == "cosine" else 0

    data_a_ptr = ffi.cast("const float*", features_a.ctypes.data)
    data_b_ptr = ffi.cast("const float*", features_b.ctypes.data)
    count_a = int(n_features * n_frames_a)
    count_b = int(n_features * n_frames_b)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_cross_similarity(
            ctx, data_a_ptr, count_a,
            data_b_ptr, count_b,
            int(n_features), int(n_frames_a), int(n_frames_b),
            metric_code, out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_cross_similarity failed with code {rc}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def rqa(rec_matrix, lmin=2, vmin=2):
    """Compute Recurrence Quantification Analysis statistics.

    This is a pure-Python implementation that operates on a binary
    recurrence matrix (typically produced by ``recurrence_matrix``).

    Parameters
    ----------
    rec_matrix : np.ndarray
        Binary recurrence matrix, shape (N, N) with values 0 or 1.
    lmin : int
        Minimum diagonal line length. Default 2.
    vmin : int
        Minimum vertical line length. Default 2.

    Returns
    -------
    dict
        Dictionary with keys: recurrence_rate, determinism, laminarity,
        average_diagonal_length, average_vertical_length,
        longest_diagonal_line, entropy.
    """
    if rec_matrix is None:
        raise ValueError("rec_matrix must be provided")

    rec_matrix = np.asarray(rec_matrix, dtype=np.float32)
    if rec_matrix.ndim != 2:
        raise ValueError(f"rec_matrix must be 2D, got {rec_matrix.ndim}D")

    n = rec_matrix.shape[0]
    if rec_matrix.shape[1] != n:
        raise ValueError("rec_matrix must be square")

    if n == 0:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "average_diagonal_length": 0.0,
            "average_vertical_length": 0.0,
            "longest_diagonal_line": 0,
            "entropy": 0.0,
        }

    # Recurrence rate (excluding main diagonal)
    mask = ~np.eye(n, dtype=bool)
    total_points = int(np.sum(rec_matrix[mask] > 0.5))
    total_off_diag = n * n - n
    rr = total_points / total_off_diag if total_off_diag > 0 else 0.0

    # Diagonal line analysis (exclude main diagonal k=0)
    diag_lengths = []
    for k in range(1, n):
        # Upper diagonal
        diag = np.diag(rec_matrix, k)
        _extract_lines(diag, lmin, diag_lengths)
        # Lower diagonal
        diag = np.diag(rec_matrix, -k)
        _extract_lines(diag, lmin, diag_lengths)

    diag_points = sum(diag_lengths)
    det = diag_points / total_points if total_points > 0 else 0.0
    avg_diag = diag_points / len(diag_lengths) if diag_lengths else 0.0
    longest_diag = max(diag_lengths) if diag_lengths else 0

    # Entropy of diagonal line length distribution
    if diag_lengths:
        from collections import Counter
        counts = Counter(diag_lengths)
        total_lines = len(diag_lengths)
        ent = 0.0
        for c in counts.values():
            p = c / total_lines
            if p > 0:
                ent -= p * np.log2(p)
    else:
        ent = 0.0

    # Vertical line analysis
    vert_lengths = []
    for j in range(n):
        col = rec_matrix[:, j].copy()
        col[j] = 0  # exclude diagonal
        _extract_lines(col, vmin, vert_lengths)

    vert_points = sum(vert_lengths)
    lam = vert_points / total_points if total_points > 0 else 0.0
    avg_vert = vert_points / len(vert_lengths) if vert_lengths else 0.0

    return {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "average_diagonal_length": float(avg_diag),
        "average_vertical_length": float(avg_vert),
        "longest_diagonal_line": int(longest_diag),
        "entropy": float(ent),
    }


def dtw(cost_matrix=None, X=None, Y=None, metric="euclidean",
        step_pattern="standard", band_width=None):
    """Compute Dynamic Time Warping.

    Either provide a pre-computed ``cost_matrix`` or two feature matrices
    ``X`` and ``Y`` (from which a Euclidean cost matrix is computed).

    Parameters
    ----------
    cost_matrix : np.ndarray or None
        Pre-computed cost matrix, shape (N, M). If provided, X and Y are
        ignored.
    X : np.ndarray or None
        Feature matrix, shape (n_features, N). Required if cost_matrix is
        None.
    Y : np.ndarray or None
        Feature matrix, shape (n_features, M). Required if cost_matrix is
        None.
    metric : str
        Distance metric for computing cost from X, Y. Currently only
        "euclidean" is supported. Default "euclidean".
    step_pattern : str
        "standard" or "symmetric2". Default "standard".
    band_width : int or None
        Sakoe-Chiba band width. None means no constraint. Default None.

    Returns
    -------
    dict
        Dictionary with keys:
        - "accumulated_cost": np.ndarray, shape (N, M)
        - "warping_path": np.ndarray, shape (L, 2) with (row, col) pairs
        - "total_cost": float
    """
    if cost_matrix is None:
        if X is None or Y is None:
            raise ValueError(
                "Either cost_matrix or both X and Y must be provided"
            )
        # Compute Euclidean cost matrix from feature matrices
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Feature dimensions must match: {X.shape[0]} vs {Y.shape[0]}"
            )
        n_features = X.shape[0]
        n_x = X.shape[1]
        n_y = Y.shape[1]
        # Compute pairwise Euclidean distances
        cost_matrix = np.zeros((n_x, n_y), dtype=np.float32)
        for i in range(n_x):
            for j in range(n_y):
                diff = X[:, i] - Y[:, j]
                cost_matrix[i, j] = np.sqrt(np.dot(diff, diff))

    cost_matrix = np.ascontiguousarray(cost_matrix, dtype=np.float32)
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got {cost_matrix.ndim}D")

    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return {
            "accumulated_cost": np.zeros((0, 0), dtype=np.float32),
            "warping_path": np.zeros((0, 2), dtype=np.int64),
            "total_cost": 0.0,
        }

    step_code = 1 if step_pattern == "symmetric2" else 0
    bw = int(band_width) if band_width is not None else 0

    data_ptr = ffi.cast("const float*", cost_matrix.ctypes.data)
    count = int(n * m)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_dtw(
            ctx, data_ptr, count,
            int(n), int(m),
            step_code, bw,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_dtw failed with code {rc}")

        accumulated_cost = buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)

    # Backtrack in Python from accumulated cost matrix
    total_cost = float(accumulated_cost[n - 1, m - 1])
    path = _dtw_backtrack(accumulated_cost)

    return {
        "accumulated_cost": accumulated_cost,
        "warping_path": np.array(path, dtype=np.int64),
        "total_cost": total_cost,
    }


def _dtw_backtrack(D):
    """Backtrack through accumulated cost matrix to find optimal warping path.

    Parameters
    ----------
    D : np.ndarray
        Accumulated cost matrix, shape (N, M).

    Returns
    -------
    list of (int, int)
        Warping path from (0, 0) to (N-1, M-1).
    """
    n, m = D.shape
    if n == 0 or m == 0:
        return []

    path = []
    i, j = n - 1, m - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag = D[i - 1, j - 1]
            up = D[i - 1, j]
            left = D[i, j - 1]

            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()
    return path


def _extract_lines(arr, min_length, out_list):
    """Extract line lengths from a binary 1D array."""
    current = 0
    for val in arr:
        if val > 0.5:
            current += 1
        else:
            if current >= min_length:
                out_list.append(current)
            current = 0
    if current >= min_length:
        out_list.append(current)
