"""Sequence decoding: Viterbi algorithm for HMM and CRF models."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def viterbi(prob, transition=None, initial=None):
    """Standard Viterbi decoding via Hidden Markov Model.

    Finds the most likely state sequence given observation probabilities,
    a transition matrix, and an initial state distribution.

    Parameters
    ----------
    prob : np.ndarray
        Observation probabilities, shape (n_frames, n_states).
        Values should be in linear (not log) domain and positive.
    transition : np.ndarray or None
        Transition probability matrix, shape (n_states, n_states).
        ``transition[i, j]`` is the probability of transitioning from
        state ``i`` to state ``j``. If None, a uniform transition matrix
        is used. Values should be in linear (not log) domain.
    initial : np.ndarray or None
        Initial state distribution, shape (n_states,).
        If None, a uniform distribution is used.

    Returns
    -------
    np.ndarray
        Optimal state sequence, shape (n_frames,), dtype int64.
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob must be 2D, got {prob.ndim}D")

    n_frames, n_states = prob.shape
    if n_frames == 0 or n_states == 0:
        return np.array([], dtype=np.int64)

    # Convert to log domain, clamping zeros
    log_prob = np.log(np.maximum(prob, 1e-10)).astype(np.float32)

    if transition is None:
        transition = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float32)
    else:
        transition = np.asarray(transition, dtype=np.float32)
    if transition.shape != (n_states, n_states):
        raise ValueError(
            f"transition must be ({n_states}, {n_states}), got {transition.shape}"
        )
    log_transition = np.log(np.maximum(transition, 1e-10)).astype(np.float32)

    if initial is None:
        initial = np.full(n_states, 1.0 / n_states, dtype=np.float32)
    else:
        initial = np.asarray(initial, dtype=np.float32)
    if initial.shape != (n_states,):
        raise ValueError(
            f"initial must be ({n_states},), got {initial.shape}"
        )
    log_initial = np.log(np.maximum(initial, 1e-10)).astype(np.float32)

    # Ensure contiguous
    log_prob = np.ascontiguousarray(log_prob)
    log_transition = np.ascontiguousarray(log_transition)
    log_initial = np.ascontiguousarray(log_initial)

    obs_ptr = ffi.cast("const float*", log_prob.ctypes.data)
    init_ptr = ffi.cast("const float*", log_initial.ctypes.data)
    trans_ptr = ffi.cast("const float*", log_transition.ctypes.data)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_viterbi(
            ctx, obs_ptr, int(n_frames), int(n_states),
            init_ptr, trans_ptr, out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_viterbi failed with code {rc}")

        path = buffer_to_numpy(out)
        return path.astype(np.int64).ravel()
    finally:
        lib.mm_destroy(ctx)


def viterbi_discriminative(prob, transition=None, initial=None):
    """Discriminative Viterbi decoding.

    Like :func:`viterbi`, but treats ``prob`` as discriminative posterior
    probabilities that are already normalized per frame. Uses the CRF
    (Conditional Random Field) Viterbi backend with log-domain scores.

    Parameters
    ----------
    prob : np.ndarray
        Per-frame posterior probabilities, shape (n_frames, n_states).
        Each row should sum to 1. Values in linear domain.
    transition : np.ndarray or None
        Transition score matrix, shape (n_states, n_states).
        If None, a uniform matrix (all zeros in log domain) is used.
        Values should be in linear domain (converted to log internally).
    initial : np.ndarray or None
        Ignored for discriminative decoding (included for API consistency).

    Returns
    -------
    np.ndarray
        Optimal state sequence, shape (n_frames,), dtype int64.
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob must be 2D, got {prob.ndim}D")

    n_frames, n_states = prob.shape
    if n_frames == 0 or n_states == 0:
        return np.array([], dtype=np.int64)

    # Convert to log domain for unary scores
    log_prob = np.log(np.maximum(prob, 1e-10)).astype(np.float32)

    if transition is None:
        # Uniform: all transitions equally likely (score = 0 in log domain)
        pairwise = np.zeros((n_states, n_states), dtype=np.float32)
    else:
        transition = np.asarray(transition, dtype=np.float32)
        if transition.shape != (n_states, n_states):
            raise ValueError(
                f"transition must be ({n_states}, {n_states}), got {transition.shape}"
            )
        pairwise = np.log(np.maximum(transition, 1e-10)).astype(np.float32)

    # Ensure contiguous
    log_prob = np.ascontiguousarray(log_prob)
    pairwise = np.ascontiguousarray(pairwise)

    unary_ptr = ffi.cast("const float*", log_prob.ctypes.data)
    pair_ptr = ffi.cast("const float*", pairwise.ctypes.data)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_viterbi_discriminative(
            ctx, unary_ptr, int(n_frames), int(n_states),
            pair_ptr, out,
        )
        if rc != 0:
            raise RuntimeError(f"mm_viterbi_discriminative failed with code {rc}")

        path = buffer_to_numpy(out)
        return path.astype(np.int64).ravel()
    finally:
        lib.mm_destroy(ctx)


def viterbi_binary(prob, transition=None):
    """Binary Viterbi decoding for two-state problems.

    Specialized interface for binary (active/inactive) state decoding.

    Parameters
    ----------
    prob : np.ndarray
        Probability of the active state at each frame, shape (n_frames,).
        Values between 0 and 1.
    transition : np.ndarray or float or None
        Either a 2x2 transition matrix, or a scalar self-loop probability.
        If scalar ``p``, the transition matrix becomes::

            [[p, 1-p],
             [1-p, p]]

        If None, defaults to self-loop probability 0.9.

    Returns
    -------
    np.ndarray
        Binary state sequence, shape (n_frames,), dtype int64.
        Values are 0 (inactive) or 1 (active).
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 1:
        raise ValueError(f"prob must be 1D, got {prob.ndim}D")

    n_frames = len(prob)
    if n_frames == 0:
        return np.array([], dtype=np.int64)

    # Expand to [n_frames, 2]: [[1-p, p], ...]
    prob_2d = np.column_stack([1.0 - prob, prob]).astype(np.float32)

    if transition is None:
        self_loop = 0.9
        trans = np.array([[self_loop, 1 - self_loop],
                          [1 - self_loop, self_loop]], dtype=np.float32)
    elif np.ndim(transition) == 0:
        # Scalar self-loop probability
        self_loop = float(transition)
        trans = np.array([[self_loop, 1 - self_loop],
                          [1 - self_loop, self_loop]], dtype=np.float32)
    else:
        trans = np.asarray(transition, dtype=np.float32)
        if trans.shape != (2, 2):
            raise ValueError(f"transition must be (2, 2), got {trans.shape}")

    return viterbi(prob_2d, transition=trans)
