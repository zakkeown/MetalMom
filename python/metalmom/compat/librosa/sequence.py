"""librosa.sequence compatibility shim."""

import numpy as np

from metalmom.sequence import viterbi, viterbi_discriminative, viterbi_binary
from metalmom.segment import dtw as _mm_dtw, rqa


def dtw(X=None, Y=None, C=None, metric='euclidean',
        step_sizes_sigma=None, weights_add=None, weights_mul=None,
        subseq=False, backtrack=True, global_constraints=False,
        band_rad=0.25, return_steps=False, **kwargs):
    """Dynamic Time Warping.

    Wraps ``metalmom.segment.dtw`` with librosa-compatible interface.

    Parameters
    ----------
    X : np.ndarray or None
        Feature matrix, shape (n_features, N).
    Y : np.ndarray or None
        Feature matrix, shape (n_features, M).
    C : np.ndarray or None
        Pre-computed cost matrix, shape (N, M).
    metric : str
        Distance metric. Default: 'euclidean'.
    backtrack : bool
        If True, return warping path. Default: True.

    Returns
    -------
    D : np.ndarray
        Accumulated cost matrix.
    wp : np.ndarray (if backtrack=True)
        Warping path, shape (L, 2).
    """
    # Map librosa parameter names to our API
    result = _mm_dtw(cost_matrix=C, X=X, Y=Y, metric=metric)

    D = result['accumulated_cost']

    if backtrack:
        wp = result['warping_path']
        return D, wp
    return D


def transition_uniform(n_states):
    """Uniform transition matrix.

    Parameters
    ----------
    n_states : int
        Number of states.

    Returns
    -------
    np.ndarray
        Uniform transition matrix, shape (n_states, n_states).
    """
    return np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)


def transition_loop(n_states, prob):
    """Self-loop transition matrix.

    Parameters
    ----------
    n_states : int
        Number of states.
    prob : float or array-like
        Self-loop probability. If scalar, same for all states.

    Returns
    -------
    np.ndarray
        Transition matrix, shape (n_states, n_states).
    """
    prob = np.atleast_1d(np.asarray(prob, dtype=np.float64))
    if len(prob) == 1:
        prob = np.full(n_states, prob[0])
    elif len(prob) != n_states:
        raise ValueError(f"prob must be scalar or length {n_states}")

    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        trans[i, i] = prob[i]
        # Distribute remaining probability uniformly among other states
        if n_states > 1:
            off_diag = (1.0 - prob[i]) / (n_states - 1)
            for j in range(n_states):
                if j != i:
                    trans[i, j] = off_diag
    return trans


def transition_cycle(n_states, prob):
    """Cyclic transition matrix.

    Parameters
    ----------
    n_states : int
        Number of states.
    prob : float or array-like
        Self-loop probability. Transitions go to the next state (cyclic).

    Returns
    -------
    np.ndarray
        Transition matrix, shape (n_states, n_states).
    """
    prob = np.atleast_1d(np.asarray(prob, dtype=np.float64))
    if len(prob) == 1:
        prob = np.full(n_states, prob[0])
    elif len(prob) != n_states:
        raise ValueError(f"prob must be scalar or length {n_states}")

    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        trans[i, i] = prob[i]
        trans[i, (i + 1) % n_states] = 1.0 - prob[i]
    return trans


def transition_local(n_states, width, window='triangle', wrap=False, **kwargs):
    """Local transition matrix with bandwidth constraint.

    Parameters
    ----------
    n_states : int
        Number of states.
    width : int or array-like
        Window width for local transitions.
    window : str
        Window type. Default: 'triangle'.
    wrap : bool
        Wrap around edges. Default: False.

    Returns
    -------
    np.ndarray
        Transition matrix, shape (n_states, n_states).
    """
    width = np.atleast_1d(np.asarray(width, dtype=int))
    if len(width) == 1:
        width = np.full(n_states, width[0])

    trans = np.zeros((n_states, n_states), dtype=np.float64)

    for i in range(n_states):
        w = int(width[i])
        half = w // 2

        # Generate window weights
        if window == 'triangle':
            weights = np.bartlett(w)
        elif window == 'hann':
            weights = np.hanning(w)
        else:
            weights = np.ones(w)

        for k_idx, k in enumerate(range(-half, -half + w)):
            j = i + k
            if wrap:
                j = j % n_states
            if 0 <= j < n_states:
                trans[i, j] = weights[k_idx]

        # Normalize row
        row_sum = trans[i].sum()
        if row_sum > 0:
            trans[i] /= row_sum

    return trans
