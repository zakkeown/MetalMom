"""Evaluation metrics for MIR tasks."""

import numpy as np
from ._native import ffi, lib


def onset_evaluate(reference, estimated, window=0.05):
    """Evaluate onset detection performance.

    Parameters
    ----------
    reference : array-like
        Reference onset times in seconds.
    estimated : array-like
        Estimated onset times in seconds.
    window : float
        Tolerance window in seconds. Default: 0.05 (50ms).

    Returns
    -------
    dict
        Dictionary with keys 'precision', 'recall', 'f_measure'.
    """
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    estimated = np.ascontiguousarray(estimated, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_p = ffi.new("float*")
        out_r = ffi.new("float*")
        out_f = ffi.new("float*")

        ref_ptr = ffi.cast("const float*", reference.ctypes.data) if len(reference) > 0 else ffi.NULL
        est_ptr = ffi.cast("const float*", estimated.ctypes.data) if len(estimated) > 0 else ffi.NULL

        status = lib.mm_onset_evaluate(
            ctx, ref_ptr, len(reference),
            est_ptr, len(estimated),
            float(window),
            out_p, out_r, out_f,
        )
        if status != 0:
            raise RuntimeError(f"mm_onset_evaluate failed with status {status}")

        return {
            'precision': float(out_p[0]),
            'recall': float(out_r[0]),
            'f_measure': float(out_f[0]),
        }
    finally:
        lib.mm_destroy(ctx)


def beat_evaluate(reference, estimated, fmeasure_window=0.07):
    """Evaluate beat tracking performance.

    Parameters
    ----------
    reference : array-like
        Reference beat times in seconds.
    estimated : array-like
        Estimated beat times in seconds.
    fmeasure_window : float
        Tolerance window for F-measure in seconds. Default: 0.07 (70ms).

    Returns
    -------
    dict
        Dictionary with keys 'f_measure', 'cemgil', 'p_score',
        'cml_c', 'cml_t', 'aml_c', 'aml_t'.
    """
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    estimated = np.ascontiguousarray(estimated, dtype=np.float32)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_f = ffi.new("float*")
        out_cemgil = ffi.new("float*")
        out_pscore = ffi.new("float*")
        out_cmlc = ffi.new("float*")
        out_cmlt = ffi.new("float*")
        out_amlc = ffi.new("float*")
        out_amlt = ffi.new("float*")

        ref_ptr = ffi.cast("const float*", reference.ctypes.data) if len(reference) > 0 else ffi.NULL
        est_ptr = ffi.cast("const float*", estimated.ctypes.data) if len(estimated) > 0 else ffi.NULL

        status = lib.mm_beat_evaluate(
            ctx, ref_ptr, len(reference),
            est_ptr, len(estimated),
            float(fmeasure_window),
            out_f, out_cemgil, out_pscore,
            out_cmlc, out_cmlt, out_amlc, out_amlt,
        )
        if status != 0:
            raise RuntimeError(f"mm_beat_evaluate failed with status {status}")

        return {
            'f_measure': float(out_f[0]),
            'cemgil': float(out_cemgil[0]),
            'p_score': float(out_pscore[0]),
            'cml_c': float(out_cmlc[0]),
            'cml_t': float(out_cmlt[0]),
            'aml_c': float(out_amlc[0]),
            'aml_t': float(out_amlt[0]),
        }
    finally:
        lib.mm_destroy(ctx)


def tempo_evaluate(ref_tempo, est_tempo, tolerance=0.08):
    """Evaluate tempo estimation.

    Parameters
    ----------
    ref_tempo : float
        Reference tempo in BPM.
    est_tempo : float
        Estimated tempo in BPM.
    tolerance : float
        Relative tolerance. Default: 0.08 (8%).

    Returns
    -------
    dict
        Dictionary with key 'p_score' (1.0 if match, 0.0 if not).
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_pscore = ffi.new("float*")

        status = lib.mm_tempo_evaluate(
            ctx, float(ref_tempo), float(est_tempo),
            float(tolerance), out_pscore,
        )
        if status != 0:
            raise RuntimeError(f"mm_tempo_evaluate failed with status {status}")

        return {
            'p_score': float(out_pscore[0]),
        }
    finally:
        lib.mm_destroy(ctx)


def chord_accuracy(reference, estimated):
    """Evaluate chord recognition accuracy.

    Parameters
    ----------
    reference : array-like
        Reference chord labels (integers, one per frame).
    estimated : array-like
        Estimated chord labels (integers, one per frame).

    Returns
    -------
    float
        Accuracy (fraction of frames with matching labels).
    """
    reference = np.ascontiguousarray(reference, dtype=np.int32)
    estimated = np.ascontiguousarray(estimated, dtype=np.int32)

    n = min(len(reference), len(estimated))
    if n == 0:
        return 0.0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_acc = ffi.new("float*")

        ref_ptr = ffi.cast("const int32_t*", reference.ctypes.data)
        est_ptr = ffi.cast("const int32_t*", estimated.ctypes.data)

        status = lib.mm_chord_accuracy(
            ctx, ref_ptr, est_ptr, n, out_acc,
        )
        if status != 0:
            raise RuntimeError(f"mm_chord_accuracy failed with status {status}")

        return float(out_acc[0])
    finally:
        lib.mm_destroy(ctx)
