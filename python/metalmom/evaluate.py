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
