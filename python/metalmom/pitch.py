"""Pitch estimation functions."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


__all__ = ["yin", "pyin"]


def yin(y, fmin, fmax, sr=22050, frame_length=2048, hop_length=None,
        trough_threshold=0.1, center=True, **kwargs):
    """Estimate fundamental frequency using YIN.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    sr : int
        Sample rate. Default: 22050.
    frame_length : int
        Analysis frame length. Default: 2048.
    hop_length : int or None
        Hop length. Default: frame_length // 4.
    trough_threshold : float
        Threshold for CMNDF. Default: 0.1.
    center : bool
        Center-pad signal. Default: True.

    Returns
    -------
    np.ndarray
        F0 estimates in Hz, shape (n_frames,). Unvoiced frames = 0.
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_hop = int(hop_length) if hop_length is not None else 0

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_yin(
            ctx, signal_ptr, len(y),
            sr, float(fmin), float(fmax),
            frame_length, c_hop,
            trough_threshold,
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_yin failed with status {status}")

        result = buffer_to_numpy(out)
        return result.ravel()
    finally:
        lib.mm_destroy(ctx)


def pyin(y, fmin, fmax, sr=22050, frame_length=2048, hop_length=None,
         n_thresholds=100, beta_parameters=(2, 18), resolution=0.1,
         switch_prob=0.01, center=True, fill_na=np.nan, **kwargs):
    """Probabilistic YIN pitch estimation with Viterbi decoding.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    sr : int
        Sample rate. Default: 22050.
    frame_length : int
        Analysis frame length. Default: 2048.
    hop_length : int or None
        Hop length. Default: frame_length // 4.
    n_thresholds : int
        Number of CMNDF thresholds. Default: 100.
    beta_parameters : tuple of (float, float)
        Alpha and beta for the beta distribution. Default: (2, 18).
    resolution : float
        Pitch resolution in semitones. Default: 0.1.
    switch_prob : float
        Voiced/unvoiced transition probability. Default: 0.01.
    center : bool
        Center-pad signal. Default: True.
    fill_na : float or None
        Value to use for unvoiced frames in f0. Default: np.nan.
        If None, unvoiced frames are left as 0.

    Returns
    -------
    f0 : np.ndarray, shape (n_frames,)
        Fundamental frequency in Hz. Unvoiced frames filled with fill_na.
    voiced_flag : np.ndarray, shape (n_frames,)
        Boolean array; True for voiced frames.
    voiced_probs : np.ndarray, shape (n_frames,)
        Voiced probability per frame in [0, 1].
    """
    if y is None:
        raise ValueError("y must be provided")

    y = np.ascontiguousarray(y, dtype=np.float32)
    c_hop = int(hop_length) if hop_length is not None else 0
    beta_alpha, beta_beta = beta_parameters

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        signal_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_pyin(
            ctx, signal_ptr, len(y),
            sr, float(fmin), float(fmax),
            frame_length, c_hop,
            n_thresholds,
            float(beta_alpha), float(beta_beta),
            float(resolution), float(switch_prob),
            1 if center else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_pyin failed with status {status}")

        result = buffer_to_numpy(out)
        # result has shape [3, n_frames]
        f0 = result[0].copy()
        voiced_flag = result[1].astype(bool)
        voiced_probs = result[2].copy()

        # Apply fill_na for unvoiced frames
        if fill_na is not None:
            f0[~voiced_flag] = fill_na

        return f0, voiced_flag, voiced_probs
    finally:
        lib.mm_destroy(ctx)
