"""Key detection from CNN activation probabilities."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


# Key labels matching madmom's CNNKeyRecognitionProcessor ordering
KEY_LABELS = [
    "A major", "A# major", "B major", "C major", "C# major", "D major",
    "D# major", "E major", "F major", "F# major", "G major", "G# major",
    "A minor", "A# minor", "B minor", "C minor", "C# minor", "D minor",
    "D# minor", "E minor", "F minor", "F# minor", "G minor", "G# minor",
]


def key_detect(activations, sr=22050):
    """Detect musical key from CNN activation probabilities.

    Takes 24-class activation probabilities (12 major + 12 minor keys)
    and determines the musical key by picking the class with highest
    probability. For multi-frame inputs, activations are averaged across
    frames before picking.

    Parameters
    ----------
    activations : np.ndarray
        Key activation probabilities.
        - Shape ``(24,)`` for single-frame detection.
        - Shape ``(n_frames, 24)`` for multi-frame (sequence) detection.
    sr : int
        Sample rate (accepted for API compatibility). Default: 22050.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'key_index'`` : int -- Detected key index (0-23).
        - ``'key_label'`` : str -- Key label (e.g. ``'C major'``).
        - ``'is_major'`` : bool -- Whether the detected key is major.
        - ``'confidence'`` : float -- Probability of the detected key.
        - ``'probabilities'`` : np.ndarray -- All 24 key probabilities
          (averaged across frames if multi-frame input).
    """
    activations = np.ascontiguousarray(activations, dtype=np.float32)

    if activations.ndim == 1:
        assert activations.shape[0] == 24, (
            f"1D activations must have length 24, got {activations.shape[0]}"
        )
        n_frames = 1
    elif activations.ndim == 2:
        assert activations.shape[1] == 24, (
            f"activations must have 24 columns, got {activations.shape[1]}"
        )
        n_frames = activations.shape[0]
        activations = activations.ravel()  # flatten to row-major [nFrames*24]
    else:
        raise ValueError(
            f"activations must be 1D or 2D, got ndim={activations.ndim}"
        )

    if n_frames == 0:
        return {
            'key_index': 0,
            'key_label': KEY_LABELS[0],
            'is_major': True,
            'confidence': 0.0,
            'probabilities': np.zeros(24, dtype=np.float32),
        }

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out_key_index = ffi.new("int32_t*")
        out_confidence = ffi.new("float*")
        out_probs = ffi.new("MMBuffer*")
        act_ptr = ffi.cast("const float*", activations.ctypes.data)

        status = lib.mm_key_detect(
            ctx, act_ptr, n_frames,
            out_key_index, out_confidence, out_probs,
        )
        if status != 0:
            raise RuntimeError(f"mm_key_detect failed with status {status}")

        key_index = int(out_key_index[0])
        confidence = float(out_confidence[0])
        probabilities = buffer_to_numpy(out_probs).ravel()
    finally:
        lib.mm_destroy(ctx)

    return {
        'key_index': key_index,
        'key_label': KEY_LABELS[key_index],
        'is_major': key_index < 12,
        'confidence': confidence,
        'probabilities': probabilities,
    }
