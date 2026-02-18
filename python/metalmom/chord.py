"""Chord recognition from CNN activation probabilities."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


# Chord labels matching madmom's CNNChordFeatureProcessor ordering:
# N (no chord), 12 major triads, 12 minor triads
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_LABELS = ["N"] + [f"{n}:maj" for n in _NOTES] + [f"{n}:min" for n in _NOTES]


def chord_detect(activations, n_classes=25, transition_scores=None,
                 self_transition_bias=1.0, fps=100.0, units='frames'):
    """Detect chords from activation probabilities using CRF decoding.

    Takes per-frame chord activation scores and decodes the most likely
    chord sequence using Viterbi decoding on a CRF with self-transition bias.

    Parameters
    ----------
    activations : np.ndarray
        Chord activation scores.
        - Shape ``(n_frames, n_classes)`` for 2D input.
        - Shape ``(n_frames * n_classes,)`` for flat 1D input.
    n_classes : int
        Number of chord classes. Default: 25
        (N + 12 major + 12 minor).
    transition_scores : np.ndarray or None
        CRF transition scores of shape ``(n_classes, n_classes)``.
        If None, uses self-transition bias on the diagonal.
    self_transition_bias : float
        Bias for staying in the same chord state. Higher values
        encourage fewer chord changes. Default: 1.0.
    fps : float
        Frames per second (used when ``units='seconds'``). Default: 100.0.
    units : str
        Output units: ``'frames'`` or ``'seconds'``. Default: ``'frames'``.

    Returns
    -------
    list of dict
        List of chord events. Each dict contains:
        - ``'start'`` : float -- Start frame or time.
        - ``'end'`` : float -- End frame or time (exclusive).
        - ``'chord_index'`` : int -- Chord class index (0 to n_classes-1).
        - ``'chord_label'`` : str -- Chord label string.
    """
    activations = np.ascontiguousarray(activations, dtype=np.float32)

    if activations.ndim == 2:
        n_frames = activations.shape[0]
        assert activations.shape[1] == n_classes, (
            f"activations must have {n_classes} columns, got {activations.shape[1]}"
        )
        activations = activations.ravel()
    elif activations.ndim == 1:
        assert activations.shape[0] % n_classes == 0, (
            f"1D activations length must be divisible by n_classes={n_classes}"
        )
        n_frames = activations.shape[0] // n_classes
    else:
        raise ValueError(
            f"activations must be 1D or 2D, got ndim={activations.ndim}"
        )

    if n_frames == 0:
        return []

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        act_ptr = ffi.cast("const float*", activations.ctypes.data)

        trans_ptr = ffi.NULL
        if transition_scores is not None:
            transition_scores = np.ascontiguousarray(
                transition_scores, dtype=np.float32
            ).ravel()
            assert transition_scores.shape[0] == n_classes * n_classes, (
                f"transition_scores must have {n_classes * n_classes} elements"
            )
            trans_ptr = ffi.cast("const float*", transition_scores.ctypes.data)

        status = lib.mm_chord_detect(
            ctx, act_ptr, n_frames, n_classes,
            trans_ptr, self_transition_bias, out,
        )
        if status != 0:
            raise RuntimeError(f"mm_chord_detect failed with status {status}")

        result = buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)

    # result is [nEvents, 3]: (startFrame, endFrame, chordIndex)
    events = []
    if result.size > 0:
        result = result.reshape(-1, 3)
        for row in result:
            start_frame = int(row[0])
            end_frame = int(row[1])
            chord_index = int(row[2])
            chord_label = CHORD_LABELS[chord_index] if chord_index < len(CHORD_LABELS) else "N"

            if units == 'seconds':
                start_val = start_frame / fps
                end_val = end_frame / fps
            else:
                start_val = float(start_frame)
                end_val = float(end_frame)

            events.append({
                'start': start_val,
                'end': end_val,
                'chord_index': chord_index,
                'chord_label': chord_label,
            })

    return events
