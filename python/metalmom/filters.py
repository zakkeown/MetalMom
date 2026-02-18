"""Filter functions: semitone bandpass filterbank."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def semitone_filterbank(y, sr=22050, midi_low=24, midi_high=119, order=4):
    """Apply a semitone bandpass filterbank to an audio signal.

    Filters the input signal through bandpass filters centered at each
    semitone in the specified MIDI range. Each filter has a constant-Q
    bandwidth of one semitone.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal (1D, float32).
    sr : int
        Sample rate in Hz. Default: 22050.
    midi_low : int
        Lowest MIDI note number. Default: 24 (C1, ~32.7 Hz).
    midi_high : int
        Highest MIDI note number. Default: 119 (B8, ~7902 Hz).
    order : int
        Filter order (number of cascaded biquad pairs). Default: 4.

    Returns
    -------
    np.ndarray
        Filtered signal, shape ``(n_semitones, n_samples)``.
    """
    y = np.ascontiguousarray(y, dtype=np.float32).ravel()

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        data_ptr = ffi.cast("const float*", y.ctypes.data)

        status = lib.mm_semitone_filterbank(
            ctx,
            data_ptr,
            len(y),
            int(sr),
            int(midi_low),
            int(midi_high),
            int(order),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_semitone_filterbank failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def semitone_frequencies(midi_low=24, midi_high=119):
    """Compute center frequencies for semitone bands.

    Parameters
    ----------
    midi_low : int
        Lowest MIDI note. Default: 24 (C1).
    midi_high : int
        Highest MIDI note. Default: 119 (B8).

    Returns
    -------
    np.ndarray
        Array of center frequencies in Hz.
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")
        status = lib.mm_semitone_frequencies(
            ctx,
            int(midi_low),
            int(midi_high),
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_semitone_frequencies failed with status {status}")

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
