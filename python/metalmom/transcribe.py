"""Piano transcription from RNN activation probabilities."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


# Note names for MIDI note conversion
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_note):
    """Convert a MIDI note number to a note name (e.g., 60 -> 'C4').

    Parameters
    ----------
    midi_note : int
        MIDI note number.

    Returns
    -------
    str
        Note name string.
    """
    note = _NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note}{octave}"


def piano_transcribe(activations, threshold=0.5, min_duration=3,
                     use_hmm=False, fps=100.0, units='frames'):
    """Transcribe piano notes from activation probabilities.

    Takes per-frame, per-pitch activation probabilities (e.g., from
    madmom's ``RNNPianoNoteProcessor``) and detects note events using
    either threshold-based or HMM-based decoding.

    Parameters
    ----------
    activations : np.ndarray
        Activation probabilities.
        - Shape ``(n_frames, 88)`` for 2D input.
        - Shape ``(n_frames * 88,)`` for flat 1D input.
    threshold : float
        Minimum activation to trigger a note (threshold mode only).
        Default: 0.5.
    min_duration : int
        Minimum note duration in frames. Default: 3.
    use_hmm : bool
        If True, use HMM-based decoding for smoother onset/offset
        detection. Default: False.
    fps : float
        Frames per second (used when ``units='seconds'``). Default: 100.0.
    units : str
        Output units: ``'frames'`` or ``'seconds'``. Default: ``'frames'``.

    Returns
    -------
    list of dict
        List of note events. Each dict contains:
        - ``'midi_note'`` : int -- MIDI note number (21-108).
        - ``'note_name'`` : str -- Note name (e.g., "C4").
        - ``'onset'`` : float -- Onset frame or time.
        - ``'offset'`` : float -- Offset frame or time (exclusive).
        - ``'velocity'`` : float -- Average activation during the note.
    """
    activations = np.ascontiguousarray(activations, dtype=np.float32)

    if activations.ndim == 2:
        n_frames = activations.shape[0]
        assert activations.shape[1] == 88, (
            f"activations must have 88 columns, got {activations.shape[1]}"
        )
        activations = activations.ravel()
    elif activations.ndim == 1:
        assert activations.shape[0] % 88 == 0, (
            "1D activations length must be divisible by 88"
        )
        n_frames = activations.shape[0] // 88
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

        status = lib.mm_piano_transcribe(
            ctx, act_ptr, n_frames,
            threshold, min_duration,
            1 if use_hmm else 0,
            out,
        )
        if status != 0:
            raise RuntimeError(f"mm_piano_transcribe failed with status {status}")

        result = buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)

    # result is [nEvents, 4]: (midiNote, onsetFrame, offsetFrame, velocity)
    events = []
    if result.size > 0:
        result = result.reshape(-1, 4)
        for row in result:
            midi_note = int(row[0])
            onset_frame = int(row[1])
            offset_frame = int(row[2])
            velocity = float(row[3])

            if units == 'seconds':
                onset_val = onset_frame / fps
                offset_val = offset_frame / fps
            else:
                onset_val = float(onset_frame)
                offset_val = float(offset_frame)

            events.append({
                'midi_note': midi_note,
                'note_name': midi_to_note_name(midi_note),
                'onset': onset_val,
                'offset': offset_val,
                'velocity': velocity,
            })

    return events
