"""madmom.features.notes compatibility shim.

Provides RNNPianoNoteProcessor backed by MetalMom.
"""

import numpy as np


class RNNPianoNoteProcessor:
    """madmom-compatible RNN piano note processor.

    In real madmom this runs an RNN ensemble to produce per-pitch
    activation probabilities for piano transcription. This shim
    delegates to MetalMom's ``piano_transcribe`` function.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 100.0.
    threshold : float
        Minimum activation to trigger a note. Default: 0.5.
    min_duration : int
        Minimum note duration in frames. Default: 3.
    use_hmm : bool
        If True, use HMM-based decoding. Default: False.
    """

    def __init__(self, fps=100.0, threshold=0.5, min_duration=3,
                 use_hmm=False, **kwargs):
        self.fps = fps
        self.threshold = threshold
        self.min_duration = min_duration
        self.use_hmm = use_hmm

    def __call__(self, activations):
        """Process activation probabilities and return note events.

        Parameters
        ----------
        activations : np.ndarray
            Per-pitch activation probabilities.
            Shape ``(n_frames, 88)`` for 2D input, or
            ``(n_frames * 88,)`` for flat 1D input.

        Returns
        -------
        np.ndarray
            Note events as a 2D array with columns:
            ``[onset_time, pitch, duration, velocity]``.
            Times are in seconds. Pitch is MIDI note number.
        """
        from metalmom.transcribe import piano_transcribe

        activations = np.asarray(activations, dtype=np.float32)

        events = piano_transcribe(
            activations,
            threshold=self.threshold,
            min_duration=self.min_duration,
            use_hmm=self.use_hmm,
            fps=self.fps,
            units='seconds',
        )

        if len(events) == 0:
            return np.empty((0, 4), dtype=np.float64)

        # Convert to madmom output format: (onset_time, pitch, duration, velocity)
        result = []
        for ev in events:
            onset = ev['onset']
            pitch = ev['midi_note']
            duration = ev['offset'] - ev['onset']
            velocity = ev['velocity']
            result.append([onset, pitch, duration, velocity])

        return np.array(result, dtype=np.float64)
