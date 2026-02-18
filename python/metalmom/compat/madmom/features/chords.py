"""madmom.features.chords compatibility shim.

Provides DeepChromaChordRecognitionProcessor backed by MetalMom.
"""

import numpy as np


class DeepChromaChordRecognitionProcessor:
    """madmom-compatible deep chroma chord recognition processor.

    In real madmom this runs a deep chroma model for chord recognition.
    This shim delegates to MetalMom's chord_detect function.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 10.0.
    """

    def __init__(self, fps=10.0, **kwargs):
        self.fps = fps

    def __call__(self, data):
        """Process data and return chord predictions.

        Parameters
        ----------
        data : np.ndarray
            Chord activation probabilities, shape (n_frames, n_classes).

        Returns
        -------
        list of tuple
            List of (start_time, end_time, chord_label) tuples.
        """
        from metalmom.chord import chord_detect

        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 1:
            # Determine number of classes -- try 25 first (madmom default)
            if len(data) % 25 == 0:
                n_classes = 25
            else:
                n_classes = data.shape[0]
                data = data.reshape(1, -1)
        elif data.ndim == 2:
            n_classes = data.shape[1]
        else:
            raise ValueError(
                f"data must be 1D or 2D, got ndim={data.ndim}"
            )

        events = chord_detect(
            data,
            n_classes=n_classes,
            fps=self.fps,
            units='seconds',
        )

        # Return as list of (start, end, label) tuples (madmom convention)
        results = []
        for event in events:
            results.append((event['start'], event['end'], event['chord_label']))

        return results

    def process(self, data):
        """Alias for __call__."""
        return self(data)
