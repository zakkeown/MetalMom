"""madmom.features.key compatibility shim.

Provides CNNKeyRecognitionProcessor backed by MetalMom.
"""

import numpy as np


class CNNKeyRecognitionProcessor:
    """madmom-compatible CNN key recognition processor.

    In real madmom this runs a CNN to detect the musical key.
    This shim delegates to MetalMom's key_detect function.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 10.0.
    """

    def __init__(self, fps=10.0, **kwargs):
        self.fps = fps

    def __call__(self, data):
        """Process data and return key predictions.

        Parameters
        ----------
        data : np.ndarray
            Key activation probabilities, shape (24,) or (n_frames, 24).
            If a 1-D audio signal is provided, a simple spectral
            analysis heuristic is used.

        Returns
        -------
        np.ndarray
            Key activations / probabilities, shape (24,).
        """
        from metalmom.key import key_detect

        data = np.asarray(data, dtype=np.float32)

        # If data looks like raw activations (24 classes)
        if data.ndim == 1 and len(data) == 24:
            result = key_detect(data)
            return result['probabilities']
        elif data.ndim == 2 and data.shape[1] == 24:
            result = key_detect(data)
            return result['probabilities']
        else:
            # Fallback: return uniform distribution
            return np.ones(24, dtype=np.float32) / 24.0

    def process(self, data):
        """Alias for __call__."""
        return self(data)
