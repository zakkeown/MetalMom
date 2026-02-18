"""madmom.features.tempo compatibility shim.

Provides TempoEstimationProcessor backed by MetalMom.
"""

import numpy as np


class TempoEstimationProcessor:
    """madmom-compatible tempo estimation processor.

    In real madmom this uses a DBN to estimate tempo from onset activations.
    This shim wraps MetalMom's neural_beat_track which accepts activation
    arrays and returns a tempo estimate via dynamic programming.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 100.0.
    min_bpm : float
        Minimum tempo in BPM. Default: 40.0.
    max_bpm : float
        Maximum tempo in BPM. Default: 250.0.
    """

    def __init__(self, fps=100.0, min_bpm=40.0, max_bpm=250.0, **kwargs):
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def __call__(self, activations):
        """Estimate tempo from onset activations.

        Parameters
        ----------
        activations : np.ndarray
            Onset activation function, shape (n_frames,).

        Returns
        -------
        np.ndarray
            Tempo estimates as (tempo, strength) pairs, shape (n, 2).
        """
        from metalmom.beat import neural_beat_track

        activations = np.asarray(activations, dtype=np.float32).ravel()

        tempo_bpm, _beats = neural_beat_track(
            activations,
            fps=self.fps,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            units='frames',
        )

        bpm = float(tempo_bpm)

        # Return as (tempo, strength) pair matching madmom format
        return np.array([[bpm, 1.0]], dtype=np.float64)
