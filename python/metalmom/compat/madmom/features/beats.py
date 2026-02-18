"""madmom.features.beats compatibility shim.

Provides RNNBeatProcessor and DBNBeatTrackingProcessor backed by MetalMom.
"""

import numpy as np


class RNNBeatProcessor:
    """madmom-compatible RNN beat processor.

    In real madmom this runs an RNN ensemble to produce beat activations.
    This shim computes onset strength from audio using MetalMom's native
    onset strength function with madmom-compatible defaults (sr=44100,
    hop=441).

    Parameters
    ----------
    fps : float
        Frames per second. Default: 100.0.
    """

    def __init__(self, fps=100.0, **kwargs):
        self.fps = fps

    def __call__(self, data):
        """Process audio and return beat activations.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal.

        Returns
        -------
        np.ndarray
            Beat activation function, shape (n_frames,).
        """
        from metalmom.onset import onset_strength
        from ..audio.signal import Signal

        if isinstance(data, (str, bytes)):
            data = Signal(data)

        audio = np.asarray(data, dtype=np.float32).ravel()
        sr = getattr(data, 'sample_rate', 44100)
        hop_size = int(round(sr / self.fps))

        env = onset_strength(y=audio, sr=sr, hop_length=hop_size)

        # Normalize to [0, 1] range
        max_val = env.max()
        if max_val > 0:
            env = env / max_val

        return env.astype(np.float32)


class DBNBeatTrackingProcessor:
    """madmom-compatible DBN beat tracking processor.

    Decodes beat positions from activation probabilities using dynamic
    programming (backed by MetalMom's neural_beat_track).

    Parameters
    ----------
    min_bpm : float
        Minimum tempo in BPM. Default: 55.0.
    max_bpm : float
        Maximum tempo in BPM. Default: 215.0.
    transition_lambda : float
        Penalty for tempo deviations. Default: 100.0.
    threshold : float
        Minimum activation threshold. Default: 0.05.
    fps : float
        Frames per second. Default: 100.0.
    """

    def __init__(self, min_bpm=55.0, max_bpm=215.0, transition_lambda=100.0,
                 threshold=0.05, fps=100.0, **kwargs):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.transition_lambda = transition_lambda
        self.threshold = threshold
        self.fps = fps

    def __call__(self, activations):
        """Process beat activations and return beat times.

        Parameters
        ----------
        activations : np.ndarray
            Beat activation probabilities, shape (n_frames,).

        Returns
        -------
        np.ndarray
            Beat times in seconds.
        """
        from metalmom.beat import neural_beat_track

        activations = np.asarray(activations, dtype=np.float32).ravel()

        _tempo, beat_frames = neural_beat_track(
            activations,
            fps=self.fps,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            transition_lambda=self.transition_lambda,
            threshold=self.threshold,
            units='frames',
        )

        # Convert frames to seconds
        beat_times = beat_frames.astype(np.float64) / self.fps
        return beat_times
