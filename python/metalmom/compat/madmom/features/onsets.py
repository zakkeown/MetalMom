"""madmom.features.onsets compatibility shim.

Provides onset detection processors backed by MetalMom.
"""

import numpy as np


class OnsetPeakPickingProcessor:
    """madmom-compatible onset peak picking processor.

    Detects onsets from an activation function using peak picking with
    local max, moving average threshold, and minimum wait constraints.

    Parameters
    ----------
    threshold : float
        Activation threshold. Default: 0.3.
    pre_max : float
        Look-back time for local maximum (seconds). Default: 0.03.
    post_max : float
        Look-ahead time for local maximum (seconds). Default: 0.03.
    pre_avg : float
        Look-back time for moving average (seconds). Default: 0.1.
    post_avg : float
        Look-ahead time for moving average (seconds). Default: 0.07.
    combine : float
        Combine onsets within this time window (seconds). Default: 0.03.
    delay : float
        Delay all onsets by this amount (seconds). Default: 0.0.
    fps : float
        Frames per second of the activation function. Default: 100.0.
    """

    def __init__(self, threshold=0.3, pre_max=0.03, post_max=0.03,
                 pre_avg=0.1, post_avg=0.07, combine=0.03, delay=0.0,
                 fps=100.0, **kwargs):
        self.threshold = threshold
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def __call__(self, activations):
        """Process activations and return onset times in seconds.

        Parameters
        ----------
        activations : np.ndarray
            Onset activation function, shape (n_frames,).

        Returns
        -------
        np.ndarray
            Onset times in seconds.
        """
        from metalmom.onset import neural_onset_detect

        activations = np.asarray(activations, dtype=np.float32).ravel()

        # Convert time parameters to frame counts
        pre_max_frames = max(1, int(round(self.pre_max * self.fps)))
        post_max_frames = max(1, int(round(self.post_max * self.fps)))
        pre_avg_frames = max(1, int(round(self.pre_avg * self.fps)))
        post_avg_frames = max(1, int(round(self.post_avg * self.fps)))
        combine_frames = max(1, int(round(self.combine * self.fps)))

        frames = neural_onset_detect(
            activations,
            fps=self.fps,
            threshold=self.threshold,
            pre_max=pre_max_frames,
            post_max=post_max_frames,
            pre_avg=pre_avg_frames,
            post_avg=post_avg_frames,
            combine='adaptive',
            wait=combine_frames,
            units='frames',
        )

        # Convert to times and add delay
        times = frames.astype(np.float64) / self.fps + self.delay
        return times


class RNNOnsetProcessor:
    """madmom-compatible RNN onset processor.

    In real madmom this runs an RNN ensemble to produce onset activations.
    This shim computes onset strength from audio using MetalMom's native
    onset strength function with madmom-compatible defaults.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 100.0.
    """

    def __init__(self, fps=100.0, **kwargs):
        self.fps = fps

    def __call__(self, data):
        """Process audio and return onset activations.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal.

        Returns
        -------
        np.ndarray
            Onset activation function, shape (n_frames,).
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
