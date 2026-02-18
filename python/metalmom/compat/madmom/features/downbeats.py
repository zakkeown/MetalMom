"""madmom.features.downbeats compatibility shim.

Provides RNNDownBeatProcessor and DBNDownBeatTrackingProcessor backed
by MetalMom.
"""

import numpy as np


class RNNDownBeatProcessor:
    """madmom-compatible RNN downbeat processor.

    In real madmom this runs an RNN ensemble to produce 3-class beat/downbeat
    activations. This shim generates approximate activations from MetalMom's
    onset strength (normalized to [0, 1]) with a simple 3-class split.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 100.0.
    """

    def __init__(self, fps=100.0, **kwargs):
        self.fps = fps

    def __call__(self, data):
        """Process audio and return downbeat activations.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal.

        Returns
        -------
        np.ndarray
            Activation probabilities, shape (n_frames, 3).
            Columns: P(no beat), P(beat), P(downbeat).
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

        # Approximate 3-class activations:
        # P(no beat) = 1 - env, P(beat) = env * 0.6, P(downbeat) = env * 0.4
        n_frames = len(env)
        activations = np.zeros((n_frames, 3), dtype=np.float32)
        activations[:, 0] = 1.0 - env           # P(no beat)
        activations[:, 1] = env * 0.6            # P(beat)
        activations[:, 2] = env * 0.4            # P(downbeat)

        return activations


class DBNDownBeatTrackingProcessor:
    """madmom-compatible DBN downbeat tracking processor.

    Decodes beat and downbeat positions from 3-class activation
    probabilities using MetalMom's downbeat_detect.

    Parameters
    ----------
    beats_per_bar : int or list
        Expected beats per bar. Default: [4].
    min_bpm : float
        Minimum tempo in BPM. Default: 55.0.
    max_bpm : float
        Maximum tempo in BPM. Default: 215.0.
    transition_lambda : float
        Penalty for tempo deviations. Default: 100.0.
    fps : float
        Frames per second. Default: 100.0.
    """

    def __init__(self, beats_per_bar=None, min_bpm=55.0, max_bpm=215.0,
                 transition_lambda=100.0, fps=100.0, **kwargs):
        if beats_per_bar is None:
            beats_per_bar = [4]
        if isinstance(beats_per_bar, int):
            beats_per_bar = [beats_per_bar]
        self.beats_per_bar = beats_per_bar
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.transition_lambda = transition_lambda
        self.fps = fps

    def __call__(self, activations):
        """Process activations and return beat/downbeat annotations.

        Parameters
        ----------
        activations : np.ndarray
            3-class activation probabilities, shape (n_frames, 3).

        Returns
        -------
        np.ndarray
            Array of shape (n_beats, 2) where each row is
            (time_in_seconds, beat_position). beat_position=1 indicates
            a downbeat.
        """
        from metalmom.beat import downbeat_detect

        activations = np.asarray(activations, dtype=np.float32)
        if activations.ndim == 1:
            n_frames = len(activations) // 3
            activations = activations.reshape(n_frames, 3)

        # Use the first beats_per_bar value
        bpb = self.beats_per_bar[0]

        beat_frames, downbeat_frames = downbeat_detect(
            activations,
            fps=self.fps,
            beats_per_bar=bpb,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            transition_lambda=self.transition_lambda,
            units='frames',
        )

        # Build output: (time, beat_position) -- 1 for downbeat
        downbeat_set = set(downbeat_frames.tolist())
        results = []
        beat_pos = 1  # 1-based beat position within bar
        for f in sorted(beat_frames):
            time = float(f) / self.fps
            if f in downbeat_set:
                beat_pos = 1
            results.append([time, float(beat_pos)])
            beat_pos += 1
            if beat_pos > bpb:
                beat_pos = 1

        if len(results) == 0:
            return np.empty((0, 2), dtype=np.float64)

        return np.array(results, dtype=np.float64)
