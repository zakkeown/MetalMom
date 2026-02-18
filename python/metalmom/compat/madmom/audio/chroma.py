"""madmom.audio.chroma compatibility shim.

Provides DeepChromaProcessor and CLPChromaProcessor backed by MetalMom.

madmom defaults:
    - sample_rate: 44100
    - fps: 10.0
"""

import numpy as np


class DeepChromaProcessor:
    """madmom-compatible deep chroma processor.

    In real madmom this runs a DNN to produce deep chroma features from
    audio. This shim delegates to MetalMom's ``chroma_stft`` function
    with madmom-compatible defaults (sr=44100, hop derived from fps).

    Parameters
    ----------
    fps : float
        Frames per second. Default: 10.0.
    fmin : float
        Minimum frequency in Hz. Default: 65.0 (C2).
    fmax : float or None
        Maximum frequency in Hz. Default: 2100.0.
    sample_rate : int
        Sample rate. Default: 44100.
    n_chroma : int
        Number of chroma bins. Default: 12.
    """

    def __init__(self, fps=10.0, fmin=65.0, fmax=2100.0,
                 sample_rate=44100, n_chroma=12, **kwargs):
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma

    def __call__(self, data):
        """Process audio and return deep chroma features.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal.

        Returns
        -------
        np.ndarray
            Chroma features, shape (n_frames, n_chroma).
        """
        from metalmom.feature import chroma_stft
        from ..audio.signal import Signal

        if isinstance(data, (str, bytes)):
            data = Signal(data, sample_rate=self.sample_rate)

        audio = np.asarray(data, dtype=np.float32).ravel()
        sr = getattr(data, 'sample_rate', self.sample_rate)
        hop_length = int(round(sr / self.fps))

        # chroma_stft returns shape (n_chroma, n_frames)
        chroma = chroma_stft(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_chroma=self.n_chroma,
        )

        # madmom convention: (n_frames, n_chroma)
        return chroma.T.astype(np.float32)


class CLPChromaProcessor:
    """madmom-compatible CLP (Chroma from Logarithmic Pitch) processor.

    In real madmom this computes chroma features from a logarithmic
    pitch representation. This shim delegates to MetalMom's
    ``chroma_cqt`` function as a close approximation.

    Parameters
    ----------
    fps : float
        Frames per second. Default: 10.0.
    fmin : float
        Minimum frequency in Hz. Default: 27.5 (A0).
    sample_rate : int
        Sample rate. Default: 44100.
    n_chroma : int
        Number of chroma bins. Default: 12.
    """

    def __init__(self, fps=10.0, fmin=27.5, sample_rate=44100,
                 n_chroma=12, **kwargs):
        self.fps = fps
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma

    def __call__(self, data):
        """Process audio and return CLP chroma features.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal.

        Returns
        -------
        np.ndarray
            Chroma features, shape (n_frames, n_chroma).
        """
        from metalmom.feature import chroma_cqt
        from ..audio.signal import Signal

        if isinstance(data, (str, bytes)):
            data = Signal(data, sample_rate=self.sample_rate)

        audio = np.asarray(data, dtype=np.float32).ravel()
        sr = getattr(data, 'sample_rate', self.sample_rate)
        hop_length = int(round(sr / self.fps))

        # chroma_cqt returns shape (n_chroma, n_frames)
        chroma = chroma_cqt(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            fmin=self.fmin,
            n_chroma=self.n_chroma,
        )

        # madmom convention: (n_frames, n_chroma)
        return chroma.T.astype(np.float32)
