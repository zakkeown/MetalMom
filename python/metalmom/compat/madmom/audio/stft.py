"""madmom.audio.stft compatibility shim.

Provides STFT class backed by MetalMom.

madmom defaults:
    - frame_size (n_fft): 2048
    - hop_size: 441
    - sample_rate: 44100
    - window: np.hanning
"""

import numpy as np


class STFT(np.ndarray):
    """madmom-compatible STFT class backed by MetalMom.

    Computes the Short-Time Fourier Transform and returns a complex
    spectrogram as an ndarray subclass.

    Can be constructed from:
    - A FramedSignal (frames are FFT'd directly)
    - A Signal or 1-D array (framed then FFT'd)
    - A file path (loaded, framed, then FFT'd)

    Parameters
    ----------
    data : FramedSignal, Signal, np.ndarray, or str
        Input data.
    frame_size : int
        FFT size. Default: 2048.
    hop_size : int
        Hop size in samples. Default: 441.
    window : callable or np.ndarray or None
        Window function. Default: np.hanning.
    sample_rate : int
        Sample rate. Default: 44100.
    """

    def __new__(cls, data, frame_size=2048, hop_size=441, window=None,
                sample_rate=None, **kwargs):
        from .signal import FramedSignal, Signal

        sr = sample_rate or getattr(data, 'sample_rate', 44100)

        if isinstance(data, FramedSignal):
            frames = data
            frame_size = data.frame_size
        elif isinstance(data, (str, bytes)):
            sig = Signal(data, sr=sr)
            frames = FramedSignal(sig, frame_size=frame_size, hop_size=hop_size)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            sig = Signal(data, sr=sr)
            frames = FramedSignal(sig, frame_size=frame_size, hop_size=hop_size)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            # Assume pre-framed data (n_frames, frame_size)
            frames = data
            frame_size = data.shape[1] if data.shape[1] > 0 else frame_size
        else:
            # Try wrapping as signal
            sig = Signal(data, sr=sr)
            frames = FramedSignal(sig, frame_size=frame_size, hop_size=hop_size)

        # Apply window function
        if window is None:
            win = np.hanning(frame_size).astype(np.float32)
        elif callable(window):
            win = np.asarray(window(frame_size), dtype=np.float32)
        else:
            win = np.asarray(window, dtype=np.float32)

        # Get frames as array
        if isinstance(frames, FramedSignal):
            frame_data = frames._data
        else:
            frame_data = np.asarray(frames, dtype=np.float32)

        if len(frame_data) == 0:
            stft_data = np.empty((0, frame_size // 2 + 1), dtype=np.complex64)
        else:
            # Apply window and compute FFT
            windowed = frame_data * win[np.newaxis, :]
            stft_data = np.fft.rfft(windowed, n=frame_size, axis=1).astype(
                np.complex64
            )

        obj = np.asarray(stft_data).view(cls)
        obj.sample_rate = sr
        obj.frame_size = frame_size
        obj.hop_size = hop_size
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.sample_rate = getattr(obj, 'sample_rate', 44100)
        self.frame_size = getattr(obj, 'frame_size', 2048)
        self.hop_size = getattr(obj, 'hop_size', 441)

    @property
    def num_frames(self):
        """Number of STFT frames."""
        return self.shape[0]

    @property
    def num_bins(self):
        """Number of frequency bins."""
        return self.shape[1] if self.ndim >= 2 else 0
