"""madmom.audio.signal compatibility shim.

Provides Signal and FramedSignal classes backed by MetalMom.

madmom defaults:
    - sample_rate: 44100 (not 22050 like librosa)
    - frame_size: 2048
    - hop_size: 441 (10 ms at 44100 Hz)
"""

import numpy as np


class Signal(np.ndarray):
    """madmom-compatible Signal class backed by MetalMom.

    Wraps a 1-D float32 audio array with a ``sample_rate`` attribute.
    Can be constructed from a file path or an existing array.

    Parameters
    ----------
    data : str or array-like
        File path to load, or existing audio samples.
    sr : int or None
        Target sample rate. Default: 44100.
    mono : bool
        Convert to mono. Default: True.
    start : float
        Start time in seconds. Default: 0.0.
    stop : float or None
        Stop time in seconds. Default: None (end of file).
    """

    def __new__(cls, data, sr=None, mono=True, start=0.0, stop=None,
                sample_rate=None, **kwargs):
        # Allow either 'sr' or 'sample_rate' (madmom uses sample_rate)
        target_sr = sample_rate or sr or 44100

        if isinstance(data, (str, bytes)):
            from metalmom import load
            duration = (stop - start) if stop is not None else None
            audio, actual_sr = load(
                data if isinstance(data, str) else data.decode(),
                sr=target_sr,
                mono=mono,
                offset=start,
                duration=duration,
            )
            obj = np.asarray(audio, dtype=np.float32).view(cls)
            obj.sample_rate = actual_sr
        else:
            obj = np.asarray(data, dtype=np.float32).view(cls)
            obj.sample_rate = target_sr

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.sample_rate = getattr(obj, 'sample_rate', 44100)

    @property
    def num_samples(self):
        """Total number of samples."""
        return len(self)

    @property
    def length(self):
        """Duration in seconds."""
        return len(self) / self.sample_rate


class FramedSignal:
    """Splits a Signal into overlapping frames (madmom convention).

    Parameters
    ----------
    signal : Signal, np.ndarray, or str
        Input signal. If a string, loaded via Signal.
    frame_size : int
        Number of samples per frame. Default: 2048.
    hop_size : int
        Hop size in samples. Default: 441 (10 ms at 44100 Hz).
    origin : str
        Frame origin: ``'center'``, ``'left'``, or ``'right'``.
        Default: ``'center'`` (matches madmom).
    """

    def __init__(self, signal, frame_size=2048, hop_size=441, origin='center',
                 **kwargs):
        if isinstance(signal, (str, bytes)):
            signal = Signal(signal, **kwargs)

        self.signal = np.asarray(signal, dtype=np.float32)
        self.sample_rate = getattr(signal, 'sample_rate', 44100)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.origin = origin

        # Pad signal for center-origin framing (matches madmom behaviour)
        if origin == 'center':
            pad_left = frame_size // 2
            pad_right = frame_size // 2
            padded = np.pad(self.signal, (pad_left, pad_right), mode='constant')
        elif origin == 'left':
            padded = np.pad(self.signal, (0, frame_size), mode='constant')
        else:  # 'right' or default
            padded = self.signal

        # Compute number of frames
        if len(padded) < frame_size:
            self._num_frames = 0
            self._data = np.empty((0, frame_size), dtype=np.float32)
        else:
            self._num_frames = 1 + (len(padded) - frame_size) // hop_size
            # Use stride tricks for efficient frame extraction
            strides = (
                padded.strides[0] * hop_size,
                padded.strides[0],
            )
            self._data = np.lib.stride_tricks.as_strided(
                padded,
                shape=(self._num_frames, frame_size),
                strides=strides,
            ).copy()  # copy to own memory (safe after padding)

    def __len__(self):
        return self._num_frames

    def __getitem__(self, index):
        return self._data[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self._data[i]

    @property
    def shape(self):
        """Shape of the framed signal: (num_frames, frame_size)."""
        return self._data.shape
