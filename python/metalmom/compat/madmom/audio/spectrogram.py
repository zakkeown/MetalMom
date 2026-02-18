"""madmom.audio.spectrogram compatibility shim.

Provides Spectrogram, FilteredSpectrogram, and LogarithmicFilteredSpectrogram
classes backed by MetalMom.

madmom defaults:
    - frame_size: 2048
    - hop_size: 441
    - sample_rate: 44100
    - num_bands (for filtered): 80
"""

import numpy as np


class Spectrogram(np.ndarray):
    """madmom-compatible Spectrogram class.

    Computes the magnitude spectrogram from an STFT, Signal, FramedSignal,
    or file path.

    Parameters
    ----------
    data : STFT, FramedSignal, Signal, np.ndarray, or str
        Input data. If complex, magnitude is taken. If real 1-D signal,
        STFT is computed first.
    frame_size : int
        FFT size. Default: 2048.
    hop_size : int
        Hop size in samples. Default: 441.
    sample_rate : int
        Sample rate. Default: 44100.
    """

    def __new__(cls, data, frame_size=2048, hop_size=441,
                sample_rate=None, **kwargs):
        from .stft import STFT
        from .signal import FramedSignal, Signal

        sr = sample_rate or getattr(data, 'sample_rate', 44100)

        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            # Complex input -- take magnitude
            mag = np.abs(data).astype(np.float32)
            sr = getattr(data, 'sample_rate', sr)
            frame_size = getattr(data, 'frame_size', frame_size)
            hop_size = getattr(data, 'hop_size', hop_size)
        elif isinstance(data, STFT):
            mag = np.abs(np.asarray(data)).astype(np.float32)
            sr = data.sample_rate
            frame_size = data.frame_size
            hop_size = data.hop_size
        elif isinstance(data, (FramedSignal, str, bytes)):
            stft = STFT(data, frame_size=frame_size, hop_size=hop_size,
                        sample_rate=sr, **kwargs)
            mag = np.abs(np.asarray(stft)).astype(np.float32)
            sr = stft.sample_rate
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            # 1-D real signal -- compute STFT first
            stft = STFT(data, frame_size=frame_size, hop_size=hop_size,
                        sample_rate=sr, **kwargs)
            mag = np.abs(np.asarray(stft)).astype(np.float32)
            sr = stft.sample_rate
        elif isinstance(data, np.ndarray) and data.ndim == 2 and not np.iscomplexobj(data):
            # 2-D real input -- assume already a magnitude spectrogram
            mag = np.asarray(data, dtype=np.float32)
        else:
            stft = STFT(data, frame_size=frame_size, hop_size=hop_size,
                        sample_rate=sr, **kwargs)
            mag = np.abs(np.asarray(stft)).astype(np.float32)
            sr = stft.sample_rate

        obj = mag.view(cls)
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


class FilteredSpectrogram(Spectrogram):
    """madmom-compatible FilteredSpectrogram class.

    Applies a mel filterbank to a magnitude spectrogram.

    Parameters
    ----------
    data : Spectrogram, STFT, Signal, np.ndarray, or str
        Input data.
    num_bands : int
        Number of filter bands (mel bands). Default: 80.
    fmin : float
        Minimum frequency for the filterbank. Default: 30.0.
    fmax : float or None
        Maximum frequency. Default: None (sr / 2).
    frame_size : int
        FFT size. Default: 2048.
    hop_size : int
        Hop size. Default: 441.
    sample_rate : int
        Sample rate. Default: 44100.
    """

    def __new__(cls, data, num_bands=80, fmin=30.0, fmax=None,
                frame_size=2048, hop_size=441, sample_rate=None,
                filterbank=None, **kwargs):
        # First get the magnitude spectrogram
        sr = sample_rate or getattr(data, 'sample_rate', 44100)

        if isinstance(data, Spectrogram):
            spec = np.asarray(data, dtype=np.float32)
            sr = data.sample_rate
            frame_size = data.frame_size
            hop_size = data.hop_size
        else:
            spec_obj = Spectrogram(data, frame_size=frame_size,
                                   hop_size=hop_size, sample_rate=sr,
                                   **kwargs)
            spec = np.asarray(spec_obj, dtype=np.float32)
            sr = spec_obj.sample_rate
            frame_size = spec_obj.frame_size
            hop_size = spec_obj.hop_size

        if spec.ndim != 2 or spec.shape[0] == 0:
            filtered = np.empty((0, num_bands), dtype=np.float32)
        elif filterbank is not None:
            # User-provided filterbank
            fb = np.asarray(filterbank, dtype=np.float32)
            filtered = spec @ fb
        else:
            # Build mel filterbank
            n_fft = frame_size
            n_freqs = spec.shape[1]  # (n_frames, n_freqs) in madmom convention

            actual_fmax = fmax if fmax is not None else sr / 2.0
            fb = _build_mel_filterbank(
                sr=sr, n_fft=n_fft, n_freqs=n_freqs,
                n_mels=num_bands, fmin=fmin, fmax=actual_fmax,
            )
            # spec shape is (n_frames, n_freqs), fb is (n_freqs, n_mels)
            filtered = spec @ fb

        obj = np.asarray(filtered, dtype=np.float32).view(cls)
        obj.sample_rate = sr
        obj.frame_size = frame_size
        obj.hop_size = hop_size
        obj.num_bands = num_bands
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.num_bands = getattr(obj, 'num_bands', 80)


class LogarithmicFilteredSpectrogram(FilteredSpectrogram):
    """madmom-compatible LogarithmicFilteredSpectrogram class.

    Applies log(1 + x) scaling to a FilteredSpectrogram.

    Parameters
    ----------
    data : FilteredSpectrogram, Spectrogram, STFT, Signal, np.ndarray, or str
        Input data.
    num_bands : int
        Number of filter bands. Default: 80.
    fmin : float
        Minimum frequency. Default: 30.0.
    fmax : float or None
        Maximum frequency. Default: None (sr / 2).
    mul : float
        Multiplier before log. Default: 1.0.
    add : float
        Addend inside log: ``log(mul * x + add)``. Default: 1.0.
    """

    def __new__(cls, data, num_bands=80, fmin=30.0, fmax=None,
                mul=1.0, add=1.0, frame_size=2048, hop_size=441,
                sample_rate=None, **kwargs):
        sr = sample_rate or getattr(data, 'sample_rate', 44100)

        if isinstance(data, FilteredSpectrogram):
            filt = np.asarray(data, dtype=np.float32)
            sr = data.sample_rate
            frame_size = data.frame_size
            hop_size = data.hop_size
            num_bands = data.num_bands
        else:
            filt_obj = FilteredSpectrogram(
                data, num_bands=num_bands, fmin=fmin, fmax=fmax,
                frame_size=frame_size, hop_size=hop_size,
                sample_rate=sr, **kwargs,
            )
            filt = np.asarray(filt_obj, dtype=np.float32)
            sr = filt_obj.sample_rate
            frame_size = filt_obj.frame_size
            hop_size = filt_obj.hop_size

        # Apply logarithmic scaling: log(mul * x + add)
        log_spec = np.log(mul * filt + add).astype(np.float32)

        obj = np.asarray(log_spec).view(cls)
        obj.sample_rate = sr
        obj.frame_size = frame_size
        obj.hop_size = hop_size
        obj.num_bands = num_bands
        return obj


def _build_mel_filterbank(sr, n_fft, n_freqs, n_mels, fmin, fmax):
    """Build a mel filterbank matrix.

    Returns shape (n_freqs, n_mels) so that ``spec @ fb`` works when
    spec is (n_frames, n_freqs).
    """
    # Mel conversion helpers
    def _hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_freqs = np.arange(n_freqs) * sr / n_fft

    fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        # Rising slope
        if f_center > f_left:
            up = (fft_freqs - f_left) / (f_center - f_left)
        else:
            up = np.zeros(n_freqs)

        # Falling slope
        if f_right > f_center:
            down = (f_right - fft_freqs) / (f_right - f_center)
        else:
            down = np.zeros(n_freqs)

        fb[:, m] = np.maximum(0.0, np.minimum(up, down))

    return fb
