from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any, Callable

def melspectrogram(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def mfcc(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def amplitude_to_db(
    S: npt.NDArray[np.float32],
    ref: float | Callable[..., float] = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> npt.NDArray[np.float32]: ...

def power_to_db(
    S: npt.NDArray[np.float32],
    ref: float | Callable[..., float] = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> npt.NDArray[np.float32]: ...

def chroma_stft(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_chroma: int = 12,
    center: bool = True,
    norm: float | None = None,
    tuning: float = 0.0,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def chroma_cqt(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.7,
    bins_per_octave: int = 36,
    n_octaves: int = 7,
    n_chroma: int = 12,
    norm: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def chroma_cens(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.7,
    bins_per_octave: int = 36,
    n_octaves: int = 7,
    n_chroma: int = 12,
    win_len_smooth: int = 41,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def chroma_vqt(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.7,
    bins_per_octave: int = 36,
    n_octaves: int = 7,
    gamma: float = 0.0,
    n_chroma: int = 12,
    norm: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def spectral_centroid(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    freq: npt.NDArray[np.float32] | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def spectral_bandwidth(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    freq: npt.NDArray[np.float32] | None = None,
    p: float = 2,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def spectral_contrast(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def spectral_rolloff(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    roll_percent: float = 0.85,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def spectral_flatness(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    power: float = 2.0,
    amin: float = 1e-10,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def rms(
    y: npt.NDArray[np.float32] | None = None,
    S: npt.NDArray[np.float32] | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def zero_crossing_rate(
    y: npt.NDArray[np.float32],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def tonnetz(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    chroma: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_chroma: int = 12,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def tempo(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    onset_envelope: npt.NDArray[np.float32] | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    start_bpm: float = 120.0,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float64]: ...

def poly_features(
    S: npt.NDArray[np.float32] | None = None,
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    order: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def delta(
    data: npt.NDArray[np.float32],
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def stack_memory(
    data: npt.NDArray[np.float32],
    n_steps: int = 2,
    delay: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def tempogram(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    onset_envelope: npt.NDArray[np.float32] | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    win_length: int = 384,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def fourier_tempogram(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    onset_envelope: npt.NDArray[np.float32] | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    win_length: int = 384,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def pcen(
    S: npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int = 512,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.06,
    eps: float = 1e-6,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def a_weighting(
    frequencies: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]: ...

def b_weighting(
    frequencies: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]: ...

def c_weighting(
    frequencies: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]: ...

def d_weighting(
    frequencies: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]: ...

def mel_to_audio(
    M: npt.NDArray[np.float32],
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    power: float = 2.0,
    n_iter: int = 32,
    fmin: float = 0.0,
    fmax: float | None = None,
    length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def mfcc_to_mel(
    M: npt.NDArray[np.float32],
    n_mels: int = 128,
    sr: int = 22050,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def mfcc_to_audio(
    M: npt.NDArray[np.float32],
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    n_iter: int = 32,
    fmin: float = 0.0,
    fmax: float | None = None,
    length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...
