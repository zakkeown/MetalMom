from __future__ import annotations
import numpy as np
import numpy.typing as npt

def semitone_filterbank(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    midi_low: int = 24,
    midi_high: int = 119,
    order: int = 4,
) -> npt.NDArray[np.float32]: ...

def semitone_frequencies(
    midi_low: int = 24,
    midi_high: int = 119,
) -> npt.NDArray[np.float32]: ...

def mel(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> npt.NDArray[np.float32]: ...

def chroma(
    sr: int = 22050,
    n_fft: int = 2048,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: float = 2,
    base_c: bool = True,
) -> npt.NDArray[np.float32]: ...

def constant_q(
    sr: int = 22050,
    fmin: float | None = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    norm: int | None = 1,
    pad_fft: bool = True,
) -> npt.NDArray[np.complex64]: ...

def mel_frequencies(
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 11025.0,
) -> npt.NDArray[np.float32]: ...

def fft_frequencies(
    sr: int = 22050,
    n_fft: int = 2048,
) -> npt.NDArray[np.float32]: ...
