from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def hpss(
    y: npt.NDArray[np.float32],
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

def harmonic(
    y: npt.NDArray[np.float32],
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def percussive(
    y: npt.NDArray[np.float32],
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def time_stretch(
    y: npt.NDArray[np.float32],
    rate: float,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def pitch_shift(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    n_steps: float = 0,
    bins_per_octave: int = 12,
    n_fft: int = 2048,
    hop_length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def trim(
    y: npt.NDArray[np.float32],
    top_db: float = 60,
    ref: Any = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], tuple[int, int]]: ...

def split(
    y: npt.NDArray[np.float32],
    top_db: float = 60,
    ref: Any = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def preemphasis(
    y: npt.NDArray[np.float32],
    coef: float = 0.97,
    zi: Any = None,
    return_zf: bool = False,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def deemphasis(
    y: npt.NDArray[np.float32],
    coef: float = 0.97,
    zi: Any = None,
    return_zf: bool = False,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def phase_vocoder(
    D: npt.NDArray[np.complex64],
    rate: float,
    hop_length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.complex64]: ...

def griffinlim(
    S: npt.NDArray[np.float32],
    n_iter: int = 32,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    length: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def griffinlim_cqt(
    C: npt.NDArray[np.float32],
    n_iter: int = 32,
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.70,
    bins_per_octave: int = 12,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...
