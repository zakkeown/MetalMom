from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def cqt(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.70,
    fmax: float | None = None,
    bins_per_octave: int = 12,
    n_fft: int = 0,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def vqt(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.70,
    fmax: float | None = None,
    bins_per_octave: int = 12,
    gamma: float = 0.0,
    n_fft: int = 0,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def hybrid_cqt(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int | None = None,
    fmin: float = 32.70,
    fmax: float | None = None,
    bins_per_octave: int = 12,
    n_fft: int = 0,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...
