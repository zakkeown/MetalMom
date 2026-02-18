from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def specshow(
    data: npt.NDArray[np.float64],
    *,
    x_coords: npt.NDArray[np.float64] | None = None,
    y_coords: npt.NDArray[np.float64] | None = None,
    x_axis: str | None = None,
    y_axis: str | None = None,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any: ...

def waveshow(
    y: npt.NDArray[np.float64],
    *,
    sr: int = 22050,
    max_points: int = 11025,
    ax: Any = None,
    offset: float = 0.0,
    **kwargs: Any,
) -> Any: ...
