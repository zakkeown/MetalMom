from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def yin(
    y: npt.NDArray[np.float32],
    fmin: float,
    fmax: float,
    sr: int = 22050,
    frame_length: int = 2048,
    hop_length: int | None = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def pyin(
    y: npt.NDArray[np.float32],
    fmin: float,
    fmax: float,
    sr: int = 22050,
    frame_length: int = 2048,
    hop_length: int | None = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    resolution: float = 0.1,
    switch_prob: float = 0.01,
    center: bool = True,
    fill_na: float | None = ...,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]: ...

def piptrack(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    win_length: int | None = None,
    center: bool = True,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

def estimate_tuning(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    resolution: float = 0.01,
    bins_per_octave: int = 12,
    center: bool = True,
    win_length: int | None = None,
    **kwargs: Any,
) -> float: ...
