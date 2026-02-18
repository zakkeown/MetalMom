from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def onset_strength(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    S: npt.NDArray[np.float32] | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    aggregate: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def onset_detect(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    onset_envelope: npt.NDArray[np.float32] | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    backtrack: bool = False,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    delta: float = 0.07,
    wait: int = 30,
    units: str = 'frames',
    **kwargs: Any,
) -> npt.NDArray[np.intp]: ...

def neural_onset_detect(
    activations: npt.NDArray[np.float32],
    fps: float = 100.0,
    threshold: float = 0.3,
    pre_max: int = 1,
    post_max: int = 1,
    pre_avg: int = 3,
    post_avg: int = 3,
    combine: str = 'adaptive',
    wait: int = 1,
    units: str = 'frames',
    hop_length: int | None = None,
    sr: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.intp]: ...
