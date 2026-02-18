from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any

def beat_track(
    y: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    onset_envelope: npt.NDArray[np.float32] | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    start_bpm: float = 120.0,
    trim: bool = True,
    units: str = 'frames',
    **kwargs: Any,
) -> tuple[float, npt.NDArray[np.intp]]: ...

def plp(
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
    tempo_min: float = 30.0,
    tempo_max: float = 300.0,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def neural_beat_track(
    activations: npt.NDArray[np.float32],
    fps: float = 100.0,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: float = 100.0,
    threshold: float = 0.05,
    trim: bool = True,
    units: str = 'frames',
    hop_length: int = 441,
    sr: int = 44100,
    **kwargs: Any,
) -> tuple[float, npt.NDArray[np.intp]]: ...

def downbeat_detect(
    activations: npt.NDArray[np.float32],
    fps: float = 100.0,
    beats_per_bar: int = 4,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: float = 100.0,
    units: str = 'frames',
    hop_length: int = 441,
    sr: int = 44100,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
