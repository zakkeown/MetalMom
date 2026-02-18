from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any, Generator

def load(
    path: str,
    sr: int | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], int]: ...

def get_duration(path: str, **kwargs: Any) -> float: ...

def get_samplerate(path: str, **kwargs: Any) -> int: ...

def resample(
    y: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def db_to_amplitude(
    S_db: npt.NDArray[np.float32],
    ref: float = 1.0,
) -> npt.NDArray[np.float32]: ...

def db_to_power(
    S_db: npt.NDArray[np.float32],
    ref: float = 1.0,
) -> npt.NDArray[np.float32]: ...

def stft(
    y: npt.NDArray[np.float32],
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
) -> npt.NDArray[np.float32]: ...

def istft(
    stft_matrix: npt.NDArray[np.complex64],
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    length: int | None = None,
) -> npt.NDArray[np.float32]: ...

def tone(
    frequency: float,
    sr: int = 22050,
    length: int | None = None,
    duration: float | None = None,
    phi: float = 0.0,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def chirp(
    fmin: float,
    fmax: float,
    sr: int = 22050,
    length: int | None = None,
    duration: float | None = None,
    linear: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def stream(
    path: str,
    block_length: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    mono: bool = True,
    sr: int | None = 22050,
    fill_value: float | None = None,
    dtype: np.dtype[Any] = ...,
    **kwargs: Any,
) -> Generator[npt.NDArray[np.float32], None, None]: ...

def clicks(
    times: npt.NDArray[np.float32] | None = None,
    sr: int = 22050,
    length: int | None = None,
    click_freq: float = 1000.0,
    click_duration: float = 0.1,
    **kwargs: Any,
) -> npt.NDArray[np.float32]: ...

def reassigned_spectrogram(
    y: npt.NDArray[np.float32],
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
