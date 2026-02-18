from __future__ import annotations
import numpy as np
import numpy.typing as npt

def hz_to_midi(
    hz: float | npt.NDArray[np.float32],
) -> float | npt.NDArray[np.float32]: ...

def midi_to_hz(
    midi: float | npt.NDArray[np.float32],
) -> float | npt.NDArray[np.float32]: ...

def hz_to_note(
    hz: float | npt.NDArray[np.float32],
) -> str | list[str]: ...

def note_to_hz(
    note: str | list[str],
) -> float | npt.NDArray[np.float32]: ...

def midi_to_note(
    midi: float | npt.NDArray[np.float32],
) -> str | list[str]: ...

def note_to_midi(
    note: str | list[str],
) -> float | npt.NDArray[np.float32]: ...

def times_to_frames(
    times: float | npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> npt.NDArray[np.intp]: ...

def frames_to_time(
    frames: int | npt.NDArray[np.float32],
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> npt.NDArray[np.float32]: ...

def times_to_samples(
    times: float | npt.NDArray[np.float64],
    sr: int = 22050,
) -> npt.NDArray[np.intp]: ...

def samples_to_time(
    samples: int | npt.NDArray[np.float64],
    sr: int = 22050,
) -> npt.NDArray[np.float32]: ...

def frames_to_samples(
    frames: int | npt.NDArray[np.intp],
    hop_length: int = 512,
    n_fft: int | None = None,
) -> npt.NDArray[np.intp]: ...

def samples_to_frames(
    samples: int | npt.NDArray[np.intp],
    hop_length: int = 512,
    n_fft: int | None = None,
) -> npt.NDArray[np.intp]: ...

def fft_frequencies(
    sr: int = 22050,
    n_fft: int = 2048,
) -> npt.NDArray[np.float32]: ...

def mel_frequencies(
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 11025.0,
) -> npt.NDArray[np.float32]: ...
