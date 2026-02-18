from __future__ import annotations
import numpy as np
import numpy.typing as npt

def midi_to_note_name(midi_note: int) -> str: ...

def piano_transcribe(
    activations: npt.NDArray[np.float32],
    threshold: float = 0.5,
    min_duration: int = 3,
    use_hmm: bool = False,
    fps: float = 100.0,
    units: str = 'frames',
) -> list[dict[str, int | str | float]]: ...
