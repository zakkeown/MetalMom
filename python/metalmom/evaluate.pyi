from __future__ import annotations
import numpy as np
import numpy.typing as npt

def onset_evaluate(
    reference: npt.NDArray[np.float32],
    estimated: npt.NDArray[np.float32],
    window: float = 0.05,
) -> dict[str, float]: ...

def beat_evaluate(
    reference: npt.NDArray[np.float32],
    estimated: npt.NDArray[np.float32],
    fmeasure_window: float = 0.07,
) -> dict[str, float]: ...

def tempo_evaluate(
    ref_tempo: float,
    est_tempo: float,
    tolerance: float = 0.08,
) -> dict[str, float]: ...

def chord_accuracy(
    reference: npt.NDArray[np.int32],
    estimated: npt.NDArray[np.int32],
) -> float: ...
