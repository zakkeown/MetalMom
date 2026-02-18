from __future__ import annotations
import numpy as np
import numpy.typing as npt

CHORD_LABELS: list[str]

def chord_detect(
    activations: npt.NDArray[np.float32],
    n_classes: int = 25,
    transition_scores: npt.NDArray[np.float32] | None = None,
    self_transition_bias: float = 1.0,
    fps: float = 100.0,
    units: str = 'frames',
) -> list[dict[str, float | int | str]]: ...
