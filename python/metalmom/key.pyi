from __future__ import annotations
import numpy as np
import numpy.typing as npt

KEY_LABELS: list[str]

def key_detect(
    activations: npt.NDArray[np.float32],
    sr: int = 22050,
) -> dict[str, int | str | bool | float | npt.NDArray[np.float32]]: ...
