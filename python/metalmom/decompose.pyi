from __future__ import annotations
import numpy as np
import numpy.typing as npt

def nmf(
    V: npt.NDArray[np.float32],
    n_components: int = 8,
    n_iter: int = 200,
    objective: str = "euclidean",
    sr: int = 22050,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

def nn_filter(
    S: npt.NDArray[np.float32],
    k: int = 10,
    metric: str = "cosine",
    aggregate: str = "mean",
    exclude_self: bool = True,
    sr: int = 22050,
) -> npt.NDArray[np.float32]: ...
