from __future__ import annotations
import numpy as np
import numpy.typing as npt

def recurrence_matrix(
    features: npt.NDArray[np.float32],
    mode: str = "knn",
    k: int = 5,
    threshold: float | None = None,
    metric: str = "euclidean",
    symmetric: bool = False,
) -> npt.NDArray[np.float32]: ...

def cross_similarity(
    features_a: npt.NDArray[np.float32],
    features_b: npt.NDArray[np.float32],
    metric: str = "euclidean",
) -> npt.NDArray[np.float32]: ...

def rqa(
    rec_matrix: npt.NDArray[np.float32],
    lmin: int = 2,
    vmin: int = 2,
) -> dict[str, float | int]: ...

def dtw(
    cost_matrix: npt.NDArray[np.float32] | None = None,
    X: npt.NDArray[np.float32] | None = None,
    Y: npt.NDArray[np.float32] | None = None,
    metric: str = "euclidean",
    step_pattern: str = "standard",
    band_width: int | None = None,
) -> dict[str, npt.NDArray[np.float32] | npt.NDArray[np.int64] | float]: ...

def agglomerative(
    features: npt.NDArray[np.float32],
    n_segments: int,
) -> npt.NDArray[np.int64]: ...
