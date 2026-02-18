from __future__ import annotations
import numpy as np
import numpy.typing as npt

def viterbi(
    prob: npt.NDArray[np.float32],
    transition: npt.NDArray[np.float32] | None = None,
    initial: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.int64]: ...

def viterbi_discriminative(
    prob: npt.NDArray[np.float32],
    transition: npt.NDArray[np.float32] | None = None,
    initial: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.int64]: ...

def viterbi_binary(
    prob: npt.NDArray[np.float32],
    transition: npt.NDArray[np.float32] | float | None = None,
) -> npt.NDArray[np.int64]: ...
