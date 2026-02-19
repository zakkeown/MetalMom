"""CoreML model inference via the MetalMom C bridge."""

import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy


def predict_model(model_path, input_array):
    """Run CoreML inference on an .mlmodel or .mlmodelc file.

    Parameters
    ----------
    model_path : str
        Path to a CoreML model file (.mlmodel) or compiled model
        directory (.mlmodelc).
    input_array : np.ndarray
        Input data as a NumPy array. Will be converted to float32
        and flattened for the C bridge. The original shape is passed
        as the input shape.

    Returns
    -------
    np.ndarray
        Model output as a NumPy array.

    Raises
    ------
    ValueError
        If model_path is empty or input_array is empty.
    RuntimeError
        If the C bridge call fails (e.g., invalid model, prediction error).
    """
    if not model_path:
        raise ValueError("model_path must not be empty")

    input_array = np.ascontiguousarray(input_array, dtype=np.float32)
    if input_array.size == 0:
        raise ValueError("input_array must not be empty")

    # Encode model path as bytes
    path_bytes = model_path.encode("utf-8")

    # Prepare input shape as int32 array
    shape = np.array(input_array.shape, dtype=np.int32)
    shape_ptr = ffi.cast("const int32_t*", shape.ctypes.data)

    # Flatten input data
    flat = input_array.ravel()
    data_ptr = ffi.cast("const float*", flat.ctypes.data)

    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")

    try:
        out = ffi.new("MMBuffer*")

        status = lib.mm_model_predict(
            ctx,
            path_bytes,
            data_ptr,
            shape_ptr,
            len(shape),
            len(flat),
            out,
        )

        if status != 0:
            error_names = {
                -1: "MM_ERR_INVALID_INPUT",
                -4: "MM_ERR_INTERNAL",
            }
            err_name = error_names.get(status, f"error code {status}")
            raise RuntimeError(
                f"mm_model_predict failed with {err_name} (status={status})"
            )

        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
