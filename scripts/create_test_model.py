#!/usr/bin/env python3
"""Create trivial CoreML test models for InferenceEngine unit tests.

Models created:
  - test_identity.mlpackage : output = input  (shape [1, 4])
  - test_double.mlpackage   : output = 2 * input  (shape [1, 4])

Usage:
    python scripts/create_test_model.py

Note: Uses neuralnetwork convert_to for Python 3.14 compatibility (mlprogram
backend requires native C extensions that are not yet built for 3.14).
"""

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import os

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "Tests", "fixtures")


def create_identity_model():
    """Create a model that returns its input unchanged: output = input."""

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
    def prog(input):
        return mb.identity(x=input, name="output")

    model = ct.convert(prog, convert_to="neuralnetwork")
    path = os.path.join(FIXTURE_DIR, "test_identity.mlmodel")
    model.save(path)
    print(f"Saved: {path}")


def create_double_model():
    """Create a model that doubles its input: output = 2 * input."""

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
    def prog(input):
        two = mb.const(val=np.float32(2.0))
        return mb.mul(x=input, y=two, name="output")

    model = ct.convert(prog, convert_to="neuralnetwork")
    path = os.path.join(FIXTURE_DIR, "test_double.mlmodel")
    model.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    create_identity_model()
    create_double_model()
    print("Done. Test models created in Tests/fixtures/")
