#!/usr/bin/env python3
"""
Convert madmom's RNNBeatProcessor BiLSTM model to CoreML format.

This script loads a madmom pkl model file using a SafeUnpickler (no madmom
import required) and converts it to a CoreML NeuralNetwork model using the
NeuralNetworkBuilder API, which supports BiLSTM layers with peephole
connections.

Architecture of beats_blstm_1.pkl:
  - Layer 0: BiLSTM (input=266, hidden=25)
  - Layer 1: BiLSTM (input=50,  hidden=25)
  - Layer 2: BiLSTM (input=50,  hidden=25)
  - Layer 3: Dense  (input=50,  output=1, sigmoid activation)

Weight convention:
  - madmom stores weights as (input_dim, hidden_size), using np.dot(data, W)
  - CoreML expects (hidden_size, input_dim), so all input/recurrent weights
    are transposed during conversion
  - CoreML gate order: [input, forget, output, cell] (IFOZ)
  - CoreML peephole order: [input, forget, output] (IFO)

CoreML NeuralNetwork LSTM expects rank-5 tensors:
  [Sequence, Batch, Channels, Height, Width]
  For our model: [Seq, 1, InputDim, 1, 1]
"""

import os
import pickle
import subprocess
import sys

import coremltools as ct
import numpy as np
from coremltools.models.neural_network import NeuralNetworkBuilder


# ---------------------------------------------------------------------------
# SafeUnpickler: load madmom pkl without importing madmom
# ---------------------------------------------------------------------------

class SafeUnpickler(pickle.Unpickler):
    """Unpickler that replaces madmom classes with lightweight stubs."""

    def find_class(self, module, name):
        if "madmom" in module:

            class Stub:
                def __init__(self, *args, **kwargs):
                    pass

                def __setstate__(self, state):
                    self.__dict__.update(
                        state if isinstance(state, dict) else {}
                    )

            Stub.__name__ = name
            Stub.__qualname__ = name
            return Stub
        return super().find_class(module, name)


def load_madmom_pkl(pkl_path):
    """Load a madmom pkl file using SafeUnpickler."""
    with open(pkl_path, "rb") as f:
        return SafeUnpickler(f, encoding="latin1").load()


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------

def extract_lstm_weights(lstm_layer):
    """
    Extract and transpose weights from a madmom LSTMLayer stub.

    Returns a dict with keys:
        W_x: list of 4 arrays [W_i, W_f, W_o, W_z], each (hidden, input)
        W_h: list of 4 arrays [R_i, R_f, R_o, R_z], each (hidden, hidden)
        b:   list of 4 arrays [b_i, b_f, b_o, b_z], each (hidden,)
        peep: list of 3 arrays [p_i, p_f, p_o], each (hidden,) or None
    """
    # CoreML gate order: Input, Forget, Output, Cell (IFOZ)
    gates = [
        lstm_layer.input_gate,    # I
        lstm_layer.forget_gate,   # F
        lstm_layer.output_gate,   # O
        lstm_layer.cell,          # Z (block input / cell candidate)
    ]

    W_x = []
    W_h = []
    b = []
    peep = []

    for g in gates:
        # madmom: (input_dim, hidden) with np.dot(data, W)
        # CoreML: (hidden, input_dim) with W @ data
        W_x.append(g.weights.T.astype(np.float32))
        W_h.append(g.recurrent_weights.T.astype(np.float32))
        b.append(g.bias.flatten().astype(np.float32))

        # Peephole: only input/forget/output gates have them (not cell)
        if hasattr(g, "peephole_weights") and g.peephole_weights is not None:
            peep.append(g.peephole_weights.flatten().astype(np.float32))

    # peep should have 3 entries (I, F, O) -- cell has no peephole
    has_peephole = len(peep) == 3

    return {
        "W_x": W_x,
        "W_h": W_h,
        "b": b,
        "peep": peep if has_peephole else None,
    }


def extract_dense_weights(dense_layer):
    """
    Extract weights from a madmom FeedForwardLayer stub.

    Returns (W, b) where W has shape (output, input) for CoreML.
    """
    # madmom: (input, output) with np.dot(data, W)
    # CoreML: (output, input)
    W = dense_layer.weights.T.astype(np.float32)
    b = dense_layer.bias.flatten().astype(np.float32)
    return W, b


# ---------------------------------------------------------------------------
# CoreML model building
# ---------------------------------------------------------------------------

def build_coreml_model(model):
    """
    Build a CoreML NeuralNetwork model from the loaded madmom stub.

    The CoreML NeuralNetwork LSTM expects rank-5 tensors:
        [Sequence, Batch, Channels, Height, Width]

    We use disable_rank5_shape_mapping=True and specify 5D shapes explicitly.

    Parameters
    ----------
    model : object
        The unpickled madmom NeuralNetwork stub with .layers attribute.

    Returns
    -------
    coremltools.models.MLModel
        The converted CoreML model.
    """
    layers = model.layers
    assert len(layers) == 4, f"Expected 4 layers, got {len(layers)}"

    # Determine dimensions from layer 0 forward input gate weights
    fwd_ig = layers[0].fwd_layer.input_gate
    input_dim = fwd_ig.weights.shape[0]   # 266
    hidden_size = fwd_ig.weights.shape[1]  # 25

    print(f"Input dimension: {input_dim}")
    print(f"Hidden size:     {hidden_size}")

    # --- Build NeuralNetworkBuilder ---
    # Input: rank-5 tensor [Seq, 1, InputDim, 1, 1]
    # We use a default seq_len of 1 and add a flexible range later.
    input_features = [
        ("input", ct.models.datatypes.Array(1, 1, input_dim, 1, 1)),
    ]
    output_features = [("output", ct.models.datatypes.Array(1))]

    builder = NeuralNetworkBuilder(
        input_features, output_features, disable_rank5_shape_mapping=True
    )

    # Make sequence dimension flexible (1 to unbounded)
    spec = builder.spec
    input_desc = spec.description.input[0]
    flex = input_desc.type.multiArrayType.shapeRange
    for lo, hi in [(1, -1), (1, 1), (input_dim, input_dim), (1, 1), (1, 1)]:
        r = flex.sizeRanges.add()
        r.lowerBound = lo
        r.upperBound = hi  # -1 = unbounded

    # Add initial hidden/cell state inputs for the first BiLSTM layer
    for state_name in [
        "input_h", "input_c", "input_h_back", "input_c_back"
    ]:
        inp = spec.description.input.add()
        inp.name = state_name
        inp.type.multiArrayType.dataType = 65600  # FLOAT32
        for dim in [1, 1, hidden_size, 1, 1]:
            inp.type.multiArrayType.shape.append(dim)

    # Track blob names through the network
    prev_output = "input"
    prev_h = "input_h"
    prev_c = "input_c"
    prev_h_back = "input_h_back"
    prev_c_back = "input_c_back"

    # --- BiLSTM layers (0, 1, 2) ---
    for i in range(3):
        bilstm_layer = layers[i]
        fwd_weights = extract_lstm_weights(bilstm_layer.fwd_layer)
        bwd_weights = extract_lstm_weights(bilstm_layer.bwd_layer)

        layer_input_size = fwd_weights["W_x"][0].shape[1]
        layer_hidden = fwd_weights["W_x"][0].shape[0]

        lstm_out = f"bilstm_{i}_out"
        lstm_h = f"bilstm_{i}_h"
        lstm_c = f"bilstm_{i}_c"
        lstm_h_back = f"bilstm_{i}_h_back"
        lstm_c_back = f"bilstm_{i}_c_back"

        print(
            f"BiLSTM layer {i}: input={layer_input_size}, "
            f"hidden={layer_hidden}, "
            f"peephole={'yes' if fwd_weights['peep'] else 'no'}"
        )

        builder.add_bidirlstm(
            name=f"bilstm_{i}",
            W_h=fwd_weights["W_h"],
            W_x=fwd_weights["W_x"],
            b=fwd_weights["b"],
            W_h_back=bwd_weights["W_h"],
            W_x_back=bwd_weights["W_x"],
            b_back=bwd_weights["b"],
            hidden_size=layer_hidden,
            input_size=layer_input_size,
            input_names=[
                prev_output, prev_h, prev_c, prev_h_back, prev_c_back
            ],
            output_names=[
                lstm_out, lstm_h, lstm_c, lstm_h_back, lstm_c_back
            ],
            peep=fwd_weights["peep"],
            peep_back=bwd_weights["peep"],
            output_all=True,
        )

        prev_output = lstm_out
        prev_h = lstm_h
        prev_c = lstm_c
        prev_h_back = lstm_h_back
        prev_c_back = lstm_c_back

    # --- Dense layer (layer 3) ---
    dense = layers[3]
    W_dense, b_dense = extract_dense_weights(dense)
    dense_input_size = W_dense.shape[1]  # 50
    dense_output_size = W_dense.shape[0]  # 1

    print(f"Dense layer: input={dense_input_size}, output={dense_output_size}")

    builder.add_inner_product(
        name="dense",
        W=W_dense,
        b=b_dense,
        input_channels=dense_input_size,
        output_channels=dense_output_size,
        has_bias=True,
        input_name=prev_output,
        output_name="dense_out",
    )

    # Sigmoid activation
    builder.add_activation(
        name="sigmoid",
        non_linearity="SIGMOID",
        input_name="dense_out",
        output_name="output",
    )

    # Build the model
    mlmodel = ct.models.MLModel(builder.spec)
    return mlmodel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Default pkl path
    default_pkl = (
        "/Users/zakkeown/Code/MetalMom/.venv/lib/python3.14/"
        "site-packages/madmom/models/beats/2015/beats_blstm_1.pkl"
    )
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else default_pkl

    if not os.path.exists(pkl_path):
        print(f"ERROR: pkl file not found: {pkl_path}")
        sys.exit(1)

    print(f"Loading model from: {pkl_path}")
    model = load_madmom_pkl(pkl_path)

    print(f"Model type: {type(model).__name__}")
    print(f"Number of layers: {len(model.layers)}")
    print()

    # Build CoreML model
    mlmodel = build_coreml_model(model)

    # Save as .mlmodel (mlpackage requires libmodelpackage)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    mlmodel_path = os.path.join(output_dir, "beats_blstm_1.mlmodel")

    mlmodel.save(mlmodel_path)
    print(f"\nSaved CoreML model to: {mlmodel_path}")

    # Try to compile to .mlmodelc (requires macOS with coremlcompiler)
    try:
        result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", mlmodel_path, output_dir],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            mlmodelc_path = os.path.join(output_dir, "beats_blstm_1.mlmodelc")
            print(f"Compiled to: {mlmodelc_path}")
        else:
            print(f"coremlcompiler error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("xcrun not found; skipping .mlmodelc compilation")
    except Exception as e:
        print(f"Compilation skipped: {e}")

    print("\nConversion complete!")
    return mlmodel_path


if __name__ == "__main__":
    main()
