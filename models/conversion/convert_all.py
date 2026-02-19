#!/usr/bin/env python3
"""
Convert all madmom neural network models to CoreML format.

Discovers all .pkl model files in the madmom installation, classifies each
by architecture type, and converts to .mlmodel files using the appropriate
NeuralNetworkBuilder methods.

Model families:
  - BiLSTM: beats/2015, onsets/2013 (brnn with LSTM), downbeats/2016 (blstm),
            onsets/2014 (brnn_pp with LSTM)
  - LSTM:   beats/2016
  - BiGRU:  downbeats/2016 (bgru)
  - BiRNN:  notes/2013, onsets/2013 (brnn with RecurrentLayer),
            onsets/2014 (brnn_pp with RecurrentLayer)
  - RNN:    onsets/2013 (rnn)
  - CNN:    onsets/2013 (cnn), key/2018, chords/2016 (cnnfeat)
  - DNN:    chroma/2016
  - CRF:    chords/2016 (cnncrf, dccrf) — saved as .npz, not .mlmodel

Usage:
    /Users/zakkeown/Code/MetalMom/.venv/bin/python models/conversion/convert_all.py
"""

import os
import sys
import time
import traceback
from glob import glob
from pathlib import Path

import numpy as np

# Suppress coremltools warnings
import warnings
warnings.filterwarnings("ignore")

import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder

# Import shared loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from madmom_loader import (
    load_model,
    classify_model,
    layer_type_name,
    extract_lstm_weights,
    extract_gru_weights,
    extract_rnn_weights,
    extract_dense_weights,
    extract_conv_weights,
    extract_batchnorm_params,
    extract_pool_params,
    extract_crf_params,
    get_activation_name,
)
from numpy_forward import _compute_min_cnn_spatial, _detect_cnn_freq_dim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MADMOM_MODELS_DIR = (
    "/Users/zakkeown/Code/MetalMom/.venv/lib/python3.14/"
    "site-packages/madmom/models"
)

OUTPUT_BASE_DIR = "/Users/zakkeown/Code/MetalMom/models/converted"


# ---------------------------------------------------------------------------
# Helper: Create NeuralNetworkBuilder with rank-5 input
# ---------------------------------------------------------------------------

def make_builder(input_dim, output_dim, seq_input=True):
    """
    Create a NeuralNetworkBuilder with proper input/output specs.

    For recurrent models (seq_input=True):
        Input:  [Seq, 1, input_dim, 1, 1] with flexible sequence dimension
        Output: [Seq, 1, output_dim, 1, 1]

    For non-recurrent models (seq_input=False):
        Input:  [1, input_dim, 1, 1] (rank-4 or rank-5)
        Output: [output_dim]
    """
    if seq_input:
        input_features = [
            ("input", ct.models.datatypes.Array(1, 1, input_dim, 1, 1)),
        ]
        output_features = [("output", ct.models.datatypes.Array(output_dim))]
        builder = NeuralNetworkBuilder(
            input_features, output_features,
            disable_rank5_shape_mapping=True
        )
        # Make sequence dimension flexible
        spec = builder.spec
        input_desc = spec.description.input[0]
        flex = input_desc.type.multiArrayType.shapeRange
        for lo, hi in [
            (1, -1), (1, 1), (input_dim, input_dim), (1, 1), (1, 1)
        ]:
            r = flex.sizeRanges.add()
            r.lowerBound = lo
            r.upperBound = hi
    else:
        input_features = [
            ("input", ct.models.datatypes.Array(input_dim)),
        ]
        output_features = [("output", ct.models.datatypes.Array(output_dim))]
        builder = NeuralNetworkBuilder(
            input_features, output_features,
            disable_rank5_shape_mapping=True
        )

    return builder


def add_state_inputs(spec, hidden_size, prefix="input"):
    """Add hidden/cell state inputs to the spec for LSTM models."""
    for state_name in [
        f"{prefix}_h", f"{prefix}_c",
        f"{prefix}_h_back", f"{prefix}_c_back",
    ]:
        inp = spec.description.input.add()
        inp.name = state_name
        inp.type.multiArrayType.dataType = 65600  # FLOAT32
        for dim in [1, 1, hidden_size, 1, 1]:
            inp.type.multiArrayType.shape.append(dim)


def add_rnn_state_inputs(spec, hidden_size, prefix="input"):
    """Add hidden state inputs (no cell state) for RNN/GRU models."""
    for state_name in [f"{prefix}_h"]:
        inp = spec.description.input.add()
        inp.name = state_name
        inp.type.multiArrayType.dataType = 65600  # FLOAT32
        for dim in [1, 1, hidden_size, 1, 1]:
            inp.type.multiArrayType.shape.append(dim)


def add_bidir_rnn_state_inputs(spec, hidden_size, prefix="input"):
    """Add forward + backward hidden state inputs for BiRNN/BiGRU."""
    for state_name in [f"{prefix}_h", f"{prefix}_h_back"]:
        inp = spec.description.input.add()
        inp.name = state_name
        inp.type.multiArrayType.dataType = 65600  # FLOAT32
        for dim in [1, 1, hidden_size, 1, 1]:
            inp.type.multiArrayType.shape.append(dim)


# ---------------------------------------------------------------------------
# Converter: BiLSTM models
# ---------------------------------------------------------------------------

def convert_bilstm_model(layers, name, output_dir):
    """
    Convert a BiLSTM model (BidirectionalLayer with LSTMLayer inside).

    Architecture: N BiLSTM layers + 1 Dense layer
    """
    # Find BiLSTM layers and dense layer
    bilstm_layers = []
    dense_layer = None
    for l in layers:
        tn = layer_type_name(l)
        if tn == "BidirectionalLayer":
            bilstm_layers.append(l)
        elif tn == "FeedForwardLayer":
            dense_layer = l

    assert len(bilstm_layers) > 0, "No BiLSTM layers found"
    assert dense_layer is not None, "No Dense layer found"

    # Get dimensions
    fwd0 = bilstm_layers[0].fwd_layer
    input_dim = fwd0.input_gate.weights.shape[0]
    hidden_size = fwd0.input_gate.weights.shape[1]
    W_dense, b_dense = extract_dense_weights(dense_layer)
    output_dim = W_dense.shape[0]

    # Build model
    builder = make_builder(input_dim, output_dim, seq_input=True)

    # Each BiLSTM layer gets its own independent zero-state inputs.
    # (Chaining output states from layer N to layer N+1 would mean layer
    # N+1 starts with the *final* hidden state of layer N, which is wrong
    # for single-shot inference where all layers start with h=0, c=0.)
    for i, bl in enumerate(bilstm_layers):
        layer_h = extract_lstm_weights(bl.fwd_layer)["W_x"][0].shape[0]
        prefix = "input" if i == 0 else f"layer{i}_init"
        add_state_inputs(builder.spec, layer_h, prefix=prefix)

    prev_output = "input"

    for i, bl in enumerate(bilstm_layers):
        fwd_w = extract_lstm_weights(bl.fwd_layer)
        bwd_w = extract_lstm_weights(bl.bwd_layer)

        layer_in = fwd_w["W_x"][0].shape[1]
        layer_h = fwd_w["W_x"][0].shape[0]

        prefix = "input" if i == 0 else f"layer{i}_init"
        state_h = f"{prefix}_h"
        state_c = f"{prefix}_c"
        state_h_b = f"{prefix}_h_back"
        state_c_b = f"{prefix}_c_back"

        out = f"bilstm_{i}_out"
        h = f"bilstm_{i}_h"
        c = f"bilstm_{i}_c"
        h_b = f"bilstm_{i}_h_back"
        c_b = f"bilstm_{i}_c_back"

        # NOTE: peephole weights are dropped because CoreML's Espresso
        # LSTM kernel crashes with peephole on macOS 26.x. The peephole
        # contribution is small for these models.
        builder.add_bidirlstm(
            name=f"bilstm_{i}",
            W_h=fwd_w["W_h"], W_x=fwd_w["W_x"], b=fwd_w["b"],
            W_h_back=bwd_w["W_h"], W_x_back=bwd_w["W_x"], b_back=bwd_w["b"],
            hidden_size=layer_h, input_size=layer_in,
            input_names=[prev_output, state_h, state_c, state_h_b, state_c_b],
            output_names=[out, h, c, h_b, c_b],
            peep=None, peep_back=None,
            output_all=True,
        )
        prev_output = out

    # Dense layer
    builder.add_inner_product(
        name="dense", W=W_dense, b=b_dense,
        input_channels=W_dense.shape[1],
        output_channels=output_dim,
        has_bias=True,
        input_name=prev_output, output_name="dense_out",
    )

    # Activation
    act = get_activation_name(dense_layer)
    if act == "SOFTMAX":
        # CoreML's Softmax layer requires output rank >= 3.
        # Recurrent models produce rank-5 intermediate tensors, but the
        # output spec is declared as rank-1 Array(output_dim).  To work
        # around this, implement softmax manually with exp + reduce_sum +
        # divide, which work at any rank.
        #
        # softmax(x)_i = exp(x_i) / sum(exp(x_j))
        # Applied along channel axis (dim 2 in rank-5 SNCHW format).
        builder.add_unary(
            name="softmax_exp",
            input_name="dense_out",
            output_name="softmax_exp_out",
            mode="exp",
        )
        # Sum along the channel axis (dim 2 in rank-5)
        builder.add_reduce_sum(
            name="softmax_sum",
            input_name="softmax_exp_out",
            output_name="softmax_sum_out",
            axes=[2],
            keepdims=True,
        )
        # Divide exp(x) / sum(exp(x))
        builder.add_divide_broadcastable(
            name="softmax_div",
            input_names=["softmax_exp_out", "softmax_sum_out"],
            output_name="output",
        )
    elif act and act != "LINEAR":
        builder.add_activation(
            name="output_activation", non_linearity=act,
            input_name="dense_out", output_name="output",
        )
    else:
        # If no activation, just rename
        builder.add_activation(
            name="output_activation", non_linearity="LINEAR",
            input_name="dense_out", output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: Unidirectional LSTM models
# ---------------------------------------------------------------------------

def convert_lstm_model(layers, name, output_dir):
    """
    Convert a unidirectional LSTM model.

    Architecture: N LSTMLayer + 1 Dense layer
    """
    lstm_layers = []
    dense_layer = None
    for l in layers:
        tn = layer_type_name(l)
        if tn == "LSTMLayer":
            lstm_layers.append(l)
        elif tn == "FeedForwardLayer":
            dense_layer = l

    assert len(lstm_layers) > 0
    assert dense_layer is not None

    input_dim = lstm_layers[0].input_gate.weights.shape[0]
    hidden_size = lstm_layers[0].input_gate.weights.shape[1]
    W_dense, b_dense = extract_dense_weights(dense_layer)
    output_dim = W_dense.shape[0]

    builder = make_builder(input_dim, output_dim, seq_input=True)

    # Each LSTM layer gets its own independent zero-state inputs.
    for i, ll in enumerate(lstm_layers):
        layer_h = extract_lstm_weights(ll)["W_x"][0].shape[0]
        prefix = "input" if i == 0 else f"layer{i}_init"
        for state_name in [f"{prefix}_h", f"{prefix}_c"]:
            inp = builder.spec.description.input.add()
            inp.name = state_name
            inp.type.multiArrayType.dataType = 65600
            for dim in [1, 1, layer_h, 1, 1]:
                inp.type.multiArrayType.shape.append(dim)

    prev_output = "input"

    for i, ll in enumerate(lstm_layers):
        w = extract_lstm_weights(ll)
        layer_in = w["W_x"][0].shape[1]
        layer_h = w["W_x"][0].shape[0]

        prefix = "input" if i == 0 else f"layer{i}_init"
        state_h = f"{prefix}_h"
        state_c = f"{prefix}_c"

        out = f"lstm_{i}_out"
        h = f"lstm_{i}_h"
        c = f"lstm_{i}_c"

        # NOTE: peephole weights are dropped because CoreML's Espresso
        # LSTM kernel crashes with peephole on macOS 26.x.
        builder.add_unilstm(
            name=f"lstm_{i}",
            W_h=w["W_h"], W_x=w["W_x"], b=w["b"],
            hidden_size=layer_h, input_size=layer_in,
            input_names=[prev_output, state_h, state_c],
            output_names=[out, h, c],
            peep=None,
            output_all=True,
        )
        prev_output = out

    # Dense
    builder.add_inner_product(
        name="dense", W=W_dense, b=b_dense,
        input_channels=W_dense.shape[1], output_channels=output_dim,
        has_bias=True,
        input_name=prev_output, output_name="dense_out",
    )

    act = get_activation_name(dense_layer)
    if act and act not in ("LINEAR", None):
        builder.add_activation(
            name="output_activation", non_linearity=act,
            input_name="dense_out", output_name="output",
        )
    else:
        builder.add_activation(
            name="output_activation", non_linearity="LINEAR",
            input_name="dense_out", output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: BiGRU models (using two unidirectional GRU layers + concat)
# ---------------------------------------------------------------------------

def convert_bigru_model(layers, name, output_dir):
    """
    Convert a BiGRU model.

    CoreML has no add_bidirgru, so we use:
      - add_gru (forward)
      - add_gru (reverse_input=True)
      - add_concat_nd to merge outputs

    Architecture: N BiGRU layers + 1 Dense layer
    """
    bigru_layers = []
    dense_layer = None
    for l in layers:
        tn = layer_type_name(l)
        if tn == "BidirectionalLayer":
            bigru_layers.append(l)
        elif tn == "FeedForwardLayer":
            dense_layer = l

    assert len(bigru_layers) > 0
    assert dense_layer is not None

    fwd0 = bigru_layers[0].fwd_layer
    input_dim = fwd0.reset_gate.weights.shape[0]
    hidden_size = fwd0.reset_gate.weights.shape[1]
    W_dense, b_dense = extract_dense_weights(dense_layer)
    output_dim = W_dense.shape[0]

    builder = make_builder(input_dim, output_dim, seq_input=True)

    # Each BiGRU layer gets its own independent zero-state inputs.
    for i, bl in enumerate(bigru_layers):
        layer_h = extract_gru_weights(bl.fwd_layer)["W_x"][0].shape[0]
        prefix = "input" if i == 0 else f"layer{i}_init"
        add_bidir_rnn_state_inputs(builder.spec, layer_h, prefix=prefix)

    prev_output = "input"

    for i, bl in enumerate(bigru_layers):
        fwd_w = extract_gru_weights(bl.fwd_layer)
        bwd_w = extract_gru_weights(bl.bwd_layer)

        layer_in = fwd_w["W_x"][0].shape[1]
        layer_h = fwd_w["W_x"][0].shape[0]

        prefix = "input" if i == 0 else f"layer{i}_init"
        state_h_fwd = f"{prefix}_h"
        state_h_bwd = f"{prefix}_h_back"

        fwd_out = f"gru_{i}_fwd_out"
        fwd_h = f"gru_{i}_fwd_h"
        bwd_out_raw = f"gru_{i}_bwd_out_raw"
        bwd_out = f"gru_{i}_bwd_out"
        bwd_h = f"gru_{i}_bwd_h"
        concat_out = f"bigru_{i}_out"

        # Forward GRU
        builder.add_gru(
            name=f"gru_{i}_fwd",
            W_h=fwd_w["W_h"], W_x=fwd_w["W_x"], b=fwd_w["b"],
            hidden_size=layer_h, input_size=layer_in,
            input_names=[prev_output, state_h_fwd],
            output_names=[fwd_out, fwd_h],
            activation="TANH", inner_activation="SIGMOID",
            output_all=True, reverse_input=False,
        )

        # Backward GRU (reverse_input=True)
        # CoreML reverse_input only reverses the INPUT, not the output.
        # So the output is in reversed time order and must be flipped
        # before concatenation to align with forward output.
        builder.add_gru(
            name=f"gru_{i}_bwd",
            W_h=bwd_w["W_h"], W_x=bwd_w["W_x"], b=bwd_w["b"],
            hidden_size=layer_h, input_size=layer_in,
            input_names=[prev_output, state_h_bwd],
            output_names=[bwd_out_raw, bwd_h],
            activation="TANH", inner_activation="SIGMOID",
            output_all=True, reverse_input=True,
        )

        # Reverse the backward output along the sequence dimension (dim 0)
        # to align with forward output's time order
        builder.add_reverse(
            name=f"gru_{i}_bwd_rev",
            input_name=bwd_out_raw,
            output_name=bwd_out,
            reverse_dim=[True, False, False, False, False],
        )

        # Concatenate forward and backward outputs along channel axis (axis=2)
        builder.add_concat_nd(
            name=f"bigru_{i}_concat",
            input_names=[fwd_out, bwd_out],
            output_name=concat_out,
            axis=2,
        )

        prev_output = concat_out

    # Dense
    builder.add_inner_product(
        name="dense", W=W_dense, b=b_dense,
        input_channels=W_dense.shape[1], output_channels=output_dim,
        has_bias=True,
        input_name=prev_output, output_name="dense_out",
    )

    act = get_activation_name(dense_layer)
    if act and act not in ("LINEAR", None):
        builder.add_activation(
            name="output_activation", non_linearity=act,
            input_name="dense_out", output_name="output",
        )
    else:
        builder.add_activation(
            name="output_activation", non_linearity="LINEAR",
            input_name="dense_out", output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: BiRNN models (using two unidirectional simple RNN + concat)
# ---------------------------------------------------------------------------

def convert_birnn_model(layers, name, output_dir):
    """
    Convert a BiRNN model (BidirectionalLayer with RecurrentLayer inside).

    CoreML has no add_bidirrnn, so we use:
      - add_simple_rnn (forward)
      - add_simple_rnn (reverse_input=True)
      - add_concat_nd to merge outputs
    """
    birnn_layers = []
    dense_layer = None
    for l in layers:
        tn = layer_type_name(l)
        if tn == "BidirectionalLayer":
            birnn_layers.append(l)
        elif tn == "FeedForwardLayer":
            dense_layer = l

    assert len(birnn_layers) > 0
    assert dense_layer is not None

    fwd0 = birnn_layers[0].fwd_layer
    input_dim = fwd0.weights.shape[0]
    hidden_size = fwd0.weights.shape[1]
    W_dense, b_dense = extract_dense_weights(dense_layer)
    output_dim = W_dense.shape[0]

    builder = make_builder(input_dim, output_dim, seq_input=True)

    # Each BiRNN layer gets its own independent zero-state inputs.
    for i, bl in enumerate(birnn_layers):
        layer_h = extract_rnn_weights(bl.fwd_layer)["W_x"].shape[0]
        prefix = "input" if i == 0 else f"layer{i}_init"
        add_bidir_rnn_state_inputs(builder.spec, layer_h, prefix=prefix)

    prev_output = "input"

    for i, bl in enumerate(birnn_layers):
        fwd_w = extract_rnn_weights(bl.fwd_layer)
        bwd_w = extract_rnn_weights(bl.bwd_layer)

        layer_in = fwd_w["W_x"].shape[1]
        layer_h = fwd_w["W_x"].shape[0]

        prefix = "input" if i == 0 else f"layer{i}_init"
        state_h_fwd = f"{prefix}_h"
        state_h_bwd = f"{prefix}_h_back"

        fwd_out = f"rnn_{i}_fwd_out"
        fwd_h = f"rnn_{i}_fwd_h"
        bwd_out_raw = f"rnn_{i}_bwd_out_raw"
        bwd_out = f"rnn_{i}_bwd_out"
        bwd_h = f"rnn_{i}_bwd_h"
        concat_out = f"birnn_{i}_out"

        # Forward RNN
        builder.add_simple_rnn(
            name=f"rnn_{i}_fwd",
            W_h=fwd_w["W_h"], W_x=fwd_w["W_x"], b=fwd_w["b"],
            hidden_size=layer_h, input_size=layer_in,
            activation="TANH",
            input_names=[prev_output, state_h_fwd],
            output_names=[fwd_out, fwd_h],
            output_all=True, reverse_input=False,
        )

        # Backward RNN (reverse_input only reverses input, not output)
        builder.add_simple_rnn(
            name=f"rnn_{i}_bwd",
            W_h=bwd_w["W_h"], W_x=bwd_w["W_x"], b=bwd_w["b"],
            hidden_size=layer_h, input_size=layer_in,
            activation="TANH",
            input_names=[prev_output, state_h_bwd],
            output_names=[bwd_out_raw, bwd_h],
            output_all=True, reverse_input=True,
        )

        # Reverse backward output along sequence dim to align with forward
        builder.add_reverse(
            name=f"rnn_{i}_bwd_rev",
            input_name=bwd_out_raw,
            output_name=bwd_out,
            reverse_dim=[True, False, False, False, False],
        )

        # Concatenate
        builder.add_concat_nd(
            name=f"birnn_{i}_concat",
            input_names=[fwd_out, bwd_out],
            output_name=concat_out,
            axis=2,
        )

        prev_output = concat_out

    # Dense
    builder.add_inner_product(
        name="dense", W=W_dense, b=b_dense,
        input_channels=W_dense.shape[1], output_channels=output_dim,
        has_bias=True,
        input_name=prev_output, output_name="dense_out",
    )

    act = get_activation_name(dense_layer)
    if act and act not in ("LINEAR", None):
        builder.add_activation(
            name="output_activation", non_linearity=act,
            input_name="dense_out", output_name="output",
        )
    else:
        builder.add_activation(
            name="output_activation", non_linearity="LINEAR",
            input_name="dense_out", output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: Simple RNN models (unidirectional)
# ---------------------------------------------------------------------------

def convert_rnn_model(layers, name, output_dir):
    """
    Convert a unidirectional simple RNN model.

    Architecture: N RecurrentLayer + 1 Dense layer
    """
    rnn_layers = []
    dense_layer = None
    for l in layers:
        tn = layer_type_name(l)
        if tn == "RecurrentLayer":
            rnn_layers.append(l)
        elif tn == "FeedForwardLayer":
            dense_layer = l

    assert len(rnn_layers) > 0
    assert dense_layer is not None

    input_dim = rnn_layers[0].weights.shape[0]
    hidden_size = rnn_layers[0].weights.shape[1]
    W_dense, b_dense = extract_dense_weights(dense_layer)
    output_dim = W_dense.shape[0]

    builder = make_builder(input_dim, output_dim, seq_input=True)

    # Each RNN layer gets its own independent zero-state inputs.
    for i, rl in enumerate(rnn_layers):
        layer_h = extract_rnn_weights(rl)["W_x"].shape[0]
        prefix = "input" if i == 0 else f"layer{i}_init"
        add_rnn_state_inputs(builder.spec, layer_h, prefix=prefix)

    prev_output = "input"

    for i, rl in enumerate(rnn_layers):
        w = extract_rnn_weights(rl)
        layer_in = w["W_x"].shape[1]
        layer_h = w["W_x"].shape[0]

        prefix = "input" if i == 0 else f"layer{i}_init"
        state_h = f"{prefix}_h"

        out = f"rnn_{i}_out"
        h = f"rnn_{i}_h"

        builder.add_simple_rnn(
            name=f"rnn_{i}",
            W_h=w["W_h"], W_x=w["W_x"], b=w["b"],
            hidden_size=layer_h, input_size=layer_in,
            activation="TANH",
            input_names=[prev_output, state_h],
            output_names=[out, h],
            output_all=True, reverse_input=False,
        )
        prev_output = out

    # Dense
    builder.add_inner_product(
        name="dense", W=W_dense, b=b_dense,
        input_channels=W_dense.shape[1], output_channels=output_dim,
        has_bias=True,
        input_name=prev_output, output_name="dense_out",
    )

    act = get_activation_name(dense_layer)
    if act and act not in ("LINEAR", None):
        builder.add_activation(
            name="output_activation", non_linearity=act,
            input_name="dense_out", output_name="output",
        )
    else:
        builder.add_activation(
            name="output_activation", non_linearity="LINEAR",
            input_name="dense_out", output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: CNN models
# ---------------------------------------------------------------------------

def _compute_cnn_coreml_input(layers):
    """Compute the CoreML input shape (C, min_h, default_h, W) and skip_bn.

    Returns
    -------
    (C, min_h, default_h, W, skip_bn) : tuple
        C, W are spatial dimensions (excluding batch).
        min_h is the minimum H for the flex range lower bound.
        default_h is the default H for the Array declaration (larger than
        min_h to avoid CoreML inference issues when actual input > default).
        skip_bn=True for onsets_cnn (3D BN applied in numpy preprocessing).
    """
    first_name = layer_type_name(layers[0])

    # Detect 3D BatchNorm = onsets_cnn
    if first_name == "BatchNormLayer":
        bn = layers[0]
        if hasattr(bn, 'mean') and bn.mean.ndim == 2:
            num_specs = bn.mean.shape[1]   # 3 (channels)
            freq_bins = bn.mean.shape[0]   # 80
            return num_specs, 15, 15, freq_bins, True

    # Standard CNN (key_cnn, chords_cnnfeat)
    first_conv = None
    for l in layers:
        if layer_type_name(l) == "ConvolutionalLayer":
            first_conv = l
            break
    assert first_conv is not None, "No ConvolutionalLayer found"

    in_channels = first_conv.weights.shape[0]
    min_h, _ = _compute_min_cnn_spatial(layers)
    freq_dim = _detect_cnn_freq_dim(layers, in_channels, min_h)

    # Use a larger default H for the Array declaration. CoreML's runtime
    # produces incorrect results when the declared default shape is much
    # smaller than the actual inference input (e.g. default=8, actual=50).
    # Setting the default to 200 ensures typical inputs are at or below
    # the default, while the flex range still allows min_h and above.
    default_h = max(min_h * 8, 200)

    return in_channels, min_h, default_h, freq_dim, False


def convert_cnn_model(layers, name, output_dir):
    """
    Convert a CNN model. Handles:
      - PadLayer -> add_padding
      - ConvolutionalLayer -> add_convolution
      - BatchNormLayer -> add_batchnorm + activation
      - MaxPoolLayer -> add_pooling
      - StrideLayer -> permute + flatten (correct feature ordering)
      - AverageLayer -> global average pooling
      - FeedForwardLayer -> add_inner_product + activation

    The CNN models operate on 2D spectrograms.
    Input shape: rank-5 [1, C, H, W, 1] where C=channels, H=time, W=freq.

    For onsets_cnn, the 3D BatchNorm is skipped (applied in numpy preprocessing
    before feeding data to CoreML).
    """
    # Compute proper input shape
    C, min_h, default_h, W, skip_bn = _compute_cnn_coreml_input(layers)

    # Use rank-4 input for CoreML CNN: (1, C, H, W)
    # CoreML Conv layers require rank >= 4. Rank-4 avoids the rank-5
    # pooling issue where the D dimension gets reduced to 0.
    # default_h is larger than min_h to avoid CoreML inference issues.
    input_features = [
        ("input", ct.models.datatypes.Array(1, C, default_h, W)),
    ]
    output_features = [("output", ct.models.datatypes.Array(1))]

    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True
    )

    # For standard CNN (not onsets), make H dimension flexible so parity
    # tests can feed any seq_len >= min_h.
    if not skip_bn:
        spec = builder.spec
        input_desc = spec.description.input[0]
        flex = input_desc.type.multiArrayType.shapeRange
        for lo, hi in [
            (1, 1), (C, C), (min_h, -1), (W, W)
        ]:
            r = flex.sizeRanges.add()
            r.lowerBound = lo
            r.upperBound = hi

    prev_name = "input"
    layer_idx = 0
    needs_flatten = False

    # Skip 3D BN for onsets_cnn (applied in numpy preprocessing)
    layer_list = layers[1:] if skip_bn else layers

    for i, l in enumerate(layer_list):
        tn = layer_type_name(l)

        if tn == "PadLayer":
            width = int(l.width)
            value = float(getattr(l, "value", 0.0))
            out_name = f"pad_{layer_idx}"
            builder.add_padding(
                name=f"pad_{layer_idx}",
                left=width, right=width, top=width, bottom=width,
                value=value,
                input_name=prev_name, output_name=out_name,
                padding_type="constant",
            )
            prev_name = out_name
            layer_idx += 1

        elif tn == "ConvolutionalLayer":
            W_conv, b, kh, kw, in_ch, out_ch, stride, pad_mode = \
                extract_conv_weights(l)
            has_nonzero_bias = np.any(b != 0)
            out_name = f"conv_{layer_idx}"
            builder.add_convolution(
                name=f"conv_{layer_idx}",
                kernel_channels=in_ch, output_channels=out_ch,
                height=kh, width=kw,
                stride_height=stride, stride_width=stride,
                border_mode="valid",  # madmom CNNs all use 'valid'
                groups=1,
                W=W_conv, b=b, has_bias=has_nonzero_bias,
                input_name=prev_name, output_name=out_name,
            )
            prev_name = out_name
            layer_idx += 1

            # Check if conv has an activation
            act = get_activation_name(l)
            if act and act not in ("LINEAR", None):
                act_name = f"conv_act_{layer_idx}"
                if act == "ELU":
                    builder.add_activation(
                        name=act_name, non_linearity="ELU",
                        input_name=prev_name, output_name=f"act_{layer_idx}",
                        params=1.0,
                    )
                else:
                    builder.add_activation(
                        name=act_name, non_linearity=act,
                        input_name=prev_name, output_name=f"act_{layer_idx}",
                    )
                prev_name = f"act_{layer_idx}"
                layer_idx += 1

        elif tn == "BatchNormLayer":
            channels, gamma, beta, mean, variance, act = \
                extract_batchnorm_params(l)
            out_name = f"bn_{layer_idx}"
            builder.add_batchnorm(
                name=f"bn_{layer_idx}",
                channels=channels,
                gamma=gamma, beta=beta,
                mean=mean, variance=variance,
                input_name=prev_name, output_name=out_name,
            )
            prev_name = out_name
            layer_idx += 1

            # BatchNorm activation
            if act and act not in ("LINEAR", None):
                act_out = f"bn_act_{layer_idx}"
                if act == "ELU":
                    builder.add_activation(
                        name=f"bn_act_{layer_idx}", non_linearity="ELU",
                        input_name=prev_name, output_name=act_out,
                        params=1.0,
                    )
                else:
                    builder.add_activation(
                        name=f"bn_act_{layer_idx}", non_linearity=act,
                        input_name=prev_name, output_name=act_out,
                    )
                prev_name = act_out
                layer_idx += 1

        elif tn == "MaxPoolLayer":
            sh, sw, strh, strw = extract_pool_params(l)
            out_name = f"pool_{layer_idx}"
            builder.add_pooling(
                name=f"pool_{layer_idx}",
                height=sh, width=sw,
                stride_height=strh, stride_width=strw,
                layer_type="MAX", padding_type="VALID",
                input_name=prev_name, output_name=out_name,
            )
            prev_name = out_name
            layer_idx += 1

        elif tn == "StrideLayer":
            # StrideLayer in madmom: transpose(0,2,1,3) then flatten.
            # The numpy code does: (N,C,H,W) -> transpose -> (N,H,C,W)
            # -> reshape to (H, C*W) -> sliding window.
            # When H == block_size (T_out=1), this is just permute+flatten.
            perm_name = f"perm_{layer_idx}"
            builder.add_permute(
                name=perm_name,
                dim=(0, 2, 1, 3),
                input_name=prev_name, output_name=perm_name,
            )
            prev_name = perm_name
            layer_idx += 1

            out_name = f"flatten_{layer_idx}"
            builder.add_flatten_to_2d(
                name=f"flatten_{layer_idx}",
                input_name=prev_name, output_name=out_name,
                axis=1,
            )
            prev_name = out_name
            layer_idx += 1
            needs_flatten = True

        elif tn == "AverageLayer":
            # Global average pooling over spatial dimensions
            out_name = f"avg_pool_{layer_idx}"
            builder.add_pooling(
                name=f"avg_pool_{layer_idx}",
                height=1, width=1,
                stride_height=1, stride_width=1,
                layer_type="AVERAGE", padding_type="VALID",
                input_name=prev_name, output_name=out_name,
                is_global=True,
            )
            prev_name = out_name
            layer_idx += 1

        elif tn == "FeedForwardLayer":
            # If we haven't flattened yet, do it now
            if not needs_flatten:
                flat_name = f"flatten_{layer_idx}"
                builder.add_flatten_to_2d(
                    name=flat_name,
                    input_name=prev_name, output_name=flat_name,
                    axis=1,
                )
                prev_name = flat_name
                layer_idx += 1
                needs_flatten = True

            W_dense, b = extract_dense_weights(l)
            in_ch = W_dense.shape[1]
            out_ch = W_dense.shape[0]

            # Check if this is the last layer
            remaining = [
                ll for ll in layer_list[i + 1:]
                if layer_type_name(ll) == "FeedForwardLayer"
            ]
            is_last = len(remaining) == 0

            out_name = "dense_out" if is_last else f"dense_{layer_idx}"
            builder.add_inner_product(
                name=f"dense_{layer_idx}",
                W=W_dense, b=b,
                input_channels=in_ch, output_channels=out_ch,
                has_bias=True,
                input_name=prev_name, output_name=out_name,
            )
            prev_name = out_name
            layer_idx += 1

            act = get_activation_name(l)
            if is_last:
                if act and act not in ("LINEAR", None):
                    builder.add_activation(
                        name="output_activation", non_linearity=act,
                        input_name=prev_name, output_name="output",
                    )
                else:
                    builder.add_activation(
                        name="output_activation", non_linearity="LINEAR",
                        input_name=prev_name, output_name="output",
                    )
                prev_name = "output"
            else:
                if act and act not in ("LINEAR", None):
                    act_out = f"dense_act_{layer_idx}"
                    builder.add_activation(
                        name=f"dense_act_{layer_idx}", non_linearity=act,
                        input_name=prev_name, output_name=act_out,
                    )
                    prev_name = act_out
                    layer_idx += 1
        else:
            print(f"  WARNING: Unknown layer type '{tn}' at index {i}, skipping")

    # If the last layer was not a Dense with output_name="output",
    # add a rename/identity layer
    if prev_name != "output":
        builder.add_activation(
            name="final_identity", non_linearity="LINEAR",
            input_name=prev_name, output_name="output",
        )

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: DNN (pure dense) models
# ---------------------------------------------------------------------------

def convert_dnn_model(layers, name, output_dir):
    """
    Convert a pure Dense/DNN model.

    Architecture: N FeedForwardLayer layers
    """
    dense_layers = [l for l in layers if layer_type_name(l) == "FeedForwardLayer"]
    assert len(dense_layers) > 0

    input_dim = dense_layers[0].weights.shape[0]
    output_dim = dense_layers[-1].weights.shape[1]

    builder = make_builder(input_dim, output_dim, seq_input=False)

    prev_name = "input"

    for i, dl in enumerate(dense_layers):
        W, b = extract_dense_weights(dl)
        in_ch = W.shape[1]
        out_ch = W.shape[0]
        is_last = (i == len(dense_layers) - 1)

        out_name = "dense_out" if is_last else f"dense_{i}_out"
        builder.add_inner_product(
            name=f"dense_{i}",
            W=W, b=b,
            input_channels=in_ch, output_channels=out_ch,
            has_bias=True,
            input_name=prev_name, output_name=out_name,
        )
        prev_name = out_name

        act = get_activation_name(dl)
        if is_last:
            if act == "SOFTMAX":
                builder.add_softmax(
                    name="output_softmax",
                    input_name=prev_name, output_name="output",
                )
            elif act and act not in ("LINEAR", None):
                builder.add_activation(
                    name="output_activation", non_linearity=act,
                    input_name=prev_name, output_name="output",
                )
            else:
                builder.add_activation(
                    name="output_activation", non_linearity="LINEAR",
                    input_name=prev_name, output_name="output",
                )
            prev_name = "output"
        else:
            if act and act not in ("LINEAR", None):
                act_out = f"dense_{i}_act"
                builder.add_activation(
                    name=f"dense_{i}_act", non_linearity=act,
                    input_name=prev_name, output_name=act_out,
                )
                prev_name = act_out

    mlmodel = ct.models.MLModel(builder.spec)
    out_path = os.path.join(output_dir, f"{name}.mlmodel")
    mlmodel.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Converter: CRF parameter files (save as .npz, not .mlmodel)
# ---------------------------------------------------------------------------

def save_crf_params(model, name, output_dir):
    """
    Save CRF parameters as a .npz file (not a CoreML model).

    CRF parameters: A (transition), tau, c, pi, W (emission weights)
    """
    params = extract_crf_params(model)
    out_path = os.path.join(output_dir, f"{name}.npz")
    np.savez(out_path, **params)
    return out_path


# ---------------------------------------------------------------------------
# Model discovery and routing
# ---------------------------------------------------------------------------

def discover_models():
    """
    Discover all .pkl model files in the madmom installation.

    Returns a list of (pkl_path, family, model_name) tuples.
    Skips patterns/ directory (not neural networks).
    """
    results = []
    for pkl_path in sorted(glob(os.path.join(MADMOM_MODELS_DIR, "**/*.pkl"),
                                recursive=True)):
        # Skip patterns directory
        if "/patterns/" in pkl_path:
            continue

        # Derive family and name from path
        rel = os.path.relpath(pkl_path, MADMOM_MODELS_DIR)
        parts = rel.replace(".pkl", "").split(os.sep)
        # e.g., beats/2015/beats_blstm_1 -> family=beats, name=beats_blstm_1
        family = parts[0]
        model_name = parts[-1]

        results.append((pkl_path, family, model_name))

    return results


def route_and_convert(pkl_path, family, model_name, base_output_dir):
    """
    Load a model, classify it, route to the appropriate converter.

    Returns (output_path, model_type) on success, raises on failure.
    """
    model = load_model(pkl_path)
    model_type, data = classify_model(model)

    # Determine output subdirectory
    output_dir = os.path.join(base_output_dir, family)
    os.makedirs(output_dir, exist_ok=True)

    if model_type == "bilstm":
        path = convert_bilstm_model(data, model_name, output_dir)
    elif model_type == "lstm":
        path = convert_lstm_model(data, model_name, output_dir)
    elif model_type == "bigru":
        path = convert_bigru_model(data, model_name, output_dir)
    elif model_type == "birnn":
        path = convert_birnn_model(data, model_name, output_dir)
    elif model_type == "rnn":
        path = convert_rnn_model(data, model_name, output_dir)
    elif model_type == "cnn":
        path = convert_cnn_model(data, model_name, output_dir)
    elif model_type == "dnn":
        path = convert_dnn_model(data, model_name, output_dir)
    elif model_type == "crf":
        path = save_crf_params(data, model_name, output_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return path, model_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("madmom -> CoreML Model Conversion")
    print("=" * 70)
    print(f"Source: {MADMOM_MODELS_DIR}")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print()

    models = discover_models()
    print(f"Discovered {len(models)} model files\n")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    results = {"success": [], "failed": []}
    start_time = time.time()

    for i, (pkl_path, family, model_name) in enumerate(models, 1):
        rel = os.path.relpath(pkl_path, MADMOM_MODELS_DIR)
        print(f"[{i}/{len(models)}] {rel}")

        try:
            out_path, model_type = route_and_convert(
                pkl_path, family, model_name, OUTPUT_BASE_DIR
            )
            size = os.path.getsize(out_path)
            ext = os.path.splitext(out_path)[1]
            print(f"  -> {model_type} -> {os.path.basename(out_path)} "
                  f"({size:,} bytes)")
            results["success"].append((model_name, model_type, out_path, size))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results["failed"].append((model_name, str(e)))

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"RESULTS: {len(results['success'])} succeeded, "
          f"{len(results['failed'])} failed ({elapsed:.1f}s)")
    print("=" * 70)

    if results["success"]:
        print(f"\nSuccessful conversions ({len(results['success'])}):")
        by_type = {}
        for name, mtype, path, size in results["success"]:
            by_type.setdefault(mtype, []).append((name, size))
        for mtype in sorted(by_type):
            items = by_type[mtype]
            total_size = sum(s for _, s in items)
            print(f"  {mtype}: {len(items)} models "
                  f"({total_size:,} bytes total)")
            for name, size in items:
                print(f"    - {name} ({size:,} bytes)")

    if results["failed"]:
        print(f"\nFailed conversions ({len(results['failed'])}):")
        for name, err in results["failed"]:
            print(f"  - {name}: {err}")

    return len(results["failed"]) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
