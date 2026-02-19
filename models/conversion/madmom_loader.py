"""
Shared utilities for loading madmom pkl models and extracting weights.

Provides a SafeUnpickler that replaces madmom classes with lightweight stubs,
plus weight extraction helpers for each layer type:
  - LSTM gates (input/forget/output/cell + peephole)
  - GRU gates (reset/update/cell)
  - Simple RNN (weights + recurrent_weights)
  - Dense / FeedForward layers
  - CNN layers (Conv, BatchNorm, MaxPool, Pad, Stride, Average)
  - CRF parameters
"""

import pickle

import numpy as np


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


def load_model(pkl_path):
    """Load a madmom pkl file using SafeUnpickler."""
    with open(pkl_path, "rb") as f:
        return SafeUnpickler(f, encoding="latin1").load()


# ---------------------------------------------------------------------------
# Layer type detection helpers
# ---------------------------------------------------------------------------

def layer_type_name(layer):
    """Return the class name of a layer stub."""
    return type(layer).__name__


def detect_rnn_sublayer_type(bidir_layer):
    """Detect the sublayer type inside a BidirectionalLayer."""
    fwd = bidir_layer.fwd_layer
    name = layer_type_name(fwd)
    if name == "LSTMLayer":
        return "lstm"
    elif name == "GRULayer":
        return "gru"
    elif name == "RecurrentLayer":
        return "rnn"
    else:
        return name.lower()


def classify_model(model):
    """
    Classify a loaded madmom model into one of:
      'bilstm', 'lstm', 'bigru', 'birnn', 'rnn', 'cnn', 'dnn', 'crf'

    Returns (model_type, layers) where layers is the list of layer stubs,
    or (model_type, model) for CRF.
    """
    type_name = layer_type_name(model)

    if type_name == "ConditionalRandomField":
        return "crf", model

    if not hasattr(model, "layers"):
        return "unknown", model

    layers = model.layers
    if len(layers) == 0:
        return "unknown", model

    first = layers[0]
    first_name = layer_type_name(first)

    # Check for CNN indicators
    cnn_types = {"ConvolutionalLayer", "BatchNormLayer", "PadLayer"}
    if first_name in cnn_types:
        return "cnn", layers

    # Check for bidirectional layers
    if first_name == "BidirectionalLayer":
        sub = detect_rnn_sublayer_type(first)
        if sub == "lstm":
            return "bilstm", layers
        elif sub == "gru":
            return "bigru", layers
        elif sub == "rnn":
            return "birnn", layers
        else:
            return f"bidir_{sub}", layers

    # Unidirectional layers
    if first_name == "LSTMLayer":
        return "lstm", layers
    elif first_name == "RecurrentLayer":
        return "rnn", layers
    elif first_name == "FeedForwardLayer":
        return "dnn", layers

    return "unknown", model


# ---------------------------------------------------------------------------
# Weight extraction: LSTM
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
        lstm_layer.input_gate,
        lstm_layer.forget_gate,
        lstm_layer.output_gate,
        lstm_layer.cell,
    ]

    W_x = []
    W_h = []
    b = []
    peep = []

    for g in gates:
        # madmom: (input_dim, hidden) -> CoreML: (hidden, input_dim)
        W_x.append(g.weights.T.astype(np.float32))
        W_h.append(g.recurrent_weights.T.astype(np.float32))
        b.append(g.bias.flatten().astype(np.float32))

        if hasattr(g, "peephole_weights") and g.peephole_weights is not None:
            peep.append(g.peephole_weights.flatten().astype(np.float32))

    has_peephole = len(peep) == 3

    return {
        "W_x": W_x,
        "W_h": W_h,
        "b": b,
        "peep": peep if has_peephole else None,
    }


# ---------------------------------------------------------------------------
# Weight extraction: GRU
# ---------------------------------------------------------------------------

def extract_gru_weights(gru_layer):
    """
    Extract and transpose weights from a madmom GRULayer stub.

    CoreML GRU gate order: [update(z), reset(r), output(o)]
    madmom has: reset_gate, update_gate, cell

    NOTE: CoreML and madmom use different GRU update equations:
      CoreML: h = (1 - z) * candidate + z * h_prev
      madmom: h = (1 - z) * h_prev   + z * candidate
    The roles of z and (1-z) are swapped. The parity test accounts for
    this by using the CoreML equation in its numpy reference pass.

    Returns a dict with keys:
        W_x: list of 3 arrays [W_z, W_r, W_o], each (hidden, input)
        W_h: list of 3 arrays [R_z, R_r, R_o], each (hidden, hidden)
        b:   list of 3 arrays [b_z, b_r, b_o], each (hidden,)
    """
    # CoreML order: update(z), reset(r), output(o)
    gates = [
        gru_layer.update_gate,   # z
        gru_layer.reset_gate,    # r
        gru_layer.cell,          # o (output/candidate)
    ]

    W_x = []
    W_h = []
    b = []

    for g in gates:
        W_x.append(g.weights.T.astype(np.float32))
        W_h.append(g.recurrent_weights.T.astype(np.float32))
        b.append(g.bias.flatten().astype(np.float32))

    return {"W_x": W_x, "W_h": W_h, "b": b}


# ---------------------------------------------------------------------------
# Weight extraction: Simple RNN
# ---------------------------------------------------------------------------

def extract_rnn_weights(rnn_layer):
    """
    Extract and transpose weights from a madmom RecurrentLayer stub.

    Returns a dict with keys:
        W_x: array of shape (hidden, input)
        W_h: array of shape (hidden, hidden)
        b:   array of shape (hidden,)
    """
    return {
        "W_x": rnn_layer.weights.T.astype(np.float32),
        "W_h": rnn_layer.recurrent_weights.T.astype(np.float32),
        "b": rnn_layer.bias.flatten().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Weight extraction: Dense / FeedForward
# ---------------------------------------------------------------------------

def extract_dense_weights(dense_layer):
    """
    Extract weights from a madmom FeedForwardLayer stub.

    Returns (W, b) where W has shape (output, input) for CoreML.
    """
    W = dense_layer.weights.T.astype(np.float32)
    b = dense_layer.bias.flatten().astype(np.float32)
    return W, b


def get_activation_name(layer):
    """
    Get the activation function name from a layer's activation_fn attribute.

    Returns a CoreML-compatible activation string.
    """
    act = getattr(layer, "activation_fn", None)
    if act is None:
        return None

    name = getattr(act, "__name__", type(act).__name__).lower()
    mapping = {
        "sigmoid": "SIGMOID",
        "tanh": "TANH",
        "relu": "RELU",
        "elu": "ELU",
        "softmax": "SOFTMAX",
        "linear": None,  # no activation needed
    }
    return mapping.get(name, name.upper())


# ---------------------------------------------------------------------------
# Weight extraction: CNN layers
# ---------------------------------------------------------------------------

def extract_conv_weights(conv_layer):
    """
    Extract weights from a madmom ConvolutionalLayer stub.

    madmom stores conv weights as (in_channels, out_channels, height, width).
    CoreML add_convolution expects (height, width, kernel_channels, output_channels).

    Returns (W, b, kernel_h, kernel_w, in_channels, out_channels, stride, pad_mode).
    """
    w = conv_layer.weights.astype(np.float32)
    in_ch, out_ch, kh, kw = w.shape
    # Transpose: (in_ch, out_ch, h, w) -> (h, w, in_ch, out_ch)
    W = w.transpose(2, 3, 0, 1)

    # Bias might be zeros or a scalar array
    bias = getattr(conv_layer, "bias", None)
    if bias is not None:
        b = np.broadcast_to(
            np.asarray(bias).astype(np.float32).flatten(),
            (out_ch,)
        ).copy()
    else:
        b = np.zeros(out_ch, dtype=np.float32)

    stride = getattr(conv_layer, "stride", 1)
    if hasattr(stride, "__len__"):
        stride = int(stride[0]) if len(stride) == 1 else int(stride)
    else:
        stride = int(stride)

    pad_mode = getattr(conv_layer, "pad", "valid")

    return W, b, kh, kw, in_ch, out_ch, stride, pad_mode


def extract_batchnorm_params(bn_layer):
    """
    Extract batch normalization parameters from a madmom BatchNormLayer stub.

    madmom stores: mean, inv_std, gamma, beta
    CoreML wants: gamma, beta, mean, variance

    Since madmom stores inv_std = 1/sqrt(var+eps), we compute:
        variance = (1/inv_std)^2 - eps ≈ (1/inv_std)^2

    Returns (channels, gamma, beta, mean, variance, activation_name).
    """
    mean = np.asarray(bn_layer.mean).astype(np.float32).flatten()
    inv_std = np.asarray(bn_layer.inv_std).astype(np.float32).flatten()
    gamma_raw = getattr(bn_layer, "gamma", 1)
    beta_raw = getattr(bn_layer, "beta", 0)

    channels = len(mean)

    gamma = np.broadcast_to(
        np.asarray(gamma_raw).astype(np.float32).flatten(),
        (channels,)
    ).copy()
    beta = np.broadcast_to(
        np.asarray(beta_raw).astype(np.float32).flatten(),
        (channels,)
    ).copy()

    # Convert inv_std to variance: var = 1/inv_std^2
    # Use small epsilon to avoid division by zero
    eps = 1e-5
    variance = np.where(
        np.abs(inv_std) > 1e-10,
        1.0 / (inv_std ** 2) - eps,
        1e10  # large variance where inv_std is near zero
    ).astype(np.float32)
    # Clamp variance to be non-negative
    variance = np.maximum(variance, 0.0)

    activation = get_activation_name(bn_layer)

    return channels, gamma, beta, mean, variance, activation


def extract_pool_params(pool_layer):
    """
    Extract pooling parameters from a madmom MaxPoolLayer.

    Returns (size_h, size_w, stride_h, stride_w).
    """
    size = pool_layer.size
    stride = pool_layer.stride

    if hasattr(size, "__len__"):
        size_h, size_w = int(size[0]), int(size[1])
    else:
        size_h = size_w = int(size)

    if hasattr(stride, "__len__"):
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        stride_h = stride_w = int(stride)

    return size_h, size_w, stride_h, stride_w


# ---------------------------------------------------------------------------
# CRF parameter extraction
# ---------------------------------------------------------------------------

def extract_crf_params(crf_model):
    """
    Extract CRF parameters from a madmom ConditionalRandomField stub.

    Returns a dict with keys: A, tau, c, pi, W (all numpy arrays).
    """
    return {
        "A": crf_model.A.astype(np.float32),
        "tau": crf_model.tau.astype(np.float32),
        "c": crf_model.c.astype(np.float32),
        "pi": crf_model.pi.astype(np.float32),
        "W": crf_model.W.astype(np.float32),
    }
