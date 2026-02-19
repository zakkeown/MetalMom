"""
Numpy forward-pass implementations for all madmom neural architecture types.

Provides reference implementations matching madmom's layer logic exactly,
used to generate golden outputs for validating CoreML-converted models.

Supported architectures:
  - LSTM (peephole), BiLSTM
  - GRU, BiGRU
  - Simple RNN, BiRNN
  - Dense / FeedForward
  - CNN: Conv2D, BatchNorm, MaxPool, Pad, AveragePool, Stride

All functions accept madmom layer stubs (loaded via SafeUnpickler) and use
the same weight layout conventions as madmom:
  - Weights shape: (input_dim, hidden_size)
  - Computation: np.dot(data, weights) + bias
  - Bidirectional layers have fwd_layer / bwd_layer attributes
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def relu(x):
    """Rectified linear unit."""
    return np.maximum(0, x)


def elu(x, alpha=1.0):
    """Exponential linear unit."""
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def get_activation_fn(layer):
    """
    Return a numpy activation callable from a madmom layer stub.

    Inspects the layer's ``activation_fn`` attribute and maps the class or
    function name to one of the numpy implementations above. Returns an
    identity function for ``linear`` or when no activation is present.

    Parameters
    ----------
    layer : object
        A madmom layer stub with an optional ``activation_fn`` attribute.

    Returns
    -------
    callable
        A numpy function ``f(x) -> ndarray``.
    """
    act = getattr(layer, "activation_fn", None)
    if act is None:
        return lambda x: x

    name = getattr(act, "__name__", type(act).__name__).lower()

    _map = {
        "sigmoid": sigmoid,
        "tanh": np.tanh,
        "relu": relu,
        "elu": elu,
        "softmax": softmax,
        "linear": lambda x: x,
    }
    return _map.get(name, lambda x: x)


# ---------------------------------------------------------------------------
# Recurrent layers: LSTM
# ---------------------------------------------------------------------------

def lstm_forward(x_seq, lstm_layer):
    """
    Full peephole LSTM forward pass matching madmom's LSTMLayer.activate().

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
        Input sequence.
    lstm_layer : object
        madmom LSTMLayer stub with ``input_gate``, ``forget_gate``, ``cell``,
        ``output_gate`` attributes. Each gate has ``weights``, ``bias``,
        ``recurrent_weights``, and optionally ``peephole_weights``.

    Returns
    -------
    np.ndarray, shape (seq_len, hidden_size)
        Output sequence.

    Notes
    -----
    madmom conventions:
      - weights shape: (input_dim, hidden_size)
      - computation: np.dot(data, weights) + bias
      - Peephole: input and forget gates use OLD cell state;
        output gate uses NEW cell state.
    """
    ig = lstm_layer.input_gate
    fg = lstm_layer.forget_gate
    og = lstm_layer.output_gate
    cell = lstm_layer.cell

    hidden_size = ig.bias.shape[0]
    seq_len = x_seq.shape[0]

    h = np.zeros(hidden_size, dtype=np.float32)
    c = np.zeros(hidden_size, dtype=np.float32)
    outputs = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for t in range(seq_len):
        x = x_seq[t].astype(np.float32)

        # Input gate: sigmoid(W_i @ x + R_i @ h + p_i * c_old + b_i)
        ig_val = np.dot(x, ig.weights) + ig.bias
        if getattr(ig, "peephole_weights", None) is not None:
            ig_val += c * ig.peephole_weights
        ig_val += np.dot(h, ig.recurrent_weights)
        ig_val = sigmoid(ig_val)

        # Forget gate: sigmoid(W_f @ x + R_f @ h + p_f * c_old + b_f)
        fg_val = np.dot(x, fg.weights) + fg.bias
        if getattr(fg, "peephole_weights", None) is not None:
            fg_val += c * fg.peephole_weights
        fg_val += np.dot(h, fg.recurrent_weights)
        fg_val = sigmoid(fg_val)

        # Cell candidate: tanh(W_c @ x + R_c @ h + b_c)
        cc_val = np.dot(x, cell.weights) + cell.bias
        cc_val += np.dot(h, cell.recurrent_weights)
        cc_val = np.tanh(cc_val)

        # New cell state
        c = fg_val * c + ig_val * cc_val

        # Output gate: sigmoid(W_o @ x + R_o @ h + p_o * c_new + b_o)
        og_val = np.dot(x, og.weights) + og.bias
        if getattr(og, "peephole_weights", None) is not None:
            og_val += c * og.peephole_weights
        og_val += np.dot(h, og.recurrent_weights)
        og_val = sigmoid(og_val)

        # Output
        h = og_val * np.tanh(c)
        outputs[t] = h

    return outputs


def bilstm_forward(x_seq, bilstm_layer):
    """
    Bidirectional LSTM forward pass matching madmom's BidirectionalLayer.

    Forward LSTM processes ``x_seq`` in order; backward LSTM processes in
    reverse. Outputs are concatenated (hstack) along the feature dimension.

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
    bilstm_layer : object
        Has ``fwd_layer`` and ``bwd_layer`` attributes (each an LSTMLayer).

    Returns
    -------
    np.ndarray, shape (seq_len, 2 * hidden_size)
    """
    fwd_out = lstm_forward(x_seq, bilstm_layer.fwd_layer)
    bwd_out = lstm_forward(x_seq[::-1], bilstm_layer.bwd_layer)
    # Reverse backward output to align with forward time
    return np.hstack((fwd_out, bwd_out[::-1]))


# ---------------------------------------------------------------------------
# Recurrent layers: GRU
# ---------------------------------------------------------------------------

def gru_forward(x_seq, gru_layer):
    """
    GRU forward pass matching madmom's GRULayer.

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
    gru_layer : object
        madmom GRULayer stub with ``update_gate``, ``reset_gate``, ``cell``
        attributes. Each has ``weights``, ``recurrent_weights``, ``bias``.

    Returns
    -------
    np.ndarray, shape (seq_len, hidden_size)

    Notes
    -----
    GRU equations (madmom convention):
      z = sigmoid(W_z @ x + R_z @ h + b_z)       # update gate
      r = sigmoid(W_r @ x + R_r @ h + b_r)       # reset gate
      h_tilde = tanh(W @ x + R @ (r * h) + b)    # candidate
      h_new = (1 - z) * h + z * h_tilde           # output
    """
    zg = gru_layer.update_gate   # z
    rg = gru_layer.reset_gate    # r
    cg = gru_layer.cell          # candidate

    hidden_size = zg.bias.shape[0]
    seq_len = x_seq.shape[0]

    h = np.zeros(hidden_size, dtype=np.float32)
    outputs = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for t in range(seq_len):
        x = x_seq[t].astype(np.float32)

        # Update gate
        z = sigmoid(
            np.dot(x, zg.weights) + np.dot(h, zg.recurrent_weights) + zg.bias
        )

        # Reset gate
        r = sigmoid(
            np.dot(x, rg.weights) + np.dot(h, rg.recurrent_weights) + rg.bias
        )

        # Candidate hidden state
        h_tilde = np.tanh(
            np.dot(x, cg.weights) + np.dot(r * h, cg.recurrent_weights) + cg.bias
        )

        # New hidden state
        h = (1.0 - z) * h + z * h_tilde
        outputs[t] = h

    return outputs


def bigru_forward(x_seq, bigru_layer):
    """
    Bidirectional GRU forward pass.

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
    bigru_layer : object
        Has ``fwd_layer`` and ``bwd_layer`` attributes (each a GRULayer).

    Returns
    -------
    np.ndarray, shape (seq_len, 2 * hidden_size)
    """
    fwd_out = gru_forward(x_seq, bigru_layer.fwd_layer)
    bwd_out = gru_forward(x_seq[::-1], bigru_layer.bwd_layer)
    return np.hstack((fwd_out, bwd_out[::-1]))


# ---------------------------------------------------------------------------
# Recurrent layers: Simple RNN
# ---------------------------------------------------------------------------

def rnn_forward(x_seq, rnn_layer):
    """
    Simple RNN forward pass matching madmom's RecurrentLayer.

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
    rnn_layer : object
        madmom RecurrentLayer stub with ``weights``, ``recurrent_weights``,
        ``bias`` attributes.

    Returns
    -------
    np.ndarray, shape (seq_len, hidden_size)

    Notes
    -----
    Equation: h_new = tanh(W @ x + R @ h + b)
    """
    hidden_size = rnn_layer.bias.shape[0]
    seq_len = x_seq.shape[0]

    h = np.zeros(hidden_size, dtype=np.float32)
    outputs = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for t in range(seq_len):
        x = x_seq[t].astype(np.float32)
        h = np.tanh(
            np.dot(x, rnn_layer.weights)
            + np.dot(h, rnn_layer.recurrent_weights)
            + rnn_layer.bias
        )
        outputs[t] = h

    return outputs


def birnn_forward(x_seq, birnn_layer):
    """
    Bidirectional simple RNN forward pass.

    Parameters
    ----------
    x_seq : np.ndarray, shape (seq_len, input_dim)
    birnn_layer : object
        Has ``fwd_layer`` and ``bwd_layer`` attributes (each a RecurrentLayer).

    Returns
    -------
    np.ndarray, shape (seq_len, 2 * hidden_size)
    """
    fwd_out = rnn_forward(x_seq, birnn_layer.fwd_layer)
    bwd_out = rnn_forward(x_seq[::-1], birnn_layer.bwd_layer)
    return np.hstack((fwd_out, bwd_out[::-1]))


# ---------------------------------------------------------------------------
# Dense / FeedForward
# ---------------------------------------------------------------------------

def dense_forward(x, dense_layer):
    """
    Dense layer forward pass matching madmom's FeedForwardLayer.

    Parameters
    ----------
    x : np.ndarray, shape (..., input_dim)
        Input tensor. Can be 1D (single sample) or 2D (seq_len, input_dim).
    dense_layer : object
        madmom FeedForwardLayer stub with ``weights``, ``bias``, and
        ``activation_fn`` attributes.

    Returns
    -------
    np.ndarray, shape (..., output_dim)

    Notes
    -----
    madmom convention: ``np.dot(data, weights) + bias``, then activation_fn.
    Uses ``get_activation_fn`` to resolve the activation (not hardcoded).
    """
    act_fn = get_activation_fn(dense_layer)
    return act_fn(np.dot(x, dense_layer.weights) + dense_layer.bias)


# ---------------------------------------------------------------------------
# CNN layers
# ---------------------------------------------------------------------------

def conv2d_forward(x, conv_layer):
    """
    2D convolution matching madmom's ConvolutionalLayer.

    Parameters
    ----------
    x : np.ndarray, shape (batch, in_ch, H, W)
        Input feature maps.
    conv_layer : object
        madmom ConvolutionalLayer stub with ``weights`` of shape
        ``(in_channels, out_channels, kH, kW)``, optional ``bias``,
        optional ``stride``, and optional ``activation_fn``.

    Returns
    -------
    np.ndarray, shape (batch, out_ch, H_out, W_out)
        Output after valid-padding convolution (+ bias + activation).

    Notes
    -----
    Uses ``valid`` padding and ``stride`` from the layer (default 1).
    Convolution is implemented as cross-correlation (matching typical NN
    convention and what madmom's underlying Theano/numpy backend does).
    """
    w = conv_layer.weights.astype(np.float32)  # (in_ch, out_ch, kH, kW)
    in_ch, out_ch, kH, kW = w.shape

    bias = getattr(conv_layer, "bias", None)
    if bias is not None:
        b = np.broadcast_to(
            np.asarray(bias).astype(np.float32).flatten(), (out_ch,)
        ).copy()
    else:
        b = np.zeros(out_ch, dtype=np.float32)

    stride = getattr(conv_layer, "stride", 1)
    if hasattr(stride, "__len__"):
        stride_h = int(stride[0])
        stride_w = int(stride[1]) if len(stride) > 1 else int(stride[0])
    else:
        stride_h = stride_w = int(stride)

    batch = x.shape[0]
    H_in, W_in = x.shape[2], x.shape[3]
    H_out = (H_in - kH) // stride_h + 1
    W_out = (W_in - kW) // stride_w + 1

    out = np.zeros((batch, out_ch, H_out, W_out), dtype=np.float32)

    for n in range(batch):
        for oc in range(out_ch):
            acc = np.zeros((H_out, W_out), dtype=np.float32)
            for ic in range(in_ch):
                # Cross-correlation (valid mode)
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        patch = x[n, ic, h_start:h_start + kH,
                                   w_start:w_start + kW]
                        acc[i, j] += np.sum(patch * w[ic, oc])
            out[n, oc] = acc + b[oc]

    # Apply activation if present
    act_fn = get_activation_fn(conv_layer)
    return act_fn(out)


def batchnorm_forward(x, bn_layer, eps=1e-5):
    """
    Batch normalization (inference mode) matching madmom's BatchNormLayer.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, H, W)
        Input feature maps.
    bn_layer : object
        madmom BatchNormLayer stub with ``mean``, ``inv_std``, ``gamma``,
        ``beta`` attributes. ``inv_std`` is used directly (not variance).
    eps : float
        Unused (madmom stores ``inv_std`` directly). Kept for API
        compatibility.

    Returns
    -------
    np.ndarray, shape (batch, channels, H, W)

    Notes
    -----
    Formula: ``gamma * (x - mean) * inv_std + beta``, broadcast over
    spatial dimensions. May have an ``activation_fn``.
    """
    mean = np.asarray(bn_layer.mean).astype(np.float32).flatten()
    inv_std = np.asarray(bn_layer.inv_std).astype(np.float32).flatten()

    gamma_raw = getattr(bn_layer, "gamma", 1)
    beta_raw = getattr(bn_layer, "beta", 0)
    channels = len(mean)

    gamma = np.broadcast_to(
        np.asarray(gamma_raw).astype(np.float32).flatten(), (channels,)
    ).copy()
    beta = np.broadcast_to(
        np.asarray(beta_raw).astype(np.float32).flatten(), (channels,)
    ).copy()

    # Reshape for broadcasting: (1, C, 1, 1)
    mean = mean.reshape(1, -1, 1, 1)
    inv_std = inv_std.reshape(1, -1, 1, 1)
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    normalized = (x - mean) * inv_std
    out = gamma * normalized + beta

    act_fn = get_activation_fn(bn_layer)
    return act_fn(out)


def maxpool_forward(x, pool_layer):
    """
    Max pooling matching madmom's MaxPoolLayer.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, H, W)
    pool_layer : object
        madmom MaxPoolLayer stub with ``size`` and ``stride`` attributes.

    Returns
    -------
    np.ndarray, shape (batch, channels, H_out, W_out)
    """
    size = pool_layer.size
    stride = pool_layer.stride

    if hasattr(size, "__len__"):
        pH, pW = int(size[0]), int(size[1])
    else:
        pH = pW = int(size)

    if hasattr(stride, "__len__"):
        sH, sW = int(stride[0]), int(stride[1])
    else:
        sH = sW = int(stride)

    batch, C, H, W = x.shape
    H_out = (H - pH) // sH + 1
    W_out = (W - pW) // sW + 1

    out = np.zeros((batch, C, H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            w_start = j * sW
            out[:, :, i, j] = x[:, :, h_start:h_start + pH,
                                 w_start:w_start + pW].max(axis=(2, 3))

    return out


def pad_forward(x, pad_layer):
    """
    Zero padding on spatial dimensions matching madmom's PadLayer.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, H, W)
    pad_layer : object
        madmom PadLayer stub with ``width`` attribute (uniform padding).

    Returns
    -------
    np.ndarray, shape (batch, channels, H + 2*width, W + 2*width)
    """
    width = int(pad_layer.width)
    value = float(getattr(pad_layer, "value", 0.0))

    # Pad only spatial dims (H, W), not batch or channel
    pad_widths = ((0, 0), (0, 0), (width, width), (width, width))
    return np.pad(x, pad_widths, mode="constant", constant_values=value)


def average_forward(x):
    """
    Global average pooling over spatial dimensions.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, H, W)

    Returns
    -------
    np.ndarray, shape (batch, channels, 1, 1)
    """
    return x.mean(axis=(2, 3), keepdims=True)


def stride_forward(x, stride_layer):
    """
    Flatten to 2D, matching madmom's StrideLayer / block interleave.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, H, W) or (batch, features)
    stride_layer : object
        madmom StrideLayer stub (``block_size`` attribute, if any).

    Returns
    -------
    np.ndarray, shape (batch, channels * H * W)
        Flattened 2D tensor.
    """
    batch = x.shape[0]
    return x.reshape(batch, -1)


# ---------------------------------------------------------------------------
# Full-model forward pass dispatcher
# ---------------------------------------------------------------------------

import os as _os
import sys as _sys

_this_dir = _os.path.dirname(_os.path.abspath(__file__))
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)
from madmom_loader import classify_model, layer_type_name  # noqa: E402


def _detect_input_dim(model_type, layers):
    """
    Detect the input dimension from the first layer's weights.

    Parameters
    ----------
    model_type : str
        One of 'bilstm', 'lstm', 'bigru', 'birnn', 'rnn', 'cnn', 'dnn'.
    layers : list
        List of madmom layer stubs.

    Returns
    -------
    int
        The input feature dimension expected by the first layer.
    """
    first = layers[0]
    name = layer_type_name(first)

    if name == "BidirectionalLayer":
        fwd = first.fwd_layer
        sub_name = layer_type_name(fwd)
        if sub_name == "LSTMLayer":
            return fwd.input_gate.weights.shape[0]
        elif sub_name == "GRULayer":
            return fwd.reset_gate.weights.shape[0]
        elif sub_name == "RecurrentLayer":
            return fwd.weights.shape[0]
        else:
            raise ValueError(f"Unknown bidirectional sublayer: {sub_name}")

    elif name == "LSTMLayer":
        return first.input_gate.weights.shape[0]

    elif name == "RecurrentLayer":
        return first.weights.shape[0]

    elif name == "FeedForwardLayer":
        return first.weights.shape[0]

    elif name in ("ConvolutionalLayer", "BatchNormLayer", "PadLayer"):
        # CNN: find the first ConvolutionalLayer to get in_channels
        for layer in layers:
            if layer_type_name(layer) == "ConvolutionalLayer":
                return layer.weights.shape[0]  # in_channels
        raise ValueError("CNN model has no ConvolutionalLayer")

    else:
        raise ValueError(f"Cannot detect input dim for layer type: {name}")


# -- recurrent model runner -------------------------------------------------

_RECURRENT_TYPES = {"BidirectionalLayer", "LSTMLayer", "RecurrentLayer"}


def _recurrent_fn_for_layer(layer):
    """Return the correct forward function for a single recurrent layer."""
    name = layer_type_name(layer)

    if name == "BidirectionalLayer":
        from madmom_loader import detect_rnn_sublayer_type
        sub = detect_rnn_sublayer_type(layer)
        if sub == "lstm":
            return bilstm_forward
        elif sub == "gru":
            return bigru_forward
        elif sub == "rnn":
            return birnn_forward
        else:
            raise ValueError(f"Unknown bidirectional sublayer type: {sub}")

    elif name == "LSTMLayer":
        return lstm_forward

    elif name == "RecurrentLayer":
        return rnn_forward

    else:
        raise ValueError(f"Not a recurrent layer: {name}")


def _run_recurrent_model(layers, seq_len=100):
    """
    Run a recurrent model (N recurrent layers + trailing dense layers).

    Parameters
    ----------
    layers : list
        Layer stubs from a recurrent madmom model.
    seq_len : int
        Number of time steps in the synthetic input.

    Returns
    -------
    np.ndarray, shape (seq_len_out, output_dim)
        Output of the final dense layer.
    """
    # Separate recurrent layers from dense layers
    recurrent_layers = []
    dense_layers = []
    for layer in layers:
        name = layer_type_name(layer)
        if name in _RECURRENT_TYPES:
            recurrent_layers.append(layer)
        elif name == "FeedForwardLayer":
            dense_layers.append(layer)
        else:
            raise ValueError(f"Unexpected layer in recurrent model: {name}")

    # Detect input dimension from the first recurrent layer
    input_dim = _detect_input_dim("recurrent", layers)

    # Generate random input
    x = np.random.randn(seq_len, input_dim).astype(np.float32) * 0.1

    # Run through recurrent layers
    for layer in recurrent_layers:
        fn = _recurrent_fn_for_layer(layer)
        x = fn(x, layer)

    # Run through dense layers
    for layer in dense_layers:
        x = dense_forward(x, layer)

    return x


# -- CNN model runner --------------------------------------------------------

def _run_cnn_model(layers, seq_len=15):
    """
    Run a CNN model through its full layer stack.

    Handles three madmom CNN variants:
      - onsets_cnn: BN(3D) -> Conv -> Pool -> Conv -> Pool -> Stride -> Dense
      - key_cnn: Pad/Conv/BN/Pool blocks -> AveragePool (no dense)
      - chords_cnnfeat: Conv/BN/Pool blocks (no dense, no avg pool)

    Parameters
    ----------
    layers : list
        Layer stubs from a CNN madmom model.
    seq_len : int
        Number of time frames in the synthetic input.

    Returns
    -------
    np.ndarray
        Output array (shape depends on model architecture).
    """
    # Detect whether this is a 3D-BN onset model or a standard 4D CNN
    first_name = layer_type_name(layers[0])

    if first_name == "BatchNormLayer" and layers[0].mean.ndim == 2:
        # Onsets CNN: BN operates on 3D input (T, H, C) then reshapes to 4D
        return _run_onsets_cnn(layers, seq_len)
    else:
        # Standard 4D CNN (key, chords)
        return _run_standard_cnn(layers, seq_len)


def _run_onsets_cnn(layers, seq_len):
    """
    Run the onsets CNN model.

    The onsets CNN has a special structure:
      1. Input is 3D: (T, 80, 3) — T frames, 80 freq bins, 3 spectrograms
      2. BatchNorm normalizes this 3D input (mean shape is (80, 3))
      3. Reshaped to 4D: (1, 3, T, 80) for convolution
      4. Conv/Pool layers process in 4D
      5. Reshape back to 2D for StrideLayer and Dense layers
    """
    bn_layer = layers[0]
    freq_bins = bn_layer.mean.shape[0]  # 80
    num_specs = bn_layer.mean.shape[1]  # 3

    # Generate 3D input
    x = np.random.randn(seq_len, freq_bins, num_specs).astype(np.float32) * 0.1

    # Apply BN on 3D data (element-wise, broadcasts over T)
    mean = np.asarray(bn_layer.mean).astype(np.float32)
    inv_std = np.asarray(bn_layer.inv_std).astype(np.float32)
    gamma_raw = getattr(bn_layer, "gamma", 1)
    beta_raw = getattr(bn_layer, "beta", 0)
    gamma = np.broadcast_to(
        np.asarray(gamma_raw).astype(np.float32).flatten(),
        (freq_bins * num_specs,),
    ).copy().reshape(freq_bins, num_specs)
    beta = np.broadcast_to(
        np.asarray(beta_raw).astype(np.float32).flatten(),
        (freq_bins * num_specs,),
    ).copy().reshape(freq_bins, num_specs)
    x = gamma * (x - mean) * inv_std + beta

    # Reshape to 4D: (1, channels=num_specs, T, freq_bins)
    # (T, 80, 3) -> transpose to (T, 3, 80) -> reshape to (1, 3, T, 80)
    x = x.transpose(0, 2, 1)  # (T, 3, 80)
    x = x.reshape(1, num_specs, seq_len, freq_bins)  # (1, 3, T, 80)

    # Process remaining layers
    for layer in layers[1:]:
        name = layer_type_name(layer)
        if name == "ConvolutionalLayer":
            x = conv2d_forward(x, layer)
        elif name == "BatchNormLayer":
            x = batchnorm_forward(x, layer)
        elif name == "MaxPoolLayer":
            x = maxpool_forward(x, layer)
        elif name == "StrideLayer":
            # Reshape 4D to 2D: (1, C, H, W) -> (H, C*W)
            # The time dimension is H (axis 2)
            batch, C, H, W = x.shape
            x = x.transpose(0, 2, 1, 3).reshape(H, C * W)  # (H, C*W)
            # Apply stride: sliding window over time
            block_size = int(layer.block_size)
            T_out = H - block_size + 1
            features = x.shape[1]
            from numpy.lib.stride_tricks import as_strided
            out = as_strided(
                x,
                shape=(T_out, features * block_size),
                strides=(x.strides[0], x.strides[1]),
            )
            x = np.array(out)  # copy to own memory
        elif name == "FeedForwardLayer":
            x = dense_forward(x, layer)
        else:
            raise ValueError(f"Unexpected layer in onsets CNN: {name}")

    return x


def _run_standard_cnn(layers, seq_len):
    """
    Run a standard 4D CNN model (key_cnn, chords_cnnfeat).

    Input is (1, in_channels, H, W) throughout the conv/bn/pool stack.
    If the model ends with AverageLayer, global average pooling is applied.
    If dense layers follow, the tensor is flattened to 2D first.

    The time dimension (seq_len) is automatically bumped to the minimum
    required by the layer stack if the caller-supplied value is too small.
    """
    # Find first conv layer to determine input shape
    first_conv = None
    for layer in layers:
        if layer_type_name(layer) == "ConvolutionalLayer":
            first_conv = layer
            break

    if first_conv is None:
        raise ValueError("No ConvolutionalLayer found in CNN model")

    in_channels = first_conv.weights.shape[0]

    # Ensure seq_len is large enough for this architecture
    min_h, _min_w = _compute_min_cnn_spatial(layers)
    if seq_len < min_h:
        seq_len = min_h

    # Determine spatial dimensions from the model structure
    # Use a reasonable default based on typical madmom CNN inputs
    # key_cnn: (1, 1, T, 168) — 168 log-freq bins
    # chords_cnnfeat: (1, 1, T, 105) — 105 semitone bins
    # Detect from layer structure: trace forward to find what works
    freq_dim = _detect_cnn_freq_dim(layers, in_channels, seq_len)

    x = np.random.randn(1, in_channels, seq_len, freq_dim).astype(np.float32) * 0.1

    for layer in layers:
        name = layer_type_name(layer)
        if name == "PadLayer":
            x = pad_forward(x, layer)
        elif name == "ConvolutionalLayer":
            x = conv2d_forward(x, layer)
        elif name == "BatchNormLayer":
            x = batchnorm_forward(x, layer)
        elif name == "MaxPoolLayer":
            x = maxpool_forward(x, layer)
        elif name == "AverageLayer":
            x = average_forward(x)
        elif name == "StrideLayer":
            x = stride_forward(x, layer)
        elif name == "FeedForwardLayer":
            # Flatten to 2D if still 4D
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
            x = dense_forward(x, layer)
        else:
            raise ValueError(f"Unexpected layer in CNN: {name}")

    # If output is still 4D, flatten to 2D
    if x.ndim == 4:
        x = x.reshape(x.shape[0], -1)

    return x


def _compute_min_cnn_spatial(layers):
    """
    Compute the minimum (H, W) input dimensions needed for a standard CNN.

    Works backwards through the layer stack: each conv/pool layer imposes a
    minimum input size so that the output is at least 1x1.  Pad layers
    *reduce* the minimum (they add pixels).

    Returns
    -------
    (min_h, min_w) : tuple of int
        The minimum height (time) and width (frequency) dimensions.
    """
    min_h = 1
    min_w = 1

    for layer in reversed(layers):
        name = layer_type_name(layer)

        if name == "ConvolutionalLayer":
            kH, kW = layer.weights.shape[2], layer.weights.shape[3]
            stride = getattr(layer, "stride", 1)
            if hasattr(stride, "__len__"):
                sH = int(stride[0])
                sW = int(stride[1]) if len(stride) > 1 else sH
            else:
                sH = sW = int(stride)
            # h_out = (h_in - kH) // sH + 1 >= min_h
            # => h_in >= (min_h - 1) * sH + kH
            min_h = (min_h - 1) * sH + kH
            min_w = (min_w - 1) * sW + kW

        elif name == "MaxPoolLayer":
            size = layer.size
            stride = layer.stride
            if hasattr(size, "__len__"):
                pH, pW = int(size[0]), int(size[1])
            else:
                pH = pW = int(size)
            if hasattr(stride, "__len__"):
                sH, sW = int(stride[0]), int(stride[1])
            else:
                sH = sW = int(stride)
            min_h = (min_h - 1) * sH + pH
            min_w = (min_w - 1) * sW + pW

        elif name == "PadLayer":
            p = int(layer.width)
            min_h = max(1, min_h - 2 * p)
            min_w = max(1, min_w - 2 * p)

        elif name == "AverageLayer":
            # Global average pooling accepts any spatial size >= 1
            min_h = max(min_h, 1)
            min_w = max(min_w, 1)

        # BatchNormLayer, StrideLayer, FeedForwardLayer don't change spatial
        # requirements (Stride/Dense flatten; they accept any input).

    return min_h, min_w


def _detect_cnn_freq_dim(layers, in_channels, seq_len):
    """
    Detect the frequency (W) dimension needed for a standard CNN.

    Traces the layer stack symbolically to find a W that produces valid
    spatial dimensions throughout. Works backwards from the last conv layer's
    kernel size requirements or from a known dense layer input size.
    """
    # Try common madmom frequency dimensions
    # key_cnn uses 168 bins, chords uses 105 bins
    candidates = [168, 105, 128, 80, 64, 48, 256]

    for freq in candidates:
        try:
            h, w = seq_len, freq
            valid = True
            for layer in layers:
                name = layer_type_name(layer)
                if name == "PadLayer":
                    p = int(layer.width)
                    h += 2 * p
                    w += 2 * p
                elif name == "ConvolutionalLayer":
                    wt = layer.weights
                    kH, kW = wt.shape[2], wt.shape[3]
                    stride = getattr(layer, "stride", 1)
                    sH = sW = int(stride) if not hasattr(stride, "__len__") else int(stride[0])
                    if hasattr(stride, "__len__") and len(stride) > 1:
                        sW = int(stride[1])
                    h = (h - kH) // sH + 1
                    w = (w - kW) // sW + 1
                    if h < 1 or w < 1:
                        valid = False
                        break
                elif name == "BatchNormLayer":
                    pass
                elif name == "MaxPoolLayer":
                    size = layer.size
                    stride = layer.stride
                    if hasattr(size, "__len__"):
                        pH, pW = int(size[0]), int(size[1])
                    else:
                        pH = pW = int(size)
                    if hasattr(stride, "__len__"):
                        sH, sW = int(stride[0]), int(stride[1])
                    else:
                        sH = sW = int(stride)
                    h = (h - pH) // sH + 1
                    w = (w - pW) // sW + 1
                    if h < 1 or w < 1:
                        valid = False
                        break
                elif name == "AverageLayer":
                    h, w = 1, 1
                elif name in ("StrideLayer", "FeedForwardLayer"):
                    break  # past spatial layers
            if valid:
                return freq
        except Exception:
            continue

    # Fallback: use 80
    return 80


# -- DNN model runner --------------------------------------------------------

def _run_dnn_model(layers, seq_len=100):
    """
    Run a DNN (dense-only) model.

    Parameters
    ----------
    layers : list
        FeedForwardLayer stubs.
    seq_len : int
        Number of input samples.

    Returns
    -------
    np.ndarray, shape (seq_len, output_dim)
    """
    input_dim = _detect_input_dim("dnn", layers)
    x = np.random.randn(seq_len, input_dim).astype(np.float32) * 0.1

    for layer in layers:
        name = layer_type_name(layer)
        if name == "FeedForwardLayer":
            x = dense_forward(x, layer)
        else:
            raise ValueError(f"Unexpected layer in DNN: {name}")

    return x


# -- main dispatcher ---------------------------------------------------------

def run_model_forward(model, seq_len=100):
    """
    Classify a loaded madmom model and run its full forward pass.

    Parameters
    ----------
    model : object
        A madmom model loaded via ``SafeUnpickler`` / ``load_model()``.
    seq_len : int
        Number of time steps for the synthetic input sequence.
        For CNN models this is the time dimension; for recurrent/DNN models
        this is the number of frames.

    Returns
    -------
    np.ndarray or None
        The output array from the forward pass. Returns None for CRF models
        (which have no neural forward pass).
    """
    model_type, data = classify_model(model)

    if model_type == "crf":
        return None

    if model_type in ("bilstm", "lstm", "bigru", "birnn", "rnn"):
        return _run_recurrent_model(data, seq_len=seq_len)

    if model_type == "cnn":
        return _run_cnn_model(data, seq_len=seq_len)

    if model_type == "dnn":
        return _run_dnn_model(data, seq_len=seq_len)

    raise ValueError(f"Unknown model type: {model_type}")


def generate_golden(model, seq_len=100, seed=42):
    """
    Generate a deterministic (input, output) pair for golden-file validation.

    Seeds the numpy RNG before running the forward pass, then re-seeds and
    re-generates the same input to return both the input and the output.

    Parameters
    ----------
    model : object
        A madmom model loaded via ``SafeUnpickler`` / ``load_model()``.
    seq_len : int
        Number of time steps in the synthetic input.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (np.ndarray, np.ndarray) or (None, None)
        (input_array, output_array). Returns (None, None) for CRF models.
    """
    model_type, data = classify_model(model)

    if model_type == "crf":
        return None, None

    # Run forward pass with seeded RNG
    np.random.seed(seed)
    output = run_model_forward(model, seq_len=seq_len)

    # Re-seed and regenerate the same input
    np.random.seed(seed)

    if model_type in ("bilstm", "lstm", "bigru", "birnn", "rnn"):
        input_dim = _detect_input_dim(model_type, data)
        input_arr = np.random.randn(seq_len, input_dim).astype(np.float32) * 0.1
    elif model_type == "dnn":
        input_dim = _detect_input_dim(model_type, data)
        input_arr = np.random.randn(seq_len, input_dim).astype(np.float32) * 0.1
    elif model_type == "cnn":
        first_name = layer_type_name(data[0])
        if first_name == "BatchNormLayer" and data[0].mean.ndim == 2:
            freq_bins = data[0].mean.shape[0]
            num_specs = data[0].mean.shape[1]
            input_arr = np.random.randn(seq_len, freq_bins, num_specs).astype(
                np.float32
            ) * 0.1
        else:
            first_conv = None
            for layer in data:
                if layer_type_name(layer) == "ConvolutionalLayer":
                    first_conv = layer
                    break
            in_channels = first_conv.weights.shape[0]
            freq_dim = _detect_cnn_freq_dim(data, in_channels, seq_len)
            input_arr = np.random.randn(1, in_channels, seq_len, freq_dim).astype(
                np.float32
            ) * 0.1
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return input_arr, output
