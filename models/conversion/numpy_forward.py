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
