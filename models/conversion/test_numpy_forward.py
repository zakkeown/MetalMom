"""
Tests for the full-model forward pass dispatcher in numpy_forward.py.

Tests one representative model per architecture type:
  - bilstm: beats/2015/beats_blstm_1.pkl
  - lstm: beats/2016/beats_lstm_1.pkl
  - bigru: downbeats/2016/downbeats_bgru_harmonic_0.pkl
  - birnn: onsets/2013/onsets_brnn_1.pkl
  - rnn: onsets/2013/onsets_rnn_1.pkl
  - cnn: onsets/2013/onsets_cnn.pkl
  - dnn: chroma/2016/chroma_dnn.pkl
  - birnn (notes): notes/2013/notes_brnn.pkl
"""

import os
import sys

import numpy as np
import pytest

# Ensure the conversion directory is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from madmom_loader import load_model, classify_model
from numpy_forward import run_model_forward, generate_golden

# Base path for madmom models
MODELS_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", ".venv", "lib", "python3.14",
    "site-packages", "madmom", "models",
)
MODELS_BASE = os.path.normpath(MODELS_BASE)


def _load(relpath):
    """Load a madmom pkl model by relative path under the models dir."""
    full = os.path.join(MODELS_BASE, relpath)
    assert os.path.isfile(full), f"Model not found: {full}"
    return load_model(full)


# ---------------------------------------------------------------------------
# Architecture: BiLSTM
# ---------------------------------------------------------------------------

class TestBiLSTM:
    """Forward pass for BiLSTM model (beats/2015/beats_blstm_1.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("beats/2015/beats_blstm_1.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "bilstm"
        assert len(layers) > 0

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)

    def test_generate_golden(self, model):
        inp, out = generate_golden(model, seq_len=20, seed=42)
        assert inp is not None and out is not None
        assert inp.shape[0] == 20
        assert out.shape[0] == 20


# ---------------------------------------------------------------------------
# Architecture: LSTM (unidirectional)
# ---------------------------------------------------------------------------

class TestLSTM:
    """Forward pass for LSTM model (beats/2016/beats_lstm_1.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("beats/2016/beats_lstm_1.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "lstm"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Architecture: BiGRU
# ---------------------------------------------------------------------------

class TestBiGRU:
    """Forward pass for BiGRU model (downbeats/2016/downbeats_bgru_harmonic_0.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("downbeats/2016/downbeats_bgru_harmonic_0.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "bigru"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Architecture: BiRNN
# ---------------------------------------------------------------------------

class TestBiRNN:
    """Forward pass for BiRNN model (onsets/2013/onsets_brnn_1.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("onsets/2013/onsets_brnn_1.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "birnn"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Architecture: RNN (unidirectional)
# ---------------------------------------------------------------------------

class TestRNN:
    """Forward pass for RNN model (onsets/2013/onsets_rnn_1.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("onsets/2013/onsets_rnn_1.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "rnn"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Architecture: CNN (onsets CNN)
# ---------------------------------------------------------------------------

class TestCNN:
    """Forward pass for CNN model (onsets/2013/onsets_cnn.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("onsets/2013/onsets_cnn.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "cnn"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=15)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=15)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=15)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=15)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=15)
        np.testing.assert_array_equal(out1, out2)

    def test_output_shape(self, model):
        """With seq_len=15, after conv/pool/stride, expect 1 output frame."""
        out = run_model_forward(model, seq_len=15)
        # 15 frames -> conv/pool -> 7 time steps -> stride(7) -> 1 frame
        assert out.shape[0] == 1
        assert out.shape[1] == 1  # single sigmoid output


# ---------------------------------------------------------------------------
# Architecture: DNN (dense-only)
# ---------------------------------------------------------------------------

class TestDNN:
    """Forward pass for DNN model (chroma/2016/chroma_dnn.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("chroma/2016/chroma_dnn.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "dnn"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)

    def test_output_shape(self, model):
        """DNN preserves sequence length; chroma_dnn outputs 12 chroma bins."""
        out = run_model_forward(model, seq_len=20)
        assert out.shape == (20, 12)


# ---------------------------------------------------------------------------
# Architecture: BiRNN (notes variant)
# ---------------------------------------------------------------------------

class TestBiRNNNotes:
    """Forward pass for BiRNN notes model (notes/2013/notes_brnn.pkl)."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("notes/2013/notes_brnn.pkl")

    def test_classify(self, model):
        mtype, layers = classify_model(model)
        assert mtype == "birnn"

    def test_output_not_none(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out is not None

    def test_output_2d(self, model):
        out = run_model_forward(model, seq_len=20)
        assert out.ndim == 2

    def test_no_nan_inf(self, model):
        out = run_model_forward(model, seq_len=20)
        assert np.all(np.isfinite(out))

    def test_deterministic(self, model):
        np.random.seed(99)
        out1 = run_model_forward(model, seq_len=20)
        np.random.seed(99)
        out2 = run_model_forward(model, seq_len=20)
        np.testing.assert_array_equal(out1, out2)

    def test_output_shape(self, model):
        """notes_brnn has 88 output units (piano keys)."""
        out = run_model_forward(model, seq_len=20)
        assert out.shape == (20, 88)


# ---------------------------------------------------------------------------
# CRF models return None
# ---------------------------------------------------------------------------

class TestCRF:
    """CRF models have no neural forward pass — dispatcher returns None."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load("chords/2016/chords_cnncrf.pkl")

    def test_classify(self, model):
        mtype, _ = classify_model(model)
        assert mtype == "crf"

    def test_returns_none(self, model):
        out = run_model_forward(model)
        assert out is None

    def test_golden_returns_none(self, model):
        inp, out = generate_golden(model)
        assert inp is None and out is None
