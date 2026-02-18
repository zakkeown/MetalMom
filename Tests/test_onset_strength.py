"""Tests for onset strength envelope."""

import os
import numpy as np
import metalmom

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def test_onset_strength_shape():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.onset_strength(y=y, sr=22050)
    assert result.ndim == 1
    assert len(result) > 0


def test_onset_strength_non_negative():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.onset_strength(y=y, sr=22050)
    assert np.all(result >= 0)


def test_onset_strength_silent():
    y = np.zeros(22050, dtype=np.float32)
    result = metalmom.onset_strength(y=y, sr=22050)
    np.testing.assert_allclose(result, 0, atol=1e-5)


def test_onset_strength_parity():
    ref = np.load(os.path.join(GOLDEN_DIR, "onset_strength.npy"))
    y = np.load(os.path.join(GOLDEN_DIR, "test_signal.npy"))
    result = metalmom.onset_strength(y=y, sr=22050)
    # Onset strength involves mel + dB + diff, wider tolerance
    np.testing.assert_allclose(result, ref, atol=1.0, rtol=0.2)


def test_onset_strength_non_aggregate():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.onset_strength(y=y, sr=22050, aggregate=False)
    assert result.ndim == 2
    assert result.shape[0] == 128  # n_mels


def test_onset_strength_finite():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.onset_strength(y=y, sr=22050)
    assert np.all(np.isfinite(result))
