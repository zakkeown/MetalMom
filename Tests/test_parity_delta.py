"""Parity tests for delta and stack_memory against librosa golden files."""

import numpy as np
import metalmom


def test_delta_shape():
    data = np.random.randn(4, 20).astype(np.float32)
    result = metalmom.feature.delta(data)
    assert result.shape == data.shape


def test_delta_constant():
    data = np.ones((4, 20), dtype=np.float32) * 5.0
    result = metalmom.feature.delta(data)
    np.testing.assert_allclose(result, 0, atol=1e-5)


def test_delta_parity():
    ref = np.load("Tests/golden/delta.npy")
    mfcc = np.load("Tests/golden/mfcc.npy")
    result = metalmom.feature.delta(mfcc)
    np.testing.assert_allclose(result, ref, atol=0.1, rtol=0.1)


def test_delta_delta_parity():
    ref = np.load("Tests/golden/delta_delta.npy")
    mfcc = np.load("Tests/golden/mfcc.npy")
    result = metalmom.feature.delta(mfcc, order=2)
    np.testing.assert_allclose(result, ref, atol=0.2, rtol=0.2)


def test_stack_memory_shape():
    data = np.random.randn(4, 20).astype(np.float32)
    result = metalmom.feature.stack_memory(data, n_steps=3)
    assert result.shape == (12, 20)


def test_stack_memory_parity():
    ref = np.load("Tests/golden/stack_memory.npy")
    mfcc = np.load("Tests/golden/mfcc.npy")
    result = metalmom.feature.stack_memory(mfcc, n_steps=3)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)
