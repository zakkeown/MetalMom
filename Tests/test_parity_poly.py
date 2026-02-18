"""Parity tests for poly_features against librosa golden files."""

import numpy as np
import metalmom


def test_poly_features_shape():
    S = np.random.randn(1025, 20).astype(np.float32)
    S = np.abs(S)  # poly_features expects non-negative spectrogram
    result = metalmom.feature.poly_features(S=S, order=1)
    assert result.shape == (2, 20)


def test_poly_features_order2_shape():
    S = np.random.randn(1025, 20).astype(np.float32)
    S = np.abs(S)
    result = metalmom.feature.poly_features(S=S, order=2)
    assert result.shape == (3, 20)


def test_poly_features_constant():
    S = np.ones((10, 5), dtype=np.float32) * 3.0
    result = metalmom.feature.poly_features(S=S, order=1)
    # Slope should be ~0
    np.testing.assert_allclose(result[0], 0, atol=1e-4)
    # Intercept should be ~3.0
    np.testing.assert_allclose(result[1], 3.0, atol=1e-4)


def test_poly_features_linear():
    # y = freq for freq[i] = i * sr / n_fft
    # With n_features=10, inferred n_fft = 18, sr=22050
    # freq[i] = i * 22050/18 = i * 1225.0
    # When y = freq, np.polyfit(freq, freq, 1) = [1.0, 0.0]
    nf, nt = 10, 3
    sr = 22050
    inferred_n_fft = (nf - 1) * 2  # = 18
    freq_scale = sr / inferred_n_fft
    S = np.zeros((nf, nt), dtype=np.float32)
    for t in range(nt):
        for f in range(nf):
            S[f, t] = f * freq_scale  # y = freq[f]
    result = metalmom.feature.poly_features(S=S, order=1)
    # Slope should be 1.0 (y = 1.0 * freq + 0.0)
    np.testing.assert_allclose(result[0], 1.0, atol=1e-3)
    # Intercept should be ~0.0
    np.testing.assert_allclose(result[1], 0.0, atol=1e-1)


def test_poly_features_parity_order1():
    ref = np.load("Tests/golden/poly_features_440hz_default.npy")
    S = np.abs(metalmom.stft(
        np.load("Tests/golden/signal_440hz_22050sr.npy"),
        n_fft=2048, hop_length=512))
    result = metalmom.feature.poly_features(S=S, order=1)
    assert result.shape == ref.shape
    np.testing.assert_allclose(result, ref, atol=1e-3, rtol=1e-3)


def test_poly_features_parity_order2():
    ref = np.load("Tests/golden/poly_features_440hz_order2.npy")
    S = np.abs(metalmom.stft(
        np.load("Tests/golden/signal_440hz_22050sr.npy"),
        n_fft=2048, hop_length=512))
    result = metalmom.feature.poly_features(S=S, order=2)
    assert result.shape == ref.shape
    np.testing.assert_allclose(result, ref, atol=1e-2, rtol=1e-2)


def test_poly_features_from_audio():
    y = np.load("Tests/golden/signal_440hz_22050sr.npy")
    result = metalmom.feature.poly_features(y=y, sr=22050, order=1)
    assert result.shape[0] == 2
    assert result.shape[1] > 0
    # All values should be finite
    assert np.all(np.isfinite(result))
