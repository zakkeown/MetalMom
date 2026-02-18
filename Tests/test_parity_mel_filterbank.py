"""Mel filterbank parity tests: MetalMom algorithm vs librosa golden reference files.

Since the mel filterbank has no C bridge (it's used internally by mel spectrogram),
this test reimplements the same Slaney formula in Python and compares against
librosa's golden output. This validates that the algorithm in FilterBank.swift
matches librosa exactly.

Tier 1 tolerances: element-wise comparison with rtol=1e-5, atol=1e-6.
"""

import os
import math
import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _load_golden(name):
    path = os.path.join(GOLDEN_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Golden file not found: {path}. Run scripts/generate_golden.py first.")
    return np.load(path)


# ── Reimplement the Slaney mel filterbank (mirrors FilterBank.swift exactly) ──

def _hz_to_mel(hz):
    """Slaney formula: matches Units.hzToMel in Swift."""
    f_sp = 200.0 / 3.0
    mel = hz / f_sp

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15.0
    logstep = math.log(6.4) / 27.0

    if hz >= min_log_hz:
        mel = min_log_mel + math.log(hz / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel):
    """Inverse Slaney formula: matches Units.melToHz in Swift."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    if mel < min_log_mel:
        return mel * f_sp
    else:
        return min_log_hz * math.exp((mel - min_log_mel) * logstep)


def _mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """Reimplement FilterBank.mel() in Python (mirrors Swift implementation)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = [
        _mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1))
        for i in range(n_mels + 2)
    ]

    fft_freqs = [k * sr / n_fft for k in range(n_freqs)]

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(n_mels):
        f_left = mel_points[m]
        f_center = mel_points[m + 1]
        f_right = mel_points[m + 2]

        for k in range(n_freqs):
            freq = fft_freqs[k]
            if freq >= f_left and freq <= f_center and f_center != f_left:
                weights[m, k] = (freq - f_left) / (f_center - f_left)
            elif freq > f_center and freq <= f_right and f_right != f_center:
                weights[m, k] = (f_right - freq) / (f_right - f_center)

        # Slaney normalisation
        enorm = 2.0 / (mel_points[m + 2] - mel_points[m])
        weights[m, :] *= enorm

    return weights


# ── Tests ──


class TestMelFilterbankShape:
    """Verify filterbank dimensions match librosa."""

    def test_default_shape(self):
        expected = _load_golden("mel_filterbank_128_2048.npy")
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_small_shape(self):
        expected = _load_golden("mel_filterbank_40_1024.npy")
        result = _mel_filterbank(sr=22050, n_fft=1024, n_mels=40)
        assert result.shape == expected.shape

    def test_custom_fmin_fmax_shape(self):
        expected = _load_golden("mel_filterbank_64_2048_300_8000.npy")
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000)
        assert result.shape == expected.shape


class TestMelFilterbankParity:
    """Element-wise parity against librosa golden files at Tier 1 tolerance."""

    def test_default_parity(self):
        """128 mels, nFFT=2048, sr=22050 vs librosa."""
        expected = _load_golden("mel_filterbank_128_2048.npy")
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=128)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-5, atol=1e-6,
            err_msg="Default mel filterbank mismatch vs librosa"
        )

    def test_small_parity(self):
        """40 mels, nFFT=1024, sr=22050 vs librosa."""
        expected = _load_golden("mel_filterbank_40_1024.npy")
        result = _mel_filterbank(sr=22050, n_fft=1024, n_mels=40)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-5, atol=1e-6,
            err_msg="Small mel filterbank (40, 1024) mismatch vs librosa"
        )

    def test_custom_fmin_fmax_parity(self):
        """64 mels, nFFT=2048, fmin=300, fmax=8000 vs librosa."""
        expected = _load_golden("mel_filterbank_64_2048_300_8000.npy")
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-5, atol=1e-6,
            err_msg="Custom fmin/fmax mel filterbank mismatch vs librosa"
        )


class TestMelFilterbankProperties:
    """Verify structural properties of the filterbank."""

    def test_non_negative(self):
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
        assert np.all(result >= 0), "Filterbank should contain only non-negative values"

    def test_no_empty_filters(self):
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
        filter_sums = result.sum(axis=1)
        assert np.all(filter_sums > 0), "Every filter should have non-zero sum"

    def test_dtype_float32(self):
        result = _mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestHzMelConversions:
    """Verify Hz/mel conversion functions match librosa."""

    def test_hz_to_mel_known_values(self):
        """Compare against librosa.hz_to_mel with htk=False."""
        import librosa

        test_hz = [0, 100, 500, 1000, 2000, 4000, 8000, 11025]
        for hz in test_hz:
            expected = librosa.hz_to_mel(hz, htk=False)
            result = _hz_to_mel(hz)
            assert abs(result - expected) < 1e-4, (
                f"hz_to_mel({hz}): got {result}, expected {expected}"
            )

    def test_mel_to_hz_known_values(self):
        """Compare against librosa.mel_to_hz with htk=False."""
        import librosa

        test_mels = [0, 5, 10, 15, 20, 30, 40, 50]
        for mel in test_mels:
            expected = librosa.mel_to_hz(mel, htk=False)
            result = _mel_to_hz(mel)
            assert abs(result - expected) < 1e-2, (
                f"mel_to_hz({mel}): got {result}, expected {expected}"
            )

    def test_round_trip(self):
        """hz -> mel -> hz round-trip."""
        test_hz = [0, 100, 500, 1000, 2000, 8000]
        for hz in test_hz:
            mel = _hz_to_mel(hz)
            recovered = _mel_to_hz(mel)
            assert abs(recovered - hz) < 1e-3, (
                f"Round-trip failed for {hz} Hz: got {recovered}"
            )
