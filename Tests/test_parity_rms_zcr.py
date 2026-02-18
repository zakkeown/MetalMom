"""RMS energy and zero-crossing rate parity tests: MetalMom vs librosa golden reference.

Tests the full pipeline for: RMS energy, zero-crossing rate.
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

RMS and ZCR are time-domain features computed by framing the signal.
Differences arise from float32 precision in frame computation.
- RMS: values in [0, ~1], tight tolerance.
- ZCR: values in [0, 1], tight tolerance.
"""

import os
import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _load_golden(name):
    path = os.path.join(GOLDEN_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Golden file not found: {path}. Run scripts/generate_golden.py first.")
    return np.load(path)


def _load_signal_440():
    return _load_golden("signal_440hz_22050sr.npy")


# -- RMS Energy --


class TestRMSShape:
    """Verify RMS dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("rms_440hz_default.npy")
        result = metalmom.rms(y=signal)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.rms(y=signal)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestRMSParity:
    """Element-wise parity against librosa golden files."""

    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("rms_440hz_default.npy")
        result = metalmom.rms(y=signal)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="RMS mismatch vs librosa"
        )


class TestRMSProperties:
    """Verify structural properties."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.rms(y=signal)
        assert np.all(np.isfinite(result)), "All RMS values should be finite"

    def test_values_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.rms(y=signal)
        assert np.all(result >= 0), "RMS values should be non-negative"

    def test_silence_near_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.rms(y=silence)
        assert np.allclose(result, 0, atol=1e-6), "RMS of silence should be ~zero"

    def test_sine_wave_rms(self):
        """RMS of a sine wave should be amplitude / sqrt(2)."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.rms(y=signal)
        # Interior frames (skip edge effects from zero-padding)
        avg = result[0, 2:-2].mean()
        expected = 1.0 / np.sqrt(2.0)
        assert abs(avg - expected) < 0.02, (
            f"Expected RMS ~{expected:.4f}, got {avg:.4f}"
        )


# -- Zero-Crossing Rate --


class TestZCRShape:
    """Verify ZCR dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("zcr_440hz_default.npy")
        result = metalmom.zero_crossing_rate(y=signal)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.zero_crossing_rate(y=signal)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestZCRParity:
    """Element-wise parity against librosa golden files."""

    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("zcr_440hz_default.npy")
        result = metalmom.zero_crossing_rate(y=signal)

        # ZCR tolerance: 1/frame_length = 1/2048 ~ 4.88e-4 per crossing difference.
        # Float32 rounding at zero can cause +-1 crossing per frame boundary,
        # so atol = 1e-3 accounts for this (about 2 crossings of tolerance).
        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-3,
            err_msg="ZCR mismatch vs librosa"
        )


class TestZCRProperties:
    """Verify structural properties."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.zero_crossing_rate(y=signal)
        assert np.all(np.isfinite(result)), "All ZCR values should be finite"

    def test_range_zero_to_one(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.zero_crossing_rate(y=signal)
        assert np.all(result >= 0), "ZCR should be >= 0"
        assert np.all(result <= 1 + 1e-6), "ZCR should be <= 1"

    def test_silence_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.zero_crossing_rate(y=silence)
        assert np.allclose(result, 0, atol=1e-6), "ZCR of silence should be ~zero"

    def test_constant_zero(self):
        import metalmom
        constant = np.ones(22050, dtype=np.float32) * 0.5
        result = metalmom.zero_crossing_rate(y=constant)
        assert np.allclose(result, 0, atol=1e-6), "ZCR of constant should be ~zero"

    def test_440hz_zcr_expected(self):
        """440 Hz sine: ZCR should be ~2*440/22050 = ~0.0399."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.zero_crossing_rate(y=signal)
        avg = result[0, 2:-2].mean()
        expected = 2.0 * 440.0 / 22050.0
        assert abs(avg - expected) < 0.005, (
            f"Expected ZCR ~{expected:.4f}, got {avg:.4f}"
        )


# -- Compat Shim Tests --


class TestRMSZCRCompat:
    """Verify the librosa compat shim works for RMS and ZCR."""

    def test_compat_rms_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "rms")

    def test_compat_zcr_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "zero_crossing_rate")

    def test_compat_rms_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.rms(y=signal)
        assert result.shape == (1, 44), f"Unexpected shape: {result.shape}"

    def test_compat_zcr_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.zero_crossing_rate(y=signal)
        assert result.shape == (1, 44), f"Unexpected shape: {result.shape}"

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.rms(y=signal)
        assert result.shape[0] == 1
