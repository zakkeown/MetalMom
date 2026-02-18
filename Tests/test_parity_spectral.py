"""Spectral descriptor parity tests: MetalMom (Swift via C bridge) vs librosa golden reference.

Tests the full pipeline for: spectral centroid, bandwidth, contrast, rolloff, flatness.
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

Spectral descriptors operate on the magnitude spectrogram. Differences arise from
float32 STFT accumulation differences between vDSP and scipy's double-precision FFT.
- Centroid/rolloff: tested in Hz, so absolute tolerance is in Hz (a few frequency bins).
- Bandwidth: similar Hz-domain tolerance.
- Contrast: dB-scale differences, wider tolerance due to log amplification of small values.
- Flatness: values in [0, 1], tight tolerance.
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


# -- Spectral Centroid --


class TestSpectralCentroidShape:
    """Verify spectral centroid dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_centroid_440hz_default.npy")
        result = metalmom.spectral_centroid(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_centroid(y=signal, sr=22050)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestSpectralCentroidParity:
    """Element-wise parity against librosa golden files."""

    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_centroid_440hz_default.npy")
        result = metalmom.spectral_centroid(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.05, atol=20.0,
            err_msg="Spectral centroid mismatch vs librosa"
        )


class TestSpectralCentroidProperties:
    """Verify structural properties."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_centroid(y=signal, sr=22050)
        assert np.all(np.isfinite(result)), "All centroid values should be finite"

    def test_values_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_centroid(y=signal, sr=22050)
        assert np.all(result >= 0), "Centroid values should be non-negative"

    def test_440hz_near_expected(self):
        """440 Hz sine centroid should be near 440 Hz."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_centroid(y=signal, sr=22050)
        avg = result[0, 2:-2].mean()
        assert 390 < avg < 650, f"Expected centroid near 440 Hz, got {avg}"

    def test_silence_near_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.spectral_centroid(y=silence, sr=22050)
        assert np.allclose(result, 0, atol=1e-6), "Centroid of silence should be ~zero"


# -- Spectral Bandwidth --


class TestSpectralBandwidthShape:
    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_bandwidth_440hz_default.npy")
        result = metalmom.spectral_bandwidth(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )


class TestSpectralBandwidthParity:
    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_bandwidth_440hz_default.npy")
        result = metalmom.spectral_bandwidth(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.10, atol=50.0,
            err_msg="Spectral bandwidth mismatch vs librosa"
        )


class TestSpectralBandwidthProperties:
    def test_values_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_bandwidth(y=signal, sr=22050)
        assert np.all(result >= 0), "Bandwidth values should be non-negative"

    def test_pure_tone_narrow(self):
        """Pure sine: bandwidth should be narrow."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_bandwidth(y=signal, sr=22050)
        avg = result[0, 2:-2].mean()
        assert avg < 500, f"Pure tone bandwidth should be narrow, got {avg}"


# -- Spectral Contrast --


class TestSpectralContrastShape:
    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_contrast_440hz_default.npy")
        result = metalmom.spectral_contrast(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )


class TestSpectralContrastParity:
    def test_default_parity(self):
        """Contrast values in dB scale. Wider tolerance due to log amplification."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_contrast_440hz_default.npy")
        result = metalmom.spectral_contrast(y=signal, sr=22050)

        # Contrast is in dB scale and depends heavily on implementation details
        # of band edge computation and quantile estimation, so use wider tolerance
        np.testing.assert_allclose(
            result, expected,
            rtol=0.5, atol=15.0,
            err_msg="Spectral contrast mismatch vs librosa"
        )


class TestSpectralContrastProperties:
    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_contrast(y=signal, sr=22050)
        assert np.all(np.isfinite(result)), "All contrast values should be finite"


# -- Spectral Rolloff --


class TestSpectralRolloffShape:
    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_rolloff_440hz_default.npy")
        result = metalmom.spectral_rolloff(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )


class TestSpectralRolloffParity:
    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_rolloff_440hz_default.npy")
        result = metalmom.spectral_rolloff(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.05, atol=50.0,
            err_msg="Spectral rolloff mismatch vs librosa"
        )


class TestSpectralRolloffProperties:
    def test_values_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_rolloff(y=signal, sr=22050)
        assert np.all(result >= 0), "Rolloff values should be non-negative"

    def test_below_nyquist(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_rolloff(y=signal, sr=22050)
        assert np.all(result <= 11025 + 1), "Rolloff should not exceed Nyquist"

    def test_silence_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.spectral_rolloff(y=silence, sr=22050)
        assert np.allclose(result, 0, atol=1e-6), "Rolloff of silence should be ~zero"


# -- Spectral Flatness --


class TestSpectralFlatnessShape:
    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_flatness_440hz_default.npy")
        result = metalmom.spectral_flatness(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )


class TestSpectralFlatnessParity:
    def test_default_parity(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("spectral_flatness_440hz_default.npy")
        result = metalmom.spectral_flatness(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.3, atol=1e-4,
            err_msg="Spectral flatness mismatch vs librosa"
        )


class TestSpectralFlatnessProperties:
    def test_range_zero_to_one(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_flatness(y=signal, sr=22050)
        assert np.all(result >= 0), "Flatness should be >= 0"
        assert np.all(result <= 1 + 1e-6), "Flatness should be <= 1"

    def test_pure_tone_low(self):
        """Pure sine: flatness should be low (tonal)."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.spectral_flatness(y=signal, sr=22050)
        avg = result[0, 2:-2].mean()
        assert avg < 0.01, f"Pure tone flatness should be very low, got {avg}"

    def test_silence_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.spectral_flatness(y=silence, sr=22050)
        assert np.allclose(result, 0, atol=1e-6), "Flatness of silence should be ~zero"


# -- Compat Shim Tests --


class TestSpectralCompat:
    """Verify the librosa compat shim works for all spectral descriptors."""

    def test_compat_centroid_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "spectral_centroid")

    def test_compat_bandwidth_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "spectral_bandwidth")

    def test_compat_contrast_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "spectral_contrast")

    def test_compat_rolloff_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "spectral_rolloff")

    def test_compat_flatness_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "spectral_flatness")

    def test_compat_centroid_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.spectral_centroid(y=signal, sr=22050)
        assert result.shape == (1, 44), f"Unexpected shape: {result.shape}"

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.spectral_centroid(y=signal, sr=22050)
        assert result.shape[0] == 1
