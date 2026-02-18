"""MFCC parity tests: MetalMom (Swift via C bridge) vs librosa golden reference.

Tests the full pipeline: mel spectrogram -> power_to_dB -> DCT-II -> truncate.
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

Tier 1 tolerances: rtol=1e-4, atol=1e-4 (float32 vDSP accumulation + DCT rounding).
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


# -- Shape Tests --


class TestMFCCShape:
    """Verify MFCC dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_default.npy")
        result = metalmom.mfcc(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_custom_n_mfcc_n_fft_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_13_1024.npy")
        result = metalmom.mfcc(y=signal, sr=22050, n_mfcc=13, n_fft=1024, n_mels=40)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_custom_fmin_fmax_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_20_64_fmin300_fmax8000.npy")
        result = metalmom.mfcc(
            y=signal, sr=22050, n_mfcc=20, n_fft=2048, n_mels=64, fmin=300, fmax=8000
        )
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.mfcc(y=signal, sr=22050)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


# -- Parity Tests --


class TestMFCCParity:
    """Element-wise parity against librosa golden files."""

    def test_default_parity(self):
        """Default params: n_mfcc=20, n_fft=2048, n_mels=128."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_default.npy")
        result = metalmom.mfcc(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="Default MFCC mismatch vs librosa"
        )

    def test_custom_n_mfcc_n_fft_parity(self):
        """n_mfcc=13, n_fft=1024, n_mels=40."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_13_1024.npy")
        result = metalmom.mfcc(y=signal, sr=22050, n_mfcc=13, n_fft=1024, n_mels=40)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="Custom (13, 1024, 40) MFCC mismatch vs librosa"
        )

    def test_custom_fmin_fmax_parity(self):
        """n_mfcc=20, n_mels=64, fmin=300, fmax=8000.

        Slightly wider tolerance (atol=5e-4) for the fmin/fmax variant because
        the narrower mel band range amplifies float32 accumulation differences
        through the STFT -> mel FB -> dB -> DCT pipeline.  The max absolute
        error is ~2e-4 on values of magnitude ~300, well within practical precision.
        """
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mfcc_440hz_20_64_fmin300_fmax8000.npy")
        result = metalmom.mfcc(
            y=signal, sr=22050, n_mfcc=20, n_fft=2048, n_mels=64, fmin=300, fmax=8000
        )

        np.testing.assert_allclose(
            result, expected,
            rtol=5e-4, atol=5e-4,
            err_msg="Custom fmin/fmax MFCC mismatch vs librosa"
        )


# -- Property Tests --


class TestMFCCProperties:
    """Verify structural properties of MFCCs."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.mfcc(y=signal, sr=22050)
        assert np.all(np.isfinite(result)), "All MFCC values should be finite"

    def test_silence_finite(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.mfcc(y=silence, sr=22050)
        assert np.all(np.isfinite(result)), "MFCC of silence should be finite"

    def test_first_coeff_relates_to_energy(self):
        """c0 should differ between loud and quiet signals."""
        import metalmom
        signal = _load_signal_440()
        quiet = signal * 0.01
        mfcc_loud = metalmom.mfcc(y=signal, sr=22050)
        mfcc_quiet = metalmom.mfcc(y=quiet, sr=22050)

        # Average c0 should be larger for the louder signal
        assert np.mean(mfcc_loud[0]) > np.mean(mfcc_quiet[0]), (
            "c0 should be larger for louder signal"
        )


# -- Compat Shim Test --


class TestMFCCCompat:
    """Verify the librosa compat shim works."""

    def test_compat_shim_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "mfcc")

    def test_compat_shim_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.mfcc(y=signal, sr=22050)
        assert result.shape == (20, 44), f"Unexpected shape: {result.shape}"
        assert result.dtype == np.float32

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.mfcc(y=signal, sr=22050)
        assert result.shape[0] == 20
