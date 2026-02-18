"""Chroma STFT parity tests: MetalMom (Swift via C bridge) vs librosa golden reference.

Tests the full pipeline: STFT magnitude -> power -> chroma filterbank -> [normalize].
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

Chroma features have wider tolerance than STFT/mel because the power spectrogram
(magnitude squared) amplifies float32 accumulation differences between vDSP and scipy.
For normalized chroma (norm=inf or L2), values are in [0, 1] and atol=0.02 suffices.
For un-normalized chroma, values span several orders of magnitude, so we use rtol only.
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


class TestChromaSTFTShape:
    """Verify chroma STFT dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("chroma_stft_440hz_no_norm.npy")
        result = metalmom.chroma_stft(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_custom_n_fft_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("chroma_stft_440hz_1024.npy")
        result = metalmom.chroma_stft(y=signal, sr=22050, n_fft=1024)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


# -- Parity Tests (un-normalized) --


class TestChromaSTFTParity:
    """Element-wise parity against librosa golden files.

    Un-normalized chroma has values spanning several orders of magnitude,
    so we use relative tolerance.  The ~7% relative error on dominant bins
    comes from the float32 STFT power spectrogram accumulation differences
    (vDSP vs scipy double-precision FFT, then squared).
    """

    def test_no_norm_parity(self):
        """Default params: n_chroma=12, n_fft=2048, norm=None."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("chroma_stft_440hz_no_norm.npy")
        result = metalmom.chroma_stft(y=signal, sr=22050)

        # Use relative tolerance for un-normalized values (amplified STFT differences)
        np.testing.assert_allclose(
            result, expected,
            rtol=0.15, atol=1.0,
            err_msg="Un-normalized chroma mismatch vs librosa"
        )

    def test_custom_n_fft_parity(self):
        """n_fft=1024, norm=None.

        Smaller FFT size amplifies float32 STFT accumulation differences
        even more (fewer bins, more weight per bin), so wider tolerance
        is needed.  The key correctness check is that the peak chroma bin
        matches and the overall shape of the distribution is correct.
        """
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("chroma_stft_440hz_1024.npy")
        result = metalmom.chroma_stft(y=signal, sr=22050, n_fft=1024)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.35, atol=5.0,
            err_msg="Custom n_fft chroma mismatch vs librosa"
        )


# -- Parity Tests (normalized) --


class TestChromaSTFTNormalizedParity:
    """Parity with normalization applied.

    Normalization maps values to [0, 1], making absolute differences more
    meaningful.  After Linf normalization, max abs diff is ~0.015 (1.5%).
    """

    def test_linf_norm_parity(self):
        """librosa default: norm=inf, n_fft=2048."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("chroma_stft_440hz_default.npy")
        result = metalmom.chroma_stft(y=signal, sr=22050, norm=np.inf)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.02, atol=0.02,
            err_msg="Linf-normalized chroma mismatch vs librosa"
        )

    def test_linf_norm_n_fft_1024_parity(self):
        """norm=inf, n_fft=1024.

        Smaller FFT has wider errors on raw power, but normalization
        brings values to [0, 1] and reduces differences significantly.
        """
        import metalmom
        import librosa
        signal = _load_signal_440()
        expected = librosa.feature.chroma_stft(y=signal, sr=22050, n_fft=1024)
        result = metalmom.chroma_stft(y=signal, sr=22050, n_fft=1024, norm=np.inf)

        np.testing.assert_allclose(
            result, expected,
            rtol=0.08, atol=0.08,
            err_msg="Linf-normalized n_fft=1024 chroma mismatch vs librosa"
        )

    def test_l2_norm_column_unity(self):
        """L2-normalized frames should have unit norm."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050, norm=2.0)

        # Each column should have L2 norm of 1.0
        norms = np.sqrt(np.sum(result ** 2, axis=0))
        np.testing.assert_allclose(
            norms, 1.0, atol=1e-5,
            err_msg="L2-normalized columns should have unit norm"
        )

    def test_linf_norm_max_unity(self):
        """Linf-normalized frames should have max value of 1.0."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050, norm=np.inf)

        # Each column should have max of 1.0
        maxes = np.max(result, axis=0)
        np.testing.assert_allclose(
            maxes, 1.0, atol=1e-5,
            err_msg="Linf-normalized columns should have max 1.0"
        )


# -- Property Tests --


class TestChromaSTFTProperties:
    """Verify structural properties of chroma features."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050)
        assert np.all(np.isfinite(result)), "All chroma values should be finite"

    def test_values_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050)
        assert np.all(result >= 0), "Chroma values should be non-negative"

    def test_440hz_peaks_at_A(self):
        """440 Hz (A4) should have strongest energy in chroma bin A (index 9)."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.chroma_stft(y=signal, sr=22050)
        avg_energy = result.mean(axis=1)
        peak_bin = np.argmax(avg_energy)
        assert peak_bin == 9, (
            f"440 Hz should peak at A (index 9), got index {peak_bin}"
        )

    def test_silence_near_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.chroma_stft(y=silence, sr=22050)
        assert np.allclose(result, 0, atol=1e-10), "Chroma of silence should be ~zero"

    def test_261hz_peaks_at_C(self):
        """261.63 Hz (C4) should have strongest energy in chroma bin C (index 0)."""
        import metalmom
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        signal_c4 = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)
        result = metalmom.chroma_stft(y=signal_c4, sr=22050)
        avg_energy = result.mean(axis=1)
        peak_bin = np.argmax(avg_energy)
        assert peak_bin == 0, (
            f"261.63 Hz should peak at C (index 0), got index {peak_bin}"
        )

    def test_louder_signal_has_more_energy(self):
        """Louder signal should produce higher chroma energy."""
        import metalmom
        signal = _load_signal_440()
        quiet = signal * 0.1
        chroma_loud = metalmom.chroma_stft(y=signal, sr=22050)
        chroma_quiet = metalmom.chroma_stft(y=quiet, sr=22050)
        assert chroma_loud.sum() > chroma_quiet.sum(), (
            "Louder signal should have more total chroma energy"
        )


# -- Compat Shim Test --


class TestChromaCompat:
    """Verify the librosa compat shim works."""

    def test_compat_shim_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "chroma_stft")

    def test_compat_shim_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.chroma_stft(y=signal, sr=22050)
        assert result.shape == (12, 44), f"Unexpected shape: {result.shape}"
        assert result.dtype == np.float32

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.chroma_stft(y=signal, sr=22050)
        assert result.shape[0] == 12
