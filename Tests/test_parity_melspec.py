"""Mel spectrogram parity tests: MetalMom (Swift via C bridge) vs librosa golden reference.

Tests the full pipeline: STFT magnitude -> power -> mel filterbank multiplication.
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

Tier 1 tolerances: rtol=1e-4, atol=1e-4 (float32 STFT + vDSP_mmul accumulation).
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


# ── Shape Tests ──


class TestMelSpectrogramShape:
    """Verify mel spectrogram dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_default.npy")
        result = metalmom.melspectrogram(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_custom_n_mels_n_fft_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_40_1024.npy")
        result = metalmom.melspectrogram(y=signal, sr=22050, n_fft=1024, n_mels=40)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_custom_fmin_fmax_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_64_fmin300_fmax8000.npy")
        result = metalmom.melspectrogram(
            y=signal, sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000
        )
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.melspectrogram(y=signal, sr=22050)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


# ── Parity Tests ──


class TestMelSpectrogramParity:
    """Element-wise parity against librosa golden files."""

    def test_default_parity(self):
        """Default params: nMels=128, nFFT=2048, power=2.0."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_default.npy")
        result = metalmom.melspectrogram(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="Default mel spectrogram mismatch vs librosa"
        )

    def test_custom_n_mels_n_fft_parity(self):
        """nMels=40, nFFT=1024."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_40_1024.npy")
        result = metalmom.melspectrogram(y=signal, sr=22050, n_fft=1024, n_mels=40)

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="Custom (40, 1024) mel spectrogram mismatch vs librosa"
        )

    def test_custom_fmin_fmax_parity(self):
        """nMels=64, nFFT=2048, fmin=300, fmax=8000."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("mel_spectrogram_440hz_64_fmin300_fmax8000.npy")
        result = metalmom.melspectrogram(
            y=signal, sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000
        )

        np.testing.assert_allclose(
            result, expected,
            rtol=1e-4, atol=1e-4,
            err_msg="Custom fmin/fmax mel spectrogram mismatch vs librosa"
        )


# ── Property Tests ──


class TestMelSpectrogramProperties:
    """Verify structural properties of the mel spectrogram."""

    def test_non_negative(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.melspectrogram(y=signal, sr=22050)
        assert np.all(result >= 0), "Mel spectrogram should be non-negative"

    def test_silence_near_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.melspectrogram(y=silence, sr=22050)
        assert np.max(result) < 1e-10, "Silent signal should produce near-zero mel spec"

    def test_energy_in_correct_bands(self):
        """440 Hz tone should have energy concentrated in lower-mid mel bands."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.melspectrogram(y=signal, sr=22050, n_mels=128)

        # Sum energy across frames for each mel band
        band_energy = result.sum(axis=1)
        peak_band = np.argmax(band_energy)

        # 440 Hz should be in roughly mel bands 10-40 (lower-mid range)
        assert 5 < peak_band < 60, (
            f"Peak energy at band {peak_band}, expected in lower-mid range"
        )


# ── Compat Shim Test ──


class TestMelSpectrogramCompat:
    """Verify the librosa compat shim works."""

    def test_compat_shim_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "melspectrogram")

    def test_compat_shim_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.melspectrogram(y=signal, sr=22050)
        assert result.shape == (128, 44), f"Unexpected shape: {result.shape}"
        assert result.dtype == np.float32

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.melspectrogram(y=signal, sr=22050)
        assert result.shape[0] == 128
