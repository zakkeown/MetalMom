"""Tonnetz parity tests: MetalMom (Swift via C bridge) vs librosa golden reference.

Tests the full pipeline: audio -> chroma_stft -> L1 normalize -> angular projection.
Uses the same 440 Hz test signal and librosa golden files as other parity tests.

Tonnetz involves chroma -> normalization -> trig projections.  Since we use
chroma_stft (matching the golden file generation), parity should be close, but
float32 accumulation differences in the chroma stage propagate through the
angular projections.
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


class TestTonnetzShape:
    """Verify tonnetz dimensions match librosa."""

    def test_default_shape(self):
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("tonnetz_440hz_default.npy")
        result = metalmom.tonnetz(y=signal, sr=22050)
        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )

    def test_dtype_float32(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.tonnetz(y=signal, sr=22050)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_6_dimensions(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.tonnetz(y=signal, sr=22050)
        assert result.shape[0] == 6, f"Expected 6 tonnetz dimensions, got {result.shape[0]}"


# -- Parity Tests --


class TestTonnetzParity:
    """Element-wise parity against librosa golden files.

    The golden file was generated using chroma_stft (not chroma_cqt),
    matching our implementation.  Tolerance accounts for float32
    chroma differences propagated through L1 normalization and
    angular projections.
    """

    def test_default_parity(self):
        """Default params: n_fft=2048, hop_length=512."""
        import metalmom
        signal = _load_signal_440()
        expected = _load_golden("tonnetz_440hz_default.npy")
        result = metalmom.tonnetz(y=signal, sr=22050)

        np.testing.assert_allclose(
            result, expected,
            atol=0.05, rtol=0.15,
            err_msg="Tonnetz mismatch vs librosa"
        )


# -- Property Tests --


class TestTonnetzProperties:
    """Verify structural properties of tonnetz features."""

    def test_values_are_finite(self):
        import metalmom
        signal = _load_signal_440()
        result = metalmom.tonnetz(y=signal, sr=22050)
        assert np.all(np.isfinite(result)), "All tonnetz values should be finite"

    def test_value_range_bounded(self):
        """Tonnetz values should be bounded by the radii (r1=1, r2=1, r3=0.5)."""
        import metalmom
        signal = _load_signal_440()
        result = metalmom.tonnetz(y=signal, sr=22050)

        # Dims 0-3 have radius 1.0, dims 4-5 have radius 0.5
        for d in range(4):
            assert np.all(np.abs(result[d]) <= 1.0 + 1e-6), (
                f"Tonnetz dim {d} exceeds radius 1.0"
            )
        for d in range(4, 6):
            assert np.all(np.abs(result[d]) <= 0.5 + 1e-6), (
                f"Tonnetz dim {d} exceeds radius 0.5"
            )

    def test_silence_near_zero(self):
        import metalmom
        silence = np.zeros(22050, dtype=np.float32)
        result = metalmom.tonnetz(y=silence, sr=22050)
        assert np.allclose(result, 0, atol=1e-6), "Tonnetz of silence should be ~zero"

    def test_different_pitches_differ(self):
        """Different pitches should produce different tonnetz features."""
        import metalmom
        signal_440 = _load_signal_440()
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        signal_261 = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)

        tonnetz_440 = metalmom.tonnetz(y=signal_440, sr=22050)
        tonnetz_261 = metalmom.tonnetz(y=signal_261, sr=22050)

        # At least one dimension should differ meaningfully
        max_diff = np.max(np.abs(tonnetz_440.mean(axis=1) - tonnetz_261.mean(axis=1)))
        assert max_diff > 0.01, "Different pitches should produce different tonnetz"

    def test_pre_computed_chroma_path(self):
        """Verify the pre-computed chroma Python path works."""
        import metalmom
        signal = _load_signal_440()

        # Compute chroma, then pass to tonnetz
        chroma = metalmom.chroma_stft(y=signal, sr=22050)
        result_from_chroma = metalmom.feature.tonnetz(chroma=chroma)

        # Compare with direct audio path
        result_from_audio = metalmom.tonnetz(y=signal, sr=22050)

        np.testing.assert_allclose(
            result_from_chroma, result_from_audio,
            atol=0.05, rtol=0.1,
            err_msg="Pre-computed chroma path should match audio path"
        )


# -- Compat Shim Tests --


class TestTonnetzCompat:
    """Verify the librosa compat shim works."""

    def test_compat_shim_import(self):
        from metalmom.compat.librosa import feature
        assert hasattr(feature, "tonnetz")

    def test_compat_shim_result(self):
        from metalmom.compat.librosa import feature
        signal = _load_signal_440()
        result = feature.tonnetz(y=signal, sr=22050)
        assert result.shape[0] == 6
        assert result.dtype == np.float32

    def test_compat_via_librosa_module(self):
        from metalmom.compat import librosa
        signal = _load_signal_440()
        result = librosa.feature.tonnetz(y=signal, sr=22050)
        assert result.shape[0] == 6
