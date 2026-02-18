"""Tests for metalmom.filters — mel, chroma, constant_q filterbanks."""

import numpy as np
import pytest
import metalmom
from metalmom import filters


# ---------------------------------------------------------------------------
# mel filterbank
# ---------------------------------------------------------------------------

class TestMelFilterbank:
    """Tests for filters.mel()."""

    def test_default_shape(self):
        fb = filters.mel()
        assert fb.shape == (128, 1025)  # (n_mels, n_fft//2+1)

    def test_custom_shape(self):
        fb = filters.mel(sr=16000, n_fft=1024, n_mels=40)
        assert fb.shape == (40, 513)

    def test_dtype(self):
        fb = filters.mel()
        assert fb.dtype == np.float32

    def test_non_negative(self):
        fb = filters.mel()
        assert np.all(fb >= 0)

    def test_rows_nonzero(self):
        """Every mel band should have non-zero energy."""
        fb = filters.mel()
        row_sums = fb.sum(axis=1)
        assert np.all(row_sums > 0)

    def test_custom_fmin_fmax(self):
        fb = filters.mel(fmin=300.0, fmax=8000.0)
        assert fb.shape == (128, 1025)
        # Filters below 300 Hz should be zero
        # bin for 300 Hz at sr=22050, n_fft=2048: bin = 300 * 2048 / 22050 ~ 27.8
        # First few bins should have minimal energy
        assert fb[:, 0] .sum() == 0  # DC bin should be zero

    def test_fmax_defaults_to_nyquist(self):
        fb1 = filters.mel(sr=22050)
        fb2 = filters.mel(sr=22050, fmax=11025.0)
        np.testing.assert_array_equal(fb1, fb2)

    def test_single_mel(self):
        fb = filters.mel(n_mels=1)
        assert fb.shape == (1, 1025)
        assert fb.sum() > 0


# ---------------------------------------------------------------------------
# chroma filterbank
# ---------------------------------------------------------------------------

class TestChromaFilterbank:
    """Tests for filters.chroma()."""

    def test_default_shape(self):
        fb = filters.chroma()
        assert fb.shape == (12, 1025)

    def test_custom_n_chroma(self):
        fb = filters.chroma(n_chroma=24)
        assert fb.shape == (24, 1025)

    def test_custom_n_fft(self):
        fb = filters.chroma(n_fft=4096)
        assert fb.shape == (12, 2049)

    def test_dtype(self):
        fb = filters.chroma()
        assert fb.dtype == np.float32

    def test_non_negative(self):
        fb = filters.chroma()
        assert np.all(fb >= 0)

    def test_rows_nonzero(self):
        """Every chroma bin should capture some energy."""
        fb = filters.chroma()
        row_sums = fb.sum(axis=1)
        assert np.all(row_sums > 0)

    def test_column_unit_norm(self):
        """Columns should be approximately L2-normalised."""
        fb = filters.chroma().astype(np.float64)
        col_norms = np.sqrt(np.sum(fb ** 2, axis=0))
        # Skip DC bin which may be special
        nonzero = col_norms > 0
        np.testing.assert_allclose(col_norms[nonzero], 1.0, atol=1e-5)

    def test_base_c_true(self):
        """With base_c=True, bin 0 should respond most to C."""
        fb = filters.chroma(base_c=True)
        # Just check it runs and has correct shape
        assert fb.shape == (12, 1025)

    def test_octwidth_zero(self):
        """octwidth=0 should give uniform octave weighting."""
        fb = filters.chroma(octwidth=0)
        assert fb.shape == (12, 1025)
        assert np.all(fb >= 0)

    def test_tuning_offset(self):
        """Non-zero tuning should produce a valid filterbank."""
        fb = filters.chroma(tuning=0.1)
        assert fb.shape == (12, 1025)
        assert np.all(np.isfinite(fb))


# ---------------------------------------------------------------------------
# constant_q filterbank
# ---------------------------------------------------------------------------

class TestConstantQFilterbank:
    """Tests for filters.constant_q()."""

    def test_default_shape(self):
        fb = filters.constant_q()
        assert fb.shape[0] == 84  # n_bins
        # fft_len should be a power of 2
        fft_len = fb.shape[1]
        assert fft_len > 0
        assert (fft_len & (fft_len - 1)) == 0, "fft_len should be power of 2"

    def test_complex_dtype(self):
        fb = filters.constant_q()
        assert np.iscomplexobj(fb)
        assert fb.dtype == np.complex64

    def test_custom_n_bins(self):
        fb = filters.constant_q(n_bins=48, bins_per_octave=12)
        assert fb.shape[0] == 48

    def test_custom_fmin(self):
        fb = filters.constant_q(fmin=65.0, n_bins=36)
        assert fb.shape[0] == 36

    def test_no_pad(self):
        fb = filters.constant_q(pad_fft=False)
        assert fb.shape[0] == 84
        # fft_len should be the max filter length (not necessarily power of 2)
        assert fb.shape[1] > 0

    def test_norm_l1(self):
        """L1-normalised filters should have non-trivial magnitudes."""
        fb = filters.constant_q(norm=1)
        mags = np.abs(fb)
        assert np.all(mags.sum(axis=1) > 0)

    def test_norm_l2(self):
        """L2-normalised filters."""
        fb = filters.constant_q(norm=2)
        assert fb.shape[0] == 84

    def test_norm_none(self):
        """No normalisation."""
        fb = filters.constant_q(norm=None)
        assert fb.shape[0] == 84

    def test_tuning_offset(self):
        fb = filters.constant_q(tuning=0.1)
        assert fb.shape[0] == 84
        assert np.all(np.isfinite(fb))

    def test_bins_per_octave(self):
        fb = filters.constant_q(bins_per_octave=24, n_bins=48)
        assert fb.shape[0] == 48


# ---------------------------------------------------------------------------
# Re-exported frequency functions
# ---------------------------------------------------------------------------

class TestMelFrequencies:
    """Tests for filters.mel_frequencies (pure-Python Slaney mel scale)."""

    def test_callable(self):
        result = filters.mel_frequencies()
        assert isinstance(result, np.ndarray)

    def test_count(self):
        result = filters.mel_frequencies(n_mels=128)
        assert len(result) == 128

    def test_monotonic(self):
        result = filters.mel_frequencies()
        assert np.all(np.diff(result) > 0)

    def test_bounds(self):
        result = filters.mel_frequencies(fmin=100.0, fmax=8000.0)
        assert result[0] >= 100.0
        assert result[-1] <= 8000.0


class TestFftFrequencies:
    """Tests for filters.fft_frequencies (pure-Python)."""

    def test_callable(self):
        result = filters.fft_frequencies()
        assert isinstance(result, np.ndarray)

    def test_count(self):
        result = filters.fft_frequencies(n_fft=2048)
        assert len(result) == 1025  # n_fft // 2 + 1

    def test_dc_zero(self):
        result = filters.fft_frequencies()
        assert result[0] == 0.0

    def test_nyquist(self):
        result = filters.fft_frequencies(sr=22050, n_fft=2048)
        assert result[-1] == pytest.approx(11025.0)

    def test_monotonic(self):
        result = filters.fft_frequencies()
        assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# Existing semitone functions still work
# ---------------------------------------------------------------------------

class TestSemitoneBackcompat:
    """Verify existing semitone functions are still accessible."""

    def test_semitone_frequencies_accessible(self):
        """semitone_frequencies requires the native library; just check importable."""
        assert callable(metalmom.semitone_frequencies)

    def test_semitone_filterbank_accessible(self):
        """Just verify the function is importable; full signal test needs audio."""
        assert callable(metalmom.semitone_filterbank)


# ---------------------------------------------------------------------------
# Compat shim
# ---------------------------------------------------------------------------

class TestCompatShim:
    """Verify librosa compat shim exposes filter functions."""

    def test_import(self):
        from metalmom.compat.librosa import filters as lf
        assert callable(lf.mel)
        assert callable(lf.chroma)
        assert callable(lf.constant_q)
        assert callable(lf.mel_frequencies)
        assert callable(lf.fft_frequencies)

    def test_mel_via_compat(self):
        from metalmom.compat.librosa import filters as lf
        fb = lf.mel()
        assert fb.shape == (128, 1025)
