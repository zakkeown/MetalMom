"""Edge case tests for the MetalMom Python bindings.

Covers dtype conversion, memory layout, signal length extremes,
idempotency, concurrency, and degenerate parameter values.
"""

import concurrent.futures

import numpy as np
import pytest

import metalmom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq=440.0, sr=22050, duration=1.0, dtype=np.float32):
    """Return a normalised sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    y = np.sin(2 * np.pi * freq * t).astype(dtype)
    return y


# ---------------------------------------------------------------------------
# 1-4  Dtype / memory-layout edge cases
# ---------------------------------------------------------------------------

class TestDtypeEdgeCases:

    def test_int32_input_converted(self):
        """Pass int32 array to stft; verify auto-conversion to float32."""
        np.random.seed(42)
        # Create int32 signal with values in a reasonable audio range
        y_int = (np.random.randn(4096) * 1000).astype(np.int32)

        result = metalmom.stft(y=y_int, n_fft=2048, hop_length=512)

        assert result.dtype == np.float32, (
            f"Expected float32 output, got {result.dtype}"
        )
        assert result.ndim == 2
        assert result.shape[0] == 1025  # n_fft/2 + 1

    def test_float64_input_large(self):
        """Pass 100k-sample float64 array; verify output is float32."""
        np.random.seed(42)
        y_f64 = np.random.randn(100_000).astype(np.float64)
        y_f64 /= np.max(np.abs(y_f64))  # normalise to [-1, 1]

        result = metalmom.stft(y=y_f64, n_fft=2048, hop_length=512)

        assert result.dtype == np.float32
        assert result.shape[0] == 1025
        assert result.shape[1] > 0

    def test_fortran_order_input(self):
        """Pass a non-C-contiguous view from a Fortran-order array.

        A 1D numpy array is always both C- and F-contiguous, so we
        place the signal into a (2, n) Fortran-order array and extract
        a row.  Rows of an F-order matrix have a stride > element size,
        which makes the resulting 1D view non-C-contiguous.
        """
        y_c = _sine(440.0, duration=0.5)
        n = len(y_c)
        # Build (2, n) Fortran-order; row 0 holds the signal.
        arr_2d = np.zeros((2, n), dtype=np.float32, order='F')
        arr_2d[0, :] = y_c
        y_row = arr_2d[0, :]
        # Row view of an F-order (2, n) array is non-contiguous
        assert not y_row.flags['C_CONTIGUOUS']

        result = metalmom.stft(y=y_row, n_fft=2048, hop_length=512)

        assert result.dtype == np.float32
        assert result.shape[0] == 1025
        assert result.shape[1] > 0

    def test_non_contiguous_slice(self):
        """Pass a non-contiguous slice (signal[::2]); verify it works."""
        y_full = _sine(440.0, duration=1.0)
        y_slice = y_full[::2]
        assert not y_slice.flags['C_CONTIGUOUS']

        result = metalmom.stft(y=y_slice, n_fft=2048, hop_length=512)

        assert result.dtype == np.float32
        assert result.shape[0] == 1025
        assert result.shape[1] > 0


# ---------------------------------------------------------------------------
# 5-7  Shape edge cases
# ---------------------------------------------------------------------------

class TestShapeEdgeCases:

    def test_short_signal_stft(self):
        """10-sample signal through stft with center padding."""
        y = np.zeros(10, dtype=np.float32)
        y[5] = 1.0  # impulse

        result = metalmom.stft(y=y, n_fft=2048, hop_length=512, center=True)

        assert result.dtype == np.float32
        assert result.shape[0] == 1025
        # With center=True padding, even a tiny signal produces at least 1 frame
        assert result.shape[1] >= 1

    def test_very_long_signal(self):
        """2-minute signal (~2.6M samples); verify correct output shape."""
        sr = 22050
        duration = 120  # seconds
        n_samples = sr * duration
        np.random.seed(42)
        y = np.random.randn(n_samples).astype(np.float32)
        y /= np.max(np.abs(y))

        n_fft = 2048
        hop_length = 512
        result = metalmom.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=True)

        assert result.dtype == np.float32
        assert result.shape[0] == n_fft // 2 + 1
        # Expected frame count with center padding:
        # ceil(n_samples / hop_length) + 1 (approximately)
        expected_frames = 1 + n_samples // hop_length
        # Allow some tolerance for rounding differences
        assert abs(result.shape[1] - expected_frames) <= 2, (
            f"Frame count {result.shape[1]} too far from expected {expected_frames}"
        )

    def test_single_sample_mel(self):
        """1-sample signal through melspectrogram; verify no crash."""
        y = np.array([0.5], dtype=np.float32)

        result = metalmom.feature.melspectrogram(y=y, sr=22050)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # Should produce some output (at least 1 frame with center padding)
        assert result.size > 0


# ---------------------------------------------------------------------------
# 8-9  Repeated / concurrent operations
# ---------------------------------------------------------------------------

class TestRepeatedConcurrent:

    def test_idempotent_stft(self):
        """Run stft twice on the same input; outputs must be bit-identical."""
        y = _sine(440.0, duration=0.5)

        result1 = metalmom.stft(y=y, n_fft=2048, hop_length=512)
        result2 = metalmom.stft(y=y, n_fft=2048, hop_length=512)

        np.testing.assert_array_equal(
            result1, result2,
            err_msg="Two identical stft calls produced different results"
        )

    def test_concurrent_stft(self):
        """Run 4 stft calls in parallel on different signals.

        mm_context is NOT thread-safe, but each call in metalmom.stft()
        creates and destroys its own context, so concurrent calls should
        be safe.
        """
        frequencies = [220, 440, 880, 1760]
        np.random.seed(42)
        signals = [_sine(f, duration=0.5) for f in frequencies]

        def run_stft(y):
            return metalmom.stft(y=y, n_fft=2048, hop_length=512)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_stft, s) for s in signals]
            results = [f.result() for f in futures]

        for i, result in enumerate(results):
            assert result.dtype == np.float32, (
                f"Signal {i} (freq={frequencies[i]}): wrong dtype {result.dtype}"
            )
            assert result.shape[0] == 1025, (
                f"Signal {i}: expected 1025 freq bins, got {result.shape[0]}"
            )
            assert result.shape[1] > 0, (
                f"Signal {i}: expected >0 frames, got {result.shape[1]}"
            )


# ---------------------------------------------------------------------------
# 10-12  Compat / degenerate-parameter edge cases
# ---------------------------------------------------------------------------

class TestCompatEdgeCases:

    def test_empty_onset_detect(self):
        """Very short silence signal; onset_detect may return empty array."""
        y = np.zeros(100, dtype=np.float32)

        frames = metalmom.onset_detect(y=y, sr=22050)

        assert isinstance(frames, np.ndarray)
        assert frames.ndim == 1
        # Silent 100-sample signal should produce 0 onsets (or at most very few)
        # The key assertion is no crash.

    def test_zero_sample_rate_degrades_gracefully(self):
        """sr=0 should either raise or produce degenerate (NaN) output.

        The native backend does not validate sr=0 upfront, so the mel
        filterbank computation produces NaN values.  We verify the call
        does not crash and that the output signals the problem via NaN.
        """
        y = _sine(440.0, duration=0.5)

        try:
            result = metalmom.feature.melspectrogram(y=y, sr=0)
            # If it doesn't raise, the output should be degenerate (NaN)
            assert np.any(np.isnan(result)), (
                "sr=0 did not raise and did not produce NaN -- unexpected"
            )
        except (RuntimeError, ValueError, ZeroDivisionError):
            # Raising is also acceptable graceful handling
            pass

    def test_single_mel_band(self):
        """melspectrogram with n_mels=1; verify shape[0] == 1."""
        y = _sine(440.0, duration=0.5)

        result = metalmom.feature.melspectrogram(y=y, sr=22050, n_mels=1)

        assert result.dtype == np.float32
        assert result.shape[0] == 1, (
            f"Expected 1 mel band, got {result.shape[0]}"
        )
        assert result.shape[1] > 0
