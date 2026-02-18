"""Tests for the Python public API (core.stft)."""

import numpy as np
import pytest


def test_stft_basic():
    """Test basic STFT computation via public API."""
    from metalmom import stft

    # 1 second of 440 Hz sine at 22050 Hz
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)

    result = stft(y)

    # Shape: (n_fft/2 + 1, n_frames)
    assert result.shape[0] == 1025
    assert result.shape[1] > 0
    assert result.dtype == np.float32


def test_stft_custom_params():
    """Test STFT with custom FFT parameters."""
    from metalmom import stft

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)

    result = stft(y, n_fft=1024, hop_length=256)

    assert result.shape[0] == 513  # 1024/2 + 1
    assert result.shape[1] > 0


def test_stft_float64_input():
    """Test that float64 input is automatically converted."""
    from metalmom import stft

    y = np.sin(np.linspace(0, 2 * np.pi, 4096))  # float64 by default
    result = stft(y)

    assert result.dtype == np.float32
    assert result.shape[0] == 1025


def test_stft_no_center():
    """Test STFT without centering."""
    from metalmom import stft

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)

    result_center = stft(y, center=True)
    result_no_center = stft(y, center=False)

    # Without centering, should have fewer frames
    assert result_no_center.shape[1] < result_center.shape[1]
