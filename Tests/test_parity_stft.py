"""STFT parity tests: MetalMom vs librosa golden reference files.

Tier 1 tolerances: element-wise comparison with rtol=1e-4, atol=1e-4.
Note: Some tolerance is expected due to:
- float32 vDSP FFT vs librosa's float64 internal computation
- Windowing implementation differences
- Padding implementation differences
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


def test_stft_shape_matches_librosa():
    """Verify MetalMom STFT output shape matches librosa."""
    from metalmom import stft

    signal = _load_golden("signal_440hz_22050sr.npy")
    expected = _load_golden("stft_440hz_default_magnitude.npy")

    result = stft(signal, n_fft=2048, hop_length=512, win_length=2048, center=True)

    assert result.shape == expected.shape, (
        f"Shape mismatch: MetalMom {result.shape} vs librosa {expected.shape}"
    )


def test_stft_magnitude_parity():
    """Verify MetalMom STFT magnitude matches librosa within Tier 1 tolerance."""
    from metalmom import stft

    signal = _load_golden("signal_440hz_22050sr.npy")
    expected = _load_golden("stft_440hz_default_magnitude.npy")

    result = stft(signal, n_fft=2048, hop_length=512, win_length=2048, center=True)

    # Tier 1: element-wise tolerance
    np.testing.assert_allclose(
        result, expected,
        rtol=1e-4, atol=1e-4,
        err_msg="STFT magnitude mismatch vs librosa"
    )


def test_stft_peak_frequency():
    """Verify the dominant frequency bin matches between MetalMom and librosa."""
    from metalmom import stft

    signal = _load_golden("signal_440hz_22050sr.npy")
    expected = _load_golden("stft_440hz_default_magnitude.npy")

    result = stft(signal, n_fft=2048, hop_length=512, win_length=2048, center=True)

    # Check peak bin in a middle frame
    mid_frame = result.shape[1] // 2
    mm_peak = np.argmax(result[:, mid_frame])
    librosa_peak = np.argmax(expected[:, mid_frame])

    assert mm_peak == librosa_peak, (
        f"Peak bin mismatch: MetalMom bin {mm_peak} vs librosa bin {librosa_peak}"
    )


def test_stft_energy_ratio():
    """Verify total energy is within 1% of librosa."""
    from metalmom import stft

    signal = _load_golden("signal_440hz_22050sr.npy")
    expected = _load_golden("stft_440hz_default_magnitude.npy")

    result = stft(signal, n_fft=2048, hop_length=512, win_length=2048, center=True)

    mm_energy = np.sum(result ** 2)
    librosa_energy = np.sum(expected ** 2)

    ratio = mm_energy / librosa_energy
    assert 0.99 < ratio < 1.01, (
        f"Energy ratio out of range: {ratio:.4f} (expected ~1.0)"
    )
