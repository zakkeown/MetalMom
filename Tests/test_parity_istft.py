"""iSTFT parity tests: MetalMom vs librosa.

Tests that MetalMom's istft produces the same output as librosa's istft
when given the same complex STFT input.

Tier 1 tolerances: rtol=1e-4, atol=1e-4.
"""

import numpy as np
import pytest


def _has_librosa():
    try:
        import librosa
        return True
    except ImportError:
        return False


requires_librosa = pytest.mark.skipif(
    not _has_librosa(), reason="librosa not installed"
)


@requires_librosa
def test_istft_round_trip_sine():
    """STFT -> iSTFT round-trip should reconstruct a sine wave."""
    import librosa
    from metalmom import istft

    sr = 22050
    duration = 1.0
    freq = 440.0
    t = np.arange(int(sr * duration)) / sr
    y = np.sin(2 * np.pi * freq * t).astype(np.float32)

    n_fft = 2048
    hop_length = n_fft // 4

    # Use librosa's complex STFT as input
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)

    # Reconstruct with MetalMom
    y_mm = istft(S, hop_length=hop_length, center=True, length=len(y))

    # Reconstruct with librosa for reference
    y_lr = librosa.istft(S, hop_length=hop_length, center=True, length=len(y))

    # Both should be close to original
    np.testing.assert_allclose(
        y_mm, y_lr, rtol=1e-4, atol=1e-4,
        err_msg="MetalMom istft vs librosa istft mismatch (sine)"
    )


@requires_librosa
def test_istft_round_trip_random():
    """STFT -> iSTFT round-trip with random signal."""
    import librosa
    from metalmom import istft

    rng = np.random.default_rng(42)
    y = rng.standard_normal(8192).astype(np.float32)

    n_fft = 1024
    hop_length = n_fft // 4

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)

    y_mm = istft(S, hop_length=hop_length, center=True, length=len(y))
    y_lr = librosa.istft(S, hop_length=hop_length, center=True, length=len(y))

    np.testing.assert_allclose(
        y_mm, y_lr, rtol=1e-4, atol=1e-4,
        err_msg="MetalMom istft vs librosa istft mismatch (random)"
    )


@requires_librosa
def test_istft_reconstruction_quality():
    """Verify round-trip reconstruction is close to original signal."""
    import librosa
    from metalmom import istft

    sr = 22050
    t = np.arange(sr) / sr
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    n_fft = 2048
    hop_length = n_fft // 4

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    y_recon = istft(S, hop_length=hop_length, center=True, length=len(y))

    # Reconstruction should be close to original
    np.testing.assert_allclose(
        y_recon, y, rtol=1e-3, atol=1e-3,
        err_msg="iSTFT reconstruction deviates from original signal"
    )


@requires_librosa
def test_istft_energy_preservation():
    """Energy of reconstructed signal should be close to original."""
    import librosa
    from metalmom import istft

    rng = np.random.default_rng(123)
    y = rng.standard_normal(16384).astype(np.float32)

    n_fft = 2048
    hop_length = n_fft // 4

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    y_recon = istft(S, hop_length=hop_length, center=True, length=len(y))

    orig_energy = np.sum(y ** 2)
    recon_energy = np.sum(y_recon ** 2)
    ratio = recon_energy / orig_energy

    assert 0.99 < ratio < 1.01, (
        f"Energy ratio out of range: {ratio:.4f} (expected ~1.0)"
    )
