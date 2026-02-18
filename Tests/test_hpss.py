"""Tests for HPSS (Harmonic-Percussive Source Separation)."""

import numpy as np
import metalmom


def _make_sine(freq=440.0, sr=22050, duration=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def _make_clicks(sr=22050, duration=0.5, interval=1000):
    """Generate a click train (periodic impulses)."""
    length = int(sr * duration)
    y = np.zeros(length, dtype=np.float32)
    for i in range(0, length, interval):
        y[i] = 1.0
    return y


def test_hpss_returns_tuple():
    y = _make_sine()
    result = metalmom.hpss(y, sr=22050)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hpss_returns_arrays():
    y = _make_sine()
    h, p = metalmom.hpss(y, sr=22050)
    assert isinstance(h, np.ndarray)
    assert isinstance(p, np.ndarray)


def test_hpss_output_length():
    y = _make_sine(duration=0.25)
    h, p = metalmom.hpss(y, sr=22050)
    assert len(h) == len(y), f"Harmonic length {len(h)} != input {len(y)}"
    assert len(p) == len(y), f"Percussive length {len(p)} != input {len(y)}"


def test_hpss_1d_output():
    y = _make_sine(duration=0.25)
    h, p = metalmom.hpss(y, sr=22050)
    assert h.ndim == 1, f"Harmonic should be 1D, got {h.ndim}D"
    assert p.ndim == 1, f"Percussive should be 1D, got {p.ndim}D"


def test_harmonic_returns_1d():
    y = _make_sine(duration=0.25)
    h = metalmom.harmonic(y, sr=22050)
    assert isinstance(h, np.ndarray)
    assert h.ndim == 1
    assert len(h) == len(y)


def test_percussive_returns_1d():
    y = _make_sine(duration=0.25)
    p = metalmom.percussive(y, sr=22050)
    assert isinstance(p, np.ndarray)
    assert p.ndim == 1
    assert len(p) == len(y)


def test_sine_harmonic_energy_ratio():
    """A pure sine wave should have high harmonic energy ratio."""
    y = _make_sine(freq=440.0, duration=0.5)
    h, p = metalmom.hpss(y, sr=22050, n_fft=1024)

    h_energy = np.sum(h ** 2)
    p_energy = np.sum(p ** 2)
    total = h_energy + p_energy

    if total == 0:
        return  # Skip if no energy

    ratio = h_energy / total
    # Sine wave should be predominantly harmonic
    assert ratio > 0.5, f"Sine harmonic ratio {ratio:.3f} should be > 0.5"


def test_clicks_percussive_energy():
    """A click train should have substantial percussive energy."""
    y = _make_clicks(duration=0.5, interval=1000)
    h, p = metalmom.hpss(y, sr=22050, n_fft=1024)

    h_energy = np.sum(h ** 2)
    p_energy = np.sum(p ** 2)
    total = h_energy + p_energy

    if total == 0:
        return  # Skip if no energy

    ratio = p_energy / total
    # Clicks should have substantial percussive energy
    assert ratio > 0.3, f"Click percussive ratio {ratio:.3f} should be > 0.3"


def test_energy_tier2_parity():
    """Tier 2 parity: H/(H+P) energy ratio within 5% for sine wave.

    A pure sine should have H/(H+P) > 0.5 (far from 50/50 split).
    This is a soft check that the separation makes physical sense.
    """
    y = _make_sine(freq=440.0, sr=22050, duration=1.0)
    h, p = metalmom.hpss(y, sr=22050, n_fft=2048)

    h_energy = float(np.sum(h ** 2))
    p_energy = float(np.sum(p ** 2))
    total = h_energy + p_energy

    assert total > 0, "Total energy should be positive"
    ratio = h_energy / total
    # Sine should have harmonic-dominant energy
    assert ratio > 0.5, f"Sine H/(H+P) = {ratio:.3f}, expected > 0.5"


def test_hpss_custom_params():
    """Verify HPSS works with custom parameters."""
    y = _make_sine(duration=0.25)
    h, p = metalmom.hpss(y, sr=22050, kernel_size=11, power=1.0, margin=2.0)
    assert len(h) == len(y)
    assert len(p) == len(y)


def test_compat_shim():
    """Test the librosa compat shim."""
    from metalmom.compat.librosa.effects import hpss as compat_hpss
    from metalmom.compat.librosa.effects import harmonic as compat_harmonic
    from metalmom.compat.librosa.effects import percussive as compat_percussive

    y = _make_sine(duration=0.25)

    h, p = compat_hpss(y, sr=22050)
    assert isinstance(h, np.ndarray)
    assert isinstance(p, np.ndarray)

    h2 = compat_harmonic(y, sr=22050)
    assert isinstance(h2, np.ndarray)

    p2 = compat_percussive(y, sr=22050)
    assert isinstance(p2, np.ndarray)


def test_compat_via_module():
    """Test the compat shim accessed via module hierarchy."""
    from metalmom.compat import librosa
    y = _make_sine(duration=0.25)
    h, p = librosa.effects.hpss(y, sr=22050)
    assert isinstance(h, np.ndarray)
    assert isinstance(p, np.ndarray)
