"""dB scaling parity tests: MetalMom vs librosa golden reference files.

Tier 1 tolerances: element-wise comparison with rtol=1e-4, atol=1e-4.
Note: Some tolerance is expected due to float32 vDSP log10 vs numpy's float64.
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


def test_amplitude_to_db_ref_max():
    """Verify amplitude_to_db with ref=np.max matches librosa.

    Uses the golden STFT magnitude as input so this test isolates the
    dB conversion from any STFT implementation differences.
    """
    from metalmom import amplitude_to_db

    S = _load_golden("stft_440hz_default_magnitude.npy")
    expected = _load_golden("stft_440hz_amplitude_db.npy")

    result = amplitude_to_db(S, ref=np.max)

    np.testing.assert_allclose(
        result, expected,
        rtol=1e-4, atol=1e-4,
        err_msg="amplitude_to_db(ref=np.max) mismatch vs librosa"
    )


def test_amplitude_to_db_ref1():
    """Verify amplitude_to_db with ref=1.0 matches librosa.

    Uses the golden STFT magnitude as input so this test isolates the
    dB conversion from any STFT implementation differences.
    """
    from metalmom import amplitude_to_db

    S = _load_golden("stft_440hz_default_magnitude.npy")
    expected = _load_golden("stft_440hz_amplitude_db_ref1.npy")

    result = amplitude_to_db(S, ref=1.0)

    np.testing.assert_allclose(
        result, expected,
        rtol=1e-4, atol=1e-4,
        err_msg="amplitude_to_db(ref=1.0) mismatch vs librosa"
    )


def test_power_to_db_ref_max():
    """Verify power_to_db with ref=np.max matches librosa.

    Uses the golden STFT magnitude as input so this test isolates the
    dB conversion from any STFT implementation differences.
    """
    from metalmom import power_to_db

    S = _load_golden("stft_440hz_default_magnitude.npy")
    expected = _load_golden("stft_440hz_power_db.npy")

    S_power = S ** 2
    result = power_to_db(S_power, ref=np.max)

    np.testing.assert_allclose(
        result, expected,
        rtol=1e-4, atol=1e-4,
        err_msg="power_to_db(ref=np.max) mismatch vs librosa"
    )


def test_amplitude_to_db_shape_preserved():
    """Verify output shape matches input for 2D spectrogram."""
    from metalmom import stft, amplitude_to_db

    signal = _load_golden("signal_440hz_22050sr.npy")
    S = stft(signal)
    result = amplitude_to_db(S)

    assert result.shape == S.shape, (
        f"Shape mismatch: input {S.shape} vs output {result.shape}"
    )


def test_amplitude_to_db_range_with_topdb():
    """Verify output range is within top_db of the maximum."""
    from metalmom import stft, amplitude_to_db

    signal = _load_golden("signal_440hz_22050sr.npy")
    S = stft(signal)
    top_db = 60.0
    result = amplitude_to_db(S, ref=np.max, top_db=top_db)

    max_val = result.max()
    min_val = result.min()
    assert min_val >= max_val - top_db - 1e-4, (
        f"Output range [{min_val:.2f}, {max_val:.2f}] exceeds top_db={top_db}"
    )


def test_db_round_trip():
    """Verify amplitude -> dB -> amplitude round trip."""
    from metalmom import amplitude_to_db, db_to_amplitude

    original = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    db = amplitude_to_db(original, ref=1.0, top_db=None)
    recovered = db_to_amplitude(db, ref=1.0)

    np.testing.assert_allclose(
        recovered, original,
        rtol=1e-4, atol=1e-5,
        err_msg="amplitude round-trip mismatch"
    )


def test_power_db_round_trip():
    """Verify power -> dB -> power round trip."""
    from metalmom import power_to_db, db_to_power

    original = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    db = power_to_db(original, ref=1.0, top_db=None)
    recovered = db_to_power(db, ref=1.0)

    np.testing.assert_allclose(
        recovered, original,
        rtol=1e-3, atol=1e-5,
        err_msg="power round-trip mismatch"
    )
