import numpy as np
import metalmom


def test_resample_identity():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.resample(y, 22050, 22050)
    np.testing.assert_array_equal(result, y)


def test_resample_downsample_length():
    y = np.sin(np.arange(44100, dtype=np.float32) * 440.0 * 2 * np.pi / 44100)
    result = metalmom.resample(y, 44100, 22050)
    assert abs(len(result) - 22050) <= 1


def test_resample_upsample_length():
    y = np.sin(np.arange(22050, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.resample(y, 22050, 44100)
    assert abs(len(result) - 44100) <= 1


def test_resample_round_trip_snr():
    """Round-trip resampling should achieve > 40 dB SNR."""
    n = 22050
    y = np.sin(np.arange(n, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    up = metalmom.resample(y, 22050, 44100)
    rt = metalmom.resample(up, 44100, 22050)

    min_len = min(len(y), len(rt))
    skip = 1000
    signal = y[skip:min_len - skip]
    noise = signal - rt[skip:min_len - skip]
    snr = 10 * np.log10(np.sum(signal**2) / max(np.sum(noise**2), 1e-20))
    assert snr > 40, f"Round-trip SNR should be > 40 dB, got {snr:.1f} dB"


def test_resample_energy_preservation():
    n = 22050
    y = np.sin(np.arange(n, dtype=np.float32) * 440.0 * 2 * np.pi / 22050)
    result = metalmom.resample(y, 22050, 44100)
    orig_rms = np.sqrt(np.mean(y**2))
    res_rms = np.sqrt(np.mean(result**2))
    assert abs(orig_rms - res_rms) < 0.05
