import numpy as np
import tempfile
import os
import pytest
import metalmom


def _create_test_wav(path, sr=22050, duration=1.0, freq=440.0):
    """Create a simple test WAV file using scipy."""
    import scipy.io.wavfile as wav
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    wav.write(path, sr, audio)
    return audio


def test_load_wav():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        ref = _create_test_wav(path)
        y, sr = metalmom.load(path, sr=None)
        assert sr == 22050
        # AVFoundation may round to packet boundaries; allow small tolerance on length
        assert abs(len(y) - len(ref)) < 1024
        # Check values match closely (up to length of shorter array)
        min_len = min(len(y), len(ref))
        np.testing.assert_allclose(y[:min_len], ref[:min_len], atol=1e-4)
    finally:
        os.unlink(path)


def test_load_with_sr():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=44100)
        y, sr = metalmom.load(path, sr=22050)
        assert sr == 22050
        # Should have roughly half the samples (1 second at 22050) with tolerance
        assert abs(len(y) - 22050) < 1024
    finally:
        os.unlink(path)


def test_load_with_offset():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, duration=2.0)
        y, sr = metalmom.load(path, sr=None, offset=1.0)
        assert sr == 22050
        assert abs(len(y) - 22050) < 1024  # ~1 second
    finally:
        os.unlink(path)


def test_load_with_duration():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, duration=3.0)
        y, sr = metalmom.load(path, sr=None, duration=1.0)
        assert sr == 22050
        assert abs(len(y) - 22050) < 1024
    finally:
        os.unlink(path)


def test_load_nonexistent():
    with pytest.raises(RuntimeError):
        metalmom.load("/nonexistent/file.wav")


def test_get_duration():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=22050, duration=2.0)
        dur = metalmom.get_duration(path)
        assert abs(dur - 2.0) < 0.01
    finally:
        os.unlink(path)


def test_get_samplerate():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=44100, duration=1.0)
        sr = metalmom.get_samplerate(path)
        assert sr == 44100
    finally:
        os.unlink(path)


def test_get_duration_nonexistent():
    with pytest.raises(RuntimeError):
        metalmom.get_duration("/nonexistent.wav")


def test_get_samplerate_nonexistent():
    with pytest.raises(RuntimeError):
        metalmom.get_samplerate("/nonexistent.wav")
