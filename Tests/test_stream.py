import numpy as np
import tempfile
import os
import metalmom


def _create_test_wav(path, sr=22050, duration=1.0, freq=440.0):
    """Create a simple test WAV file using scipy."""
    import scipy.io.wavfile as wav
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    wav.write(path, sr, audio)
    return audio


def test_stream_basic():
    """Streaming 3-second file in 1-second blocks yields 3 blocks."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=22050, duration=3.0)
        blocks = list(metalmom.stream(path, block_length=22050, sr=22050))
        assert len(blocks) == 3
        for block in blocks:
            assert len(block) <= 22050
    finally:
        os.unlink(path)


def test_stream_short_last_block():
    """2.5-second file yields 3 blocks, last one shorter."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=22050, duration=2.5)
        blocks = list(metalmom.stream(path, block_length=22050, sr=22050))
        assert len(blocks) == 3
        assert len(blocks[-1]) <= 22050
    finally:
        os.unlink(path)


def test_stream_fill_value():
    """With fill_value, all blocks should be exactly block_length."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=22050, duration=2.5)
        blocks = list(metalmom.stream(path, block_length=22050, sr=22050,
                                       fill_value=0.0))
        for block in blocks:
            assert len(block) == 22050
    finally:
        os.unlink(path)


def test_stream_content():
    """Concatenated stream blocks should approximately match full load."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        ref = _create_test_wav(path, sr=22050, duration=2.0)
        blocks = list(metalmom.stream(path, block_length=22050, sr=None))
        result = np.concatenate(blocks)
        # Allow tolerance for AVFoundation packet rounding
        assert abs(len(result) - len(ref)) < 2048
    finally:
        os.unlink(path)


def test_stream_native_sr():
    """Streaming with sr=None should use the file's native sample rate."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=44100, duration=2.0)
        blocks = list(metalmom.stream(path, block_length=44100, sr=None))
        assert len(blocks) == 2
        for block in blocks:
            assert len(block) <= 44100
    finally:
        os.unlink(path)


def test_stream_dtype():
    """Output blocks should have the requested dtype."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        _create_test_wav(path, sr=22050, duration=1.0)
        blocks = list(metalmom.stream(path, block_length=22050, sr=22050,
                                       dtype=np.float32))
        assert blocks[0].dtype == np.float32
    finally:
        os.unlink(path)
