"""Tests for signal generation functions (tone, chirp, clicks)."""

import numpy as np
import metalmom


def test_tone_shape():
    y = metalmom.tone(440.0, sr=22050, length=22050)
    assert len(y) == 22050


def test_tone_frequency():
    y = metalmom.tone(440.0, sr=22050, length=22050)
    crossings = np.sum(np.diff(np.sign(y)) != 0)
    assert abs(crossings - 880) < 5


def test_tone_phase_offset():
    y = metalmom.tone(440.0, sr=22050, length=100, phi=np.pi / 2)
    # sin(pi/2) = 1
    assert abs(y[0] - 1.0) < 0.01


def test_tone_duration():
    y = metalmom.tone(440.0, sr=22050, duration=0.5)
    assert len(y) == 11025


def test_tone_bounded():
    y = metalmom.tone(440.0, sr=22050, length=22050)
    assert np.all(np.abs(y) <= 1.01)


def test_chirp_shape():
    y = metalmom.chirp(100.0, 1000.0, sr=22050, length=22050)
    assert len(y) == 22050


def test_chirp_bounded():
    y = metalmom.chirp(100.0, 1000.0, sr=22050, length=22050)
    assert np.all(np.abs(y) <= 1.01)


def test_chirp_log_bounded():
    y = metalmom.chirp(100.0, 1000.0, sr=22050, length=22050, linear=False)
    assert np.all(np.abs(y) <= 1.01)


def test_chirp_duration():
    y = metalmom.chirp(100.0, 1000.0, sr=22050, duration=0.5)
    assert len(y) == 11025


def test_clicks_shape():
    y = metalmom.clicks(times=[0.0, 0.5, 1.0], sr=22050, length=33075)
    assert len(y) == 33075


def test_clicks_nonzero():
    y = metalmom.clicks(times=[0.0], sr=22050, length=22050)
    assert np.any(np.abs(y[:2205]) > 0.01)
