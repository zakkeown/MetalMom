"""Real audio tests using synthetic signals.

Verifies that MetalMom produces physically meaningful results for signals
with known spectral, temporal, and harmonic properties. All signals are
synthesized (no recorded audio files needed).
"""

import numpy as np
import pytest
import metalmom

SR = 22050


def _make_chirp(f0, f1, duration, sr=SR):
    """Generate a linear chirp sweeping from f0 to f1 Hz."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # Instantaneous phase integral for linear chirp
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))
    return np.sin(phase).astype(np.float32)


def _make_click_train(n_clicks, interval, duration, sr=SR, click_freq=1000.0,
                      click_duration=0.001, amplitude=0.9, start_offset=0.15):
    """Generate a signal with periodic short sine bursts (clicks).

    Parameters
    ----------
    n_clicks : int
        Number of clicks.
    interval : float
        Time between clicks in seconds.
    duration : float
        Total signal duration in seconds.
    sr : int
        Sample rate.
    click_freq : float
        Frequency of the sine burst (Hz).
    click_duration : float
        Duration of each click in seconds.
    amplitude : float
        Peak amplitude of each click.
    start_offset : float
        Time of the first click in seconds.

    Returns
    -------
    y : np.ndarray
        Signal with click bursts, shape (n_samples,), dtype float32.
    """
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    burst_len = int(click_duration * sr)
    burst = amplitude * np.sin(
        np.arange(burst_len, dtype=np.float32) * click_freq * 2 * np.pi / sr
    ).astype(np.float32)

    for i in range(n_clicks):
        t = start_offset + i * interval
        idx = int(t * sr)
        end = min(idx + burst_len, n)
        seg = end - idx
        if seg > 0:
            y[idx:end] = burst[:seg]
    return y


def _make_mixture(duration, sr=SR, seed=42):
    """Generate a music-like mixture: bass + melody + noise floor.

    Returns a float32 signal normalized to [-1, 1].
    """
    rng = np.random.RandomState(seed)
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr

    # Bass: 110 Hz
    bass = 0.3 * np.sin(2.0 * np.pi * 110.0 * t).astype(np.float32)

    # Melody: random note changes every 0.5s
    note_freqs = np.array([220, 261.6, 293.7, 329.6, 349.2, 392.0,
                           440.0, 523.3, 587.3, 659.3, 698.5, 784.0, 880.0],
                          dtype=np.float32)
    samples_per_note = sr // 2
    melody = np.zeros(n, dtype=np.float32)
    for start in range(0, n, samples_per_note):
        end = min(start + samples_per_note, n)
        freq = rng.choice(note_freqs)
        seg_t = np.arange(end - start, dtype=np.float32) / sr
        melody[start:end] = 0.2 * np.sin(2.0 * np.pi * freq * seg_t).astype(np.float32)

    # Noise floor
    noise = 0.02 * rng.randn(n).astype(np.float32)

    y = bass + melody + noise
    # Normalize to [-1, 1]
    peak = np.abs(y).max()
    if peak > 0:
        y /= peak
    return y


# ---------------------------------------------------------------------------
# Test 1: Chirp spectral centroid increases
# ---------------------------------------------------------------------------

def test_chirp_centroid_increases():
    """A chirp sweeping 200 -> 4000 Hz should have increasing spectral centroid."""
    y = _make_chirp(200.0, 4000.0, duration=2.0)
    centroid = metalmom.feature.spectral_centroid(y=y, sr=SR)

    assert centroid.ndim == 2, f"Expected 2D centroid, got shape {centroid.shape}"
    n_frames = centroid.shape[1]
    assert n_frames > 4, f"Need enough frames, got {n_frames}"

    # Flatten to 1D for easier slicing
    c = centroid.ravel()
    half = n_frames // 2
    first_half_mean = np.mean(c[:half])
    second_half_mean = np.mean(c[half:])

    assert second_half_mean > first_half_mean, (
        f"Chirp centroid should increase: first half mean = {first_half_mean:.1f} Hz, "
        f"second half mean = {second_half_mean:.1f} Hz"
    )


# ---------------------------------------------------------------------------
# Test 2: Click train onset detection
# ---------------------------------------------------------------------------

def test_click_train_onset():
    """Periodic clicks should be detected as onsets."""
    n_clicks = 10
    y = _make_click_train(
        n_clicks=n_clicks,
        interval=0.3,
        duration=3.0,
        click_freq=1000.0,
        click_duration=0.001,
        amplitude=0.9,
    )

    frames = metalmom.onset.onset_detect(
        y=y, sr=SR, hop_length=512,
        delta=0.03, wait=5,
    )

    assert isinstance(frames, np.ndarray), "onset_detect should return ndarray"
    detected = len(frames)
    # Require at least 50% of clicks found
    threshold = n_clicks // 2
    assert detected >= threshold, (
        f"Click train: detected {detected} onsets, expected >= {threshold} "
        f"out of {n_clicks} clicks"
    )


# ---------------------------------------------------------------------------
# Test 3: Full pipeline no crash (smoke test)
# ---------------------------------------------------------------------------

def test_full_pipeline_no_crash():
    """Run every major pipeline stage on a realistic mixture signal.

    Verifies: no crash, all outputs finite, reasonable shapes.
    """
    y = _make_mixture(duration=3.0, seed=42)

    # STFT
    S = metalmom.stft(y=y, n_fft=2048)
    assert S.ndim == 2, f"STFT should be 2D, got shape {S.shape}"
    assert S.shape[0] == 1025, f"STFT should have n_fft/2+1 bins, got {S.shape[0]}"
    assert S.shape[1] > 0, "STFT should have frames"
    assert np.all(np.isfinite(S)), "STFT output contains non-finite values"

    # Mel spectrogram
    mel = metalmom.feature.melspectrogram(y=y, sr=SR)
    assert mel.ndim == 2, f"Mel should be 2D, got shape {mel.shape}"
    assert mel.shape[0] == 128, f"Mel should have 128 bands, got {mel.shape[0]}"
    assert mel.shape[1] > 0, "Mel should have frames"
    assert np.all(np.isfinite(mel)), "Mel output contains non-finite values"

    # MFCC
    mfcc = metalmom.feature.mfcc(y=y, sr=SR)
    assert mfcc.ndim == 2, f"MFCC should be 2D, got shape {mfcc.shape}"
    assert mfcc.shape[0] == 20, f"MFCC should have 20 coefficients, got {mfcc.shape[0]}"
    assert mfcc.shape[1] > 0, "MFCC should have frames"
    assert np.all(np.isfinite(mfcc)), "MFCC output contains non-finite values"

    # Chroma
    chroma = metalmom.feature.chroma_stft(y=y, sr=SR)
    assert chroma.ndim == 2, f"Chroma should be 2D, got shape {chroma.shape}"
    assert chroma.shape[0] == 12, f"Chroma should have 12 bins, got {chroma.shape[0]}"
    assert chroma.shape[1] > 0, "Chroma should have frames"
    assert np.all(np.isfinite(chroma)), "Chroma output contains non-finite values"

    # Onset detection
    onset_frames = metalmom.onset.onset_detect(y=y, sr=SR)
    assert isinstance(onset_frames, np.ndarray), "onset_detect should return ndarray"
    if len(onset_frames) > 0:
        assert np.all(np.isfinite(onset_frames)), "Onset frames contain non-finite values"
        assert np.all(onset_frames >= 0), "Onset frame indices should be non-negative"

    # Beat tracking
    tempo, beats = metalmom.beat.beat_track(y=y, sr=SR)
    assert np.isfinite(tempo), f"Tempo should be finite, got {tempo}"
    assert tempo > 0, f"Tempo should be positive for a signal with energy, got {tempo}"
    assert isinstance(beats, np.ndarray), "beat_track should return ndarray for beats"
    if len(beats) > 0:
        assert np.all(np.isfinite(beats)), "Beat locations contain non-finite values"
