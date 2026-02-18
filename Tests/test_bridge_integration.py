"""Integration tests for neural/ML bridge functions with synthetic inputs.

Tests the following bridge functions that accept pre-computed activation
arrays and decode them via the Swift native backend:

  - neural_beat_track (DBN beat decoding from activation probabilities)
  - neural_onset_detect (peak-picking from onset activations)
  - downbeat_detect (beat + downbeat decoding from 3-class activations)
  - key_detect (key classification from 24-class activations)
  - chord_detect (CRF chord sequence decoding from per-frame activations)
  - piano_transcribe (note event extraction from 88-key activations)
  - griffinlim_cqt (CQT magnitude inversion via Griffin-Lim)
"""

import pytest
import numpy as np
import metalmom
from metalmom.beat import neural_beat_track, downbeat_detect
from metalmom.onset import neural_onset_detect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _periodic_activation(n_frames, period, peak_value=0.9, base_value=0.05,
                         peak_width=3, dtype=np.float32):
    """Create a 1D activation array with periodic Gaussian-ish peaks."""
    act = np.full(n_frames, base_value, dtype=dtype)
    for center in range(0, n_frames, period):
        for offset in range(-peak_width, peak_width + 1):
            idx = center + offset
            if 0 <= idx < n_frames:
                weight = np.exp(-0.5 * (offset / max(peak_width / 2, 1)) ** 2)
                act[idx] = max(act[idx], peak_value * weight)
    return act


# ---------------------------------------------------------------------------
# 1. neural_beat_track -- periodic activations at 120 BPM
# ---------------------------------------------------------------------------

def test_neural_beat_track_periodic():
    """Create 10s activation at 100fps with 120 BPM peaks (every 50 frames).

    Verify: tempo in [80, 160], at least 5 beats detected, frames sorted
    and non-negative.
    """
    np.random.seed(42)
    fps = 100.0
    n_frames = 1000  # 10 seconds at 100 fps
    period = 50      # 120 BPM = 2 beats/sec = 1 beat every 50 frames at 100fps

    activations = _periodic_activation(n_frames, period)
    # Add slight noise for realism
    activations += np.random.uniform(0, 0.02, n_frames).astype(np.float32)
    activations = np.clip(activations, 0.0, 1.0)

    tempo, beats = neural_beat_track(
        activations, fps=fps, min_bpm=55.0, max_bpm=215.0,
        transition_lambda=100.0, threshold=0.05, trim=True, units='frames',
    )

    # Return types
    assert isinstance(tempo, float), f"tempo should be float, got {type(tempo)}"
    assert isinstance(beats, np.ndarray), f"beats should be ndarray, got {type(beats)}"

    # Tempo in a reasonable range around 120 BPM
    assert 80.0 <= tempo <= 160.0, f"Expected tempo near 120, got {tempo}"

    # Should detect a decent number of beats in 10 seconds
    assert len(beats) >= 5, f"Expected >= 5 beats, got {len(beats)}"

    # Beats should be sorted and non-negative
    if len(beats) > 1:
        assert np.all(np.diff(beats) > 0), "Beat frames should be strictly increasing"
    if len(beats) > 0:
        assert np.all(beats >= 0), "Beat frames should be non-negative"
        assert np.all(beats < n_frames), "Beat frames should be within range"


# ---------------------------------------------------------------------------
# 2. neural_onset_detect -- sharp peaks
# ---------------------------------------------------------------------------

def test_neural_onset_detect_peaks():
    """Create 500-frame activation with 5 sharp peaks.

    Peaks at frames 50, 150, 250, 350, 450.
    Verify: at least 3 onsets detected, frames sorted and non-negative.
    """
    np.random.seed(42)
    n_frames = 500
    peak_positions = [50, 150, 250, 350, 450]

    activations = np.full(n_frames, 0.05, dtype=np.float32)
    for pos in peak_positions:
        # Sharp peak with small spread
        for offset in range(-2, 3):
            idx = pos + offset
            if 0 <= idx < n_frames:
                weight = np.exp(-0.5 * (offset / 1.0) ** 2)
                activations[idx] = max(activations[idx], 0.9 * weight)

    onsets = neural_onset_detect(
        activations, fps=100.0, threshold=0.3,
        pre_max=1, post_max=1, pre_avg=3, post_avg=3,
        combine='adaptive', wait=1, units='frames',
    )

    # Return type
    assert isinstance(onsets, np.ndarray), f"onsets should be ndarray, got {type(onsets)}"

    # Should find at least 3 of the 5 peaks
    assert len(onsets) >= 3, f"Expected >= 3 onsets, got {len(onsets)}"

    # Onsets should be sorted and non-negative
    if len(onsets) > 1:
        assert np.all(np.diff(onsets) > 0), "Onset frames should be strictly increasing"
    if len(onsets) > 0:
        assert np.all(onsets >= 0), "Onset frames should be non-negative"
        assert np.all(onsets < n_frames), "Onset frames should be within range"


# ---------------------------------------------------------------------------
# 3. downbeat_detect -- periodic beat + downbeat pattern
# ---------------------------------------------------------------------------

def test_downbeat_detect_pattern():
    """Create 1000-frame x 3-class activation with beat pattern.

    Every 50 frames: high P(beat). Every 4th beat (200 frames): high P(downbeat).
    Background frames: high P(no-beat).
    Verify: some beats and some downbeats detected.
    """
    np.random.seed(42)
    n_frames = 1000
    n_classes = 3
    beat_period = 50
    beats_per_bar = 4

    # Start with high P(no-beat) everywhere
    activations = np.zeros((n_frames, n_classes), dtype=np.float32)
    activations[:, 0] = 0.85  # P(no-beat) high by default
    activations[:, 1] = 0.10  # P(beat) low
    activations[:, 2] = 0.05  # P(downbeat) low

    beat_count = 0
    for frame in range(0, n_frames, beat_period):
        spread = 3
        for offset in range(-spread, spread + 1):
            idx = frame + offset
            if 0 <= idx < n_frames:
                weight = np.exp(-0.5 * (offset / max(spread / 2, 1)) ** 2)
                if beat_count % beats_per_bar == 0:
                    # Downbeat
                    activations[idx, 0] = 0.05
                    activations[idx, 1] = 0.15 * weight
                    activations[idx, 2] = 0.80 * weight
                else:
                    # Regular beat
                    activations[idx, 0] = 0.05
                    activations[idx, 1] = 0.85 * weight
                    activations[idx, 2] = 0.10 * weight
        beat_count += 1

    beat_frames, downbeat_frames = downbeat_detect(
        activations, fps=100.0, beats_per_bar=beats_per_bar,
        min_bpm=55.0, max_bpm=215.0, transition_lambda=100.0,
        units='frames',
    )

    # Return types
    assert isinstance(beat_frames, np.ndarray), \
        f"beat_frames should be ndarray, got {type(beat_frames)}"
    assert isinstance(downbeat_frames, np.ndarray), \
        f"downbeat_frames should be ndarray, got {type(downbeat_frames)}"

    # Should detect some beats and some downbeats
    assert len(beat_frames) >= 3, \
        f"Expected >= 3 beats, got {len(beat_frames)}"
    assert len(downbeat_frames) >= 1, \
        f"Expected >= 1 downbeats, got {len(downbeat_frames)}"

    # Downbeats should be a subset of beats (or at least close to beat positions)
    # Frames should be non-negative and within range
    if len(beat_frames) > 0:
        assert np.all(beat_frames >= 0), "Beat frames should be non-negative"
        assert np.all(beat_frames < n_frames), "Beat frames should be in range"
    if len(downbeat_frames) > 0:
        assert np.all(downbeat_frames >= 0), "Downbeat frames should be non-negative"
        assert np.all(downbeat_frames < n_frames), "Downbeat frames should be in range"


# ---------------------------------------------------------------------------
# 4. key_detect -- dominant C major (index 3 in madmom ordering: A, A#, B, C...)
# ---------------------------------------------------------------------------

def test_key_detect_dominant():
    """Create 24-class activation where one key is clearly dominant.

    Using the madmom key ordering (A major=0, ..., C major=3, ...),
    set class 3 (C major) to 0.9 and the rest to ~0.004.
    Verify: key_index matches the dominant class, confidence > 0.5.
    """
    np.random.seed(42)
    activations = np.full(24, 0.004, dtype=np.float32)

    # C major is index 3 in madmom ordering (A, A#, B, C, C#, ...)
    dominant_idx = 3
    activations[dominant_idx] = 0.9

    # Normalize to roughly sum to 1 (soft probability distribution)
    activations = activations / activations.sum()

    result = metalmom.key_detect(activations, sr=22050)

    # Return type
    assert isinstance(result, dict), f"result should be dict, got {type(result)}"

    # Required keys present
    for key in ('key_index', 'key_label', 'is_major', 'confidence', 'probabilities'):
        assert key in result, f"Missing key '{key}' in result"

    # Detected key should match our dominant class
    assert result['key_index'] == dominant_idx, \
        f"Expected key_index={dominant_idx}, got {result['key_index']}"

    # Confidence should be high
    assert result['confidence'] > 0.5, \
        f"Expected confidence > 0.5, got {result['confidence']}"

    # Probabilities array should have 24 elements
    assert len(result['probabilities']) == 24, \
        f"Expected 24 probabilities, got {len(result['probabilities'])}"

    # Key label should indicate major (index < 12)
    assert result['is_major'] is True, \
        f"Expected is_major=True for index {dominant_idx}"


# ---------------------------------------------------------------------------
# 5. chord_detect -- three-chord sequence
# ---------------------------------------------------------------------------

def test_chord_detect_sequence():
    """Create 300-frame x 25-class activations with three distinct chords.

    Frames 0-99: class 1 dominant (C:maj).
    Frames 100-199: class 2 dominant (C#:maj).
    Frames 200-299: class 3 dominant (D:maj).
    Verify: at least 2 chord events detected with correct structure.
    """
    np.random.seed(42)
    n_frames = 300
    n_classes = 25

    activations = np.full((n_frames, n_classes), 0.01, dtype=np.float32)

    # Three chord regions
    regions = [(0, 100, 1), (100, 200, 2), (200, 300, 3)]
    for start, end, cls_idx in regions:
        activations[start:end, cls_idx] = 0.9

    events = metalmom.chord_detect(
        activations, n_classes=n_classes,
        self_transition_bias=1.0, fps=100.0, units='frames',
    )

    # Return type
    assert isinstance(events, list), f"events should be list, got {type(events)}"

    # Should detect at least 2 chord changes (3 regions -> >= 2 transitions)
    assert len(events) >= 2, f"Expected >= 2 chord events, got {len(events)}"

    # Verify event structure
    for event in events:
        assert isinstance(event, dict), f"Each event should be a dict, got {type(event)}"
        assert 'start' in event, "Event missing 'start'"
        assert 'end' in event, "Event missing 'end'"
        assert 'chord_index' in event, "Event missing 'chord_index'"
        assert 'chord_label' in event, "Event missing 'chord_label'"
        assert event['start'] >= 0, "start should be non-negative"
        assert event['end'] > event['start'], "end should be > start"
        assert 0 <= event['chord_index'] < n_classes, \
            f"chord_index {event['chord_index']} out of range"


# ---------------------------------------------------------------------------
# 6. piano_transcribe -- two isolated notes
# ---------------------------------------------------------------------------

def test_piano_transcribe_notes():
    """Create 200-frame x 88-key activations with two active notes.

    MIDI 60 (C4, piano key index 39) active frames 10-50.
    MIDI 64 (E4, piano key index 43) active frames 60-100.
    Verify: at least 1 note event detected with correct structure.
    """
    np.random.seed(42)
    n_frames = 200
    n_keys = 88

    activations = np.full((n_frames, n_keys), 0.02, dtype=np.float32)

    # MIDI note 60 = C4 -> piano index = 60 - 21 = 39
    # MIDI note 64 = E4 -> piano index = 64 - 21 = 43
    activations[10:51, 39] = 0.85  # note 60, frames 10-50 inclusive
    activations[60:101, 43] = 0.85  # note 64, frames 60-100 inclusive

    events = metalmom.piano_transcribe(
        activations, threshold=0.5, min_duration=3,
        use_hmm=False, fps=100.0, units='frames',
    )

    # Return type
    assert isinstance(events, list), f"events should be list, got {type(events)}"

    # Should detect at least 1 note event (ideally 2)
    assert len(events) >= 1, f"Expected >= 1 note events, got {len(events)}"

    # Verify event structure
    for event in events:
        assert isinstance(event, dict), f"Each event should be a dict, got {type(event)}"
        assert 'midi_note' in event, "Event missing 'midi_note'"
        assert 'note_name' in event, "Event missing 'note_name'"
        assert 'onset' in event, "Event missing 'onset'"
        assert 'offset' in event, "Event missing 'offset'"
        assert 'velocity' in event, "Event missing 'velocity'"
        assert 21 <= event['midi_note'] <= 108, \
            f"MIDI note {event['midi_note']} out of piano range [21, 108]"
        assert event['offset'] > event['onset'], "offset should be > onset"
        assert event['velocity'] > 0, "velocity should be positive"


# ---------------------------------------------------------------------------
# 7. griffinlim_cqt -- CQT magnitude inversion roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="mm_griffinlim_cqt not declared in cffi _native.py definitions",
    raises=AttributeError,
    strict=True,
)
def test_griffinlim_cqt_roundtrip():
    """Generate a 1s 440Hz sine, compute CQT, invert with griffinlim_cqt.

    Verify: output is 1D, finite, non-empty float32 array.

    Note: Currently xfail because the mm_griffinlim_cqt bridge symbol is
    implemented in Swift (Bridge.swift) and declared in metalmom.h, but the
    cffi definition in _native.py is missing. Remove xfail after adding the
    cffi declaration.
    """
    np.random.seed(42)
    sr = 22050
    duration = 1.0
    freq = 440.0

    # Generate test signal
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    signal = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Compute CQT magnitude (returns shape (n_bins, n_frames))
    cqt_mag = metalmom.cqt(y=signal, sr=sr)
    assert cqt_mag.ndim == 2, f"CQT should be 2D, got {cqt_mag.ndim}D"
    assert cqt_mag.shape[0] > 0 and cqt_mag.shape[1] > 0, \
        f"CQT should be non-empty, got shape {cqt_mag.shape}"

    # Take magnitude (CQT may return complex or already be magnitude)
    cqt_abs = np.abs(cqt_mag).astype(np.float32)

    # Reconstruct audio from CQT magnitude
    audio = metalmom.griffinlim_cqt(cqt_abs, n_iter=32, sr=sr)

    # Output should be 1D float array
    assert isinstance(audio, np.ndarray), f"audio should be ndarray, got {type(audio)}"
    assert audio.ndim == 1, f"audio should be 1D, got {audio.ndim}D"
    assert len(audio) > 0, "Reconstructed audio should be non-empty"
    assert np.all(np.isfinite(audio)), "Audio should contain only finite values"
    assert audio.dtype in (np.float32, np.float64), \
        f"Audio dtype should be float, got {audio.dtype}"
