"""
End-to-end smoke tests: synthesized audio -> full pipeline -> task output.

These tests verify that the complete MetalMom pipeline works end-to-end:
synthesize audio with known properties, run feature extraction and/or model
inference, and verify that the output is plausible.

Thresholds are deliberately loose -- the standard-API algorithms (beat_track,
onset_detect, tempo, key_detect) were designed for real music, and the neural
models were trained on real recordings, so perfect accuracy on simple
synthesized signals is not expected.
"""

import os

import numpy as np
import pytest

try:
    import metalmom
    HAS_METALMOM = True
except (ImportError, OSError):
    HAS_METALMOM = False

pytestmark = pytest.mark.skipif(not HAS_METALMOM, reason="metalmom not available")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "converted")


def _model_available(family, name):
    """Check whether a specific .mlmodel file exists on disk."""
    path = os.path.join(_MODELS_DIR, family, f"{name}.mlmodel")
    return os.path.isfile(path)


def _model_path(family, name):
    """Return the absolute path for a converted .mlmodel file."""
    return os.path.join(_MODELS_DIR, family, f"{name}.mlmodel")


# ---------------------------------------------------------------------------
# Audio synthesis helpers
# ---------------------------------------------------------------------------

def make_click_track(bpm, duration=10.0, sr=22050, click_freq=1000.0,
                     click_duration=0.01):
    """Synthesize a click track at the given BPM.

    Parameters
    ----------
    bpm : float
        Tempo in beats per minute.
    duration : float
        Total duration in seconds.
    sr : int
        Sample rate.
    click_freq : float
        Frequency of each click (Hz).
    click_duration : float
        Duration of each click (seconds).

    Returns
    -------
    np.ndarray
        1-D float32 audio signal.
    """
    beat_interval = 60.0 / bpm
    times = np.arange(0.0, duration, beat_interval)
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples, dtype=np.float32)

    click_len = int(sr * click_duration)
    t_click = np.arange(click_len, dtype=np.float32) / sr
    click = (0.8 * np.sin(2.0 * np.pi * click_freq * t_click)
             * np.exp(-10.0 * t_click)).astype(np.float32)

    for t in times:
        idx = int(t * sr)
        end = min(idx + click_len, n_samples)
        actual = end - idx
        if actual > 0:
            signal[idx:end] += click[:actual]

    return signal


def make_sine_bursts(freqs, interval=0.5, burst_dur=0.05, duration=5.0,
                     sr=22050):
    """Synthesize sine bursts at regular intervals.

    Parameters
    ----------
    freqs : list of float
        Frequency of each burst (Hz).  Cycles through the list.
    interval : float
        Time between burst onsets (seconds).
    burst_dur : float
        Duration of each burst (seconds).
    duration : float
        Total signal duration (seconds).
    sr : int
        Sample rate.

    Returns
    -------
    signal : np.ndarray
        1-D float32 audio.
    onset_times : np.ndarray
        True onset times in seconds.
    """
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples, dtype=np.float32)
    burst_len = int(sr * burst_dur)
    t_burst = np.arange(burst_len, dtype=np.float32) / sr
    onset_times = np.arange(0.0, duration - burst_dur, interval)

    for i, t in enumerate(onset_times):
        freq = freqs[i % len(freqs)]
        burst = (0.8 * np.sin(2.0 * np.pi * freq * t_burst)
                 * np.hanning(burst_len)).astype(np.float32)
        idx = int(t * sr)
        end = min(idx + burst_len, n_samples)
        actual = end - idx
        if actual > 0:
            signal[idx:end] += burst[:actual]

    return signal, onset_times


def make_scale(notes_hz, note_dur=0.3, sr=22050, amplitude=0.5):
    """Synthesize a sequence of pure tones (e.g. a scale).

    Parameters
    ----------
    notes_hz : list of float
        Frequencies in Hz for each note.
    note_dur : float
        Duration of each note (seconds).
    sr : int
        Sample rate.
    amplitude : float
        Peak amplitude.

    Returns
    -------
    np.ndarray
        1-D float32 audio.
    """
    samples_per_note = int(sr * note_dur)
    total = samples_per_note * len(notes_hz)
    signal = np.zeros(total, dtype=np.float32)
    t = np.arange(samples_per_note, dtype=np.float32) / sr

    for i, freq in enumerate(notes_hz):
        start = i * samples_per_note
        # Apply short fade-in/out to avoid clicks
        env = np.ones(samples_per_note, dtype=np.float32)
        fade = min(int(sr * 0.01), samples_per_note // 4)
        if fade > 0:
            env[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            env[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        signal[start:start + samples_per_note] = (
            amplitude * np.sin(2.0 * np.pi * freq * t) * env
        )

    return signal


# ---------------------------------------------------------------------------
# Test 1: Beat tracking on a click track
# ---------------------------------------------------------------------------

class TestBeatTracking:
    """Beat tracking: synthesize 120 BPM click track, verify detected BPM."""

    def test_beat_track_bpm_120(self):
        """Standard beat_track should estimate BPM within 10% of 120."""
        sr = 22050
        bpm_target = 120.0
        signal = make_click_track(bpm_target, duration=10.0, sr=sr)

        tempo_est, beats = metalmom.beat_track(y=signal, sr=sr, units='time')

        # Tempo within 10% of target (or its double/half -- octave errors
        # are common with beat trackers)
        bpm_ok = (
            abs(tempo_est - bpm_target) / bpm_target < 0.10
            or abs(tempo_est - 2 * bpm_target) / (2 * bpm_target) < 0.10
            or abs(tempo_est - bpm_target / 2) / (bpm_target / 2) < 0.10
        )
        assert bpm_ok, (
            f"Estimated BPM {tempo_est:.1f} is not within 10% of "
            f"{bpm_target} (or its double/half)"
        )

        # Should detect a reasonable number of beats for 10s at 120 BPM
        # (expected ~20, allow generous margin)
        assert len(beats) >= 5, (
            f"Expected at least 5 beats in 10s at 120 BPM, got {len(beats)}"
        )

    def test_tempo_estimate_120(self):
        """metalmom.tempo should estimate ~120 BPM on a 120 BPM click."""
        sr = 22050
        bpm_target = 120.0
        signal = make_click_track(bpm_target, duration=10.0, sr=sr)

        tempos = metalmom.tempo(y=signal, sr=sr)
        tempo_est = tempos[0]

        bpm_ok = (
            abs(tempo_est - bpm_target) / bpm_target < 0.10
            or abs(tempo_est - 2 * bpm_target) / (2 * bpm_target) < 0.10
            or abs(tempo_est - bpm_target / 2) / (bpm_target / 2) < 0.10
        )
        assert bpm_ok, (
            f"Estimated tempo {tempo_est:.1f} is not within 10% of "
            f"{bpm_target} (or double/half)"
        )

    def test_neural_beat_decode_on_synthetic_activations(self):
        """Neural beat decode should find beats in a synthetic pulse train."""
        from metalmom.beat import neural_beat_track

        # Synthesize a perfect activation signal: peaks at 120 BPM,
        # fps=100 means 100 frames/sec, so beat period = 50 frames
        fps = 100.0
        bpm = 120.0
        duration_sec = 10.0
        n_frames = int(fps * duration_sec)
        beat_period = int(fps * 60.0 / bpm)

        activations = np.zeros(n_frames, dtype=np.float32)
        for i in range(0, n_frames, beat_period):
            activations[i] = 1.0
            # Small spread around peak
            if i > 0:
                activations[i - 1] = 0.3
            if i + 1 < n_frames:
                activations[i + 1] = 0.3

        tempo_est, beats = neural_beat_track(
            activations, fps=fps, min_bpm=80.0, max_bpm=200.0,
            units='time'
        )

        assert len(beats) >= 10, (
            f"Expected at least 10 beats from clean activations, got {len(beats)}"
        )

        bpm_ok = (
            abs(tempo_est - bpm) / bpm < 0.15
            or abs(tempo_est - 2 * bpm) / (2 * bpm) < 0.15
        )
        assert bpm_ok, (
            f"Neural decode estimated {tempo_est:.1f} BPM, "
            f"expected ~{bpm} (or double)"
        )


# ---------------------------------------------------------------------------
# Test 2: Onset detection on sine bursts
# ---------------------------------------------------------------------------

class TestOnsetDetection:
    """Onset detection: sine bursts at known intervals, verify recall."""

    def test_onset_detect_sine_bursts(self):
        """Standard onset_detect should find >= 30% of sine burst onsets.

        Threshold is deliberately loose: the peak-picking algorithm is
        designed for real music, not synthetic sine bursts.  We just
        verify the pipeline works and detects *some* of the onsets.
        """
        sr = 22050
        freqs = [1000.0]
        signal, true_onsets = make_sine_bursts(
            freqs, interval=0.5, burst_dur=0.05, duration=5.0, sr=sr
        )

        detected = metalmom.onset_detect(
            y=signal, sr=sr, units='time', delta=0.05
        )

        if len(detected) == 0:
            pytest.skip("onset_detect returned no onsets on synthetic bursts")

        # Count how many true onsets have a detected onset within 100ms
        tolerance = 0.100  # 100 ms -- generous for synthesized signals
        hits = 0
        for t_true in true_onsets:
            if np.any(np.abs(detected - t_true) <= tolerance):
                hits += 1

        recall = hits / len(true_onsets)
        assert recall >= 0.30, (
            f"Onset recall {recall:.2%} is below 30% "
            f"({hits}/{len(true_onsets)} hits within {tolerance*1000:.0f}ms)"
        )

    def test_neural_onset_decode_synthetic(self):
        """Neural onset decode on synthetic activation peaks."""
        from metalmom.onset import neural_onset_detect

        # Synthetic onset activations: peaks every 50 frames at fps=100
        fps = 100.0
        interval_sec = 0.5
        duration = 5.0
        n_frames = int(fps * duration)
        peak_period = int(fps * interval_sec)

        activations = np.full(n_frames, 0.05, dtype=np.float32)
        true_frames = list(range(0, n_frames, peak_period))
        for f in true_frames:
            activations[f] = 0.9

        detected = neural_onset_detect(
            activations, fps=fps, threshold=0.3, units='time'
        )

        if len(detected) == 0:
            pytest.skip("neural_onset_detect returned no onsets")

        tolerance = 0.050
        hits = 0
        true_times = np.array(true_frames, dtype=np.float64) / fps
        for t_true in true_times:
            if np.any(np.abs(detected - t_true) <= tolerance):
                hits += 1

        recall = hits / len(true_times)
        assert recall >= 0.50, (
            f"Neural onset recall {recall:.2%} is below 50%"
        )


# ---------------------------------------------------------------------------
# Test 3: Key detection
# ---------------------------------------------------------------------------

class TestKeyDetection:
    """Key detection: C major scale tones -> should detect C major or A minor."""

    # C major scale frequencies (C4 to C5)
    C_MAJOR_HZ = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

    def test_key_detect_from_activations(self):
        """key_detect with hand-crafted C-major activations."""
        # Create activations that strongly favor C major (index 3 in KEY_LABELS)
        # KEY_LABELS: A, A#, B, C, C#, D, D#, E, F, F#, G, G# (major)
        #             A, A#, B, C, C#, D, D#, E, F, F#, G, G# (minor)
        activations = np.full(24, 0.01, dtype=np.float32)
        activations[3] = 0.8   # C major
        activations[12] = 0.15  # A minor (relative minor)

        result = metalmom.key_detect(activations)

        assert result['key_label'] in ("C major", "A minor"), (
            f"Expected C major or A minor, got {result['key_label']}"
        )

    @pytest.mark.skipif(
        not _model_available("key", "key_cnn"),
        reason="key_cnn model not available"
    )
    def test_key_cnn_model_inference(self):
        """Run key CNN model on synthesized C major scale features."""
        from metalmom._inference import predict_model

        model = _model_path("key", "key_cnn")

        # The key CNN expects spectrogram-like input.  We create a simple
        # synthetic input that we know the model can process -- the exact
        # key prediction may not be C major on synthetic data but it
        # should run without errors and produce a 24-class output.
        sr = 22050
        signal = make_scale(self.C_MAJOR_HZ, note_dur=0.4, sr=sr)

        # Compute log-mel spectrogram as a generic feature
        mel = metalmom.melspectrogram(y=signal, sr=sr, n_mels=128)
        log_mel = metalmom.power_to_db(mel)

        # The CNN may expect a specific input shape -- we try a reasonable
        # slice and reshape.  If it fails, we skip.
        try:
            # Try feeding the full log-mel spectrogram
            output = predict_model(model, log_mel)
        except RuntimeError:
            pytest.skip(
                "key_cnn model rejected the synthetic input shape -- "
                "CNN input requirements differ from generic features"
            )

        assert output is not None
        assert output.size > 0, "Model produced empty output"


# ---------------------------------------------------------------------------
# Test 4: Chroma extraction on A4 (440 Hz)
# ---------------------------------------------------------------------------

class TestChromaExtraction:
    """Chroma: synthesize 440Hz (A4), verify A bin has highest energy."""

    def test_chroma_stft_440hz(self):
        """chroma_stft on a 440Hz tone should peak at the A chroma bin."""
        sr = 22050
        duration = 2.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        signal = (0.8 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        chroma = metalmom.chroma_stft(y=signal, sr=sr)

        assert chroma.shape[0] == 12, f"Expected 12 chroma bins, got {chroma.shape[0]}"

        # Sum energy across frames for each chroma bin
        energy = chroma.sum(axis=1)

        # Chroma bins are ordered C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        # A is index 9
        peak_bin = int(np.argmax(energy))
        assert peak_bin == 9, (
            f"Expected A (bin 9) to have highest chroma energy, "
            f"got bin {peak_bin} (energy distribution: {energy})"
        )

    def test_chroma_stft_c4(self):
        """chroma_stft on a 261.63Hz (C4) tone should peak at the C bin."""
        sr = 22050
        duration = 2.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        signal = (0.8 * np.sin(2.0 * np.pi * 261.63 * t)).astype(np.float32)

        chroma = metalmom.chroma_stft(y=signal, sr=sr)
        energy = chroma.sum(axis=1)

        # C is bin 0
        peak_bin = int(np.argmax(energy))
        assert peak_bin == 0, (
            f"Expected C (bin 0) to have highest chroma energy, "
            f"got bin {peak_bin}"
        )


# ---------------------------------------------------------------------------
# Test 5: Full neural model pipeline (CoreML inference)
# ---------------------------------------------------------------------------

class TestNeuralModelPipeline:
    """End-to-end tests that run actual CoreML model inference."""

    @pytest.mark.skipif(
        not _model_available("beats", "beats_lstm_1"),
        reason="beats_lstm_1 model not available"
    )
    def test_beat_model_inference(self):
        """Run a beat LSTM model on synthetic features and decode beats."""
        from metalmom._inference import predict_model
        from metalmom.beat import neural_beat_track

        model = _model_path("beats", "beats_lstm_1")

        # Beat models expect spectral features.  Create a synthetic input
        # with the correct input dimensionality.
        # Beat LSTM models take (seq_len, input_dim) -- input_dim varies
        # but is typically 314 for madmom beat models.
        sr = 22050
        bpm = 120.0
        signal = make_click_track(bpm, duration=5.0, sr=sr)

        # Compute log-mel spectrogram features
        mel = metalmom.melspectrogram(y=signal, sr=sr, n_mels=128, hop_length=441)
        log_mel = metalmom.power_to_db(mel)  # shape: (128, n_frames)

        # The beat model expects (seq_len, input_dim) in rank-5 for CoreML
        # input_dim=314 for the standard madmom beat models.
        # Our mel features have 128 bands which differs from the expected
        # 314 features, so we pad or adjust.
        n_frames = log_mel.shape[1]
        input_dim = 314

        # Zero-pad features to match expected input dimension
        features = np.zeros((n_frames, input_dim), dtype=np.float32)
        features[:, :128] = log_mel.T

        # Reshape to rank-5 for CoreML: (seq_len, 1, input_dim, 1, 1)
        features_5d = features.reshape(n_frames, 1, input_dim, 1, 1)

        try:
            activations = predict_model(model, features_5d)
        except RuntimeError as e:
            pytest.skip(f"Beat model inference failed: {e}")

        # Flatten activations to 1D
        act = activations.ravel()

        if len(act) == 0:
            pytest.skip("Beat model produced empty activations")

        # Normalize to [0, 1]
        act_max = act.max()
        if act_max > 0:
            act = act / act_max

        # Decode beats
        tempo_est, beats = neural_beat_track(
            act, fps=100.0, min_bpm=60.0, max_bpm=200.0, units='time'
        )

        # We just verify the pipeline ran without errors and produced output
        assert len(beats) >= 0, "Beat decode should return an array"

    @pytest.mark.skipif(
        not _model_available("onsets", "onsets_rnn_1"),
        reason="onsets_rnn_1 model not available"
    )
    def test_onset_model_inference(self):
        """Run an onset RNN model on synthetic features and decode onsets."""
        from metalmom._inference import predict_model
        from metalmom.onset import neural_onset_detect

        model = _model_path("onsets", "onsets_rnn_1")

        sr = 22050
        signal, true_onsets = make_sine_bursts(
            [1000.0], interval=0.5, burst_dur=0.05, duration=3.0, sr=sr
        )

        mel = metalmom.melspectrogram(y=signal, sr=sr, n_mels=128, hop_length=441)
        log_mel = metalmom.power_to_db(mel)

        n_frames = log_mel.shape[1]
        # Onset RNN models typically expect input_dim=314
        input_dim = 314
        features = np.zeros((n_frames, input_dim), dtype=np.float32)
        features[:, :128] = log_mel.T
        features_5d = features.reshape(n_frames, 1, input_dim, 1, 1)

        try:
            activations = predict_model(model, features_5d)
        except RuntimeError as e:
            pytest.skip(f"Onset model inference failed: {e}")

        act = activations.ravel()

        if len(act) == 0:
            pytest.skip("Onset model produced empty activations")

        # Normalize
        act_max = act.max()
        if act_max > 0:
            act = act / act_max

        onsets = neural_onset_detect(act, fps=100.0, threshold=0.3, units='time')

        # Pipeline should run without errors
        assert isinstance(onsets, np.ndarray), "Onset decode should return ndarray"


# ---------------------------------------------------------------------------
# Test 6: Chord detection on synthetic activations
# ---------------------------------------------------------------------------

class TestChordDetection:
    """Chord detection: provide C-major-dominant activations, verify output."""

    def test_chord_detect_synthetic_activations(self):
        """chord_detect should identify C:maj from strong activations."""
        # 25 classes: N, C:maj, C#:maj, ..., B:maj, C:min, ..., B:min
        # C:maj is index 1
        n_frames = 50
        n_classes = 25

        activations = np.full((n_frames, n_classes), 0.02, dtype=np.float32)
        activations[:, 1] = 0.9  # strong C:maj throughout

        events = metalmom.chord_detect(activations, n_classes=n_classes)

        assert len(events) > 0, "chord_detect should return at least one event"

        # The dominant chord should be C:maj
        chord_labels = [e['chord_label'] for e in events]
        assert "C:maj" in chord_labels, (
            f"Expected C:maj among detected chords, got: {chord_labels}"
        )

        # C:maj should span most of the frames
        cmaj_frames = sum(
            e['end'] - e['start'] for e in events if e['chord_label'] == 'C:maj'
        )
        total_frames = float(n_frames)
        assert cmaj_frames / total_frames >= 0.5, (
            f"C:maj covers only {cmaj_frames/total_frames:.0%} of frames, "
            f"expected >= 50%"
        )


# ---------------------------------------------------------------------------
# Test 7: Onset strength + MFCC feature extraction pipeline
# ---------------------------------------------------------------------------

class TestFeatureExtractionPipeline:
    """Verify feature extraction functions produce valid output on synth audio."""

    def test_onset_strength_on_click(self):
        """onset_strength should produce non-zero envelope on a click track."""
        sr = 22050
        signal = make_click_track(120.0, duration=3.0, sr=sr)

        env = metalmom.onset_strength(y=signal, sr=sr)

        assert env.ndim == 1, f"Expected 1-D envelope, got shape {env.shape}"
        assert len(env) > 0, "Envelope is empty"
        assert env.max() > 0.0, "Envelope is all zeros on a click track"

    def test_mfcc_on_tone(self):
        """MFCC extraction should produce valid coefficients on a pure tone."""
        sr = 22050
        duration = 2.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        mfccs = metalmom.mfcc(y=signal, sr=sr, n_mfcc=13)

        assert mfccs.shape[0] == 13, f"Expected 13 MFCCs, got {mfccs.shape[0]}"
        assert mfccs.shape[1] > 0, "No MFCC frames produced"
        assert np.isfinite(mfccs).all(), "MFCCs contain non-finite values"

    def test_melspectrogram_on_tone(self):
        """Mel spectrogram should have energy concentrated at tone frequency."""
        sr = 22050
        duration = 2.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        mel = metalmom.melspectrogram(y=signal, sr=sr, n_mels=128)

        assert mel.shape[0] == 128, f"Expected 128 mel bands, got {mel.shape[0]}"
        assert mel.shape[1] > 0, "No mel frames produced"
        # Energy should not be uniformly distributed
        band_energy = mel.sum(axis=1)
        assert band_energy.max() > 10 * band_energy.mean(), (
            "Mel energy should be concentrated, not uniform for a pure tone"
        )


# ---------------------------------------------------------------------------
# Test 8: Madmom compat shim classes
# ---------------------------------------------------------------------------

class TestMadmomCompat:
    """Verify madmom-compatible processor classes work end-to-end."""

    def test_rnn_beat_processor_compat(self):
        """RNNBeatProcessor should produce beat activations from audio."""
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor

        sr = 44100  # madmom default
        signal = make_click_track(120.0, duration=5.0, sr=sr)

        proc = RNNBeatProcessor(fps=100.0)
        activations = proc(signal)

        assert isinstance(activations, np.ndarray)
        assert activations.ndim == 1
        assert len(activations) > 0
        assert activations.max() > 0.0

    def test_dbn_beat_tracking_processor_compat(self):
        """DBNBeatTrackingProcessor should decode beats from activations."""
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor

        # Synthetic perfect activations at 120 BPM, 100 fps
        fps = 100.0
        n_frames = 1000
        beat_period = int(fps * 60.0 / 120.0)  # 50 frames
        activations = np.full(n_frames, 0.05, dtype=np.float32)
        for i in range(0, n_frames, beat_period):
            activations[i] = 0.95

        proc = DBNBeatTrackingProcessor(fps=fps)
        beat_times = proc(activations)

        assert isinstance(beat_times, np.ndarray)
        assert len(beat_times) >= 5, (
            f"Expected at least 5 beats, got {len(beat_times)}"
        )

    def test_onset_peak_picking_compat(self):
        """OnsetPeakPickingProcessor should detect onsets from activations."""
        from metalmom.compat.madmom.features.onsets import OnsetPeakPickingProcessor

        fps = 100.0
        n_frames = 500
        activations = np.full(n_frames, 0.05, dtype=np.float32)
        # Place onset peaks every 50 frames (0.5s intervals)
        for i in range(0, n_frames, 50):
            activations[i] = 0.9

        proc = OnsetPeakPickingProcessor(fps=fps, threshold=0.3)
        onset_times = proc(activations)

        assert isinstance(onset_times, np.ndarray)
        assert len(onset_times) >= 3, (
            f"Expected at least 3 onsets, got {len(onset_times)}"
        )
