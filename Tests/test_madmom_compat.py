"""Tests for the madmom compatibility shim.

Verifies that madmom-compatible classes backed by MetalMom produce
correct shapes, types, and default parameter values.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignal:
    """Tests for madmom.audio.signal.Signal."""

    def test_signal_from_array(self):
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.random.randn(44100).astype(np.float32)
        sig = Signal(data, sr=44100)
        assert isinstance(sig, np.ndarray)
        assert sig.shape == (44100,)
        assert sig.sample_rate == 44100
        assert sig.dtype == np.float32

    def test_signal_default_sr(self):
        """Default sample rate should be 44100 (madmom convention, not 22050)."""
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.zeros(1000, dtype=np.float32)
        sig = Signal(data)
        assert sig.sample_rate == 44100

    def test_signal_sample_rate_kwarg(self):
        """madmom uses 'sample_rate' kwarg, not 'sr'."""
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.zeros(1000, dtype=np.float32)
        sig = Signal(data, sample_rate=16000)
        assert sig.sample_rate == 16000

    def test_signal_is_ndarray_subclass(self):
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.random.randn(1000).astype(np.float32)
        sig = Signal(data)
        assert isinstance(sig, np.ndarray)
        # Should support normal array operations
        assert sig.mean() == pytest.approx(data.mean(), abs=1e-6)

    def test_signal_array_finalize(self):
        """Slicing should preserve sample_rate attribute."""
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.random.randn(1000).astype(np.float32)
        sig = Signal(data, sr=48000)
        sliced = sig[:500]
        assert hasattr(sliced, 'sample_rate')
        assert sliced.sample_rate == 48000

    def test_signal_num_samples(self):
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.zeros(44100, dtype=np.float32)
        sig = Signal(data, sr=44100)
        assert sig.num_samples == 44100

    def test_signal_length_property(self):
        from metalmom.compat.madmom.audio.signal import Signal
        data = np.zeros(44100, dtype=np.float32)
        sig = Signal(data, sr=44100)
        assert sig.length == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# FramedSignal tests
# ---------------------------------------------------------------------------

class TestFramedSignal:
    """Tests for madmom.audio.signal.FramedSignal."""

    def test_framed_signal_shape(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        sr = 44100
        duration = 1.0  # 1 second
        data = np.random.randn(int(sr * duration)).astype(np.float32)
        sig = Signal(data, sr=sr)
        frames = FramedSignal(sig, frame_size=2048, hop_size=441)

        # Each frame should have frame_size samples
        assert frames.shape[1] == 2048
        assert len(frames) > 0

    def test_framed_signal_default_params(self):
        """Default frame_size=2048, hop_size=441 (madmom convention)."""
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.zeros(44100, dtype=np.float32)
        sig = Signal(data, sr=44100)
        frames = FramedSignal(sig)

        assert frames.frame_size == 2048
        assert frames.hop_size == 441

    def test_framed_signal_frame_count(self):
        """Verify frame count calculation."""
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        # With center-origin, the signal is padded
        sr = 44100
        n_samples = 44100  # 1 second
        frame_size = 2048
        hop_size = 441

        sig = Signal(np.zeros(n_samples, dtype=np.float32), sr=sr)
        frames = FramedSignal(sig, frame_size=frame_size, hop_size=hop_size)

        # With center padding: padded_len = n_samples + frame_size
        padded_len = n_samples + frame_size
        expected_frames = 1 + (padded_len - frame_size) // hop_size
        assert len(frames) == expected_frames

    def test_framed_signal_custom_params(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.random.randn(22050).astype(np.float32)
        sig = Signal(data, sr=22050)
        frames = FramedSignal(sig, frame_size=1024, hop_size=256)

        assert frames.frame_size == 1024
        assert frames.hop_size == 256
        assert frames.shape[1] == 1024

    def test_framed_signal_getitem(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.random.randn(44100).astype(np.float32)
        sig = Signal(data, sr=44100)
        frames = FramedSignal(sig, frame_size=2048, hop_size=441)

        frame0 = frames[0]
        assert frame0.shape == (2048,)
        assert frame0.dtype == np.float32

    def test_framed_signal_iterable(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.random.randn(4410).astype(np.float32)
        sig = Signal(data, sr=44100)
        frames = FramedSignal(sig, frame_size=2048, hop_size=441)

        count = 0
        for frame in frames:
            assert frame.shape == (2048,)
            count += 1
        assert count == len(frames)

    def test_framed_signal_sample_rate(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.zeros(44100, dtype=np.float32)
        sig = Signal(data, sr=44100)
        frames = FramedSignal(sig)
        assert frames.sample_rate == 44100

    def test_framed_signal_from_array(self):
        """FramedSignal should accept a raw numpy array wrapped in Signal."""
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        data = np.random.randn(22050).astype(np.float32)
        sig = Signal(data, sr=22050)
        frames = FramedSignal(sig, frame_size=1024, hop_size=512)
        assert len(frames) > 0
        assert frames[0].shape == (1024,)


# ---------------------------------------------------------------------------
# STFT tests
# ---------------------------------------------------------------------------

class TestSTFT:
    """Tests for madmom.audio.stft.STFT."""

    def test_stft_produces_complex(self):
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.random.randn(44100).astype(np.float32)
        stft = STFT(data, frame_size=2048, hop_size=441, sample_rate=44100)

        assert np.iscomplexobj(stft)
        assert stft.ndim == 2
        # Shape should be (n_frames, n_fft // 2 + 1)
        assert stft.shape[1] == 2048 // 2 + 1

    def test_stft_default_params(self):
        """STFT defaults should match madmom: frame_size=2048, hop_size=441."""
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.random.randn(44100).astype(np.float32)
        stft = STFT(data, sample_rate=44100)

        assert stft.frame_size == 2048
        assert stft.hop_size == 441
        assert stft.sample_rate == 44100

    def test_stft_from_framed_signal(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        from metalmom.compat.madmom.audio.stft import STFT

        data = np.random.randn(44100).astype(np.float32)
        sig = Signal(data, sr=44100)
        frames = FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = STFT(frames)

        assert np.iscomplexobj(stft)
        assert stft.shape[0] == len(frames)
        assert stft.shape[1] == 2048 // 2 + 1

    def test_stft_num_frames_and_bins(self):
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.random.randn(44100).astype(np.float32)
        stft = STFT(data, frame_size=1024, hop_size=512, sample_rate=44100)

        assert stft.num_bins == 1024 // 2 + 1
        assert stft.num_frames == stft.shape[0]
        assert stft.num_frames > 0


# ---------------------------------------------------------------------------
# Spectrogram tests
# ---------------------------------------------------------------------------

class TestSpectrogram:
    """Tests for madmom.audio.spectrogram.Spectrogram."""

    def test_spectrogram_magnitude(self):
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram
        data = np.random.randn(44100).astype(np.float32)
        spec = Spectrogram(data, frame_size=2048, hop_size=441, sample_rate=44100)

        assert spec.dtype == np.float32
        assert spec.ndim == 2
        # Magnitude should be non-negative
        assert np.all(spec >= 0)

    def test_spectrogram_from_complex(self):
        """Spectrogram from complex STFT data takes magnitude."""
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram

        data = np.random.randn(44100).astype(np.float32)
        stft = STFT(data, frame_size=2048, hop_size=441, sample_rate=44100)
        spec = Spectrogram(stft)

        assert not np.iscomplexobj(spec)
        assert spec.shape == stft.shape
        np.testing.assert_allclose(spec, np.abs(np.asarray(stft)), atol=1e-6)

    def test_spectrogram_default_params(self):
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram
        data = np.random.randn(44100).astype(np.float32)
        spec = Spectrogram(data, sample_rate=44100)

        assert spec.frame_size == 2048
        assert spec.hop_size == 441
        assert spec.sample_rate == 44100


# ---------------------------------------------------------------------------
# FilteredSpectrogram tests
# ---------------------------------------------------------------------------

class TestFilteredSpectrogram:
    """Tests for madmom.audio.spectrogram.FilteredSpectrogram."""

    def test_filtered_spectrogram_shape(self):
        from metalmom.compat.madmom.audio.spectrogram import FilteredSpectrogram
        data = np.random.randn(44100).astype(np.float32)
        filt = FilteredSpectrogram(
            data, num_bands=80, frame_size=2048, hop_size=441,
            sample_rate=44100,
        )

        assert filt.ndim == 2
        # Second dimension should be num_bands
        assert filt.shape[1] == 80
        assert filt.dtype == np.float32

    def test_filtered_spectrogram_default_bands(self):
        """Default num_bands should be 80."""
        from metalmom.compat.madmom.audio.spectrogram import FilteredSpectrogram
        data = np.random.randn(44100).astype(np.float32)
        filt = FilteredSpectrogram(data, sample_rate=44100)

        assert filt.num_bands == 80
        assert filt.shape[1] == 80

    def test_filtered_spectrogram_custom_bands(self):
        from metalmom.compat.madmom.audio.spectrogram import FilteredSpectrogram
        data = np.random.randn(44100).astype(np.float32)
        filt = FilteredSpectrogram(data, num_bands=40, sample_rate=44100)

        assert filt.num_bands == 40
        assert filt.shape[1] == 40

    def test_filtered_spectrogram_nonnegative(self):
        from metalmom.compat.madmom.audio.spectrogram import FilteredSpectrogram
        data = np.random.randn(44100).astype(np.float32)
        filt = FilteredSpectrogram(data, sample_rate=44100)
        assert np.all(filt >= 0)


# ---------------------------------------------------------------------------
# LogarithmicFilteredSpectrogram tests
# ---------------------------------------------------------------------------

class TestLogFilteredSpectrogram:
    """Tests for madmom.audio.spectrogram.LogarithmicFilteredSpectrogram."""

    def test_log_filtered_spectrogram(self):
        from metalmom.compat.madmom.audio.spectrogram import (
            FilteredSpectrogram, LogarithmicFilteredSpectrogram,
        )
        data = np.random.randn(44100).astype(np.float32)
        filt = FilteredSpectrogram(data, num_bands=80, sample_rate=44100)
        log_filt = LogarithmicFilteredSpectrogram(filt)

        assert log_filt.shape == filt.shape
        assert log_filt.dtype == np.float32

        # log(1 + x) should be applied: verify manually
        expected = np.log(1.0 * np.asarray(filt, dtype=np.float32) + 1.0)
        np.testing.assert_allclose(log_filt, expected, atol=1e-5)

    def test_log_filtered_from_raw_audio(self):
        from metalmom.compat.madmom.audio.spectrogram import (
            LogarithmicFilteredSpectrogram,
        )
        data = np.random.randn(44100).astype(np.float32)
        log_filt = LogarithmicFilteredSpectrogram(
            data, num_bands=80, sample_rate=44100,
        )

        assert log_filt.ndim == 2
        assert log_filt.shape[1] == 80
        # Log(1+x) is always >= 0 when x >= 0
        assert np.all(log_filt >= 0)


# ---------------------------------------------------------------------------
# Processor tests: callable interface
# ---------------------------------------------------------------------------

class TestProcessorsCallable:
    """All madmom processors should be callable (implement __call__)."""

    def test_onset_peak_picking_callable(self):
        from metalmom.compat.madmom.features.onsets import OnsetPeakPickingProcessor
        proc = OnsetPeakPickingProcessor()
        assert callable(proc)

    def test_rnn_onset_processor_callable(self):
        from metalmom.compat.madmom.features.onsets import RNNOnsetProcessor
        proc = RNNOnsetProcessor()
        assert callable(proc)

    def test_rnn_beat_processor_callable(self):
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor
        proc = RNNBeatProcessor()
        assert callable(proc)

    def test_dbn_beat_tracking_callable(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor()
        assert callable(proc)

    def test_rnn_downbeat_processor_callable(self):
        from metalmom.compat.madmom.features.downbeats import RNNDownBeatProcessor
        proc = RNNDownBeatProcessor()
        assert callable(proc)

    def test_dbn_downbeat_tracking_callable(self):
        from metalmom.compat.madmom.features.downbeats import DBNDownBeatTrackingProcessor
        proc = DBNDownBeatTrackingProcessor()
        assert callable(proc)

    def test_cnn_key_recognition_callable(self):
        from metalmom.compat.madmom.features.key import CNNKeyRecognitionProcessor
        proc = CNNKeyRecognitionProcessor()
        assert callable(proc)

    def test_deep_chroma_chord_recognition_callable(self):
        from metalmom.compat.madmom.features.chords import DeepChromaChordRecognitionProcessor
        proc = DeepChromaChordRecognitionProcessor()
        assert callable(proc)


# ---------------------------------------------------------------------------
# Processor execution tests
# ---------------------------------------------------------------------------

class TestOnsetPeakPicking:
    """Tests for OnsetPeakPickingProcessor execution."""

    def test_peak_picking_returns_times(self):
        from metalmom.compat.madmom.features.onsets import OnsetPeakPickingProcessor
        proc = OnsetPeakPickingProcessor(fps=100.0, threshold=0.3)

        # Create synthetic activation with clear peaks
        act = np.zeros(1000, dtype=np.float32)
        act[100] = 0.9
        act[300] = 0.8
        act[500] = 0.7
        act[700] = 0.85

        times = proc(act)
        assert isinstance(times, np.ndarray)
        assert len(times) > 0
        # All times should be non-negative
        assert np.all(times >= 0)

    def test_peak_picking_default_fps(self):
        from metalmom.compat.madmom.features.onsets import OnsetPeakPickingProcessor
        proc = OnsetPeakPickingProcessor()
        assert proc.fps == 100.0


class TestKeyProcessor:
    """Tests for CNNKeyRecognitionProcessor."""

    def test_key_from_activations(self):
        from metalmom.compat.madmom.features.key import CNNKeyRecognitionProcessor
        proc = CNNKeyRecognitionProcessor()

        # Simulate activation probabilities for C major (index 3)
        act = np.zeros(24, dtype=np.float32)
        act[3] = 0.9  # C major
        result = proc(act)

        assert isinstance(result, np.ndarray)
        assert result.shape == (24,)

    def test_key_from_multi_frame(self):
        from metalmom.compat.madmom.features.key import CNNKeyRecognitionProcessor
        proc = CNNKeyRecognitionProcessor()

        # Multi-frame activations
        act = np.random.rand(10, 24).astype(np.float32)
        result = proc(act)
        assert result.shape == (24,)


class TestChordProcessor:
    """Tests for DeepChromaChordRecognitionProcessor."""

    def test_chord_from_activations(self):
        from metalmom.compat.madmom.features.chords import DeepChromaChordRecognitionProcessor
        proc = DeepChromaChordRecognitionProcessor(fps=10.0)

        # Simulate activations: 10 frames, 25 classes
        act = np.random.rand(10, 25).astype(np.float32)
        # Make one chord clearly dominant
        act[:, 1] = 2.0  # C major dominant

        result = proc(act)
        assert isinstance(result, list)
        # Each result should be (start, end, label)
        for item in result:
            assert len(item) == 3


# ---------------------------------------------------------------------------
# Default parameter value tests (madmom conventions)
# ---------------------------------------------------------------------------

class TestMadmomDefaults:
    """Verify default parameter values match madmom conventions."""

    def test_signal_default_sr_is_44100(self):
        from metalmom.compat.madmom.audio.signal import Signal
        sig = Signal(np.zeros(100, dtype=np.float32))
        assert sig.sample_rate == 44100

    def test_framed_signal_default_hop_is_441(self):
        from metalmom.compat.madmom.audio.signal import FramedSignal
        data = np.zeros(44100, dtype=np.float32)
        frames = FramedSignal(data)
        assert frames.hop_size == 441

    def test_framed_signal_default_frame_size_is_2048(self):
        from metalmom.compat.madmom.audio.signal import FramedSignal
        data = np.zeros(44100, dtype=np.float32)
        frames = FramedSignal(data)
        assert frames.frame_size == 2048

    def test_stft_default_frame_size_is_2048(self):
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.zeros(44100, dtype=np.float32)
        stft = STFT(data, sample_rate=44100)
        assert stft.frame_size == 2048

    def test_stft_default_hop_is_441(self):
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.zeros(44100, dtype=np.float32)
        stft = STFT(data, sample_rate=44100)
        assert stft.hop_size == 441

    def test_stft_default_sr_is_44100(self):
        from metalmom.compat.madmom.audio.stft import STFT
        data = np.zeros(44100, dtype=np.float32)
        stft = STFT(data)
        assert stft.sample_rate == 44100

    def test_spectrogram_default_sr_is_44100(self):
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram
        data = np.zeros(44100, dtype=np.float32)
        spec = Spectrogram(data)
        assert spec.sample_rate == 44100

    def test_dbn_beat_default_fps_is_100(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor()
        assert proc.fps == 100.0

    def test_dbn_beat_default_min_bpm(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor()
        assert proc.min_bpm == 55.0

    def test_dbn_beat_default_max_bpm(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor()
        assert proc.max_bpm == 215.0


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify the package structure can be imported like real madmom."""

    def test_import_audio_signal(self):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        assert Signal is not None
        assert FramedSignal is not None

    def test_import_audio_stft(self):
        from metalmom.compat.madmom.audio.stft import STFT
        assert STFT is not None

    def test_import_audio_spectrogram(self):
        from metalmom.compat.madmom.audio.spectrogram import (
            Spectrogram, FilteredSpectrogram, LogarithmicFilteredSpectrogram,
        )
        assert Spectrogram is not None
        assert FilteredSpectrogram is not None
        assert LogarithmicFilteredSpectrogram is not None

    def test_import_features_onsets(self):
        from metalmom.compat.madmom.features.onsets import (
            OnsetPeakPickingProcessor, RNNOnsetProcessor,
        )
        assert OnsetPeakPickingProcessor is not None
        assert RNNOnsetProcessor is not None

    def test_import_features_beats(self):
        from metalmom.compat.madmom.features.beats import (
            RNNBeatProcessor, DBNBeatTrackingProcessor,
        )
        assert RNNBeatProcessor is not None
        assert DBNBeatTrackingProcessor is not None

    def test_import_features_downbeats(self):
        from metalmom.compat.madmom.features.downbeats import (
            RNNDownBeatProcessor, DBNDownBeatTrackingProcessor,
        )
        assert RNNDownBeatProcessor is not None
        assert DBNDownBeatTrackingProcessor is not None

    def test_import_features_key(self):
        from metalmom.compat.madmom.features.key import CNNKeyRecognitionProcessor
        assert CNNKeyRecognitionProcessor is not None

    def test_import_features_chords(self):
        from metalmom.compat.madmom.features.chords import DeepChromaChordRecognitionProcessor
        assert DeepChromaChordRecognitionProcessor is not None

    def test_import_via_compat(self):
        """Should be importable via metalmom.compat.madmom path."""
        from metalmom.compat import madmom
        assert hasattr(madmom, 'audio')
        assert hasattr(madmom, 'features')
        assert hasattr(madmom.audio, 'signal')
        assert hasattr(madmom.audio, 'stft')
        assert hasattr(madmom.audio, 'spectrogram')
        assert hasattr(madmom.features, 'onsets')
        assert hasattr(madmom.features, 'beats')
        assert hasattr(madmom.features, 'downbeats')
        assert hasattr(madmom.features, 'key')
        assert hasattr(madmom.features, 'chords')
