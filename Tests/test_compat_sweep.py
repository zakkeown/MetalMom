"""Tests for complete librosa and madmom compatibility shim coverage.

Verifies that all compat modules are importable and key functions/classes
are accessible via the compat API paths.
"""

import pytest


# ---------------------------------------------------------------------------
# librosa compat: top-level imports
# ---------------------------------------------------------------------------

class TestLibrosaTopLevel:
    """Verify top-level librosa compat imports."""

    def test_import_from_compat(self):
        from metalmom.compat import librosa
        assert librosa is not None

    def test_stft(self):
        from metalmom.compat import librosa
        assert callable(librosa.stft)

    def test_istft(self):
        from metalmom.compat import librosa
        assert callable(librosa.istft)

    def test_load(self):
        from metalmom.compat import librosa
        assert callable(librosa.load)

    def test_resample(self):
        from metalmom.compat import librosa
        assert callable(librosa.resample)

    def test_get_duration(self):
        from metalmom.compat import librosa
        assert callable(librosa.get_duration)

    def test_get_samplerate(self):
        from metalmom.compat import librosa
        assert callable(librosa.get_samplerate)

    def test_reassigned_spectrogram(self):
        from metalmom.compat import librosa
        assert callable(librosa.reassigned_spectrogram)

    def test_phase_vocoder(self):
        from metalmom.compat import librosa
        assert callable(librosa.phase_vocoder)

    def test_griffinlim(self):
        from metalmom.compat import librosa
        assert callable(librosa.griffinlim)

    def test_cqt(self):
        from metalmom.compat import librosa
        assert callable(librosa.cqt)

    def test_vqt(self):
        from metalmom.compat import librosa
        assert callable(librosa.vqt)

    def test_hybrid_cqt(self):
        from metalmom.compat import librosa
        assert callable(librosa.hybrid_cqt)


# ---------------------------------------------------------------------------
# librosa.core module
# ---------------------------------------------------------------------------

class TestLibrosaCore:
    """Verify librosa.core compat functions."""

    def test_reassigned_spectrogram(self):
        from metalmom.compat.librosa.core import reassigned_spectrogram
        assert callable(reassigned_spectrogram)

    def test_phase_vocoder(self):
        from metalmom.compat.librosa.core import phase_vocoder
        assert callable(phase_vocoder)

    def test_griffinlim(self):
        from metalmom.compat.librosa.core import griffinlim
        assert callable(griffinlim)

    def test_stft(self):
        from metalmom.compat.librosa.core import stft
        assert callable(stft)

    def test_istft(self):
        from metalmom.compat.librosa.core import istft
        assert callable(istft)

    def test_load(self):
        from metalmom.compat.librosa.core import load
        assert callable(load)

    def test_amplitude_to_db(self):
        from metalmom.compat.librosa.core import amplitude_to_db
        assert callable(amplitude_to_db)

    def test_power_to_db(self):
        from metalmom.compat.librosa.core import power_to_db
        assert callable(power_to_db)

    def test_tone(self):
        from metalmom.compat.librosa.core import tone
        assert callable(tone)

    def test_chirp(self):
        from metalmom.compat.librosa.core import chirp
        assert callable(chirp)

    def test_clicks(self):
        from metalmom.compat.librosa.core import clicks
        assert callable(clicks)

    def test_stream(self):
        from metalmom.compat.librosa.core import stream
        assert callable(stream)

    def test_db_to_amplitude(self):
        from metalmom.compat.librosa.core import db_to_amplitude
        assert callable(db_to_amplitude)

    def test_db_to_power(self):
        from metalmom.compat.librosa.core import db_to_power
        assert callable(db_to_power)


# ---------------------------------------------------------------------------
# librosa.feature module
# ---------------------------------------------------------------------------

class TestLibrosaFeature:
    """Verify librosa.feature compat functions."""

    def test_melspectrogram(self):
        from metalmom.compat.librosa.feature import melspectrogram
        assert callable(melspectrogram)

    def test_mfcc(self):
        from metalmom.compat.librosa.feature import mfcc
        assert callable(mfcc)

    def test_chroma_stft(self):
        from metalmom.compat.librosa.feature import chroma_stft
        assert callable(chroma_stft)

    def test_chroma_cqt(self):
        from metalmom.compat.librosa.feature import chroma_cqt
        assert callable(chroma_cqt)

    def test_chroma_cens(self):
        from metalmom.compat.librosa.feature import chroma_cens
        assert callable(chroma_cens)

    def test_chroma_vqt(self):
        from metalmom.compat.librosa.feature import chroma_vqt
        assert callable(chroma_vqt)

    def test_spectral_centroid(self):
        from metalmom.compat.librosa.feature import spectral_centroid
        assert callable(spectral_centroid)

    def test_spectral_bandwidth(self):
        from metalmom.compat.librosa.feature import spectral_bandwidth
        assert callable(spectral_bandwidth)

    def test_spectral_contrast(self):
        from metalmom.compat.librosa.feature import spectral_contrast
        assert callable(spectral_contrast)

    def test_spectral_rolloff(self):
        from metalmom.compat.librosa.feature import spectral_rolloff
        assert callable(spectral_rolloff)

    def test_spectral_flatness(self):
        from metalmom.compat.librosa.feature import spectral_flatness
        assert callable(spectral_flatness)

    def test_rms(self):
        from metalmom.compat.librosa.feature import rms
        assert callable(rms)

    def test_zero_crossing_rate(self):
        from metalmom.compat.librosa.feature import zero_crossing_rate
        assert callable(zero_crossing_rate)

    def test_tonnetz(self):
        from metalmom.compat.librosa.feature import tonnetz
        assert callable(tonnetz)

    def test_delta(self):
        from metalmom.compat.librosa.feature import delta
        assert callable(delta)

    def test_stack_memory(self):
        from metalmom.compat.librosa.feature import stack_memory
        assert callable(stack_memory)

    def test_poly_features(self):
        from metalmom.compat.librosa.feature import poly_features
        assert callable(poly_features)

    def test_tempo(self):
        from metalmom.compat.librosa.feature import tempo
        assert callable(tempo)

    def test_tempogram(self):
        from metalmom.compat.librosa.feature import tempogram
        assert callable(tempogram)

    def test_fourier_tempogram(self):
        from metalmom.compat.librosa.feature import fourier_tempogram
        assert callable(fourier_tempogram)

    def test_pcen(self):
        from metalmom.compat.librosa.feature import pcen
        assert callable(pcen)

    def test_mel_to_audio(self):
        from metalmom.compat.librosa.feature import mel_to_audio
        assert callable(mel_to_audio)

    def test_mfcc_to_mel(self):
        from metalmom.compat.librosa.feature import mfcc_to_mel
        assert callable(mfcc_to_mel)

    def test_mfcc_to_audio(self):
        from metalmom.compat.librosa.feature import mfcc_to_audio
        assert callable(mfcc_to_audio)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.feature.melspectrogram)
        assert callable(librosa.feature.pcen)


# ---------------------------------------------------------------------------
# librosa.onset module
# ---------------------------------------------------------------------------

class TestLibrosaOnset:
    """Verify librosa.onset compat functions."""

    def test_onset_strength(self):
        from metalmom.compat.librosa.onset import onset_strength
        assert callable(onset_strength)

    def test_onset_detect(self):
        from metalmom.compat.librosa.onset import onset_detect
        assert callable(onset_detect)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.onset.onset_strength)


# ---------------------------------------------------------------------------
# librosa.beat module
# ---------------------------------------------------------------------------

class TestLibrosaBeat:
    """Verify librosa.beat compat functions."""

    def test_beat_track(self):
        from metalmom.compat.librosa.beat import beat_track
        assert callable(beat_track)

    def test_plp(self):
        from metalmom.compat.librosa.beat import plp
        assert callable(plp)


# ---------------------------------------------------------------------------
# librosa.pitch module
# ---------------------------------------------------------------------------

class TestLibrosaPitch:
    """Verify librosa.pitch compat functions."""

    def test_yin(self):
        from metalmom.compat.librosa.pitch import yin
        assert callable(yin)

    def test_pyin(self):
        from metalmom.compat.librosa.pitch import pyin
        assert callable(pyin)

    def test_piptrack(self):
        from metalmom.compat.librosa.pitch import piptrack
        assert callable(piptrack)

    def test_estimate_tuning(self):
        from metalmom.compat.librosa.pitch import estimate_tuning
        assert callable(estimate_tuning)


# ---------------------------------------------------------------------------
# librosa.effects module
# ---------------------------------------------------------------------------

class TestLibrosaEffects:
    """Verify librosa.effects compat functions."""

    def test_hpss(self):
        from metalmom.compat.librosa.effects import hpss
        assert callable(hpss)

    def test_harmonic(self):
        from metalmom.compat.librosa.effects import harmonic
        assert callable(harmonic)

    def test_percussive(self):
        from metalmom.compat.librosa.effects import percussive
        assert callable(percussive)

    def test_time_stretch(self):
        from metalmom.compat.librosa.effects import time_stretch
        assert callable(time_stretch)

    def test_pitch_shift(self):
        from metalmom.compat.librosa.effects import pitch_shift
        assert callable(pitch_shift)

    def test_trim(self):
        from metalmom.compat.librosa.effects import trim
        assert callable(trim)

    def test_split(self):
        from metalmom.compat.librosa.effects import split
        assert callable(split)

    def test_preemphasis(self):
        from metalmom.compat.librosa.effects import preemphasis
        assert callable(preemphasis)

    def test_deemphasis(self):
        from metalmom.compat.librosa.effects import deemphasis
        assert callable(deemphasis)

    def test_phase_vocoder(self):
        from metalmom.compat.librosa.effects import phase_vocoder
        assert callable(phase_vocoder)

    def test_griffinlim(self):
        from metalmom.compat.librosa.effects import griffinlim
        assert callable(griffinlim)


# ---------------------------------------------------------------------------
# librosa.decompose module
# ---------------------------------------------------------------------------

class TestLibrosaDecompose:
    """Verify librosa.decompose compat functions."""

    def test_nmf(self):
        from metalmom.compat.librosa.decompose import nmf
        assert callable(nmf)

    def test_nn_filter(self):
        from metalmom.compat.librosa.decompose import nn_filter
        assert callable(nn_filter)


# ---------------------------------------------------------------------------
# librosa.sequence module
# ---------------------------------------------------------------------------

class TestLibrosaSequence:
    """Verify librosa.sequence compat functions."""

    def test_viterbi(self):
        from metalmom.compat.librosa.sequence import viterbi
        assert callable(viterbi)

    def test_viterbi_discriminative(self):
        from metalmom.compat.librosa.sequence import viterbi_discriminative
        assert callable(viterbi_discriminative)

    def test_viterbi_binary(self):
        from metalmom.compat.librosa.sequence import viterbi_binary
        assert callable(viterbi_binary)


# ---------------------------------------------------------------------------
# librosa.filters module
# ---------------------------------------------------------------------------

class TestLibrosaFilters:
    """Verify librosa.filters compat functions."""

    def test_mel(self):
        from metalmom.compat.librosa.filters import mel
        assert callable(mel)

    def test_chroma(self):
        from metalmom.compat.librosa.filters import chroma
        assert callable(chroma)

    def test_constant_q(self):
        from metalmom.compat.librosa.filters import constant_q
        assert callable(constant_q)

    def test_mel_frequencies(self):
        from metalmom.compat.librosa.filters import mel_frequencies
        assert callable(mel_frequencies)

    def test_fft_frequencies(self):
        from metalmom.compat.librosa.filters import fft_frequencies
        assert callable(fft_frequencies)

    def test_semitone_filterbank(self):
        from metalmom.compat.librosa.filters import semitone_filterbank
        assert callable(semitone_filterbank)

    def test_semitone_frequencies(self):
        from metalmom.compat.librosa.filters import semitone_frequencies
        assert callable(semitone_frequencies)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.filters.mel)


# ---------------------------------------------------------------------------
# librosa.segment module (new)
# ---------------------------------------------------------------------------

class TestLibrosaSegment:
    """Verify librosa.segment compat functions."""

    def test_recurrence_matrix(self):
        from metalmom.compat.librosa.segment import recurrence_matrix
        assert callable(recurrence_matrix)

    def test_cross_similarity(self):
        from metalmom.compat.librosa.segment import cross_similarity
        assert callable(cross_similarity)

    def test_agglomerative(self):
        from metalmom.compat.librosa.segment import agglomerative
        assert callable(agglomerative)

    def test_dtw(self):
        from metalmom.compat.librosa.segment import dtw
        assert callable(dtw)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.segment.recurrence_matrix)
        assert callable(librosa.segment.dtw)


# ---------------------------------------------------------------------------
# librosa.display module (new)
# ---------------------------------------------------------------------------

class TestLibrosaDisplay:
    """Verify librosa.display compat functions."""

    def test_specshow(self):
        from metalmom.compat.librosa.display import specshow
        assert callable(specshow)

    def test_waveshow(self):
        from metalmom.compat.librosa.display import waveshow
        assert callable(waveshow)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.display.specshow)
        assert callable(librosa.display.waveshow)


# ---------------------------------------------------------------------------
# librosa.convert module (new)
# ---------------------------------------------------------------------------

class TestLibrosaConvert:
    """Verify librosa.convert compat functions."""

    def test_hz_to_midi(self):
        from metalmom.compat.librosa.convert import hz_to_midi
        assert callable(hz_to_midi)

    def test_midi_to_hz(self):
        from metalmom.compat.librosa.convert import midi_to_hz
        assert callable(midi_to_hz)

    def test_hz_to_note(self):
        from metalmom.compat.librosa.convert import hz_to_note
        assert callable(hz_to_note)

    def test_note_to_hz(self):
        from metalmom.compat.librosa.convert import note_to_hz
        assert callable(note_to_hz)

    def test_midi_to_note(self):
        from metalmom.compat.librosa.convert import midi_to_note
        assert callable(midi_to_note)

    def test_note_to_midi(self):
        from metalmom.compat.librosa.convert import note_to_midi
        assert callable(note_to_midi)

    def test_times_to_frames(self):
        from metalmom.compat.librosa.convert import times_to_frames
        assert callable(times_to_frames)

    def test_frames_to_time(self):
        from metalmom.compat.librosa.convert import frames_to_time
        assert callable(frames_to_time)

    def test_times_to_samples(self):
        from metalmom.compat.librosa.convert import times_to_samples
        assert callable(times_to_samples)

    def test_samples_to_time(self):
        from metalmom.compat.librosa.convert import samples_to_time
        assert callable(samples_to_time)

    def test_frames_to_samples(self):
        from metalmom.compat.librosa.convert import frames_to_samples
        assert callable(frames_to_samples)

    def test_samples_to_frames(self):
        from metalmom.compat.librosa.convert import samples_to_frames
        assert callable(samples_to_frames)

    def test_fft_frequencies(self):
        from metalmom.compat.librosa.convert import fft_frequencies
        assert callable(fft_frequencies)

    def test_mel_frequencies(self):
        from metalmom.compat.librosa.convert import mel_frequencies
        assert callable(mel_frequencies)

    def test_via_module_attribute(self):
        from metalmom.compat import librosa
        assert callable(librosa.convert.hz_to_midi)
        assert callable(librosa.convert.midi_to_hz)


# ---------------------------------------------------------------------------
# madmom compat: package imports
# ---------------------------------------------------------------------------

class TestMadmomImports:
    """Verify madmom compat package structure."""

    def test_import_from_compat(self):
        from metalmom.compat import madmom
        assert madmom is not None

    def test_audio_module(self):
        from metalmom.compat import madmom
        assert hasattr(madmom, 'audio')

    def test_features_module(self):
        from metalmom.compat import madmom
        assert hasattr(madmom, 'features')


# ---------------------------------------------------------------------------
# madmom.audio compat
# ---------------------------------------------------------------------------

class TestMadmomAudio:
    """Verify madmom.audio compat classes."""

    def test_signal(self):
        from metalmom.compat.madmom.audio.signal import Signal
        assert Signal is not None

    def test_framed_signal(self):
        from metalmom.compat.madmom.audio.signal import FramedSignal
        assert FramedSignal is not None

    def test_stft(self):
        from metalmom.compat.madmom.audio.stft import STFT
        assert STFT is not None

    def test_spectrogram(self):
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram
        assert Spectrogram is not None

    def test_filtered_spectrogram(self):
        from metalmom.compat.madmom.audio.spectrogram import FilteredSpectrogram
        assert FilteredSpectrogram is not None

    def test_log_filtered_spectrogram(self):
        from metalmom.compat.madmom.audio.spectrogram import LogarithmicFilteredSpectrogram
        assert LogarithmicFilteredSpectrogram is not None

    def test_via_module_path(self):
        from metalmom.compat import madmom
        assert hasattr(madmom.audio, 'signal')
        assert hasattr(madmom.audio, 'stft')
        assert hasattr(madmom.audio, 'spectrogram')
        assert hasattr(madmom.audio.signal, 'Signal')


# ---------------------------------------------------------------------------
# madmom.features compat
# ---------------------------------------------------------------------------

class TestMadmomFeatures:
    """Verify madmom.features compat classes."""

    def test_onset_peak_picking(self):
        from metalmom.compat.madmom.features.onsets import OnsetPeakPickingProcessor
        assert callable(OnsetPeakPickingProcessor)

    def test_rnn_onset(self):
        from metalmom.compat.madmom.features.onsets import RNNOnsetProcessor
        assert callable(RNNOnsetProcessor)

    def test_rnn_beat(self):
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor
        assert callable(RNNBeatProcessor)

    def test_dbn_beat_tracking(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor
        assert callable(DBNBeatTrackingProcessor)

    def test_rnn_downbeat(self):
        from metalmom.compat.madmom.features.downbeats import RNNDownBeatProcessor
        assert callable(RNNDownBeatProcessor)

    def test_dbn_downbeat_tracking(self):
        from metalmom.compat.madmom.features.downbeats import DBNDownBeatTrackingProcessor
        assert callable(DBNDownBeatTrackingProcessor)

    def test_cnn_key(self):
        from metalmom.compat.madmom.features.key import CNNKeyRecognitionProcessor
        assert callable(CNNKeyRecognitionProcessor)

    def test_deep_chroma_chord(self):
        from metalmom.compat.madmom.features.chords import DeepChromaChordRecognitionProcessor
        assert callable(DeepChromaChordRecognitionProcessor)

    def test_via_module_path(self):
        from metalmom.compat import madmom
        assert hasattr(madmom.features, 'onsets')
        assert hasattr(madmom.features, 'beats')
        assert hasattr(madmom.features, 'downbeats')
        assert hasattr(madmom.features, 'key')
        assert hasattr(madmom.features, 'chords')
        assert hasattr(madmom.features, 'tempo')


# ---------------------------------------------------------------------------
# madmom.features.tempo (new)
# ---------------------------------------------------------------------------

class TestMadmomTempo:
    """Verify madmom.features.tempo compat class."""

    def test_import(self):
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor
        assert callable(TempoEstimationProcessor)

    def test_callable(self):
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor
        proc = TempoEstimationProcessor()
        assert callable(proc)

    def test_default_fps(self):
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor
        proc = TempoEstimationProcessor()
        assert proc.fps == 100.0

    def test_default_bpm_range(self):
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor
        proc = TempoEstimationProcessor()
        assert proc.min_bpm == 40.0
        assert proc.max_bpm == 250.0

    def test_custom_params(self):
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor
        proc = TempoEstimationProcessor(fps=50.0, min_bpm=60.0, max_bpm=200.0)
        assert proc.fps == 50.0
        assert proc.min_bpm == 60.0
        assert proc.max_bpm == 200.0

    def test_via_module_path(self):
        from metalmom.compat import madmom
        assert hasattr(madmom.features.tempo, 'TempoEstimationProcessor')

    def test_call_with_activations(self):
        import numpy as np
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor

        proc = TempoEstimationProcessor(fps=100.0)

        # Create synthetic onset activations with ~120 BPM pattern
        # 100 fps * 10 seconds = 1000 frames
        act = np.zeros(1000, dtype=np.float32)
        # 120 BPM = 2 beats/sec = beat every 50 frames at 100fps
        for i in range(0, 1000, 50):
            act[i] = 1.0

        result = proc(act)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2  # (tempo, strength) pairs


# ---------------------------------------------------------------------------
# Cross-module consistency: ensure the same function is reachable via
# multiple compat paths (just like real librosa)
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    """Verify functions accessible from multiple paths match."""

    def test_phase_vocoder_same_in_core_and_effects(self):
        from metalmom.compat.librosa.core import phase_vocoder as pv_core
        from metalmom.compat.librosa.effects import phase_vocoder as pv_effects
        assert pv_core is pv_effects

    def test_griffinlim_same_in_core_and_effects(self):
        from metalmom.compat.librosa.core import griffinlim as gl_core
        from metalmom.compat.librosa.effects import griffinlim as gl_effects
        assert gl_core is gl_effects

    def test_fft_frequencies_in_convert_and_filters(self):
        from metalmom.compat.librosa.convert import fft_frequencies as ff_convert
        from metalmom.compat.librosa.filters import fft_frequencies as ff_filters
        # These may come from different source modules (convert vs filters)
        # but both should be callable
        assert callable(ff_convert)
        assert callable(ff_filters)

    def test_mel_frequencies_in_convert_and_filters(self):
        from metalmom.compat.librosa.convert import mel_frequencies as mf_convert
        from metalmom.compat.librosa.filters import mel_frequencies as mf_filters
        assert callable(mf_convert)
        assert callable(mf_filters)
