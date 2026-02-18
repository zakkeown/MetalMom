"""End-to-end integration tests for librosa and madmom compat shims.

Exercises realistic user workflows through the compat API, verifying that
common analysis pipelines produce correctly shaped and typed outputs.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared test signal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tone_22k():
    """1-second 440 Hz sine at 22050 Hz sample rate."""
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


@pytest.fixture
def tone_44k():
    """1-second 440 Hz sine at 44100 Hz sample rate."""
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


@pytest.fixture
def click_train_22k():
    """Click train at 120 BPM (2 Hz) for 4 seconds at 22050 Hz."""
    sr = 22050
    duration = 4.0
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    # Clicks every 0.5 seconds = 120 BPM
    interval_samples = int(sr * 0.5)
    for i in range(0, n_samples, interval_samples):
        end = min(i + 100, n_samples)
        y[i:end] = 0.9
    return y, sr


# ===========================================================================
# librosa compat tests
# ===========================================================================

class TestLibrosaSTFTPipeline:
    """STFT -> inverse STFT -> mel spectrogram -> MFCC -> dB pipeline."""

    def test_stft_shape_and_dtype(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        # n_fft=2048 default -> 1025 freq bins
        assert S.shape[0] == 1025
        assert S.ndim == 2
        # MetalMom STFT returns magnitude (float32), not complex
        assert S.dtype == np.float32

    def test_istft_reconstruction(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        y_hat = librosa.istft(S)
        assert y_hat.ndim == 1
        assert len(y_hat) > 0

    def test_mel_spectrogram_shape(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        assert mel.shape[0] == 128  # default n_mels
        assert mel.ndim == 2
        assert mel.dtype in (np.float32, np.float64)

    def test_mfcc_shape(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        assert mfcc.shape[0] == 20  # default n_mfcc
        assert mfcc.ndim == 2

    def test_amplitude_to_db(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S))
        assert S_db.shape == S.shape
        # dB values should be non-positive for normalized signals
        assert np.isfinite(S_db).all()

    def test_power_to_db(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        S_power = np.abs(S) ** 2
        S_db = librosa.power_to_db(S_power)
        assert S_db.shape == S_power.shape
        assert np.isfinite(S_db).all()

    def test_magphase(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        mag, phase = librosa.magphase(S)
        assert mag.shape == S.shape
        assert phase.shape == S.shape
        assert not np.iscomplexobj(mag)
        assert np.iscomplexobj(phase)
        # Phase should have unit magnitude
        np.testing.assert_allclose(np.abs(phase), 1.0, atol=1e-6)


class TestLibrosaFeaturePipeline:
    """Feature extraction pipeline: chroma, spectral features, RMS."""

    def test_chroma_stft(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        assert chroma.shape[0] == 12
        assert chroma.ndim == 2

    def test_spectral_centroid(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        assert cent.ndim == 2
        assert cent.shape[0] == 1  # single row
        # For a 440 Hz tone, centroid should be near 440 Hz
        mean_cent = np.mean(cent)
        assert 200 < mean_cent < 1000

    def test_spectral_bandwidth(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        assert bw.ndim == 2
        assert bw.shape[0] == 1

    def test_spectral_contrast(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        assert contrast.ndim == 2
        # Default: 6 sub-bands + 1 valley = 7 rows
        assert contrast.shape[0] == 7

    def test_spectral_rolloff(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        assert rolloff.ndim == 2
        assert rolloff.shape[0] == 1

    def test_spectral_flatness(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        flat = librosa.feature.spectral_flatness(y=y)
        assert flat.ndim == 2
        assert flat.shape[0] == 1

    def test_rms(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        rms_val = librosa.feature.rms(y=y)
        assert rms_val.ndim == 2
        assert rms_val.shape[0] == 1
        # RMS of a sine wave ~ amplitude / sqrt(2) ~ 0.707
        mean_rms = np.mean(rms_val)
        assert 0.3 < mean_rms < 1.0

    def test_zero_crossing_rate(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        zcr = librosa.feature.zero_crossing_rate(y)
        assert zcr.ndim == 2
        assert zcr.shape[0] == 1

    def test_tonnetz(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        ton = librosa.feature.tonnetz(y=y, sr=sr)
        assert ton.ndim == 2
        assert ton.shape[0] == 6  # 6 tonal dimensions

    def test_delta(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        d = librosa.feature.delta(mfcc)
        assert d.shape == mfcc.shape

    def test_stack_memory(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        stacked = librosa.feature.stack_memory(mfcc, n_steps=3)
        assert stacked.shape[0] == mfcc.shape[0] * 3


class TestLibrosaOnsetBeatPipeline:
    """Onset detection and beat tracking pipeline."""

    def test_onset_strength(self, click_train_22k):
        from metalmom.compat import librosa

        y, sr = click_train_22k
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        assert onset_env.ndim == 1
        assert len(onset_env) > 0

    def test_onset_detect(self, click_train_22k):
        from metalmom.compat import librosa

        y, sr = click_train_22k
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        assert onsets.ndim == 1
        # Click train should produce some onsets
        assert len(onsets) >= 1

    def test_beat_track(self, click_train_22k):
        from metalmom.compat import librosa

        y, sr = click_train_22k
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        assert isinstance(tempo, (int, float, np.integer, np.floating))
        assert beats.ndim == 1

    def test_tempo(self, click_train_22k):
        from metalmom.compat import librosa

        y, sr = click_train_22k
        t = librosa.feature.tempo(y=y, sr=sr)
        assert np.isscalar(t) or (hasattr(t, 'ndim') and t.size >= 1)


class TestLibrosaEffectsPipeline:
    """Audio effects pipeline: HPSS, trim, time-stretch."""

    def test_hpss(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        assert y_harmonic.shape == y.shape
        assert y_percussive.shape == y.shape

    def test_harmonic_only(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        y_h = librosa.effects.harmonic(y)
        assert y_h.shape == y.shape

    def test_percussive_only(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        y_p = librosa.effects.percussive(y)
        assert y_p.shape == y.shape

    def test_trim(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        # Add silence at start and end
        y_padded = np.concatenate([
            np.zeros(2000, dtype=np.float32),
            y,
            np.zeros(2000, dtype=np.float32),
        ])
        y_trimmed, indices = librosa.effects.trim(y_padded)
        assert len(y_trimmed) <= len(y_padded)
        assert len(indices) == 2  # (start, end)

    def test_split(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        # Insert silence gap
        y_gapped = np.concatenate([
            y[:5000],
            np.zeros(5000, dtype=np.float32),
            y[:5000],
        ])
        intervals = librosa.effects.split(y_gapped)
        assert intervals.ndim == 2
        assert intervals.shape[1] == 2


class TestLibrosaDisplayPipeline:
    """Display functions (non-visual, backend='Agg')."""

    def test_specshow(self, tone_22k):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(np.abs(S), x_axis='time',
                                       y_axis='log', ax=ax, sr=sr)
        assert img is not None
        plt.close(fig)

    def test_specshow_mel(self, tone_22k):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from metalmom.compat import librosa

        y, sr = tone_22k
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel_db, x_axis='time',
                                       y_axis='mel', ax=ax, sr=sr)
        assert img is not None
        plt.close(fig)

    def test_specshow_chroma(self, tone_22k):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from metalmom.compat import librosa

        y, sr = tone_22k
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(chroma, x_axis='time',
                                       y_axis='chroma', ax=ax, sr=sr)
        assert img is not None
        plt.close(fig)

    def test_waveshow(self, tone_22k):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from metalmom.compat import librosa

        y, sr = tone_22k
        fig, ax = plt.subplots()
        artist = librosa.display.waveshow(y, sr=sr, ax=ax)
        assert artist is not None
        plt.close(fig)


class TestLibrosaFiltersPipeline:
    """Filter bank construction."""

    def test_mel_filterbank(self):
        from metalmom.compat import librosa

        mel_fb = librosa.filters.mel(sr=22050, n_fft=2048)
        assert mel_fb.shape == (128, 1025)
        assert mel_fb.dtype in (np.float32, np.float64)
        # Filter bank values should be non-negative
        assert np.all(mel_fb >= 0)

    def test_chroma_filterbank(self):
        from metalmom.compat import librosa

        chroma_fb = librosa.filters.chroma(sr=22050, n_fft=2048)
        assert chroma_fb.shape == (12, 1025)
        assert np.all(chroma_fb >= 0)

    def test_mel_filterbank_custom(self):
        from metalmom.compat import librosa

        mel_fb = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=80)
        assert mel_fb.shape[0] == 80
        assert mel_fb.shape[1] == 2049  # n_fft/2 + 1

    def test_constant_q_filterbank(self):
        from metalmom.compat import librosa

        cq_fb = librosa.filters.constant_q(sr=22050)
        assert cq_fb is not None
        assert cq_fb.ndim == 2


class TestLibrosaConvertPipeline:
    """Unit conversion functions."""

    @pytest.mark.xfail(reason="mm_hz_to_midi C bridge symbol pending dylib rebuild")
    def test_hz_to_midi(self):
        from metalmom.compat import librosa

        midi = librosa.hz_to_midi(440.0)
        assert abs(float(midi) - 69.0) < 0.01

    @pytest.mark.xfail(reason="mm_midi_to_hz C bridge symbol pending dylib rebuild")
    def test_midi_to_hz(self):
        from metalmom.compat import librosa

        hz = librosa.midi_to_hz(69)
        assert abs(float(hz) - 440.0) < 0.01

    @pytest.mark.xfail(reason="mm_hz_to_midi/mm_midi_to_hz C bridge symbols pending dylib rebuild")
    def test_hz_midi_roundtrip(self):
        from metalmom.compat import librosa

        freqs = np.array([261.63, 440.0, 880.0], dtype=np.float32)
        midi = librosa.hz_to_midi(freqs)
        hz_back = librosa.midi_to_hz(midi)
        np.testing.assert_allclose(hz_back, freqs, rtol=1e-3)

    def test_hz_to_note(self):
        from metalmom.compat import librosa

        note = librosa.hz_to_note(440.0)
        assert isinstance(note, (str, np.str_, list))

    def test_note_to_hz(self):
        from metalmom.compat import librosa

        hz = librosa.note_to_hz('A4')
        assert abs(float(hz) - 440.0) < 1.0

    def test_hz_to_mel(self):
        from metalmom.compat import librosa

        mel_val = librosa.hz_to_mel(440.0)
        assert mel_val > 0

    def test_mel_to_hz(self):
        from metalmom.compat import librosa

        hz = librosa.mel_to_hz(librosa.hz_to_mel(440.0))
        assert abs(float(hz) - 440.0) < 1.0

    @pytest.mark.xfail(reason="mm_fft_frequencies C bridge symbol pending dylib rebuild")
    def test_fft_frequencies(self):
        from metalmom.compat import librosa

        freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
        assert len(freqs) == 1025
        assert freqs[0] == 0.0
        np.testing.assert_allclose(freqs[-1], 22050 / 2.0, rtol=1e-3)

    @pytest.mark.xfail(reason="mm_mel_frequencies C bridge symbol pending dylib rebuild")
    def test_mel_frequencies(self):
        from metalmom.compat import librosa

        freqs = librosa.mel_frequencies(n_mels=128)
        assert len(freqs) == 128
        assert freqs[0] >= 0.0

    @pytest.mark.xfail(reason="mm_frames_to_time C bridge symbol pending dylib rebuild")
    def test_frames_to_time(self):
        from metalmom.compat import librosa

        frames = np.arange(10)
        times = librosa.frames_to_time(frames, sr=22050, hop_length=512)
        assert len(times) == 10
        assert times[0] == 0.0
        np.testing.assert_allclose(times[1], 512 / 22050, rtol=1e-3)

    def test_samples_to_time(self):
        from metalmom.compat import librosa

        samples = np.array([0, 22050, 44100])
        times = librosa.samples_to_time(samples, sr=22050)
        np.testing.assert_allclose(times, [0.0, 1.0, 2.0], rtol=1e-3)

    def test_hz_to_octs(self):
        from metalmom.compat import librosa

        octs = librosa.hz_to_octs(440.0)
        assert isinstance(octs, (float, np.floating))

    def test_octs_to_hz(self):
        from metalmom.compat import librosa

        octs = librosa.hz_to_octs(440.0)
        hz = librosa.octs_to_hz(octs)
        assert abs(float(hz) - 440.0) < 1.0


class TestLibrosaDecomposeSegmentPipeline:
    """Decompose (NMF, nn_filter) and segment (recurrence_matrix) pipeline."""

    def test_nmf(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = np.abs(librosa.stft(y))
        W, H = librosa.decompose.nmf(S, n_components=4)
        assert W.shape[0] == S.shape[0]
        assert W.shape[1] == 4
        assert H.shape[0] == 4
        assert H.shape[1] == S.shape[1]

    def test_nn_filter(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = np.abs(librosa.stft(y))
        S_filtered = librosa.decompose.nn_filter(S)
        assert S_filtered.shape == S.shape

    def test_recurrence_matrix(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        R = librosa.segment.recurrence_matrix(mfcc)
        n_frames = mfcc.shape[1]
        assert R.shape == (n_frames, n_frames)

    def test_cross_similarity(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        # Cross-similarity of feature matrix with itself
        cs = librosa.segment.cross_similarity(mfcc, mfcc)
        n_frames = mfcc.shape[1]
        assert cs.shape == (n_frames, n_frames)


class TestLibrosaSequencePipeline:
    """Sequence decoding: Viterbi and transition matrices."""

    def test_viterbi(self):
        from metalmom.compat import librosa

        np.random.seed(42)
        n_frames, n_states = 10, 3
        prob = np.random.rand(n_frames, n_states).astype(np.float32)
        prob /= prob.sum(axis=1, keepdims=True)
        trans = np.eye(n_states, dtype=np.float32) * 0.9 + 0.1 / n_states
        path = librosa.sequence.viterbi(prob, trans)
        assert len(path) == n_frames
        assert all(0 <= s < n_states for s in path)

    def test_viterbi_discriminative(self):
        from metalmom.compat import librosa

        np.random.seed(42)
        n_frames, n_states = 10, 3
        prob = np.random.rand(n_frames, n_states).astype(np.float32)
        prob /= prob.sum(axis=1, keepdims=True)
        trans = np.eye(n_states, dtype=np.float32) * 0.9 + 0.1 / n_states
        path = librosa.sequence.viterbi_discriminative(prob, trans)
        assert len(path) == n_frames

    def test_viterbi_binary(self):
        from metalmom.compat import librosa

        np.random.seed(42)
        n_frames = 10
        prob = np.random.rand(n_frames).astype(np.float32)
        trans = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        path = librosa.sequence.viterbi_binary(prob, trans)
        assert len(path) == n_frames
        assert set(path).issubset({0, 1})

    def test_transition_uniform(self):
        from metalmom.compat import librosa

        T = librosa.sequence.transition_uniform(4)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_transition_loop(self):
        from metalmom.compat import librosa

        T = librosa.sequence.transition_loop(3, 0.9)
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.diag(T), 0.9, atol=1e-6)

    def test_transition_cycle(self):
        from metalmom.compat import librosa

        T = librosa.sequence.transition_cycle(3, 0.8)
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_dtw(self):
        from metalmom.compat import librosa

        np.random.seed(42)
        X = np.random.randn(5, 10).astype(np.float32)
        Y = np.random.randn(5, 12).astype(np.float32)
        D, wp = librosa.sequence.dtw(X=X, Y=Y)
        assert D.ndim == 2
        assert wp.ndim == 2
        assert wp.shape[1] == 2


class TestLibrosaMiscPipeline:
    """Miscellaneous top-level functions: zero_crossings, autocorrelate, etc."""

    def test_zero_crossings(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        zc = librosa.zero_crossings(y)
        assert zc.ndim == 1
        assert zc.dtype == bool
        assert len(zc) == len(y)

    def test_autocorrelate(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        acf = librosa.autocorrelate(y)
        assert acf.ndim == 1
        # Autocorrelation at lag 0 should be maximum
        assert acf[0] >= acf[1]

    def test_to_mono(self):
        from metalmom.compat import librosa

        stereo = np.random.randn(2, 1000).astype(np.float32)
        mono = librosa.to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 1000

    def test_lpc(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        coeffs = librosa.lpc(y[:512], order=10)
        assert len(coeffs) == 11  # order + 1
        assert coeffs[0] == pytest.approx(1.0, abs=1e-5)

    def test_times_like(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        times = librosa.times_like(S)
        assert len(times) == S.shape[1]

    def test_samples_like(self, tone_22k):
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        samples = librosa.samples_like(S)
        assert len(samples) == S.shape[1]


class TestLibrosaFullPipeline:
    """Full analysis pipeline combining multiple steps."""

    def test_full_stft_to_mfcc_to_delta(self, tone_22k):
        """STFT -> mel spectrogram -> MFCC -> delta -> delta-delta."""
        from metalmom.compat import librosa

        y, sr = tone_22k
        S = librosa.stft(y)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        assert S.ndim == 2
        assert mel.shape[0] == 128
        assert mel_db.shape == mel.shape
        assert mfcc.shape[0] == 13
        assert delta_mfcc.shape == mfcc.shape
        assert delta2_mfcc.shape == mfcc.shape

    def test_onset_to_beat_pipeline(self, click_train_22k):
        """onset_strength -> onset_detect and beat_track."""
        from metalmom.compat import librosa

        y, sr = click_train_22k
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        assert onset_env.ndim == 1
        assert onsets.ndim == 1
        assert beats.ndim == 1
        assert tempo > 0

    def test_hpss_then_features(self, tone_22k):
        """HPSS -> extract features from harmonic component."""
        from metalmom.compat import librosa

        y, sr = tone_22k
        y_h, y_p = librosa.effects.hpss(y)
        chroma_h = librosa.feature.chroma_stft(y=y_h, sr=sr)
        tonnetz_h = librosa.feature.tonnetz(y=y_h, sr=sr)

        assert chroma_h.shape[0] == 12
        assert tonnetz_h.shape[0] == 6
        # Both should have the same number of frames
        assert chroma_h.shape[1] == tonnetz_h.shape[1]


# ===========================================================================
# madmom compat tests
# ===========================================================================

class TestMadmomAudioPipeline:
    """madmom audio processing: Signal -> FramedSignal -> STFT -> Spectrogram."""

    def test_signal_from_array(self, tone_44k):
        from metalmom.compat.madmom.audio.signal import Signal

        y, sr = tone_44k
        sig = Signal(y, sample_rate=sr)
        assert sig.sample_rate == sr
        assert len(sig) == len(y)
        assert sig.num_samples == len(y)
        assert sig.length == pytest.approx(1.0, abs=0.01)

    def test_framed_signal(self, tone_44k):
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal

        y, sr = tone_44k
        sig = Signal(y, sample_rate=sr)
        frames = FramedSignal(sig, frame_size=2048, hop_size=441)
        assert len(frames) > 0
        assert frames.shape[1] == 2048
        # Check individual frame access
        frame_0 = frames[0]
        assert len(frame_0) == 2048

    def test_stft(self, tone_44k):
        from metalmom.compat.madmom.audio.signal import Signal
        from metalmom.compat.madmom.audio.stft import STFT

        y, sr = tone_44k
        sig = Signal(y, sample_rate=sr)
        stft = STFT(sig)
        assert stft.ndim == 2
        assert np.iscomplexobj(stft)
        assert stft.shape[1] == 1025  # n_fft=2048 -> 1025 bins

    def test_stft_from_raw_array(self, tone_44k):
        from metalmom.compat.madmom.audio.stft import STFT

        y, sr = tone_44k
        stft = STFT(y, sample_rate=sr)
        assert stft.ndim == 2
        assert np.iscomplexobj(stft)

    def test_spectrogram(self, tone_44k):
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import Spectrogram

        y, sr = tone_44k
        stft = STFT(y, sample_rate=sr)
        spec = Spectrogram(stft)
        assert spec.ndim == 2
        assert not np.iscomplexobj(spec)
        assert spec.shape == stft.shape
        # All values non-negative (magnitude)
        assert np.all(spec >= 0)

    def test_filtered_spectrogram(self, tone_44k):
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import (
            Spectrogram, FilteredSpectrogram,
        )

        y, sr = tone_44k
        stft = STFT(y, sample_rate=sr)
        spec = Spectrogram(stft)
        filtered = FilteredSpectrogram(spec)
        assert filtered.ndim == 2
        # Default 80 mel bands
        assert filtered.shape[1] == 80

    def test_logarithmic_filtered_spectrogram(self, tone_44k):
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import (
            Spectrogram, LogarithmicFilteredSpectrogram,
        )

        y, sr = tone_44k
        stft = STFT(y, sample_rate=sr)
        spec = Spectrogram(stft)
        log_spec = LogarithmicFilteredSpectrogram(spec)
        assert log_spec.ndim == 2
        assert log_spec.shape[1] == 80

    def test_full_audio_chain(self, tone_44k):
        """Signal -> FramedSignal -> STFT -> Spectrogram -> Filtered -> Log."""
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import (
            Spectrogram, FilteredSpectrogram, LogarithmicFilteredSpectrogram,
        )

        y, sr = tone_44k
        sig = Signal(y, sample_rate=sr)
        assert sig.sample_rate == sr

        frames = FramedSignal(sig, frame_size=2048, hop_size=441)
        assert len(frames) > 0

        stft = STFT(sig)
        assert stft.ndim == 2 and np.iscomplexobj(stft)

        spec = Spectrogram(stft)
        assert not np.iscomplexobj(spec)

        filtered = FilteredSpectrogram(spec)
        assert filtered.shape[1] == 80

        log_spec = LogarithmicFilteredSpectrogram(spec)
        assert log_spec.shape == filtered.shape

    def test_random_noise_signal(self):
        """Process random noise through the full chain."""
        from metalmom.compat.madmom.audio.signal import Signal, FramedSignal
        from metalmom.compat.madmom.audio.stft import STFT
        from metalmom.compat.madmom.audio.spectrogram import (
            Spectrogram, FilteredSpectrogram, LogarithmicFilteredSpectrogram,
        )

        y = np.random.randn(44100).astype(np.float32)
        sig = Signal(y, sample_rate=44100)
        assert sig.sample_rate == 44100

        frames = FramedSignal(sig, frame_size=2048, hop_size=441)
        assert len(frames) > 0

        stft = STFT(sig)
        assert stft.ndim == 2 and np.iscomplexobj(stft)

        spec = Spectrogram(stft)
        filtered = FilteredSpectrogram(spec)
        log_spec = LogarithmicFilteredSpectrogram(spec)
        assert log_spec.shape == filtered.shape


class TestMadmomProcessorPipeline:
    """madmom processor chain: RNNBeatProcessor -> DBNBeatTrackingProcessor."""

    def test_rnn_beat_processor_callable(self):
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor

        rnn = RNNBeatProcessor()
        assert callable(rnn)

    def test_dbn_beat_tracking_processor_callable(self):
        from metalmom.compat.madmom.features.beats import DBNBeatTrackingProcessor

        dbn = DBNBeatTrackingProcessor()
        assert callable(dbn)

    def test_rnn_beat_processor(self, tone_44k):
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor

        y, sr = tone_44k
        rnn = RNNBeatProcessor()
        activations = rnn(y)
        assert activations.ndim == 1
        assert len(activations) > 0
        # Activations should be normalized to [0, 1]
        assert np.all(activations >= 0)
        assert np.all(activations <= 1.01)

    def test_dbn_beat_tracking_processor(self, tone_44k):
        from metalmom.compat.madmom.features.beats import (
            RNNBeatProcessor, DBNBeatTrackingProcessor,
        )

        y, sr = tone_44k
        rnn = RNNBeatProcessor()
        dbn = DBNBeatTrackingProcessor()

        activations = rnn(y)
        beats = dbn(activations)
        assert beats.ndim == 1
        # Beat times should be non-negative
        if len(beats) > 0:
            assert np.all(beats >= 0)

    def test_onset_peak_picking_processor(self, tone_44k):
        from metalmom.compat.madmom.features.onsets import (
            RNNOnsetProcessor, OnsetPeakPickingProcessor,
        )

        y, sr = tone_44k
        rnn = RNNOnsetProcessor()
        picker = OnsetPeakPickingProcessor()

        assert callable(rnn)
        assert callable(picker)

        activations = rnn(y)
        assert activations.ndim == 1

        onsets = picker(activations)
        assert onsets.ndim == 1
        if len(onsets) > 0:
            assert np.all(onsets >= 0)

    def test_rnn_downbeat_processor(self, tone_44k):
        from metalmom.compat.madmom.features.downbeats import RNNDownBeatProcessor

        y, sr = tone_44k
        proc = RNNDownBeatProcessor()
        assert callable(proc)

        activations = proc(y)
        assert activations.ndim == 2
        assert activations.shape[1] == 3  # 3-class: no-beat, beat, downbeat

    def test_tempo_estimation_processor(self, tone_44k):
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor
        from metalmom.compat.madmom.features.tempo import TempoEstimationProcessor

        y, sr = tone_44k
        rnn = RNNBeatProcessor()
        tempo_proc = TempoEstimationProcessor()

        activations = rnn(y)
        result = tempo_proc(activations)
        assert result.ndim == 2
        assert result.shape[1] == 2  # (tempo, strength)
        assert result[0, 0] > 0  # tempo > 0
        assert result[0, 1] > 0  # strength > 0

    def test_full_processor_chain(self):
        """RNNBeatProcessor -> DBNBeatTrackingProcessor on click train."""
        from metalmom.compat.madmom.audio.signal import Signal
        from metalmom.compat.madmom.features.beats import (
            RNNBeatProcessor, DBNBeatTrackingProcessor,
        )

        # Create a click train at 44100 Hz
        sr = 44100
        duration = 3.0
        n_samples = int(sr * duration)
        y = np.zeros(n_samples, dtype=np.float32)
        interval = int(sr * 0.5)  # 120 BPM
        for i in range(0, n_samples, interval):
            end = min(i + 100, n_samples)
            y[i:end] = 0.9

        sig = Signal(y, sample_rate=sr)

        rnn = RNNBeatProcessor()
        dbn = DBNBeatTrackingProcessor()

        activations = rnn(sig)
        assert activations.ndim == 1

        beats = dbn(activations)
        assert beats.ndim == 1


class TestMadmomEvaluationPipeline:
    """madmom evaluation: onset, beat, and tempo metrics."""

    def test_onset_evaluation_perfect(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0])
        det = np.array([0.5, 1.0, 1.5, 2.0])
        e = OnsetEvaluation(det, ref, window=0.05)
        assert e.fmeasure == pytest.approx(1.0, abs=0.01)
        assert e.precision == pytest.approx(1.0, abs=0.01)
        assert e.recall == pytest.approx(1.0, abs=0.01)

    def test_onset_evaluation_close_matches(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0])
        det = np.array([0.5, 1.01, 1.49, 2.0])
        e = OnsetEvaluation(det, ref, window=0.05)
        assert e.fmeasure > 0.9

    def test_onset_evaluation_missed_detections(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0])
        det = np.array([0.5, 2.0])  # missed 1.0, 1.5
        e = OnsetEvaluation(det, ref, window=0.05)
        assert e.precision > e.recall  # high precision, low recall

    def test_onset_evaluation_false_alarms(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        ref = np.array([0.5, 1.0])
        det = np.array([0.5, 0.7, 1.0, 1.3])  # 2 false alarms
        e = OnsetEvaluation(det, ref, window=0.05)
        assert e.recall > e.precision  # high recall, low precision

    def test_onset_evaluation_properties(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0])
        det = np.array([0.5, 1.01, 1.49, 2.0])
        e = OnsetEvaluation(det, ref, window=0.05)
        # All properties should be accessible
        assert 0 <= e.fmeasure <= 1
        assert 0 <= e.precision <= 1
        assert 0 <= e.recall <= 1
        assert isinstance(e.num_tp, int)
        assert isinstance(e.num_fp, int)
        assert isinstance(e.num_fn, int)

    def test_beat_evaluation(self):
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0])
        det = np.array([0.5, 1.01, 1.49, 2.0])
        e = BeatEvaluation(det, ref)
        assert e.fmeasure > 0.5
        # Access all properties
        assert 0 <= e.cemgil <= 1.01
        assert isinstance(e.p_score, (int, float, np.floating))
        assert isinstance(e.cmlc, (int, float, np.floating))
        assert isinstance(e.cmlt, (int, float, np.floating))
        assert isinstance(e.amlc, (int, float, np.floating))
        assert isinstance(e.amlt, (int, float, np.floating))

    def test_beat_evaluation_perfect(self):
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation

        ref = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        det = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        e = BeatEvaluation(det, ref)
        assert e.fmeasure == pytest.approx(1.0, abs=0.01)

    def test_tempo_evaluation_exact(self):
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        e = TempoEvaluation(120.0, 120.0)
        assert e.acc1 == 1.0

    def test_tempo_evaluation_wrong(self):
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        e = TempoEvaluation(100.0, 120.0)
        assert e.acc1 == 0.0

    def test_tempo_evaluation_double_tempo(self):
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        e = TempoEvaluation(240.0, 120.0)
        # acc1 should be 0 (strict), acc2 should be 1 (allows double)
        assert e.acc1 == 0.0
        assert e.acc2 > 0.5

    def test_tempo_evaluation_repr(self):
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        e = TempoEvaluation(120.0, 120.0)
        r = repr(e)
        assert 'TempoEvaluation' in r
        assert 'acc1' in r
        assert 'acc2' in r
