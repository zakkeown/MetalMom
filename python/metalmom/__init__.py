"""MetalMom: GPU-accelerated audio/music analysis on Apple Metal."""

__version__ = "0.1.0"

from .core import load, resample, stft, istft, db_to_amplitude, db_to_power, tone, chirp, clicks, get_duration, get_samplerate, stream, reassigned_spectrogram
from .feature import (
    amplitude_to_db, power_to_db, melspectrogram, mfcc, chroma_stft,
    chroma_cqt, chroma_cens, chroma_vqt,
    spectral_centroid, spectral_bandwidth, spectral_contrast,
    spectral_rolloff, spectral_flatness,
    rms, zero_crossing_rate, tonnetz,
    delta, stack_memory, poly_features,
    tempo, tempogram, fourier_tempogram,
    mel_to_audio, mfcc_to_mel, mfcc_to_audio,
)
from .evaluate import onset_evaluate, beat_evaluate, tempo_evaluate, chord_accuracy
from .onset import onset_strength, onset_detect
from .beat import beat_track, plp
from .pitch import yin, pyin, piptrack, estimate_tuning
from .effects import hpss, harmonic, percussive, time_stretch, pitch_shift, trim, split, preemphasis, deemphasis, phase_vocoder, griffinlim
from .key import key_detect
from .chord import chord_detect
from .transcribe import piano_transcribe
from .cqt import cqt, vqt, hybrid_cqt
from .decompose import nmf, nn_filter
from .segment import recurrence_matrix, cross_similarity, rqa, dtw, agglomerative
from .sequence import viterbi, viterbi_discriminative, viterbi_binary
from .convert import (
    hz_to_midi, midi_to_hz, hz_to_note, note_to_hz, midi_to_note, note_to_midi,
    times_to_frames, frames_to_time, times_to_samples, samples_to_time,
    frames_to_samples, samples_to_frames,
    fft_frequencies, mel_frequencies,
)
from .filters import semitone_filterbank, semitone_frequencies
