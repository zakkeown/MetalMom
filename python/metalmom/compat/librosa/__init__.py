"""librosa compatibility shim.

Drop-in replacement for common librosa functions backed by MetalMom.

Usage::

    from metalmom.compat import librosa
    S = librosa.stft(y)
    y_hat = librosa.istft(S)
    mel = librosa.feature.melspectrogram(y=y, sr=22050)
"""

# -- Core audio I/O and transforms --
from .core import (
    load, resample, stft, istft, get_duration, get_samplerate,
    reassigned_spectrogram, phase_vocoder, griffinlim, griffinlim_cqt,
    # Signal generation
    tone, chirp, clicks, stream,
    # dB conversions (top-level in librosa)
    amplitude_to_db, power_to_db, db_to_amplitude, db_to_power,
    # PCEN (top-level in librosa)
    pcen,
    # Pitch estimation (top-level in librosa)
    yin, pyin, piptrack, estimate_tuning,
    # Unit conversions (top-level in librosa)
    hz_to_midi, midi_to_hz, hz_to_note, note_to_hz,
    midi_to_note, note_to_midi,
    fft_frequencies, mel_frequencies,
    frames_to_time, samples_to_time,
    frames_to_samples, samples_to_frames,
    # Pure-Python helpers
    magphase, to_mono, zero_crossings, autocorrelate, lpc,
    cqt_frequencies, tempo_frequencies, fourier_tempo_frequencies,
    times_like, samples_like,
    time_to_frames, time_to_samples,
    hz_to_mel, mel_to_hz, hz_to_octs, octs_to_hz,
    salience, interp_harmonics, f0_harmonics, pitch_tuning,
)

# -- CQT / VQT --
from metalmom.cqt import cqt, vqt, hybrid_cqt

# -- Submodules --
from . import feature
from . import filters
from . import onset
from . import beat
from . import pitch
from . import effects
from . import decompose
from . import sequence
from . import segment
from . import display
from . import convert
