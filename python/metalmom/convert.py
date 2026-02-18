"""Unit conversions: Hz/MIDI/note/oct/mel, time/frame/sample, frequency bins."""

import re
import numpy as np
from ._native import ffi, lib
from ._buffer import buffer_to_numpy

# Note names indexed by pitch class (0 = C, 11 = B)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Regex for parsing note names: letter + optional accidental + octave (may be negative)
_NOTE_RE = re.compile(r"^([A-Ga-g])(#|b)?(-?\d+)$")


# ---------------------------------------------------------------------------
# Hz <-> MIDI
# ---------------------------------------------------------------------------

def hz_to_midi(hz):
    """Convert frequency in Hz to MIDI note number.

    Parameters
    ----------
    hz : float or np.ndarray
        Frequency/frequencies in Hz.

    Returns
    -------
    float or np.ndarray
        MIDI note number(s). Non-positive Hz values produce NaN.
    """
    hz = np.asarray(hz, dtype=np.float32)
    data_ptr = ffi.cast("const float*", hz.ctypes.data)
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_hz_to_midi(ctx, data_ptr, int(hz.size), out)
        if rc != 0:
            raise RuntimeError(f"mm_hz_to_midi failed with code {rc}")
        result = buffer_to_numpy(out)
        return float(result) if hz.ndim == 0 else result
    finally:
        lib.mm_destroy(ctx)


def midi_to_hz(midi):
    """Convert MIDI note number to frequency in Hz.

    Parameters
    ----------
    midi : float or np.ndarray
        MIDI note number(s).

    Returns
    -------
    float or np.ndarray
        Frequency/frequencies in Hz.
    """
    midi = np.asarray(midi, dtype=np.float32)
    data_ptr = ffi.cast("const float*", midi.ctypes.data)
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_midi_to_hz(ctx, data_ptr, int(midi.size), out)
        if rc != 0:
            raise RuntimeError(f"mm_midi_to_hz failed with code {rc}")
        result = buffer_to_numpy(out)
        return float(result) if midi.ndim == 0 else result
    finally:
        lib.mm_destroy(ctx)


# ---------------------------------------------------------------------------
# Hz <-> Note name
# ---------------------------------------------------------------------------

def hz_to_note(hz):
    """Convert frequency in Hz to note name string.

    Parameters
    ----------
    hz : float or array-like
        Frequency/frequencies in Hz.

    Returns
    -------
    str or list of str
        Note name(s), e.g. "A4", "C#3".
    """
    hz = np.atleast_1d(np.asarray(hz, dtype=np.float32))
    midi = np.array([12.0 * np.log2(h / 440.0) + 69.0 if h > 0 else np.nan for h in hz])
    notes = [_midi_to_note_scalar(m) for m in midi]
    return notes[0] if len(notes) == 1 else notes


def note_to_hz(note):
    """Convert note name string to frequency in Hz.

    Parameters
    ----------
    note : str or list of str
        Note name(s), e.g. "A4", "C#3".

    Returns
    -------
    float or np.ndarray
        Frequency/frequencies in Hz.
    """
    if isinstance(note, str):
        m = note_to_midi(note)
        return 440.0 * 2.0 ** ((m - 69.0) / 12.0)
    return np.array([note_to_hz(n) for n in note], dtype=np.float32)


# ---------------------------------------------------------------------------
# MIDI <-> Note name
# ---------------------------------------------------------------------------

def midi_to_note(midi):
    """Convert MIDI note number to note name string.

    Parameters
    ----------
    midi : float or array-like
        MIDI note number(s).

    Returns
    -------
    str or list of str
        Note name(s), e.g. "A4".
    """
    midi = np.atleast_1d(np.asarray(midi, dtype=np.float32))
    notes = [_midi_to_note_scalar(m) for m in midi]
    return notes[0] if len(notes) == 1 else notes


def note_to_midi(note):
    """Convert note name string to MIDI note number.

    Parameters
    ----------
    note : str or list of str
        Note name(s), e.g. "A4", "C#3", "Db2".

    Returns
    -------
    float or np.ndarray
        MIDI note number(s).
    """
    if isinstance(note, str):
        return _note_to_midi_scalar(note)
    return np.array([_note_to_midi_scalar(n) for n in note], dtype=np.float32)


def _midi_to_note_scalar(midi):
    """Convert a single MIDI value to note name."""
    if np.isnan(midi):
        return "nan"
    rounded = int(round(midi))
    pitch_class = rounded % 12
    if pitch_class < 0:
        pitch_class += 12
    octave = (rounded // 12) - 1
    return f"{_NOTE_NAMES[pitch_class]}{octave}"


def _note_to_midi_scalar(note):
    """Convert a single note name to MIDI number."""
    m = _NOTE_RE.match(note.strip())
    if not m:
        return float("nan")

    letter, accidental, octave_str = m.groups()

    base_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = base_map.get(letter.upper())
    if base is None:
        return float("nan")

    acc = 0
    if accidental == "#":
        acc = 1
    elif accidental == "b":
        acc = -1

    octave = int(octave_str)
    return float((octave + 1) * 12 + base + acc)


# ---------------------------------------------------------------------------
# Time <-> Frame
# ---------------------------------------------------------------------------

def times_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    """Convert time values in seconds to frame indices.

    Parameters
    ----------
    times : float or np.ndarray
        Time value(s) in seconds.
    sr : int
        Sample rate. Default 22050.
    hop_length : int
        Hop length. Default 512.
    n_fft : int, optional
        FFT size (unused, for compat).

    Returns
    -------
    np.ndarray
        Frame indices (int).
    """
    times = np.atleast_1d(np.asarray(times, dtype=np.float32)).ravel()
    data_ptr = ffi.cast("const float*", times.ctypes.data)
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_times_to_frames(ctx, data_ptr, int(times.size), int(sr), int(hop_length), out)
        if rc != 0:
            raise RuntimeError(f"mm_times_to_frames failed with code {rc}")
        result = buffer_to_numpy(out)
        return result.astype(np.intp)
    finally:
        lib.mm_destroy(ctx)


def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    """Convert frame indices to time values in seconds.

    Parameters
    ----------
    frames : int or np.ndarray
        Frame index/indices.
    sr : int
        Sample rate. Default 22050.
    hop_length : int
        Hop length. Default 512.
    n_fft : int, optional
        FFT size (unused, for compat).

    Returns
    -------
    np.ndarray
        Time value(s) in seconds.
    """
    frames = np.atleast_1d(np.asarray(frames, dtype=np.float32)).ravel()
    data_ptr = ffi.cast("const float*", frames.ctypes.data)
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_frames_to_time(ctx, data_ptr, int(frames.size), int(sr), int(hop_length), out)
        if rc != 0:
            raise RuntimeError(f"mm_frames_to_time failed with code {rc}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


# ---------------------------------------------------------------------------
# Time <-> Sample
# ---------------------------------------------------------------------------

def times_to_samples(times, sr=22050):
    """Convert time values in seconds to sample indices.

    Parameters
    ----------
    times : float or np.ndarray
        Time value(s) in seconds.
    sr : int
        Sample rate. Default 22050.

    Returns
    -------
    np.ndarray
        Sample indices (int).
    """
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    return np.floor(times * sr).astype(np.intp)


def samples_to_time(samples, sr=22050):
    """Convert sample indices to time values in seconds.

    Parameters
    ----------
    samples : int or np.ndarray
        Sample index/indices.
    sr : int
        Sample rate. Default 22050.

    Returns
    -------
    np.ndarray
        Time value(s) in seconds.
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=np.float64))
    return (samples / sr).astype(np.float32)


# ---------------------------------------------------------------------------
# Frame <-> Sample
# ---------------------------------------------------------------------------

def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Convert frame indices to sample indices.

    Parameters
    ----------
    frames : int or np.ndarray
        Frame index/indices.
    hop_length : int
        Hop length. Default 512.
    n_fft : int, optional
        FFT size (unused, for compat).

    Returns
    -------
    np.ndarray
        Sample indices (int).
    """
    frames = np.atleast_1d(np.asarray(frames, dtype=np.intp))
    return frames * hop_length


def samples_to_frames(samples, hop_length=512, n_fft=None):
    """Convert sample indices to frame indices.

    Parameters
    ----------
    samples : int or np.ndarray
        Sample index/indices.
    hop_length : int
        Hop length. Default 512.
    n_fft : int, optional
        FFT size (unused, for compat).

    Returns
    -------
    np.ndarray
        Frame indices (int).
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=np.intp))
    return samples // hop_length


# ---------------------------------------------------------------------------
# Frequency bin generation
# ---------------------------------------------------------------------------

def fft_frequencies(sr=22050, n_fft=2048):
    """Array of FFT bin center frequencies.

    Parameters
    ----------
    sr : int
        Sample rate. Default 22050.
    n_fft : int
        FFT window size. Default 2048.

    Returns
    -------
    np.ndarray
        Center frequencies for each FFT bin, shape (n_fft/2 + 1,).
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_fft_frequencies(ctx, int(sr), int(n_fft), out)
        if rc != 0:
            raise RuntimeError(f"mm_fft_frequencies failed with code {rc}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """Array of mel-spaced frequencies.

    Parameters
    ----------
    n_mels : int
        Number of mel frequencies. Default 128.
    fmin : float
        Minimum frequency in Hz. Default 0.
    fmax : float
        Maximum frequency in Hz. Default 11025.

    Returns
    -------
    np.ndarray
        Mel-spaced frequencies in Hz, shape (n_mels,).
    """
    ctx = lib.mm_init()
    if ctx == ffi.NULL:
        raise RuntimeError("Failed to initialize MetalMom context")
    try:
        out = ffi.new("MMBuffer*")
        rc = lib.mm_mel_frequencies(ctx, int(n_mels), float(fmin), float(fmax), out)
        if rc != 0:
            raise RuntimeError(f"mm_mel_frequencies failed with code {rc}")
        return buffer_to_numpy(out)
    finally:
        lib.mm_destroy(ctx)
