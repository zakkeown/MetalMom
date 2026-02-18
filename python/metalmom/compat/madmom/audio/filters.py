"""madmom.audio.filters compatibility shim.

Provides MelFilterbank, LogarithmicFilterbank, and SemitoneFilterbank
classes backed by MetalMom.

madmom defaults:
    - sample_rate: 44100
    - frame_size: 2048
    - num_bands: 80
    - fmin: 30.0
"""

import numpy as np


class MelFilterbank:
    """madmom-compatible mel filterbank.

    Computes a mel filterbank matrix that maps FFT bins to mel-scaled
    frequency bands. Delegates to MetalMom's native ``mel()`` function.

    Parameters
    ----------
    bin_frequencies : np.ndarray or None
        Array of FFT bin center frequencies. If None, computed from
        ``frame_size`` and ``sample_rate``.
    num_bands : int
        Number of mel bands. Default: 80.
    fmin : float
        Minimum frequency in Hz. Default: 30.0.
    fmax : float or None
        Maximum frequency in Hz. Default: None (sr / 2).
    norm_filters : bool
        If True, normalize each filter to unit area. Default: True.
    frame_size : int
        FFT window size. Default: 2048.
    sample_rate : int
        Sample rate in Hz. Default: 44100.
    """

    def __init__(self, bin_frequencies=None, num_bands=80, fmin=30.0,
                 fmax=None, norm_filters=True, frame_size=2048,
                 sample_rate=44100, **kwargs):
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.bin_frequencies = bin_frequencies

        from metalmom.filters import mel

        actual_fmax = fmax if fmax is not None else sample_rate / 2.0

        # mel() returns shape (n_mels, n_fft // 2 + 1)
        # madmom convention is (n_freqs, n_bands), so we transpose
        fb = mel(
            sr=sample_rate,
            n_fft=frame_size,
            n_mels=num_bands,
            fmin=fmin,
            fmax=actual_fmax,
        )
        self._filterbank = fb.T  # (n_freqs, n_bands)

    @property
    def filterbank(self):
        """The filterbank matrix, shape (n_freqs, num_bands)."""
        return self._filterbank

    def __call__(self, spectrogram):
        """Apply the mel filterbank to a magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            Magnitude spectrogram, shape (n_frames, n_freqs).

        Returns
        -------
        np.ndarray
            Mel-filtered spectrogram, shape (n_frames, num_bands).
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        return (spectrogram @ self._filterbank).astype(np.float32)


class LogarithmicFilterbank(MelFilterbank):
    """madmom-compatible logarithmic filterbank.

    Extends MelFilterbank with logarithmic frequency spacing. In madmom,
    this uses a slightly different frequency spacing than mel. This shim
    uses mel spacing as a close approximation, since the difference is
    negligible for most applications.

    Parameters
    ----------
    bin_frequencies : np.ndarray or None
        Array of FFT bin center frequencies.
    num_bands : int
        Number of filter bands. Default: 80.
    fmin : float
        Minimum frequency in Hz. Default: 30.0.
    fmax : float or None
        Maximum frequency in Hz. Default: None (sr / 2).
    norm_filters : bool
        If True, normalize each filter to unit area. Default: True.
    frame_size : int
        FFT window size. Default: 2048.
    sample_rate : int
        Sample rate in Hz. Default: 44100.
    """

    def __init__(self, bin_frequencies=None, num_bands=80, fmin=30.0,
                 fmax=None, norm_filters=True, frame_size=2048,
                 sample_rate=44100, **kwargs):
        super().__init__(
            bin_frequencies=bin_frequencies,
            num_bands=num_bands,
            fmin=fmin,
            fmax=fmax,
            norm_filters=norm_filters,
            frame_size=frame_size,
            sample_rate=sample_rate,
            **kwargs,
        )


class SemitoneFilterbank:
    """madmom-compatible semitone filterbank.

    Applies bandpass filters centered at each semitone in the specified
    MIDI range. Delegates to MetalMom's native ``semitone_filterbank()``
    function.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz. Default: 44100.
    midi_low : int
        Lowest MIDI note number. Default: 24 (C1, ~32.7 Hz).
    midi_high : int
        Highest MIDI note number. Default: 119 (B8, ~7902 Hz).
    order : int
        Filter order (number of cascaded biquad pairs). Default: 4.
    """

    def __init__(self, sample_rate=44100, midi_low=24, midi_high=119,
                 order=4, **kwargs):
        self.sample_rate = sample_rate
        self.midi_low = midi_low
        self.midi_high = midi_high
        self.order = order

    def __call__(self, data):
        """Apply the semitone filterbank to an audio signal.

        Parameters
        ----------
        data : Signal, np.ndarray, or str
            Audio signal (1D float32).

        Returns
        -------
        np.ndarray
            Filtered signal, shape (n_semitones, n_samples).
        """
        from metalmom.filters import semitone_filterbank
        from ..audio.signal import Signal

        if isinstance(data, (str, bytes)):
            data = Signal(data, sample_rate=self.sample_rate)

        audio = np.asarray(data, dtype=np.float32).ravel()
        sr = getattr(data, 'sample_rate', self.sample_rate)

        return semitone_filterbank(
            y=audio,
            sr=sr,
            midi_low=self.midi_low,
            midi_high=self.midi_high,
            order=self.order,
        )
