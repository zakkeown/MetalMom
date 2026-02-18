"""Display functions for spectrograms and waveforms.

Provides ``specshow`` for 2D spectrogram-like arrays and ``waveshow`` for
1D audio waveforms.  Matplotlib is imported lazily so the module can be
imported even when matplotlib is not installed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_matplotlib():
    """Lazy-import matplotlib and return ``plt``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for display functions. "
            "Install it with:  pip install matplotlib"
        ) from exc
    return plt


def _hz_to_note_label(hz):
    """Convert a frequency in Hz to a note-name string (e.g. 'A4').

    Uses a pure-Python MIDI-to-note conversion so that this module does
    not depend on the native C bridge.
    """
    _NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B"]
    if hz <= 0:
        return ""
    midi = 12.0 * np.log2(hz / 440.0) + 69.0
    rounded = int(round(midi))
    pitch_class = rounded % 12
    if pitch_class < 0:
        pitch_class += 12
    octave = (rounded // 12) - 1
    return f"{_NOTE_NAMES[pitch_class]}{octave}"


def _time_ticks(n_frames, sr, hop_length):
    """Return an array of time values (seconds) for *n_frames* frames."""
    return np.arange(n_frames + 1, dtype=np.float64) * hop_length / sr


def _frame_ticks(n_frames):
    """Return an array of frame indices."""
    return np.arange(n_frames + 1, dtype=np.float64)


def _fft_freqs(n_freqs, sr, n_fft):
    """Return FFT bin centre frequencies, shape ``(n_freqs,)``."""
    return np.linspace(0, sr / 2.0, n_freqs, endpoint=True, dtype=np.float64)


def _mel_freqs(n_mels, fmin, fmax):
    """Return mel-spaced frequencies using the HTK formula (pure Python)."""
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels, dtype=np.float64)
    return _mel_to_hz(mels)


def _log_freqs(n_freqs, fmin, fmax):
    """Return log-spaced frequencies between *fmin* and *fmax*."""
    if fmin <= 0:
        fmin = 1.0
    return np.geomspace(fmin, fmax, n_freqs, dtype=np.float64)


def _cqt_freqs(n_bins, fmin=None, bins_per_octave=12):
    """Return CQT centre frequencies."""
    if fmin is None:
        fmin = 32.703  # C1
    return fmin * 2.0 ** (np.arange(n_bins, dtype=np.float64) / bins_per_octave)


def _chroma_labels(n_chroma=12):
    """Return chroma pitch-class labels starting from C."""
    full = ["C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"]
    if n_chroma <= 12:
        step = max(1, 12 // n_chroma)
        return [full[i * step % 12] for i in range(n_chroma)]
    return [str(i) for i in range(n_chroma)]


# ---------------------------------------------------------------------------
# specshow
# ---------------------------------------------------------------------------

def specshow(data, *, x_coords=None, y_coords=None,
             x_axis=None, y_axis=None,
             sr=22050, hop_length=512, n_fft=None,
             fmin=None, fmax=None,
             ax=None, **kwargs):
    """Display a spectrogram-like 2D array as a colour-mesh image.

    Parameters
    ----------
    data : np.ndarray, shape ``(n_rows, n_cols)``
        The 2D data to display (e.g. a spectrogram, mel spectrogram,
        chroma, or any feature matrix).
    x_coords : np.ndarray, optional
        Explicit x-axis coordinates.  If provided, *x_axis* is ignored
        for coordinate computation (but still used for labelling).
    y_coords : np.ndarray, optional
        Explicit y-axis coordinates.  If provided, *y_axis* is ignored
        for coordinate computation (but still used for labelling).
    x_axis : str or None
        Type of the x-axis.  One of ``'time'``, ``'frames'``, ``'hz'``,
        ``'mel'``, ``'log'``, ``'fft'``, ``'fft_note'``, or ``None``.
    y_axis : str or None
        Type of the y-axis.  One of ``'linear'``, ``'hz'``, ``'mel'``,
        ``'log'``, ``'fft'``, ``'fft_note'``, ``'cqt_hz'``,
        ``'cqt_note'``, ``'chroma'``, or ``None``.
    sr : int
        Sample rate.  Default: 22050.
    hop_length : int
        Hop length.  Default: 512.
    n_fft : int or None
        FFT window size.  Inferred from *data* shape when ``None``.
    fmin : float or None
        Minimum frequency (Hz) for mel/log/cqt axes.  Default: 0.
    fmax : float or None
        Maximum frequency (Hz) for mel/log/fft axes.  Default: ``sr / 2``.
    ax : matplotlib.axes.Axes or None
        Target axes.  If ``None``, uses ``plt.gca()``.
    **kwargs
        Extra keyword arguments forwarded to
        ``matplotlib.axes.Axes.pcolormesh``.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh object returned by ``pcolormesh``.
    """
    plt = _check_matplotlib()

    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n_rows, n_cols = data.shape

    if n_fft is None:
        n_fft = 2 * (n_rows - 1) if n_rows > 1 else 2048
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = sr / 2.0

    # -- x coordinates -------------------------------------------------------
    if x_coords is not None:
        x = np.asarray(x_coords, dtype=np.float64)
    elif x_axis == "time":
        x = _time_ticks(n_cols, sr, hop_length)
    elif x_axis == "frames":
        x = _frame_ticks(n_cols)
    elif x_axis in ("hz", "fft", "fft_note"):
        x = _fft_freqs(n_cols + 1, sr, n_fft)
    elif x_axis == "mel":
        x = _mel_freqs(n_cols + 1, fmin, fmax)
    elif x_axis == "log":
        x = _log_freqs(n_cols + 1, max(fmin, 1.0), fmax)
    else:
        x = np.arange(n_cols + 1, dtype=np.float64)

    # -- y coordinates -------------------------------------------------------
    if y_coords is not None:
        y = np.asarray(y_coords, dtype=np.float64)
    elif y_axis in ("linear", "hz", "fft"):
        y = _fft_freqs(n_rows + 1, sr, n_fft)
    elif y_axis == "fft_note":
        y = _fft_freqs(n_rows + 1, sr, n_fft)
    elif y_axis == "mel":
        y = _mel_freqs(n_rows + 1, fmin, fmax)
    elif y_axis == "log":
        y = _log_freqs(n_rows + 1, max(fmin, 1.0), fmax)
    elif y_axis in ("cqt_hz", "cqt_note"):
        y = _cqt_freqs(n_rows + 1, fmin=fmin if fmin > 0 else None)
    elif y_axis == "chroma":
        y = np.arange(n_rows + 1, dtype=np.float64)
    else:
        y = np.arange(n_rows + 1, dtype=np.float64)

    # -- render --------------------------------------------------------------
    if ax is None:
        ax = plt.gca()

    # Set sensible defaults for pcolormesh
    kwargs.setdefault("shading", "flat")

    mesh = ax.pcolormesh(x, y, data, **kwargs)

    # -- axis labels and formatting ------------------------------------------
    _format_x_axis(ax, x_axis, sr, n_fft)
    _format_y_axis(ax, y_axis, y, sr, n_fft)

    return mesh


def _format_x_axis(ax, x_axis, sr, n_fft):
    """Set x-axis label and formatting based on *x_axis* type."""
    if x_axis == "time":
        ax.set_xlabel("Time (s)")
    elif x_axis == "frames":
        ax.set_xlabel("Frames")
    elif x_axis in ("hz", "fft"):
        ax.set_xlabel("Hz")
    elif x_axis == "fft_note":
        ax.set_xlabel("Note")
    elif x_axis == "mel":
        ax.set_xlabel("Mel")
    elif x_axis == "log":
        ax.set_xlabel("Hz")


def _format_y_axis(ax, y_axis, y_coords, sr, n_fft):
    """Set y-axis label and formatting based on *y_axis* type."""
    if y_axis in ("linear", "hz", "fft"):
        ax.set_ylabel("Hz")
    elif y_axis == "fft_note":
        ax.set_ylabel("Note")
        # Set note-name tick labels at selected frequencies
        _set_note_ticks(ax, y_coords, axis="y")
    elif y_axis == "mel":
        ax.set_ylabel("Mel")
    elif y_axis == "log":
        ax.set_ylabel("Hz")
        ax.set_yscale("symlog", linthresh=1.0)
    elif y_axis == "cqt_hz":
        ax.set_ylabel("Hz")
    elif y_axis == "cqt_note":
        ax.set_ylabel("Note")
        _set_note_ticks(ax, y_coords, axis="y")
    elif y_axis == "chroma":
        ax.set_ylabel("Pitch class")
        n_chroma = max(1, len(y_coords) - 1)
        labels = _chroma_labels(n_chroma)
        centres = np.arange(n_chroma) + 0.5
        ax.set_yticks(centres)
        ax.set_yticklabels(labels)


def _set_note_ticks(ax, freqs, axis="y", max_ticks=12):
    """Place note-name tick labels at musically meaningful frequencies.

    Selects a subset of *freqs* spaced roughly evenly in log-space and
    converts each to a note name.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    positive = freqs[freqs > 0]
    if len(positive) == 0:
        return

    # Choose up to *max_ticks* representative frequencies
    n = min(max_ticks, len(positive))
    indices = np.linspace(0, len(positive) - 1, n, dtype=int)
    chosen = positive[indices]

    labels = [_hz_to_note_label(f) for f in chosen]

    if axis == "y":
        ax.set_yticks(chosen)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(chosen)
        ax.set_xticklabels(labels)


# ---------------------------------------------------------------------------
# waveshow
# ---------------------------------------------------------------------------

def waveshow(y, *, sr=22050, max_points=11025, ax=None, offset=0.0,
             **kwargs):
    """Display a waveform (1D audio signal).

    For long signals (``len(y) > max_points``), an envelope (min/max per
    segment) is drawn using ``fill_between``.  Short signals are plotted
    directly with ``ax.plot``.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D).
    sr : int
        Sample rate.  Default: 22050.
    max_points : int
        Maximum number of display points before switching to envelope
        mode.  Default: 11025.
    ax : matplotlib.axes.Axes or None
        Target axes.  If ``None``, uses ``plt.gca()``.
    offset : float
        Time offset in seconds added to the x-axis.  Default: 0.0.
    **kwargs
        Extra keyword arguments forwarded to ``ax.plot`` or
        ``ax.fill_between``.

    Returns
    -------
    matplotlib artist(s)
        The ``Line2D`` from ``plot`` or the ``PolyCollection`` from
        ``fill_between``.
    """
    plt = _check_matplotlib()

    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples = len(y)

    if ax is None:
        ax = plt.gca()

    if n_samples == 0:
        # Nothing to draw -- return an empty line
        line, = ax.plot([], [], **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return line

    if n_samples <= max_points:
        # Direct plot
        times = np.arange(n_samples, dtype=np.float64) / sr + offset
        kwargs.setdefault("linewidth", 0.5)
        line, = ax.plot(times, y, **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return line

    # Envelope mode: downsample to *max_points* segments
    n_segments = max(1, max_points)
    seg_size = n_samples // n_segments

    # Trim to an even multiple of seg_size
    usable = seg_size * n_segments
    y_trimmed = y[:usable].reshape(n_segments, seg_size)

    y_max = y_trimmed.max(axis=1)
    y_min = y_trimmed.min(axis=1)

    # Time axis: centre of each segment
    sample_centres = (np.arange(n_segments, dtype=np.float64) + 0.5) * seg_size
    times = sample_centres / sr + offset

    kwargs.setdefault("alpha", 0.5)
    fill = ax.fill_between(times, y_min, y_max, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fill
