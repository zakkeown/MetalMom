"""Tests for metalmom.display (specshow, waveshow)."""

import numpy as np
import pytest

# Use non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh, PolyCollection
from matplotlib.lines import Line2D

from metalmom.display import specshow, waveshow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Automatically close all figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def spectrogram():
    """A small random spectrogram-like 2D array (128 freq bins x 20 frames)."""
    rng = np.random.default_rng(42)
    return rng.random((128, 20)).astype(np.float32)


@pytest.fixture
def short_signal():
    """A short audio signal (< default max_points)."""
    return np.sin(2.0 * np.pi * 440.0 * np.arange(5000) / 22050).astype(np.float32)


@pytest.fixture
def long_signal():
    """A long audio signal (> default max_points of 11025)."""
    return np.sin(2.0 * np.pi * 440.0 * np.arange(44100) / 22050).astype(np.float32)


# ---------------------------------------------------------------------------
# specshow tests
# ---------------------------------------------------------------------------

class TestSpecshow:

    def test_returns_quadmesh(self, spectrogram):
        """specshow should return a QuadMesh object."""
        mesh = specshow(spectrogram)
        assert isinstance(mesh, QuadMesh)

    def test_custom_ax(self, spectrogram):
        """specshow should draw on the provided axes."""
        fig, ax = plt.subplots()
        mesh = specshow(spectrogram, ax=ax)
        assert isinstance(mesh, QuadMesh)
        # The mesh should belong to the provided axes
        assert mesh.axes is ax

    def test_x_axis_time(self, spectrogram):
        """x_axis='time' should label the x-axis 'Time (s)'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="time", ax=ax)
        assert ax.get_xlabel() == "Time (s)"

    def test_x_axis_frames(self, spectrogram):
        """x_axis='frames' should label the x-axis 'Frames'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="frames", ax=ax)
        assert ax.get_xlabel() == "Frames"

    def test_x_axis_hz(self, spectrogram):
        """x_axis='hz' should label the x-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="hz", ax=ax)
        assert ax.get_xlabel() == "Hz"

    def test_y_axis_linear(self, spectrogram):
        """y_axis='linear' should label the y-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="linear", ax=ax)
        assert ax.get_ylabel() == "Hz"

    def test_y_axis_hz(self, spectrogram):
        """y_axis='hz' should label the y-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="hz", ax=ax)
        assert ax.get_ylabel() == "Hz"

    def test_y_axis_mel(self, spectrogram):
        """y_axis='mel' should label the y-axis 'Mel'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="mel", ax=ax)
        assert ax.get_ylabel() == "Mel"

    def test_y_axis_log(self, spectrogram):
        """y_axis='log' should label the y-axis 'Hz' with log scale."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="log", ax=ax)
        assert ax.get_ylabel() == "Hz"

    def test_y_axis_fft(self, spectrogram):
        """y_axis='fft' should label the y-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="fft", ax=ax)
        assert ax.get_ylabel() == "Hz"

    def test_y_axis_fft_note(self, spectrogram):
        """y_axis='fft_note' should label the y-axis 'Note'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="fft_note", ax=ax)
        assert ax.get_ylabel() == "Note"

    def test_y_axis_cqt_hz(self, spectrogram):
        """y_axis='cqt_hz' should label the y-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="cqt_hz", ax=ax)
        assert ax.get_ylabel() == "Hz"

    def test_y_axis_cqt_note(self, spectrogram):
        """y_axis='cqt_note' should label the y-axis 'Note'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, y_axis="cqt_note", ax=ax)
        assert ax.get_ylabel() == "Note"

    def test_y_axis_chroma(self):
        """y_axis='chroma' should label the y-axis 'Pitch class'."""
        data = np.random.default_rng(0).random((12, 20)).astype(np.float32)
        fig, ax = plt.subplots()
        specshow(data, y_axis="chroma", ax=ax)
        assert ax.get_ylabel() == "Pitch class"
        # Should have 12 tick labels (C, C#, D, ...)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(labels) == 12
        assert labels[0] == "C"

    def test_no_axis_type(self, spectrogram):
        """With no axis type specified, specshow should still work."""
        mesh = specshow(spectrogram)
        assert isinstance(mesh, QuadMesh)

    def test_custom_x_coords(self, spectrogram):
        """Explicit x_coords should be used."""
        n_cols = spectrogram.shape[1]
        x_coords = np.linspace(0, 10, n_cols + 1)
        fig, ax = plt.subplots()
        mesh = specshow(spectrogram, x_coords=x_coords, ax=ax)
        assert isinstance(mesh, QuadMesh)

    def test_custom_y_coords(self, spectrogram):
        """Explicit y_coords should be used."""
        n_rows = spectrogram.shape[0]
        y_coords = np.linspace(0, 8000, n_rows + 1)
        fig, ax = plt.subplots()
        mesh = specshow(spectrogram, y_coords=y_coords, ax=ax)
        assert isinstance(mesh, QuadMesh)

    def test_kwargs_forwarded(self, spectrogram):
        """Extra kwargs should be forwarded to pcolormesh."""
        fig, ax = plt.subplots()
        mesh = specshow(spectrogram, ax=ax, cmap="magma", vmin=-10, vmax=10)
        assert isinstance(mesh, QuadMesh)

    def test_single_frame(self):
        """Single-frame (1 column) data should not error."""
        data = np.random.default_rng(0).random((64, 1)).astype(np.float32)
        mesh = specshow(data, x_axis="time")
        assert isinstance(mesh, QuadMesh)

    def test_single_row(self):
        """Single-row data should not error."""
        data = np.random.default_rng(0).random((1, 20)).astype(np.float32)
        mesh = specshow(data)
        assert isinstance(mesh, QuadMesh)

    def test_sr_and_hop_affect_time_axis(self, spectrogram):
        """Changing sr and hop_length should change time coordinates."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        specshow(spectrogram, x_axis="time", sr=22050, hop_length=512, ax=ax1)
        specshow(spectrogram, x_axis="time", sr=44100, hop_length=256, ax=ax2)
        # The x-limits should differ
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()
        assert xlim1[1] != xlim2[1]

    def test_x_axis_mel(self, spectrogram):
        """x_axis='mel' should label the x-axis 'Mel'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="mel", ax=ax)
        assert ax.get_xlabel() == "Mel"

    def test_x_axis_log(self, spectrogram):
        """x_axis='log' should label the x-axis 'Hz'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="log", ax=ax)
        assert ax.get_xlabel() == "Hz"

    def test_x_axis_fft_note(self, spectrogram):
        """x_axis='fft_note' should label the x-axis 'Note'."""
        fig, ax = plt.subplots()
        specshow(spectrogram, x_axis="fft_note", ax=ax)
        assert ax.get_xlabel() == "Note"


# ---------------------------------------------------------------------------
# waveshow tests
# ---------------------------------------------------------------------------

class TestWaveshow:

    def test_short_signal_returns_line(self, short_signal):
        """Short signals should be plotted with ax.plot, returning Line2D."""
        result = waveshow(short_signal)
        assert isinstance(result, Line2D)

    def test_long_signal_returns_polycollection(self, long_signal):
        """Long signals should use fill_between, returning PolyCollection."""
        result = waveshow(long_signal)
        assert isinstance(result, PolyCollection)

    def test_custom_ax(self, short_signal):
        """waveshow should draw on the provided axes."""
        fig, ax = plt.subplots()
        result = waveshow(short_signal, ax=ax)
        assert isinstance(result, Line2D)
        assert result.axes is ax

    def test_axis_labels(self, short_signal):
        """x and y axis labels should be set."""
        fig, ax = plt.subplots()
        waveshow(short_signal, ax=ax)
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Amplitude"

    def test_envelope_axis_labels(self, long_signal):
        """Envelope mode should also set axis labels."""
        fig, ax = plt.subplots()
        waveshow(long_signal, ax=ax)
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Amplitude"

    def test_offset(self, short_signal):
        """offset should shift the time axis."""
        fig, ax = plt.subplots()
        result = waveshow(short_signal, sr=22050, offset=1.0, ax=ax)
        # The line data x-values should start at the offset
        xdata = result.get_xdata()
        assert xdata[0] == pytest.approx(1.0, abs=1e-6)

    def test_empty_signal(self):
        """Empty signal should not crash."""
        result = waveshow(np.array([], dtype=np.float32))
        assert isinstance(result, Line2D)

    def test_max_points_threshold(self, long_signal):
        """Setting max_points high enough should avoid envelope mode."""
        result = waveshow(long_signal, max_points=100000)
        assert isinstance(result, Line2D)

    def test_max_points_low(self, short_signal):
        """Setting max_points very low should force envelope mode."""
        result = waveshow(short_signal, max_points=100)
        assert isinstance(result, PolyCollection)

    def test_kwargs_forwarded_plot(self, short_signal):
        """Extra kwargs should be forwarded to ax.plot."""
        fig, ax = plt.subplots()
        result = waveshow(short_signal, ax=ax, color="red")
        assert isinstance(result, Line2D)

    def test_kwargs_forwarded_fill(self, long_signal):
        """Extra kwargs should be forwarded to ax.fill_between."""
        fig, ax = plt.subplots()
        result = waveshow(long_signal, ax=ax, color="blue")
        assert isinstance(result, PolyCollection)

    def test_sr_affects_time(self, short_signal):
        """Different sample rates should change the time axis range."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        waveshow(short_signal, sr=22050, ax=ax1)
        waveshow(short_signal, sr=44100, ax=ax2)
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()
        # Duration at 44100 should be about half of 22050
        assert xlim2[1] < xlim1[1]
