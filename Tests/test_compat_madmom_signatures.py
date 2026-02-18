"""Automated madmom signature compatibility verification tests.

Since real madmom cannot be imported on Python 3.10+ (uses removed
``collections.MutableSequence``), we hardcode the expected API surface based
on madmom 0.16.1 documentation and verify our compat shim against it.

Checks:
 1. Class existence and importability
 2. Callability (__call__ for processors)
 3. Constructor parameter coverage (via inspect.signature)
 4. Expected attributes on constructed instances
 5. ndarray subclass status for audio containers
 6. Full package hierarchy importability
 7. madmom-standard default values (sr=44100, fps=100, etc.)
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Expected madmom 0.16.1 API (hardcoded reference)
# ---------------------------------------------------------------------------

EXPECTED_CLASSES: dict[str, dict[str, dict[str, Any]]] = {
    "madmom.audio.signal": {
        "Signal": {
            "init_params": ["data", "sample_rate"],
            "attributes": ["sample_rate"],
            "is_ndarray_subclass": True,
        },
        "FramedSignal": {
            "init_params": ["signal", "frame_size", "hop_size"],
            "attributes": ["frame_size", "hop_size", "signal"],
            "supports_iteration": True,
        },
    },
    "madmom.audio.stft": {
        "STFT": {
            "init_params": ["signal"],
            "is_ndarray_subclass": True,
        },
    },
    "madmom.audio.spectrogram": {
        "Spectrogram": {
            "init_params": ["stft"],
            "is_ndarray_subclass": True,
        },
        "FilteredSpectrogram": {
            "init_params": ["spectrogram"],
            "is_ndarray_subclass": True,
        },
        "LogarithmicFilteredSpectrogram": {
            "init_params": ["spectrogram"],
            "is_ndarray_subclass": True,
        },
    },
    "madmom.features.onsets": {
        "OnsetPeakPickingProcessor": {
            "init_params": [],
            "is_callable": True,
        },
        "RNNOnsetProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.features.beats": {
        "RNNBeatProcessor": {
            "init_params": [],
            "is_callable": True,
        },
        "DBNBeatTrackingProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.features.downbeats": {
        "RNNDownBeatProcessor": {
            "init_params": [],
            "is_callable": True,
        },
        "DBNDownBeatTrackingProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.features.key": {
        "CNNKeyRecognitionProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.features.chords": {
        "DeepChromaChordRecognitionProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.features.tempo": {
        "TempoEstimationProcessor": {
            "init_params": [],
            "is_callable": True,
        },
    },
    "madmom.evaluation.onsets": {
        "OnsetEvaluation": {
            "init_params": ["detections", "annotations"],
            "attributes": ["fmeasure", "precision", "recall"],
        },
    },
    "madmom.evaluation.beats": {
        "BeatEvaluation": {
            "init_params": ["detections", "annotations"],
            "attributes": ["fmeasure", "cemgil"],
        },
    },
    "madmom.evaluation.tempo": {
        "TempoEvaluation": {
            "init_params": ["detections", "annotations"],
            "attributes": ["acc1", "acc2"],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compat_module(madmom_module: str) -> str:
    """Convert a madmom module path to our compat module path.

    ``madmom.audio.signal`` -> ``metalmom.compat.madmom.audio.signal``
    """
    return "metalmom.compat." + madmom_module


def _import_class(madmom_module: str, class_name: str):
    """Import *class_name* from our compat shim. Returns None on failure."""
    mod_path = _compat_module(madmom_module)
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, class_name, None)
    except Exception:
        return None


def _get_init_params(cls) -> dict[str, dict[str, Any]]:
    """Extract __init__ (or __new__ for ndarray subclasses) parameter info."""
    # ndarray subclasses define __new__ instead of __init__
    if issubclass(cls, np.ndarray) and hasattr(cls, "__new__"):
        func = cls.__new__
    else:
        func = cls.__init__

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {}

    info: dict[str, dict[str, Any]] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        info[name] = {
            "kind": p.kind,
            "default": p.default,
            "has_default": p.default is not inspect.Parameter.empty,
        }
    return info


def _has_var_keyword(params: dict[str, dict[str, Any]]) -> bool:
    """Return True if the signature contains **kwargs."""
    return any(
        v["kind"] == inspect.Parameter.VAR_KEYWORD for v in params.values()
    )


def _named_params(params: dict[str, dict[str, Any]]) -> set[str]:
    """Return the set of explicitly named parameter names (no *args/**kwargs)."""
    return {
        name
        for name, info in params.items()
        if info["kind"]
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


# ---------------------------------------------------------------------------
# Build flat test cases from EXPECTED_CLASSES
# ---------------------------------------------------------------------------

# (madmom_module, class_name, spec_dict)
_ALL_CLASSES: list[tuple[str, str, dict[str, Any]]] = []
for mod, classes in EXPECTED_CLASSES.items():
    for cls_name, spec in classes.items():
        _ALL_CLASSES.append((mod, cls_name, spec))

_CLASS_IDS = [f"{mod}.{cls}" for mod, cls, _ in _ALL_CLASSES]


# Known parameter-name mappings between madmom and our shim.
# madmom's first positional arg may have a different name in our shim
# (e.g. madmom says "signal" for STFT, we say "data").
_PARAM_ALIASES: dict[tuple[str, str], dict[str, str]] = {
    # (module, class) -> {expected_name: our_name}
    ("madmom.audio.stft", "STFT"): {"signal": "data"},
    ("madmom.audio.spectrogram", "Spectrogram"): {"stft": "data"},
    ("madmom.audio.spectrogram", "FilteredSpectrogram"): {
        "spectrogram": "data",
    },
    ("madmom.audio.spectrogram", "LogarithmicFilteredSpectrogram"): {
        "spectrogram": "data",
    },
    ("madmom.audio.signal", "Signal"): {"sample_rate": "sample_rate"},
}


# ---------------------------------------------------------------------------
# 1. test_class_exists  -- each class is importable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "madmom_mod, class_name, spec",
    _ALL_CLASSES,
    ids=_CLASS_IDS,
)
def test_class_exists(madmom_mod, class_name, spec):
    """Each expected madmom class should be importable from our compat shim."""
    cls = _import_class(madmom_mod, class_name)
    assert cls is not None, (
        f"Cannot import {class_name} from "
        f"{_compat_module(madmom_mod)}"
    )


# ---------------------------------------------------------------------------
# 2. test_class_callable  -- processors have __call__
# ---------------------------------------------------------------------------

_CALLABLE_CLASSES = [
    (mod, cls_name, spec)
    for mod, cls_name, spec in _ALL_CLASSES
    if spec.get("is_callable", False)
]
_CALLABLE_IDS = [f"{mod}.{cls}" for mod, cls, _ in _CALLABLE_CLASSES]


@pytest.mark.parametrize(
    "madmom_mod, class_name, spec",
    _CALLABLE_CLASSES,
    ids=_CALLABLE_IDS,
)
def test_class_callable(madmom_mod, class_name, spec):
    """Processor classes must be callable (have __call__)."""
    cls = _import_class(madmom_mod, class_name)
    if cls is None:
        pytest.skip(f"Cannot import {class_name}")

    # Instantiate with no args (all processors have default-only inits)
    instance = cls()
    assert callable(instance), (
        f"{class_name} instance is not callable "
        f"(missing __call__)"
    )


# ---------------------------------------------------------------------------
# 3. test_init_params  -- constructor accepts expected params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "madmom_mod, class_name, spec",
    _ALL_CLASSES,
    ids=_CLASS_IDS,
)
def test_init_params(madmom_mod, class_name, spec):
    """Constructor should accept at least the expected parameter names."""
    cls = _import_class(madmom_mod, class_name)
    if cls is None:
        pytest.skip(f"Cannot import {class_name}")

    expected_params = spec.get("init_params", [])
    if not expected_params:
        # Processors with no required init params -- just check instantiation
        try:
            _obj = cls()
        except Exception as exc:
            pytest.fail(
                f"{class_name}() raised {type(exc).__name__}: {exc}"
            )
        return

    our_params = _get_init_params(cls)
    our_named = _named_params(our_params)
    has_kwargs = _has_var_keyword(our_params)

    aliases = _PARAM_ALIASES.get((madmom_mod, class_name), {})

    missing = []
    for param in expected_params:
        # Check direct match, alias, or **kwargs acceptance
        aliased = aliases.get(param, param)
        if aliased in our_named or param in our_named or has_kwargs:
            continue
        missing.append(param)

    if missing:
        pytest.fail(
            f"{class_name} constructor missing expected param(s): "
            f"{missing}.  Our params: {sorted(our_named)}"
        )


# ---------------------------------------------------------------------------
# 4. test_attributes  -- expected attributes exist after construction
# ---------------------------------------------------------------------------

# Build test cases for classes that have expected attributes.
_ATTR_CLASSES: list[tuple[str, str, list[str]]] = []
for mod, cls_name, spec in _ALL_CLASSES:
    attrs = spec.get("attributes", [])
    if attrs:
        _ATTR_CLASSES.append((mod, cls_name, attrs))

_ATTR_IDS = [f"{mod}.{cls}" for mod, cls, _ in _ATTR_CLASSES]


def _make_test_instance(madmom_mod: str, class_name: str):
    """Create a test instance of the given class with minimal test data."""
    cls = _import_class(madmom_mod, class_name)
    if cls is None:
        return None

    # Audio classes need test data
    if class_name == "Signal":
        return cls(np.zeros(4410, dtype=np.float32), sample_rate=44100)
    elif class_name == "FramedSignal":
        sig = np.zeros(4410, dtype=np.float32)
        return cls(sig, frame_size=2048, hop_size=441)
    elif class_name == "STFT":
        sig = np.zeros(4410, dtype=np.float32)
        return cls(sig)
    elif class_name in ("Spectrogram", "FilteredSpectrogram",
                        "LogarithmicFilteredSpectrogram"):
        # Build from a small 1-D signal
        sig = np.zeros(4410, dtype=np.float32)
        return cls(sig)
    elif class_name == "OnsetEvaluation":
        return cls(
            detections=np.array([0.5, 1.0, 1.5]),
            annotations=np.array([0.5, 1.0, 1.5]),
        )
    elif class_name == "BeatEvaluation":
        return cls(
            detections=np.array([0.5, 1.0, 1.5]),
            annotations=np.array([0.5, 1.0, 1.5]),
        )
    elif class_name == "TempoEvaluation":
        return cls(detections=120.0, annotations=120.0)
    else:
        # Processors -- no-arg construction
        try:
            return cls()
        except Exception:
            return None


@pytest.mark.parametrize(
    "madmom_mod, class_name, expected_attrs",
    _ATTR_CLASSES,
    ids=_ATTR_IDS,
)
def test_attributes(madmom_mod, class_name, expected_attrs):
    """Constructed instances expose the expected attributes."""
    instance = _make_test_instance(madmom_mod, class_name)
    if instance is None:
        pytest.skip(f"Cannot construct {class_name}")

    missing = []
    for attr in expected_attrs:
        if not hasattr(instance, attr):
            missing.append(attr)

    if missing:
        pytest.fail(
            f"{class_name} instance missing expected attribute(s): "
            f"{missing}"
        )


# ---------------------------------------------------------------------------
# 5. test_ndarray_subclass  -- Signal/STFT/Spectrogram are ndarray subclasses
# ---------------------------------------------------------------------------

_NDARRAY_CLASSES = [
    (mod, cls_name)
    for mod, cls_name, spec in _ALL_CLASSES
    if spec.get("is_ndarray_subclass", False)
]
_NDARRAY_IDS = [f"{mod}.{cls}" for mod, cls in _NDARRAY_CLASSES]


@pytest.mark.parametrize(
    "madmom_mod, class_name",
    _NDARRAY_CLASSES,
    ids=_NDARRAY_IDS,
)
def test_ndarray_subclass(madmom_mod, class_name):
    """Audio container classes must be np.ndarray subclasses."""
    cls = _import_class(madmom_mod, class_name)
    if cls is None:
        pytest.skip(f"Cannot import {class_name}")

    assert issubclass(cls, np.ndarray), (
        f"{class_name} is not a subclass of np.ndarray"
    )

    # Also verify an instance is an ndarray
    instance = _make_test_instance(madmom_mod, class_name)
    if instance is not None:
        assert isinstance(instance, np.ndarray), (
            f"{class_name} instance is not an ndarray"
        )


# ---------------------------------------------------------------------------
# 6. test_module_structure  -- full package hierarchy is importable
# ---------------------------------------------------------------------------

EXPECTED_MODULES = [
    "metalmom.compat.madmom",
    "metalmom.compat.madmom.audio",
    "metalmom.compat.madmom.audio.signal",
    "metalmom.compat.madmom.audio.stft",
    "metalmom.compat.madmom.audio.spectrogram",
    "metalmom.compat.madmom.features",
    "metalmom.compat.madmom.features.onsets",
    "metalmom.compat.madmom.features.beats",
    "metalmom.compat.madmom.features.downbeats",
    "metalmom.compat.madmom.features.key",
    "metalmom.compat.madmom.features.chords",
    "metalmom.compat.madmom.features.tempo",
    "metalmom.compat.madmom.evaluation",
    "metalmom.compat.madmom.evaluation.onsets",
    "metalmom.compat.madmom.evaluation.beats",
    "metalmom.compat.madmom.evaluation.tempo",
]


@pytest.mark.parametrize("module_path", EXPECTED_MODULES)
def test_module_structure(module_path):
    """The full madmom package hierarchy must be importable."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"Cannot import {module_path}: {exc}")

    assert mod is not None


# ---------------------------------------------------------------------------
# 7. test_default_values  -- madmom-standard defaults
# ---------------------------------------------------------------------------

class TestDefaultValues:
    """Verify madmom-standard default values are used throughout."""

    def test_signal_default_sample_rate(self):
        """Signal default sample_rate should be 44100 (not librosa's 22050)."""
        from metalmom.compat.madmom.audio.signal import Signal

        sig = Signal(np.zeros(100, dtype=np.float32))
        assert sig.sample_rate == 44100

    def test_framed_signal_default_frame_size(self):
        """FramedSignal default frame_size should be 2048."""
        from metalmom.compat.madmom.audio.signal import FramedSignal

        sig = np.zeros(4410, dtype=np.float32)
        framed = FramedSignal(sig)
        assert framed.frame_size == 2048

    def test_framed_signal_default_hop_size(self):
        """FramedSignal default hop_size should be 441 (10 ms at 44100 Hz)."""
        from metalmom.compat.madmom.audio.signal import FramedSignal

        sig = np.zeros(4410, dtype=np.float32)
        framed = FramedSignal(sig)
        assert framed.hop_size == 441

    def test_stft_default_frame_size(self):
        """STFT default frame_size should be 2048."""
        from metalmom.compat.madmom.audio.stft import STFT

        params = _get_init_params(STFT)
        named = {
            n: info for n, info in params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        assert "frame_size" in named
        assert named["frame_size"]["default"] == 2048

    def test_stft_default_hop_size(self):
        """STFT default hop_size should be 441."""
        from metalmom.compat.madmom.audio.stft import STFT

        params = _get_init_params(STFT)
        named = {
            n: info for n, info in params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        assert "hop_size" in named
        assert named["hop_size"]["default"] == 441

    def test_stft_default_sample_rate(self):
        """STFT default sample_rate should be 44100."""
        from metalmom.compat.madmom.audio.stft import STFT

        sig = np.zeros(4410, dtype=np.float32)
        stft = STFT(sig)
        assert stft.sample_rate == 44100

    def test_onset_peak_picking_default_fps(self):
        """OnsetPeakPickingProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.onsets import (
            OnsetPeakPickingProcessor,
        )

        proc = OnsetPeakPickingProcessor()
        assert proc.fps == 100.0

    def test_rnn_onset_default_fps(self):
        """RNNOnsetProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.onsets import RNNOnsetProcessor

        proc = RNNOnsetProcessor()
        assert proc.fps == 100.0

    def test_rnn_beat_default_fps(self):
        """RNNBeatProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.beats import RNNBeatProcessor

        proc = RNNBeatProcessor()
        assert proc.fps == 100.0

    def test_dbn_beat_default_fps(self):
        """DBNBeatTrackingProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.beats import (
            DBNBeatTrackingProcessor,
        )

        proc = DBNBeatTrackingProcessor()
        assert proc.fps == 100.0

    def test_dbn_beat_default_min_bpm(self):
        """DBNBeatTrackingProcessor default min_bpm should be 55.0."""
        from metalmom.compat.madmom.features.beats import (
            DBNBeatTrackingProcessor,
        )

        proc = DBNBeatTrackingProcessor()
        assert proc.min_bpm == 55.0

    def test_dbn_beat_default_max_bpm(self):
        """DBNBeatTrackingProcessor default max_bpm should be 215.0."""
        from metalmom.compat.madmom.features.beats import (
            DBNBeatTrackingProcessor,
        )

        proc = DBNBeatTrackingProcessor()
        assert proc.max_bpm == 215.0

    def test_rnn_downbeat_default_fps(self):
        """RNNDownBeatProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.downbeats import (
            RNNDownBeatProcessor,
        )

        proc = RNNDownBeatProcessor()
        assert proc.fps == 100.0

    def test_dbn_downbeat_default_beats_per_bar(self):
        """DBNDownBeatTrackingProcessor default beats_per_bar should be [4]."""
        from metalmom.compat.madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
        )

        proc = DBNDownBeatTrackingProcessor()
        assert proc.beats_per_bar == [4]

    def test_tempo_estimation_default_fps(self):
        """TempoEstimationProcessor default fps should be 100.0."""
        from metalmom.compat.madmom.features.tempo import (
            TempoEstimationProcessor,
        )

        proc = TempoEstimationProcessor()
        assert proc.fps == 100.0

    def test_key_recognition_default_fps(self):
        """CNNKeyRecognitionProcessor default fps should be 10.0."""
        from metalmom.compat.madmom.features.key import (
            CNNKeyRecognitionProcessor,
        )

        proc = CNNKeyRecognitionProcessor()
        assert proc.fps == 10.0

    def test_chord_recognition_default_fps(self):
        """DeepChromaChordRecognitionProcessor default fps should be 10.0."""
        from metalmom.compat.madmom.features.chords import (
            DeepChromaChordRecognitionProcessor,
        )

        proc = DeepChromaChordRecognitionProcessor()
        assert proc.fps == 10.0

    def test_onset_eval_default_window(self):
        """OnsetEvaluation default window should be 0.025."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        params = _get_init_params(OnsetEvaluation)
        named = {
            n: info for n, info in params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        assert "window" in named
        assert named["window"]["default"] == 0.025

    def test_beat_eval_default_fmeasure_window(self):
        """BeatEvaluation default fmeasure_window should be 0.07."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation

        params = _get_init_params(BeatEvaluation)
        named = {
            n: info for n, info in params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        assert "fmeasure_window" in named
        assert named["fmeasure_window"]["default"] == 0.07

    def test_tempo_eval_default_tolerance(self):
        """TempoEvaluation default tolerance should be 0.08."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        params = _get_init_params(TempoEvaluation)
        named = {
            n: info for n, info in params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        assert "tolerance" in named
        assert named["tolerance"]["default"] == 0.08


# ---------------------------------------------------------------------------
# Extra: FramedSignal supports iteration
# ---------------------------------------------------------------------------

def test_framed_signal_iteration():
    """FramedSignal should support iteration over frames."""
    from metalmom.compat.madmom.audio.signal import FramedSignal

    sig = np.random.randn(4410).astype(np.float32)
    framed = FramedSignal(sig, frame_size=2048, hop_size=441)

    # Must be iterable
    frames = list(framed)
    assert len(frames) > 0
    assert len(frames) == len(framed)

    # Each frame should be the correct size
    for frame in frames:
        assert frame.shape == (2048,)


# ---------------------------------------------------------------------------
# Extra: evaluation classes produce valid numeric outputs
# ---------------------------------------------------------------------------

class TestEvaluationOutputs:
    """Verify evaluation classes produce sensible numeric results."""

    def test_onset_eval_perfect_score(self):
        """Perfect onset detections should yield fmeasure=1.0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation

        times = np.array([0.5, 1.0, 1.5, 2.0])
        ev = OnsetEvaluation(detections=times, annotations=times)
        assert ev.fmeasure == pytest.approx(1.0, abs=0.01)
        assert ev.precision == pytest.approx(1.0, abs=0.01)
        assert ev.recall == pytest.approx(1.0, abs=0.01)

    def test_beat_eval_perfect_score(self):
        """Perfect beat detections should yield fmeasure=1.0."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation

        times = np.array([0.5, 1.0, 1.5, 2.0])
        ev = BeatEvaluation(detections=times, annotations=times)
        assert ev.fmeasure == pytest.approx(1.0, abs=0.01)
        assert ev.cemgil >= 0.0

    def test_tempo_eval_perfect_score(self):
        """Perfect tempo estimate should yield acc1=1.0."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation

        ev = TempoEvaluation(detections=120.0, annotations=120.0)
        assert ev.acc1 == pytest.approx(1.0)
        assert ev.acc2 >= ev.acc1


# ---------------------------------------------------------------------------
# Summary test
# ---------------------------------------------------------------------------

def test_overall_coverage_report():
    """Print an aggregate coverage summary for all madmom compat classes.

    This test always passes -- its purpose is to produce a readable report
    via pytest output (use ``-s`` or ``--tb=short`` to see it).
    """
    total_expected = 0
    total_found = 0
    rows: list[tuple[str, str, bool, str]] = []

    for madmom_mod, classes in EXPECTED_CLASSES.items():
        for cls_name, spec in classes.items():
            total_expected += 1
            cls = _import_class(madmom_mod, cls_name)
            found = cls is not None
            if found:
                total_found += 1
            status = "OK" if found else "MISSING"
            rows.append((madmom_mod, cls_name, found, status))

    lines = [
        "",
        "=" * 78,
        "  madmom Compat Signature Coverage Report",
        "=" * 78,
        f"  {'Module':<40} {'Class':<30} {'Status':>6}",
        "-" * 78,
    ]
    for mod, cls, found, status in rows:
        marker = "   " if found else " !!"
        lines.append(f"{marker}{mod:<39} {cls:<30} {status:>6}")
    lines.append("-" * 78)
    pct = (total_found / total_expected * 100) if total_expected else 100.0
    lines.append(
        f"   Classes: {total_found}/{total_expected} ({pct:.0f}%)"
    )
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)
    warnings.warn(f"\n{report}", stacklevel=1)
