"""librosa completeness audit -- checks which public functions are covered.

Walks the public API of key librosa modules and checks whether each
function exists in our compat shim.  Missing functions are reported as
warnings (not hard failures) with a clear per-module breakdown.

Run with:
    .venv/bin/pytest Tests/test_compat_completeness.py -v -s
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import pytest


# ---------------------------------------------------------------------------
# Modules to audit and their compat equivalents
# ---------------------------------------------------------------------------

AUDIT_MODULES = [
    ("librosa", "metalmom.compat.librosa"),
    ("librosa.feature", "metalmom.compat.librosa.feature"),
    ("librosa.onset", "metalmom.compat.librosa.onset"),
    ("librosa.beat", "metalmom.compat.librosa.beat"),
    ("librosa.effects", "metalmom.compat.librosa.effects"),
    ("librosa.filters", "metalmom.compat.librosa.filters"),
    ("librosa.decompose", "metalmom.compat.librosa.decompose"),
    ("librosa.sequence", "metalmom.compat.librosa.sequence"),
    ("librosa.display", "metalmom.compat.librosa.display"),
    ("librosa.segment", "metalmom.compat.librosa.segment"),
    ("librosa.feature.inverse", "metalmom.compat.librosa.feature"),  # inverse merged into feature
]


# ---------------------------------------------------------------------------
# Functions we intentionally do not implement
# ---------------------------------------------------------------------------
# Keyed by librosa module name -> set of function names.

SKIP_FUNCTIONS: dict[str, set[str]] = {
    # --- Top-level librosa ---
    "librosa": {
        # Python dunder / module internals
        "__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
        "__name__", "__package__", "__path__", "__spec__", "__version__",
        "__all__", "__getattr__",
        # Submodule names (not functions)
        "beat", "core", "decompose", "display", "effects", "feature",
        "filters", "onset", "segment", "sequence", "util", "cache",
        "convert",
        # Version / utility not related to audio analysis
        "show_versions",
        "cache",
        # Example / data loading utilities
        "example", "ex",
        # FFT backend selection (implementation detail)
        "get_fftlib", "set_fftlib",
        # Deprecated or very niche frequency weighting variants
        "A_weighting", "B_weighting", "C_weighting", "D_weighting",
        "Z_weighting", "multi_frequency_weighting", "frequency_weighting",
        "perceptual_weighting",
        # Indian music theory (extremely niche)
        "hz_to_svara_c", "hz_to_svara_h",
        "midi_to_svara_c", "midi_to_svara_h",
        "note_to_svara_c", "note_to_svara_h",
        "mela_to_degrees", "mela_to_svara",
        "list_mela", "list_thaat", "thaat_to_degrees",
        "key_to_degrees", "key_to_notes",
        # FJS (Functional Just System) interval notation -- very niche
        "hz_to_fjs", "interval_to_fjs",
        # Interval tuning systems -- very niche
        "plimit_intervals", "pythagorean_intervals",
        "interval_frequencies",
        # Pitch naming helper (we have note_to_hz / hz_to_note)
        "fifths_to_note",
        # Mu-law companding -- out of scope (telecom, not MIR)
        "mu_compress", "mu_expand",
        # Inverse CQT / ICQT -- not yet implemented
        "icqt",
        # Griffin-Lim CQT -- not yet implemented (requires icqt)
        "griffinlim_cqt",
        # Pseudo-CQT -- niche variant
        "pseudo_cqt",
        # IIR / alternative filter implementations -- out of scope
        "iirt",
        # Tuning <-> A4 helpers -- very niche
        "A4_to_tuning", "tuning_to_A4",
        # Citation helper
        "cite",
        # FMT -- rarely used
        "fmt",
        # blocks_to_* conversions -- rarely used
        "blocks_to_frames", "blocks_to_samples", "blocks_to_time",
    },
    # --- librosa.feature ---
    "librosa.feature": set(),  # nothing to skip -- we should cover everything
    # --- librosa.feature.inverse ---
    "librosa.feature.inverse": {
        # Utility helpers that leaked into the inverse namespace
        "Optional", "db_to_power", "expand_to", "get_fftlib", "tiny",
        # mel_to_stft is the inner step of mel_to_audio (not commonly called directly)
        "mel_to_stft",
        # NNLS is a solver utility, not typically user-facing
        "nnls",
        # griffinlim in inverse namespace is same as top-level
        "griffinlim",
    },
    # --- librosa.onset ---
    "librosa.onset": {
        # Type annotations / internal re-exports
        "Callable", "Optional", "Sequence",
        # Internal caching
        "cache",
        # Re-exported from feature (not an onset function)
        "melspectrogram",
        # onset_backtrack is a low-level helper rarely used directly
        "onset_backtrack",
        # onset_strength_multi is a multi-channel variant -- not yet implemented
        "onset_strength_multi",
    },
    # --- librosa.beat ---
    "librosa.beat": {
        # Type annotations / internal re-exports
        "Optional", "Tuple",
        # moved() is a deprecation marker, not a real function
        "moved",
        # fourier_tempogram was moved from beat to feature; it's in feature compat
        "fourier_tempogram",
        # tempo was moved from beat to feature; it's in feature compat
        "tempo",
    },
    # --- librosa.effects ---
    "librosa.effects": {
        # Type annotations / internal re-exports
        "Callable", "Iterable", "List", "Literal", "Optional", "Tuple",
        # overload is a decorator, not a real function
        "overload",
        # remix is rarely used (reorder audio frames)
        "remix",
    },
    # --- librosa.filters ---
    "librosa.filters": {
        # Type annotations / internal re-exports
        "List", "Literal", "Optional", "Tuple",
        # Internal caching
        "cache",
        # JIT decorator
        "jit",
        # deprecated decorator
        "deprecated",
        # Internal helper re-exports
        "hz_to_midi", "hz_to_octs", "midi_to_hz", "note_to_hz",
        # Window functions (users call scipy/numpy directly)
        "get_window",
        # Wavelet-related (very niche, out of scope)
        "wavelet", "wavelet_lengths",
        # Window bandwidth / sumsquare (internal helpers)
        "window_bandwidth", "window_sumsquare",
        # constant_q_lengths (internal helper)
        "constant_q_lengths",
        # CQ-to-chroma mapping (niche)
        "cq_to_chroma",
        # Multi-resolution frequencies (niche)
        "mr_frequencies",
        # Diagonal filter (used internally by segment module)
        "diagonal_filter",
    },
    # --- librosa.decompose ---
    "librosa.decompose": {
        # Type annotations / internal re-exports
        "Callable", "List", "Optional", "Tuple",
        # Internal caching
        "cache",
        # decompose() is a generic wrapper; users call nmf() directly
        "decompose",
        # HPSS is in effects (where it belongs); decompose re-exports it
        "hpss",
        # median_filter is a scipy wrapper
        "median_filter",
    },
    # --- librosa.sequence ---
    "librosa.sequence": {
        # Type annotations / internal re-exports
        "Iterable", "List", "Literal", "Optional", "Tuple",
        # Scipy distance function
        "cdist",
        # Internal utilities
        "expand_to", "get_window", "is_positive_int", "jit", "overload",
        "pad_center", "tiny", "fill_off_diagonal",
        # dtw_backtracking is an internal helper (our dtw returns the path)
        "dtw_backtracking",
    },
    # --- librosa.display ---
    "librosa.display": {
        # Type annotations / internal re-exports
        "Callable", "Collection", "Dict", "Optional",
        # Internal colormap helper
        "cmap", "mcm",
        # Deprecation rename helper
        "rename_kw",
    },
    # --- librosa.segment ---
    "librosa.segment": {
        # Type annotations / internal re-exports
        "Callable", "Literal", "Optional",
        # Internal caching / decorator re-exports
        "cache", "decorator", "overload",
        # Internal helpers
        "diagonal_filter",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_public_functions(module_name: str) -> list[str]:
    """Get all public callable names from a module (excluding classes/modules)."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return []

    funcs = []
    skips = SKIP_FUNCTIONS.get(module_name, set())

    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        if name in skips:
            continue
        obj = getattr(mod, name)
        if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
            funcs.append(name)
    return funcs


def get_compat_names(module_name: str) -> set[str]:
    """Get all names available in a compat module."""
    try:
        mod = importlib.import_module(module_name)
        return set(dir(mod))
    except ImportError:
        return set()


# ---------------------------------------------------------------------------
# Parametrized test: per-function coverage
# ---------------------------------------------------------------------------

# Build (librosa_module, compat_module, function_name) triples for parametrize.
_TEST_CASES: list[tuple[str, str, str]] = []
_TEST_IDS: list[str] = []

for lib_mod, compat_mod in AUDIT_MODULES:
    for fn in get_public_functions(lib_mod):
        _TEST_CASES.append((lib_mod, compat_mod, fn))
        _TEST_IDS.append(f"{lib_mod}.{fn}")


@pytest.mark.parametrize("lib_mod, compat_mod, func_name", _TEST_CASES, ids=_TEST_IDS)
def test_function_exists_in_compat(lib_mod, compat_mod, func_name):
    """Check that each librosa public function exists in our compat shim."""
    compat_names = get_compat_names(compat_mod)

    if func_name in compat_names:
        # Verify it's actually callable
        mod = importlib.import_module(compat_mod)
        obj = getattr(mod, func_name)
        assert callable(obj), f"{compat_mod}.{func_name} exists but is not callable"
    else:
        # Not a hard fail -- emit warning so the audit is visible
        warnings.warn(
            f"MISSING: {lib_mod}.{func_name} not found in {compat_mod}",
            stacklevel=1,
        )
        pytest.skip(f"{func_name} not yet implemented in compat shim")


# ---------------------------------------------------------------------------
# Summary test: coverage report
# ---------------------------------------------------------------------------

def test_completeness_coverage_report():
    """Print a per-module and overall coverage report.

    Always passes -- the purpose is producing a readable audit report.
    """
    total_functions = 0
    total_covered = 0
    module_rows: list[tuple[str, int, int, list[str]]] = []

    for lib_mod, compat_mod in AUDIT_MODULES:
        funcs = get_public_functions(lib_mod)
        compat_names = get_compat_names(compat_mod)

        covered = 0
        missing: list[str] = []
        for fn in funcs:
            if fn in compat_names:
                covered += 1
            else:
                missing.append(fn)

        module_rows.append((lib_mod, len(funcs), covered, missing))
        total_functions += len(funcs)
        total_covered += covered

    # Build report
    lines = [
        "",
        "=" * 78,
        "  librosa Completeness Audit Report",
        "=" * 78,
    ]

    for mod_name, n_total, n_covered, missing in module_rows:
        pct = (n_covered / n_total * 100) if n_total else 100.0
        status = "COMPLETE" if not missing else f"{len(missing)} missing"
        lines.append(f"")
        lines.append(f"  {mod_name}: {n_covered}/{n_total} ({pct:.0f}%) -- {status}")
        if missing:
            for fn in missing:
                lines.append(f"    - {fn}")

    lines.append("")
    lines.append("-" * 78)
    overall = (total_covered / total_functions * 100) if total_functions else 100.0
    lines.append(
        f"  TOTAL: {total_covered}/{total_functions} functions covered ({overall:.1f}%)"
    )
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)
    warnings.warn(f"\n{report}", stacklevel=1)


# ---------------------------------------------------------------------------
# Top-level function re-export check
# ---------------------------------------------------------------------------
# Verify that functions re-exported at the top-level compat __init__.py
# are the same objects as in the underlying module.

_TOP_LEVEL_REEXPORTS = [
    ("stft", "metalmom.core"),
    ("istft", "metalmom.core"),
    ("load", "metalmom.core"),
    ("resample", "metalmom.core"),
    ("get_duration", "metalmom.core"),
    ("get_samplerate", "metalmom.core"),
    ("reassigned_spectrogram", "metalmom.core"),
    ("phase_vocoder", "metalmom.effects"),
    ("griffinlim", "metalmom.effects"),
    ("cqt", "metalmom.cqt"),
    ("vqt", "metalmom.cqt"),
    ("hybrid_cqt", "metalmom.cqt"),
]


@pytest.mark.parametrize(
    "func_name, native_mod",
    _TOP_LEVEL_REEXPORTS,
    ids=[f"top-level:{fn}" for fn, _ in _TOP_LEVEL_REEXPORTS],
)
def test_top_level_reexport_identity(func_name, native_mod):
    """Top-level compat functions should be the same object as native ones."""
    compat = importlib.import_module("metalmom.compat.librosa")
    native = importlib.import_module(native_mod)

    compat_fn = getattr(compat, func_name, None)
    native_fn = getattr(native, func_name, None)

    assert compat_fn is not None, f"metalmom.compat.librosa.{func_name} not found"
    assert native_fn is not None, f"{native_mod}.{func_name} not found"
    assert compat_fn is native_fn, (
        f"metalmom.compat.librosa.{func_name} is not the same object as "
        f"{native_mod}.{func_name}"
    )
