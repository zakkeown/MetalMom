#!/usr/bin/env python3
"""Check API surface drift between librosa/madmom and MetalMom compat shims.

Walks the public API of librosa and madmom, compares against what
metalmom.compat exposes, and writes a drift report to api-drift-report.json.

Exit codes:
    0  No NEW drift detected (pre-existing gaps are fine)
    1  New uncovered APIs found since last baseline

Usage:
    python scripts/ci/check_api_surface.py [--baseline api-drift-report.json]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Modules to audit: (upstream_module, compat_module)
# ---------------------------------------------------------------------------

LIBROSA_MODULES = [
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
    ("librosa.convert", "metalmom.compat.librosa.convert"),
    ("librosa.pitch", "metalmom.compat.librosa.pitch"),
]

MADMOM_MODULES = [
    ("madmom.audio.signal", "metalmom.compat.madmom.audio.signal"),
    ("madmom.audio.stft", "metalmom.compat.madmom.audio.stft"),
    ("madmom.audio.spectrogram", "metalmom.compat.madmom.audio.spectrogram"),
    ("madmom.features.onsets", "metalmom.compat.madmom.features.onsets"),
    ("madmom.features.beats", "metalmom.compat.madmom.features.beats"),
    ("madmom.features.downbeats", "metalmom.compat.madmom.features.downbeats"),
    ("madmom.features.key", "metalmom.compat.madmom.features.key"),
    ("madmom.features.chords", "metalmom.compat.madmom.features.chords"),
    ("madmom.features.tempo", "metalmom.compat.madmom.features.tempo"),
    ("madmom.evaluation.onsets", "metalmom.compat.madmom.evaluation.onsets"),
    ("madmom.evaluation.beats", "metalmom.compat.madmom.evaluation.beats"),
    ("madmom.evaluation.tempo", "metalmom.compat.madmom.evaluation.tempo"),
]

# ---------------------------------------------------------------------------
# Names to always skip (internals, typing, decorators, etc.)
# ---------------------------------------------------------------------------

GLOBAL_SKIP = {
    # Python dunder / module internals
    "__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
    "__name__", "__package__", "__path__", "__spec__", "__version__",
    "__all__", "__getattr__",
    # Common typing re-exports that leak into dir()
    "Any", "Callable", "Collection", "Dict", "Iterable", "List", "Literal",
    "Optional", "Sequence", "Tuple", "TypeVar", "Union",
    # Common decorator / utility re-exports
    "overload", "cache", "jit", "deprecated", "decorator",
    # Exception / error classes re-exported across submodules
    "LibrosaError", "ParameterError",
    # itertools re-exports
    "product",
    # Deprecation helper class
    "Deprecated",
}

# Per-module skip sets -- functions we intentionally do not implement.
# Matches the existing SKIP_FUNCTIONS from test_compat_completeness.py.
MODULE_SKIP: dict[str, set[str]] = {
    "librosa": {
        # Submodule names
        "beat", "core", "decompose", "display", "effects", "feature",
        "filters", "onset", "segment", "sequence", "util", "cache",
        "convert",
        # Version / utility
        "show_versions", "cite",
        # Example data
        "example", "ex",
        # FFT backend
        "get_fftlib", "set_fftlib",
        # Deprecated / very niche frequency weighting
        "A_weighting", "B_weighting", "C_weighting", "D_weighting",
        "Z_weighting", "multi_frequency_weighting", "frequency_weighting",
        "perceptual_weighting",
        # Indian music theory
        "hz_to_svara_c", "hz_to_svara_h", "midi_to_svara_c", "midi_to_svara_h",
        "note_to_svara_c", "note_to_svara_h", "mela_to_degrees", "mela_to_svara",
        "list_mela", "list_thaat", "thaat_to_degrees",
        "key_to_degrees", "key_to_notes",
        # FJS notation
        "hz_to_fjs", "interval_to_fjs",
        # Interval tuning
        "plimit_intervals", "pythagorean_intervals", "interval_frequencies",
        # Misc niche
        "fifths_to_note", "mu_compress", "mu_expand",
        "icqt", "griffinlim_cqt", "pseudo_cqt", "iirt",
        "A4_to_tuning", "tuning_to_A4", "fmt",
        "blocks_to_frames", "blocks_to_samples", "blocks_to_time",
    },
    "librosa.feature": set(),
    "librosa.feature.inverse": {
        "Optional", "db_to_power", "expand_to", "get_fftlib", "tiny",
        "mel_to_stft", "nnls", "griffinlim",
    },
    "librosa.onset": {
        "melspectrogram", "onset_backtrack", "onset_strength_multi",
    },
    "librosa.beat": {
        "moved", "fourier_tempogram", "tempo",
    },
    "librosa.effects": {
        "remix",
    },
    "librosa.filters": {
        "hz_to_midi", "hz_to_octs", "midi_to_hz", "note_to_hz",
        "get_window", "wavelet", "wavelet_lengths",
        "window_bandwidth", "window_sumsquare",
        "constant_q_lengths", "cq_to_chroma", "mr_frequencies",
        "diagonal_filter",
    },
    "librosa.decompose": {
        "decompose", "hpss", "median_filter",
    },
    "librosa.sequence": {
        "cdist", "expand_to", "get_window", "is_positive_int",
        "pad_center", "tiny", "fill_off_diagonal", "dtw_backtracking",
    },
    "librosa.display": {
        "cmap", "mcm", "rename_kw",
        # Matplotlib formatter classes (internal display helpers)
        "AdaptiveWaveplot", "ChromaFJSFormatter", "ChromaFormatter",
        "ChromaSvaraFormatter", "FJSFormatter", "LogHzFormatter",
        "NoteFormatter", "SvaraFormatter", "TimeFormatter",
        "TonnetzFormatter",
    },
    "librosa.segment": {
        "diagonal_filter",
    },
}


# ---------------------------------------------------------------------------
# API extraction
# ---------------------------------------------------------------------------

def get_public_names(module_name: str) -> list[str]:
    """Return sorted public callable/class names from a module."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return []

    skip = GLOBAL_SKIP | MODULE_SKIP.get(module_name, set())
    names = []
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        if name in skip:
            continue
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        # Keep callables (functions, classes) -- skip plain modules/constants
        if callable(obj) or inspect.isclass(obj):
            names.append(name)
    return names


def get_compat_names(module_name: str) -> set[str]:
    """Return all names available in a compat module."""
    try:
        mod = importlib.import_module(module_name)
        return {n for n in dir(mod) if not n.startswith("_")}
    except ImportError:
        return set()


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def check_modules(module_pairs: list[tuple[str, str]], library: str) -> dict:
    """Check API coverage for a list of (upstream, compat) module pairs.

    Returns a dict keyed by upstream module name with:
        - upstream_count: total public names
        - covered: list of covered names
        - missing: list of missing names
    """
    results: dict[str, dict] = {}
    for upstream_mod, compat_mod in module_pairs:
        upstream_names = get_public_names(upstream_mod)
        compat_names = get_compat_names(compat_mod)

        covered = [n for n in upstream_names if n in compat_names]
        missing = [n for n in upstream_names if n not in compat_names]

        results[upstream_mod] = {
            "upstream_count": len(upstream_names),
            "covered": covered,
            "missing": missing,
        }
    return results


def compute_drift(current_missing: dict[str, list[str]],
                  baseline_missing: dict[str, list[str]]) -> dict[str, list[str]]:
    """Compute new missing names that weren't in the baseline.

    Returns a dict of module -> list of NEW missing names.
    """
    new_drift: dict[str, list[str]] = {}
    for module, missing in current_missing.items():
        baseline = set(baseline_missing.get(module, []))
        new_items = [n for n in missing if n not in baseline]
        if new_items:
            new_drift[module] = new_items
    return new_drift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Check API surface drift")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to previous api-drift-report.json for delta detection",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("api-drift-report.json"),
        help="Output path for the report (default: api-drift-report.json)",
    )
    args = parser.parse_args()

    # -- Check librosa --
    librosa_available = True
    try:
        import librosa  # noqa: F401
        librosa_version = librosa.__version__
    except ImportError:
        librosa_available = False
        librosa_version = "N/A"
        print("WARNING: librosa not installed, skipping librosa API check")

    # -- Check madmom --
    madmom_available = True
    try:
        import madmom  # noqa: F401
        madmom_version = madmom.__version__
    except (ImportError, AttributeError, Exception) as e:
        madmom_available = False
        madmom_version = "N/A"
        print(f"WARNING: madmom not available ({e}), skipping madmom API check")

    # -- Run checks --
    librosa_results = check_modules(LIBROSA_MODULES, "librosa") if librosa_available else {}
    madmom_results = check_modules(MADMOM_MODULES, "madmom") if madmom_available else {}

    all_results = {**librosa_results, **madmom_results}

    # Flatten missing per module
    current_missing: dict[str, list[str]] = {}
    for mod, info in all_results.items():
        if info["missing"]:
            current_missing[mod] = info["missing"]

    # -- Load baseline if provided --
    baseline_missing: dict[str, list[str]] = {}
    if args.baseline and args.baseline.exists():
        try:
            baseline = json.loads(args.baseline.read_text())
            baseline_missing = baseline.get("missing_by_module", {})
            print(f"Loaded baseline from {args.baseline}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Could not parse baseline: {e}")

    # -- Compute drift --
    new_drift = compute_drift(current_missing, baseline_missing)

    # -- Compute summary stats --
    total_upstream = sum(info["upstream_count"] for info in all_results.values())
    total_covered = sum(len(info["covered"]) for info in all_results.values())
    total_missing = sum(len(info["missing"]) for info in all_results.values())
    total_new = sum(len(v) for v in new_drift.values())

    # -- Build report --
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "librosa_version": librosa_version,
        "madmom_version": madmom_version,
        "summary": {
            "total_upstream_apis": total_upstream,
            "total_covered": total_covered,
            "total_missing": total_missing,
            "total_new_drift": total_new,
            "coverage_pct": round(total_covered / total_upstream * 100, 1) if total_upstream else 100.0,
        },
        "missing_by_module": current_missing,
        "new_drift_by_module": new_drift,
        "modules": {
            mod: {
                "upstream_count": info["upstream_count"],
                "covered_count": len(info["covered"]),
                "missing_count": len(info["missing"]),
                "missing": info["missing"],
            }
            for mod, info in all_results.items()
        },
    }

    # -- Write report --
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nAPI drift report written to {args.output}")

    # -- Print summary --
    print(f"\n{'=' * 60}")
    print(f"  API Drift Report")
    print(f"{'=' * 60}")
    print(f"  librosa {librosa_version}  |  madmom {madmom_version}")
    print(f"  Coverage: {total_covered}/{total_upstream} ({report['summary']['coverage_pct']}%)")
    print(f"  Pre-existing gaps: {total_missing}")
    print(f"  NEW drift items:   {total_new}")
    print(f"{'=' * 60}")

    if new_drift:
        print("\nNEW uncovered APIs:")
        for mod, names in sorted(new_drift.items()):
            print(f"\n  {mod}:")
            for name in names:
                print(f"    - {name}")
        print()

    if current_missing and not new_drift:
        print("\nPre-existing gaps (not new):")
        for mod, names in sorted(current_missing.items()):
            print(f"  {mod}: {len(names)} missing")
        print()

    # Exit 1 if there is new drift
    if total_new > 0:
        print(f"FAIL: {total_new} new uncovered API(s) detected")
        return 1

    print("OK: No new API drift detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
