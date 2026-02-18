"""Automated librosa signature compatibility verification tests.

Uses inspect.signature() to compare MetalMom compat function signatures against
real librosa functions.  For each function pair we check:

 1. Our function accepts at least the same parameter names as librosa
    (or has **kwargs to accept any).
 2. We do not require (no-default) a parameter that librosa has as optional.
 3. A coverage percentage is reported per function.

Intentional divergences (e.g. our function uses **kwargs where librosa has
many explicit params, or we renamed a param) are reported as warnings rather
than hard failures.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_param_info(func) -> dict[str, dict[str, Any]]:
    """Extract parameter metadata from *func*."""
    sig = inspect.signature(func)
    info: dict[str, dict[str, Any]] = {}
    for name, p in sig.parameters.items():
        info[name] = {
            "kind": p.kind,
            "default": p.default,
            "has_default": p.default is not inspect.Parameter.empty,
        }
    return info


def _has_var_keyword(params: dict[str, dict[str, Any]]) -> bool:
    """Return True if the signature contains **kwargs."""
    return any(v["kind"] == inspect.Parameter.VAR_KEYWORD for v in params.values())


def _import_func(module_path: str, func_name: str):
    """Import *func_name* from *module_path*, return None on failure."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name, None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Function pair table
# ---------------------------------------------------------------------------
# Each entry:  (librosa_module, func_name, metalmom_module, metalmom_func_name)
#
# The metalmom_module is the *underlying* module (not the compat shim) because
# the compat shim re-exports via plain import -- the signature is identical.

FUNCTION_PAIRS: list[tuple[str, str, str, str]] = [
    # -- core --
    ("librosa", "load", "metalmom.core", "load"),
    ("librosa", "stft", "metalmom.core", "stft"),
    ("librosa", "istft", "metalmom.core", "istft"),
    ("librosa", "resample", "metalmom.core", "resample"),
    ("librosa", "get_duration", "metalmom.core", "get_duration"),
    ("librosa", "get_samplerate", "metalmom.core", "get_samplerate"),
    ("librosa", "tone", "metalmom.core", "tone"),
    ("librosa", "chirp", "metalmom.core", "chirp"),
    ("librosa", "clicks", "metalmom.core", "clicks"),
    ("librosa", "stream", "metalmom.core", "stream"),
    ("librosa", "reassigned_spectrogram", "metalmom.core", "reassigned_spectrogram"),
    ("librosa", "phase_vocoder", "metalmom.effects", "phase_vocoder"),
    ("librosa", "griffinlim", "metalmom.effects", "griffinlim"),
    ("librosa", "amplitude_to_db", "metalmom.feature", "amplitude_to_db"),
    ("librosa", "power_to_db", "metalmom.feature", "power_to_db"),
    ("librosa", "db_to_amplitude", "metalmom.core", "db_to_amplitude"),
    ("librosa", "db_to_power", "metalmom.core", "db_to_power"),
    # -- feature --
    ("librosa.feature", "melspectrogram", "metalmom.feature", "melspectrogram"),
    ("librosa.feature", "mfcc", "metalmom.feature", "mfcc"),
    ("librosa.feature", "chroma_stft", "metalmom.feature", "chroma_stft"),
    ("librosa.feature", "chroma_cqt", "metalmom.feature", "chroma_cqt"),
    ("librosa.feature", "chroma_cens", "metalmom.feature", "chroma_cens"),
    ("librosa.feature", "chroma_vqt", "metalmom.feature", "chroma_vqt"),
    ("librosa.feature", "spectral_centroid", "metalmom.feature", "spectral_centroid"),
    ("librosa.feature", "spectral_bandwidth", "metalmom.feature", "spectral_bandwidth"),
    ("librosa.feature", "spectral_contrast", "metalmom.feature", "spectral_contrast"),
    ("librosa.feature", "spectral_rolloff", "metalmom.feature", "spectral_rolloff"),
    ("librosa.feature", "spectral_flatness", "metalmom.feature", "spectral_flatness"),
    ("librosa.feature", "rms", "metalmom.feature", "rms"),
    ("librosa.feature", "zero_crossing_rate", "metalmom.feature", "zero_crossing_rate"),
    ("librosa.feature", "tonnetz", "metalmom.feature", "tonnetz"),
    ("librosa.feature", "delta", "metalmom.feature", "delta"),
    ("librosa.feature", "poly_features", "metalmom.feature", "poly_features"),
    ("librosa.feature", "stack_memory", "metalmom.feature", "stack_memory"),
    ("librosa.feature", "tempo", "metalmom.feature", "tempo"),
    ("librosa.feature", "tempogram", "metalmom.feature", "tempogram"),
    ("librosa.feature", "fourier_tempogram", "metalmom.feature", "fourier_tempogram"),
    ("librosa", "pcen", "metalmom.feature", "pcen"),
    ("librosa.feature.inverse", "mel_to_audio", "metalmom.feature", "mel_to_audio"),
    ("librosa.feature.inverse", "mfcc_to_mel", "metalmom.feature", "mfcc_to_mel"),
    ("librosa.feature.inverse", "mfcc_to_audio", "metalmom.feature", "mfcc_to_audio"),
    # -- onset --
    ("librosa.onset", "onset_strength", "metalmom.onset", "onset_strength"),
    ("librosa.onset", "onset_detect", "metalmom.onset", "onset_detect"),
    # -- beat --
    ("librosa.beat", "beat_track", "metalmom.beat", "beat_track"),
    ("librosa.beat", "plp", "metalmom.beat", "plp"),
    # -- pitch --
    ("librosa", "yin", "metalmom.pitch", "yin"),
    ("librosa", "pyin", "metalmom.pitch", "pyin"),
    ("librosa", "piptrack", "metalmom.pitch", "piptrack"),
    ("librosa", "estimate_tuning", "metalmom.pitch", "estimate_tuning"),
    # -- effects --
    ("librosa.effects", "hpss", "metalmom.effects", "hpss"),
    ("librosa.effects", "harmonic", "metalmom.effects", "harmonic"),
    ("librosa.effects", "percussive", "metalmom.effects", "percussive"),
    ("librosa.effects", "time_stretch", "metalmom.effects", "time_stretch"),
    ("librosa.effects", "pitch_shift", "metalmom.effects", "pitch_shift"),
    ("librosa.effects", "trim", "metalmom.effects", "trim"),
    ("librosa.effects", "split", "metalmom.effects", "split"),
    ("librosa.effects", "preemphasis", "metalmom.effects", "preemphasis"),
    ("librosa.effects", "deemphasis", "metalmom.effects", "deemphasis"),
    # -- filters --
    ("librosa.filters", "mel", "metalmom.filters", "mel"),
    ("librosa.filters", "chroma", "metalmom.filters", "chroma"),
    ("librosa.filters", "constant_q", "metalmom.filters", "constant_q"),
    # -- decompose --
    ("librosa.decompose", "nn_filter", "metalmom.decompose", "nn_filter"),
    # -- sequence --
    ("librosa.sequence", "viterbi", "metalmom.sequence", "viterbi"),
    ("librosa.sequence", "viterbi_discriminative", "metalmom.sequence", "viterbi_discriminative"),
    ("librosa.sequence", "viterbi_binary", "metalmom.sequence", "viterbi_binary"),
    # -- display --
    ("librosa.display", "specshow", "metalmom.display", "specshow"),
    ("librosa.display", "waveshow", "metalmom.display", "waveshow"),
    # -- convert --
    ("librosa", "hz_to_midi", "metalmom.convert", "hz_to_midi"),
    ("librosa", "midi_to_hz", "metalmom.convert", "midi_to_hz"),
    ("librosa", "hz_to_note", "metalmom.convert", "hz_to_note"),
    ("librosa", "note_to_hz", "metalmom.convert", "note_to_hz"),
    ("librosa", "midi_to_note", "metalmom.convert", "midi_to_note"),
    ("librosa", "note_to_midi", "metalmom.convert", "note_to_midi"),
    ("librosa", "frames_to_time", "metalmom.convert", "frames_to_time"),
    ("librosa", "samples_to_time", "metalmom.convert", "samples_to_time"),
    ("librosa", "frames_to_samples", "metalmom.convert", "frames_to_samples"),
    ("librosa", "samples_to_frames", "metalmom.convert", "samples_to_frames"),
    ("librosa", "fft_frequencies", "metalmom.convert", "fft_frequencies"),
    ("librosa", "mel_frequencies", "metalmom.convert", "mel_frequencies"),
    # -- segment --
    ("librosa.segment", "recurrence_matrix", "metalmom.segment", "recurrence_matrix"),
    ("librosa.segment", "cross_similarity", "metalmom.segment", "cross_similarity"),
    ("librosa.segment", "agglomerative", "metalmom.segment", "agglomerative"),
]


# Build a human-readable test ID for each pair.
_IDS = [
    f"{lib_mod}.{fn}" for lib_mod, fn, _mm_mod, _mm_fn in FUNCTION_PAIRS
]


# ---------------------------------------------------------------------------
# Known intentional differences
# ---------------------------------------------------------------------------
# (librosa_module, func_name) -> set of parameter names we intentionally skip
# or rename.  These will be reported as warnings, not failures.

KNOWN_DIVERGENCES: dict[tuple[str, str], set[str]] = {
    # librosa.stft has window/dtype/pad_mode/out; we keep it minimal
    ("librosa", "stft"): {"window", "dtype", "pad_mode", "out"},
    # librosa.istft has window/dtype/out/n_fft; we keep it minimal
    ("librosa", "istft"): {"window", "dtype", "out", "n_fft"},
    # librosa.load has dtype/res_type; we accept via **kwargs
    ("librosa", "load"): {"dtype", "res_type"},
    # librosa.filters.mel has htk/norm/dtype -- we omit these
    ("librosa.filters", "mel"): {"htk", "norm", "dtype"},
    # librosa.filters.chroma has norm/dtype -- we omit
    ("librosa.filters", "chroma"): {"norm", "dtype"},
    # librosa.filters.constant_q many extras
    ("librosa.filters", "constant_q"): {"window", "filter_scale", "norm", "dtype", "gamma", "kwargs", "tuning"},
    # librosa.decompose.nn_filter uses rec/axis/kwargs -- we use k/metric/exclude_self/sr
    ("librosa.decompose", "nn_filter"): {"rec", "axis", "kwargs"},
    # librosa.sequence.viterbi: p_init vs initial, return_logp
    ("librosa.sequence", "viterbi"): {"p_init", "return_logp"},
    ("librosa.sequence", "viterbi_discriminative"): {"p_state", "p_init", "return_logp"},
    ("librosa.sequence", "viterbi_binary"): {"p_state", "p_init", "return_logp"},
    # librosa.segment.recurrence_matrix: different param naming
    ("librosa.segment", "recurrence_matrix"): {"data", "k", "width", "sym", "sparse", "mode", "bandwidth", "self", "axis", "full"},
    # librosa.segment.cross_similarity: different param naming
    ("librosa.segment", "cross_similarity"): {"data", "data_ref", "k", "sparse", "mode", "bandwidth", "full"},
    # librosa.segment.agglomerative: different param naming
    ("librosa.segment", "agglomerative"): {"data", "k", "clusterer", "axis"},
    # convert functions: param name differences (frequencies vs hz, notes vs midi)
    ("librosa", "hz_to_midi"): {"frequencies"},
    ("librosa", "midi_to_hz"): {"notes"},
    ("librosa", "hz_to_note"): {"frequencies", "kwargs"},
    ("librosa", "note_to_hz"): {"note", "kwargs"},
    ("librosa", "midi_to_note"): {"midi", "octave", "cents", "key", "unicode"},
    ("librosa", "note_to_midi"): {"note", "round_midi"},
    # display.specshow: many librosa-specific params
    ("librosa.display", "specshow"): {
        "tempo_min", "tempo_max", "tuning", "bins_per_octave", "key",
        "Sa", "mela", "thaat", "auto_aspect", "htk", "unicode",
        "intervals", "unison", "win_length",
    },
    # display.waveshow: some extras
    ("librosa.display", "waveshow"): {"marker", "where", "label", "transpose", "x_axis"},
    # griffinlim: momentum/init/random_state/n_fft/pad_mode/dtype
    ("librosa", "griffinlim"): {"momentum", "init", "random_state", "n_fft", "pad_mode", "dtype"},
    # reassigned_spectrogram: many librosa-specific params
    ("librosa", "reassigned_spectrogram"): {
        "S", "window", "reassign_frequencies", "reassign_times",
        "ref_power", "fill_nan", "clip", "dtype", "pad_mode",
    },
    # stream: offset/duration differences
    ("librosa", "stream"): {"offset", "duration"},
    # clicks: frames/hop_length/click param
    ("librosa", "clicks"): {"frames", "hop_length", "click"},
    # chroma_cqt: C/threshold/cqt_mode/window
    ("librosa.feature", "chroma_cqt"): {"C", "threshold", "cqt_mode", "window"},
    # chroma_cens: C/cqt_mode/window/norm/smoothing_window
    ("librosa.feature", "chroma_cens"): {"C", "cqt_mode", "window", "norm", "smoothing_window"},
    # chroma_vqt: V/intervals/threshold
    ("librosa.feature", "chroma_vqt"): {"V", "intervals", "threshold"},
    # mel_frequencies: htk flag not exposed
    ("librosa", "mel_frequencies"): {"htk"},
    # sequence.dtw: different param naming scheme
    ("librosa.sequence", "dtw"): set(),  # handled at module level -- see below
}


# Known cases where MetalMom requires a param that librosa has as optional.
# (librosa_module, func_name) -> set of parameter names that are intentionally
# required in MetalMom but optional in librosa.
KNOWN_REQUIRED_DIVERGENCES: dict[tuple[str, str], set[str]] = {
    # MetalMom get_duration requires a file path; librosa can also compute
    # from y/sr/S so path is optional there.
    ("librosa", "get_duration"): {"path"},
}


# ---------------------------------------------------------------------------
# Parametrised test: signature compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lib_mod, func_name, mm_mod, mm_func",
    FUNCTION_PAIRS,
    ids=_IDS,
)
def test_signature_param_coverage(lib_mod, func_name, mm_mod, mm_func):
    """Verify MetalMom exposes at least the same parameter names as librosa.

    * If our function has **kwargs it implicitly accepts any parameter name
      from librosa -- that counts as covered.
    * Parameters listed in KNOWN_DIVERGENCES are reported as warnings, not
      failures.
    * If we *require* (no default) a parameter that librosa has as optional,
      that is a hard failure.
    """
    librosa_func = _import_func(lib_mod, func_name)
    mm_func_obj = _import_func(mm_mod, mm_func)

    if librosa_func is None:
        pytest.skip(f"librosa function {lib_mod}.{func_name} not found in this version")
    if mm_func_obj is None:
        pytest.fail(f"MetalMom function {mm_mod}.{mm_func} could not be imported")

    lib_params = _get_param_info(librosa_func)
    mm_params = _get_param_info(mm_func_obj)
    mm_has_kwargs = _has_var_keyword(mm_params)

    # Parameter names in librosa (excluding *args and **kwargs themselves)
    lib_named = {
        name for name, info in lib_params.items()
        if info["kind"] not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    mm_named = {
        name for name, info in mm_params.items()
        if info["kind"] not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }

    known = KNOWN_DIVERGENCES.get((lib_mod, func_name), set())

    # --- coverage analysis ---
    covered = set()
    missing = set()
    for p in lib_named:
        if p in mm_named or mm_has_kwargs:
            covered.add(p)
        elif p in known:
            covered.add(p)  # known divergence counts as acknowledged
        else:
            missing.add(p)

    coverage_pct = (len(covered) / len(lib_named) * 100) if lib_named else 100.0

    # Emit warnings for known divergences that are truly absent
    known_absent = known & lib_named - mm_named
    if known_absent and not mm_has_kwargs:
        warnings.warn(
            f"[{lib_mod}.{func_name}] Known divergences (not in MetalMom explicit params): "
            f"{sorted(known_absent)}",
            stacklevel=1,
        )

    # --- hard failure: missing non-known parameters ---
    if missing:
        pytest.fail(
            f"{lib_mod}.{func_name}: {len(missing)} librosa param(s) not covered by "
            f"MetalMom {mm_mod}.{mm_func}: {sorted(missing)}  "
            f"(coverage {coverage_pct:.0f}%)"
        )


@pytest.mark.parametrize(
    "lib_mod, func_name, mm_mod, mm_func",
    FUNCTION_PAIRS,
    ids=_IDS,
)
def test_no_new_required_params(lib_mod, func_name, mm_mod, mm_func):
    """Ensure MetalMom does not require params that librosa treats as optional.

    If librosa has ``param=default`` (optional) and our version has ``param``
    (required, no default), callers porting from librosa will break.
    """
    librosa_func = _import_func(lib_mod, func_name)
    mm_func_obj = _import_func(mm_mod, mm_func)

    if librosa_func is None or mm_func_obj is None:
        pytest.skip("function not importable")

    lib_params = _get_param_info(librosa_func)
    mm_params = _get_param_info(mm_func_obj)

    known_req = KNOWN_REQUIRED_DIVERGENCES.get((lib_mod, func_name), set())

    violations: list[str] = []
    for name, mm_info in mm_params.items():
        if mm_info["kind"] in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Is this param required in MetalMom?
        if mm_info["has_default"]:
            continue
        # Does librosa have it as optional?
        lib_info = lib_params.get(name)
        if lib_info is not None and lib_info["has_default"]:
            if name in known_req:
                warnings.warn(
                    f"[{lib_mod}.{func_name}] Known required divergence: "
                    f"'{name}' is required in MetalMom but optional in librosa",
                    stacklevel=1,
                )
            else:
                violations.append(name)

    if violations:
        pytest.fail(
            f"{mm_mod}.{mm_func}: requires param(s) that librosa has as optional: "
            f"{sorted(violations)}"
        )


# ---------------------------------------------------------------------------
# Summary test: aggregate coverage report
# ---------------------------------------------------------------------------

def test_overall_coverage_report():
    """Print an aggregate coverage summary for all function pairs.

    This test always passes -- its purpose is to produce a readable report
    via pytest output (use ``-s`` or ``--tb=short`` to see it).
    """
    total_lib_params = 0
    total_covered = 0
    rows: list[tuple[str, int, int, float]] = []

    for lib_mod, func_name, mm_mod, mm_func in FUNCTION_PAIRS:
        librosa_func = _import_func(lib_mod, func_name)
        mm_func_obj = _import_func(mm_mod, mm_func)
        if librosa_func is None or mm_func_obj is None:
            continue

        lib_params = _get_param_info(librosa_func)
        mm_params = _get_param_info(mm_func_obj)
        mm_has_kwargs = _has_var_keyword(mm_params)

        lib_named = {
            name for name, info in lib_params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        mm_named = {
            name for name, info in mm_params.items()
            if info["kind"] not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        known = KNOWN_DIVERGENCES.get((lib_mod, func_name), set())

        covered = 0
        for p in lib_named:
            if p in mm_named or mm_has_kwargs or p in known:
                covered += 1

        pct = (covered / len(lib_named) * 100) if lib_named else 100.0
        rows.append((f"{lib_mod}.{func_name}", len(lib_named), covered, pct))
        total_lib_params += len(lib_named)
        total_covered += covered

    # Build report
    lines = [
        "",
        "=" * 78,
        "  librosa Signature Coverage Report",
        "=" * 78,
        f"  {'Function':<50} {'librosa':>7} {'covered':>7} {'%':>6}",
        "-" * 78,
    ]
    for fn, n_lib, n_cov, pct in sorted(rows, key=lambda r: r[3]):
        marker = " !!" if pct < 100 else "   "
        lines.append(f"{marker}{fn:<49} {n_lib:>7} {n_cov:>7} {pct:>5.0f}%")
    lines.append("-" * 78)
    overall = (total_covered / total_lib_params * 100) if total_lib_params else 100.0
    lines.append(
        f"   {'TOTAL':<49} {total_lib_params:>7} {total_covered:>7} {overall:>5.1f}%"
    )
    lines.append(f"   Functions checked: {len(rows)}")
    lines.append("=" * 78)

    report = "\n".join(lines)
    # Use print so it shows with -s; also emit as a warning so it shows in
    # the short summary without -s.
    print(report)
    warnings.warn(f"\n{report}", stacklevel=1)


# ---------------------------------------------------------------------------
# Extra: verify compat shim re-exports resolve to the same objects
# ---------------------------------------------------------------------------

_COMPAT_REIMPORT_PAIRS: list[tuple[str, str, str, str]] = [
    ("metalmom.compat.librosa.core", "stft", "metalmom.core", "stft"),
    ("metalmom.compat.librosa.core", "istft", "metalmom.core", "istft"),
    ("metalmom.compat.librosa.core", "load", "metalmom.core", "load"),
    ("metalmom.compat.librosa.feature", "melspectrogram", "metalmom.feature", "melspectrogram"),
    ("metalmom.compat.librosa.feature", "mfcc", "metalmom.feature", "mfcc"),
    ("metalmom.compat.librosa.onset", "onset_strength", "metalmom.onset", "onset_strength"),
    ("metalmom.compat.librosa.beat", "beat_track", "metalmom.beat", "beat_track"),
    ("metalmom.compat.librosa.effects", "hpss", "metalmom.effects", "hpss"),
    ("metalmom.compat.librosa.effects", "trim", "metalmom.effects", "trim"),
    ("metalmom.compat.librosa.filters", "mel", "metalmom.filters", "mel"),
    ("metalmom.compat.librosa.sequence", "viterbi", "metalmom.sequence", "viterbi"),
    ("metalmom.compat.librosa.display", "specshow", "metalmom.display", "specshow"),
    ("metalmom.compat.librosa.segment", "recurrence_matrix", "metalmom.segment", "recurrence_matrix"),
    ("metalmom.compat.librosa.pitch", "yin", "metalmom.pitch", "yin"),
    ("metalmom.compat.librosa.decompose", "nn_filter", "metalmom.decompose", "nn_filter"),
]

_REIMPORT_IDS = [
    f"compat:{cm}.{fn}" for cm, fn, _, _ in _COMPAT_REIMPORT_PAIRS
]


@pytest.mark.parametrize(
    "compat_mod, func_name, native_mod, native_func",
    _COMPAT_REIMPORT_PAIRS,
    ids=_REIMPORT_IDS,
)
def test_compat_reexport_identity(compat_mod, func_name, native_mod, native_func):
    """The compat shim should re-export the exact same function object."""
    compat_fn = _import_func(compat_mod, func_name)
    native_fn = _import_func(native_mod, native_func)
    if compat_fn is None:
        pytest.fail(f"Cannot import {compat_mod}.{func_name}")
    if native_fn is None:
        pytest.fail(f"Cannot import {native_mod}.{native_func}")
    assert compat_fn is native_fn, (
        f"{compat_mod}.{func_name} is not the same object as "
        f"{native_mod}.{native_func}"
    )
