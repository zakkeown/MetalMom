# MetalMom Polish Sprint — 2026-02-18

## Overview

Three tasks to polish the project after completing all 11 implementation phases + stress tests.

## Task 1: Fix `mm_griffinlim_cqt` cffi Gap

The `mm_griffinlim_cqt` function is exported via `@_cdecl` in `Bridge.swift` but missing from the cffi `cdef` block in `_native.py`. One xfail test remains because of this.

**Changes:**
- Add cdef declaration to `python/metalmom/_native.py`
- Add Python wrapper in the appropriate module
- Remove xfail marker from `Tests/test_bridge_integration.py`
- Verify test passes

## Task 2: Python Type Stubs (Public API)

Add `.pyi` stub files for IDE autocomplete. Scope: public API only (not `_native.py` or `_buffer.py`).

**Files to create:**
- `python/metalmom/py.typed` — PEP 561 marker
- `python/metalmom/core.pyi`
- `python/metalmom/feature.pyi`
- `python/metalmom/compat/librosa.pyi`
- `python/metalmom/compat/madmom.pyi` (if applicable)
- `python/metalmom/__init__.pyi`

**Conventions:**
- Use `numpy.typing.NDArray[numpy.float32]` for audio arrays
- Match all existing function signatures exactly
- Include docstring summaries where helpful

## Task 3: Benchmark Suite

Create `benchmarks/` directory with a runner script that times core operations and outputs JSON reports.

**Operations to benchmark:**
- STFT, mel spectrogram, MFCC, CQT
- Onset detection, beat tracking
- Multiple input sizes: 1s, 5s, 30s at 44100 Hz

**Output:** JSON report with timestamps, hardware info, per-operation timing.

**Comparison:** Optional `--compare` flag to diff against a previous report.

**Scope:** CPU-only timing, librosa comparison. No CI integration yet (future work).
