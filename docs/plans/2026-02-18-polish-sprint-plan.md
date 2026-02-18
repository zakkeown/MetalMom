# Polish Sprint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the last xfail test (griffinlim_cqt cffi gap), add Python type stubs for IDE support, and create a benchmark suite with JSON output.

**Architecture:** Three independent tasks executed sequentially. Task 1 is a one-line cffi fix. Task 2 creates `.pyi` stub files for all public Python modules. Task 3 creates a `benchmarks/` directory with a runner script that times core operations against librosa.

**Tech Stack:** Python (cffi, numpy, typing), pytest, librosa (for benchmark comparison)

---

## Task 1: Fix `mm_griffinlim_cqt` cffi Gap

**Files:**
- Modify: `python/metalmom/_native.py` (after line ~380, the `/* Griffin-Lim */` section)
- Modify: `Tests/test_bridge_integration.py` (line ~340, remove xfail decorator)

**Step 1: Add the cffi cdef declaration**

In `python/metalmom/_native.py`, find the `/* Griffin-Lim */` section (around line 376). After the existing `mm_griffinlim` declaration, add:

```c
    int32_t mm_griffinlim_cqt(mm_context ctx, const float* mag_data, int64_t mag_count,
                               int32_t n_bins, int32_t n_frames, int32_t sample_rate,
                               int32_t n_iter, int32_t hop_length,
                               float f_min, int32_t bins_per_octave, MMBuffer* out);
```

This matches the `@_cdecl` signature in `Sources/MetalMomBridge/Bridge.swift:2804-2817`.

**Step 2: Remove the xfail marker from the test**

In `Tests/test_bridge_integration.py`, remove the `@pytest.mark.xfail(...)` decorator from `test_griffinlim_cqt_roundtrip` (around line 340-345). Keep the test function itself unchanged.

**Step 3: Build the dylib and run the test**

Run: `swift build -c release && ./scripts/build_dylib.sh`
Then: `.venv/bin/pytest Tests/test_bridge_integration.py::test_griffinlim_cqt_roundtrip -v`
Expected: PASS

**Step 4: Commit**

```bash
git add python/metalmom/_native.py Tests/test_bridge_integration.py
git commit -m "fix: add mm_griffinlim_cqt to cffi cdef (closes last xfail)"
```

---

## Task 2: Python Type Stubs (Public API)

**Files to create:**
- `python/metalmom/py.typed` (empty PEP 561 marker)
- `python/metalmom/__init__.pyi`
- `python/metalmom/core.pyi`
- `python/metalmom/feature.pyi`
- `python/metalmom/onset.pyi`
- `python/metalmom/beat.pyi`
- `python/metalmom/pitch.pyi`
- `python/metalmom/effects.pyi`
- `python/metalmom/cqt.pyi`
- `python/metalmom/decompose.pyi`
- `python/metalmom/segment.pyi`
- `python/metalmom/sequence.pyi`
- `python/metalmom/convert.pyi`
- `python/metalmom/display.pyi`
- `python/metalmom/filters.pyi`
- `python/metalmom/evaluate.pyi`
- `python/metalmom/key.pyi`
- `python/metalmom/chord.pyi`
- `python/metalmom/transcribe.pyi`

**Approach:** Read each `.py` file, extract all public function signatures (no leading `_`), and write corresponding `.pyi` stubs with proper numpy typing. Use `numpy.typing.NDArray[numpy.float32]` for audio arrays, `numpy.typing.NDArray[numpy.floating[Any]]` for general float arrays.

**Step 1: Create `py.typed` marker**

Create an empty file at `python/metalmom/py.typed`. This is the PEP 561 marker that tells type checkers the package ships inline types.

**Step 2: Create stub files**

For each module listed above, read the corresponding `.py` file and create a `.pyi` stub with:
- All public function signatures with type annotations
- `from __future__ import annotations` at the top
- Proper imports: `import numpy as np`, `import numpy.typing as npt`, `from typing import ...`
- Use `...` as the function body (standard stub convention)
- Match default values exactly from the source

**Conventions for type annotations:**
- Audio signals (1D float32): `npt.NDArray[np.float32]`
- Spectrograms (2D): `npt.NDArray[np.float32]`
- Frequency arrays: `npt.NDArray[np.float64]`
- Integer arrays (frame indices): `npt.NDArray[np.intp]`
- Optional parameters with None default: `X | None = None`
- Return tuples: `tuple[npt.NDArray[np.float32], int]`
- `**kwargs`: keep as `**kwargs: Any`

**Step 3: Verify stubs don't break imports**

Run: `.venv/bin/python -c "import metalmom; print(metalmom.stft)"`
Expected: function reference printed, no import errors

**Step 4: Run full test suite to confirm nothing breaks**

Run: `.venv/bin/pytest Tests/ -x -q`
Expected: all tests pass (no regressions from adding .pyi files)

**Step 5: Commit**

```bash
git add python/metalmom/py.typed python/metalmom/*.pyi
git commit -m "feat: add PEP 561 type stubs for all public modules"
```

---

## Task 3: Benchmark Suite

**Files to create:**
- `benchmarks/run_benchmarks.py`
- `benchmarks/__init__.py` (empty, makes it importable)
- `benchmarks/results/.gitkeep`

**Step 1: Create the benchmarks directory structure**

```bash
mkdir -p benchmarks/results
touch benchmarks/__init__.py benchmarks/results/.gitkeep
```

**Step 2: Write `benchmarks/run_benchmarks.py`**

The script should:
1. Generate test signals at multiple durations (1s, 5s, 30s at 44100 Hz)
2. Time MetalMom operations: stft, melspectrogram, mfcc, cqt, onset_strength, beat_track
3. Time librosa equivalents on the same signals (if librosa available)
4. Collect results into a dict with:
   - `timestamp` (ISO 8601)
   - `hardware` (platform info, CPU, RAM)
   - `python_version`
   - `metalmom_version`
   - `results`: list of `{operation, duration_s, input_size, time_ms, iterations}`
   - `librosa_results`: same structure (or null if librosa unavailable)
5. Write JSON to `benchmarks/results/bench_YYYYMMDD_HHMMSS.json`
6. Print a summary table to stdout
7. Support `--compare <path>` flag to diff against a previous JSON report

**Timing pattern:**
```python
import time

def benchmark(fn, *args, iterations=5, warmup=1, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "iterations": iterations,
    }
```

**Operations to benchmark:**
| Operation | MetalMom call | librosa equivalent |
|-----------|---------------|-------------------|
| STFT | `metalmom.stft(y, n_fft=2048)` | `librosa.stft(y, n_fft=2048)` |
| Mel spectrogram | `metalmom.melspectrogram(y=y, sr=sr)` | `librosa.feature.melspectrogram(y=y, sr=sr)` |
| MFCC | `metalmom.mfcc(y=y, sr=sr)` | `librosa.feature.mfcc(y=y, sr=sr)` |
| CQT | `metalmom.cqt(y, sr=sr)` | `librosa.cqt(y, sr=sr)` |
| Onset strength | `metalmom.onset_strength(y=y, sr=sr)` | `librosa.onset.onset_strength(y=y, sr=sr)` |
| Beat tracking | `metalmom.beat_track(y=y, sr=sr)` | `librosa.beat.beat_track(y=y, sr=sr)` |

**Signal generation:**
```python
def make_signal(duration_s, sr=44100):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr
```

**Step 3: Test the benchmark runner**

Run: `.venv/bin/python benchmarks/run_benchmarks.py`
Expected: JSON file created in `benchmarks/results/`, summary table printed to stdout

**Step 4: Test the comparison mode**

Run: `.venv/bin/python benchmarks/run_benchmarks.py --compare benchmarks/results/<previous>.json`
Expected: side-by-side comparison printed showing speedup/slowdown percentages

**Step 5: Commit**

```bash
git add benchmarks/
git commit -m "feat: add benchmark suite with JSON output and librosa comparison"
```
