#!/usr/bin/env python3
"""MetalMom benchmark suite — times core operations and compares against librosa."""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(duration_s, sr=44100):
    """Generate a 440 Hz sine wave test signal."""
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


def benchmark(fn, *args, iterations=5, warmup=1, **kwargs):
    """Time a function with warmup iterations."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return {
        "mean_ms": round(sum(times) / len(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Operation definitions
# ---------------------------------------------------------------------------

DURATIONS = [1, 5, 30]

def _metalmom_operations(mm):
    """Return list of (name, callable(y, sr)) for MetalMom."""
    return [
        ("STFT",             lambda y, sr: mm.stft(y, n_fft=2048)),
        ("Mel spectrogram",  lambda y, sr: mm.melspectrogram(y=y, sr=sr)),
        ("MFCC",             lambda y, sr: mm.mfcc(y=y, sr=sr)),
        ("CQT",              lambda y, sr: mm.cqt(y, sr=sr)),
        ("Onset strength",   lambda y, sr: mm.onset_strength(y=y, sr=sr)),
        ("Beat tracking",    lambda y, sr: mm.beat_track(y=y, sr=sr)),
    ]


def _librosa_operations(lr):
    """Return list of (name, callable(y, sr)) for librosa."""
    return [
        ("STFT",             lambda y, sr: lr.stft(y, n_fft=2048)),
        ("Mel spectrogram",  lambda y, sr: lr.feature.melspectrogram(y=y, sr=sr)),
        ("MFCC",             lambda y, sr: lr.feature.mfcc(y=y, sr=sr)),
        ("CQT",              lambda y, sr: lr.cqt(y, sr=sr)),
        ("Onset strength",   lambda y, sr: lr.onset.onset_strength(y=y, sr=sr)),
        ("Beat tracking",    lambda y, sr: lr.beat.beat_track(y=y, sr=sr)),
    ]


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

def run_metalmom_benchmarks(mm, signals, iterations=5, warmup=1):
    """Benchmark all MetalMom operations across signal durations."""
    ops = _metalmom_operations(mm)
    results = []
    for dur_s, (y, sr) in zip(DURATIONS, signals):
        for name, fn in ops:
            print(f"  MetalMom | {name:20s} | {dur_s:3d}s ...", end="", flush=True)
            try:
                timing = benchmark(fn, y, sr, iterations=iterations, warmup=warmup)
                timing["operation"] = name
                timing["duration_s"] = dur_s
                results.append(timing)
                print(f"  {timing['mean_ms']:10.3f} ms")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                results.append({
                    "operation": name,
                    "duration_s": dur_s,
                    "mean_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "iterations": iterations,
                    "error": str(exc),
                })
    return results


def run_librosa_benchmarks(lr, signals, iterations=5, warmup=1):
    """Benchmark all librosa operations across signal durations."""
    ops = _librosa_operations(lr)
    results = []
    for dur_s, (y, sr) in zip(DURATIONS, signals):
        for name, fn in ops:
            print(f"  librosa  | {name:20s} | {dur_s:3d}s ...", end="", flush=True)
            try:
                timing = benchmark(fn, y, sr, iterations=iterations, warmup=warmup)
                timing["operation"] = name
                timing["duration_s"] = dur_s
                results.append(timing)
                print(f"  {timing['mean_ms']:10.3f} ms")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                results.append({
                    "operation": name,
                    "duration_s": dur_s,
                    "mean_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "iterations": iterations,
                    "error": str(exc),
                })
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(mm_results, lr_results):
    """Print a formatted comparison table."""
    print()
    print("MetalMom Benchmark Results")
    print("==========================")
    header = f"{'Operation':20s} | {'Duration':>8s} | {'MetalMom (ms)':>14s} | {'librosa (ms)':>13s} | {'Speedup':>7s}"
    sep    = f"{'-'*20}-+-{'-'*8}-+-{'-'*14}-+-{'-'*13}-+-{'-'*7}"
    print(header)
    print(sep)

    # Index librosa results by (operation, duration_s)
    lr_map = {}
    if lr_results:
        for r in lr_results:
            lr_map[(r["operation"], r["duration_s"])] = r

    for r in mm_results:
        op = r["operation"]
        dur = r["duration_s"]
        mm_val = r["mean_ms"]
        lr_entry = lr_map.get((op, dur))
        lr_val = lr_entry.get("mean_ms") if lr_entry else None

        mm_str = f"{mm_val:14.3f}" if mm_val is not None else "         FAILED"
        lr_str = f"{lr_val:13.3f}" if lr_val is not None else "          N/A"

        if mm_val is not None and lr_val is not None and mm_val > 0:
            speedup = lr_val / mm_val
            sp_str = f"{speedup:6.1f}x"
        else:
            sp_str = "    N/A"

        print(f"{op:20s} | {dur:>6d}s  | {mm_str} | {lr_str} | {sp_str}")

    print()


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def compare_results(current_path, previous_path):
    """Compare two benchmark JSON files and print a delta table."""
    with open(current_path) as f:
        current = json.load(f)
    with open(previous_path) as f:
        previous = json.load(f)

    print()
    print(f"Comparison: current vs previous ({previous_path})")
    header = f"{'Operation':20s} | {'Duration':>8s} | {'Previous (ms)':>14s} | {'Current (ms)':>13s} | {'Change':>8s}"
    sep    = f"{'-'*20}-+-{'-'*8}-+-{'-'*14}-+-{'-'*13}-+-{'-'*8}"
    print(header)
    print(sep)

    # Index previous results
    prev_map = {}
    for r in previous.get("results", []):
        prev_map[(r["operation"], r["duration_s"])] = r

    for r in current.get("results", []):
        op = r["operation"]
        dur = r["duration_s"]
        cur_val = r["mean_ms"]
        prev_entry = prev_map.get((op, dur))
        prev_val = prev_entry.get("mean_ms") if prev_entry else None

        prev_str = f"{prev_val:14.3f}" if prev_val is not None else "           N/A"
        cur_str  = f"{cur_val:13.3f}" if cur_val is not None else "          N/A"

        if cur_val is not None and prev_val is not None and prev_val > 0:
            pct = ((cur_val - prev_val) / prev_val) * 100
            sign = "+" if pct >= 0 else ""
            chg_str = f"{sign}{pct:.1f}%"
        else:
            chg_str = "     N/A"

        print(f"{op:20s} | {dur:>6d}s  | {prev_str} | {cur_str} | {chg_str:>8s}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MetalMom benchmark suite")
    parser.add_argument(
        "--compare", metavar="PATH",
        help="Path to a previous benchmark JSON to compare against",
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of timed iterations per benchmark (default: 5)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup iterations per benchmark (default: 1)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Import MetalMom
    # ------------------------------------------------------------------
    try:
        import metalmom as mm
    except ImportError:
        print("ERROR: Could not import metalmom. Build the dylib first:")
        print("  swift build -c release && ./scripts/build_dylib.sh")
        sys.exit(1)

    mm_version = getattr(mm, "__version__", "unknown")

    # ------------------------------------------------------------------
    # Try to import librosa (optional)
    # ------------------------------------------------------------------
    lr = None
    try:
        import librosa
        lr = librosa
        print(f"librosa {librosa.__version__} found -- will benchmark for comparison.")
    except ImportError:
        print("librosa not available -- skipping librosa benchmarks.")

    # ------------------------------------------------------------------
    # Generate test signals
    # ------------------------------------------------------------------
    print(f"\nGenerating test signals at {DURATIONS} seconds ...")
    signals = [make_signal(d) for d in DURATIONS]

    # ------------------------------------------------------------------
    # Run MetalMom benchmarks
    # ------------------------------------------------------------------
    print(f"\nRunning MetalMom benchmarks (iterations={args.iterations}, warmup={args.warmup}):")
    mm_results = run_metalmom_benchmarks(mm, signals, iterations=args.iterations, warmup=args.warmup)

    # ------------------------------------------------------------------
    # Run librosa benchmarks (if available)
    # ------------------------------------------------------------------
    lr_results = None
    if lr is not None:
        print(f"\nRunning librosa benchmarks (iterations={args.iterations}, warmup={args.warmup}):")
        lr_results = run_librosa_benchmarks(lr, signals, iterations=args.iterations, warmup=args.warmup)

    # ------------------------------------------------------------------
    # Collect report
    # ------------------------------------------------------------------
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "python_version": platform.python_version(),
        "metalmom_version": mm_version,
        "results": mm_results,
        "librosa_results": lr_results,
    }

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"bench_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults written to {out_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_summary(mm_results, lr_results)

    # ------------------------------------------------------------------
    # Comparison mode
    # ------------------------------------------------------------------
    if args.compare:
        compare_results(str(out_path), args.compare)

    return str(out_path)


if __name__ == "__main__":
    main()
