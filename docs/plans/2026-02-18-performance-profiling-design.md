# Performance Profiling Deep-Dive — Design

**Date:** 2026-02-18
**Goal:** Add OSSignposter instrumentation to MetalMom's Swift engine, profile a real-song analysis pipeline on M3 Max, identify the top bottlenecks, and fix them.

## Context

MetalMom benchmarks show 2.3–2.7x speedup over librosa on 30s synthetic signals. With Metal GPU acceleration, we should be capable of much more on real audio. The current bottleneck picture is unknown — all benchmarking is wall-clock only with no per-stage or GPU-level visibility.

**Known architectural concerns:**
- Every GPU operation calls `cmdBuf.commit() + waitUntilCompleted()` — fully synchronous, no pipelining
- MTLBuffers allocated per call — no pooling or reuse
- Smart dispatch thresholds were calibrated once via unit tests, never validated on real workloads
- No Apple Instruments integration (no os_signpost/OSSignposter)

**Test audio:**
- `~/Desktop/moth_30s.m4a` — 30s excerpt for quick iteration
- `~/Desktop/inamorata.m4a` — 24MB full song for realistic profiling

## Design

### 1. OSSignposter Instrumentation

Add a lightweight `Profiler` utility in MetalMomCore wrapping `OSSignposter`. Always compiled in — signposts have near-zero overhead when Instruments isn't attached.

**Single shared signposter:**
- Subsystem: `"com.metalmom.engine"`
- Categories: `"bridge"`, `"engine"`, `"gpu"`

**Three instrumentation tiers:**

| Tier | Where | What it shows |
|------|-------|---------------|
| Bridge | Every `@_cdecl` in MetalMomBridge | Full call round-trip including cffi overhead |
| Engine | Major stages in MetalMomCore (windowing, FFT, filterbank, peak picking, autocorrelation) | Where within an operation time is spent |
| GPU | `SmartDispatcher.dispatch()`, `cmdBuf.commit()/waitUntilCompleted()` | GPU vs CPU routing decisions, sync wait time |

### 2. Profiling Workload Target

A Swift executable target (`ProfilingRunner`) for attaching Instruments. Not shipped — dev tool only.

**Pipeline (mirrors common user workflow):**
1. Load audio file (AVFoundation)
2. STFT (n_fft=2048, hop_length=512)
3. Mel spectrogram
4. MFCC (13 coefficients)
5. Chroma STFT
6. Onset strength
7. Beat tracking

**Why Swift target, not Python:**
- Instruments attaches cleanly to a native process
- No cffi/Python interpreter noise in traces
- Python path can be profiled separately if cffi overhead looks suspicious

**Usage:**
```
swift run ProfilingRunner --file ~/Desktop/moth_30s.m4a    # quick iteration
swift run ProfilingRunner --file ~/Desktop/inamorata.m4a   # real workload
```

Prints wall-clock per-step times to stdout as a sanity check alongside signpost data.

### 3. Measure-Then-Fix Cycle

Strictly: profile first, optimize second.

**Deliverables:**
1. OSSignposter instrumentation in MetalMomCore (permanent infrastructure)
2. Swift ProfilingRunner target with real audio workload
3. Baseline Instruments trace on M3 Max with `inamorata.m4a`
4. Bottleneck report (`docs/profiling-report.md`) — top bottlenecks with Instruments evidence
5. Fix the top 3 bottlenecks (specific fixes depend on trace, likely candidates below)
6. After-trace to measure improvement

**Likely bottleneck candidates (to be confirmed by profiling):**
- Synchronous `waitUntilCompleted()` stalls → pipeline GPU commands
- Per-call MTLBuffer allocation → buffer pooling
- Redundant CPU↔GPU copies in compound operations → fused pipelines
- Suboptimal dispatch thresholds → re-calibrate with real audio

## Non-Goals

- Optimizing Python/cffi overhead (measure first, likely not dominant)
- Changing the public API
- Adding new features
- CoreML/neural pipeline profiling (separate effort)
