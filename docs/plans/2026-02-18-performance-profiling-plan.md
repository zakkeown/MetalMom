# Performance Profiling Deep-Dive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OSSignposter instrumentation to MetalMom's Swift engine, build a profiling workload runner, capture baseline traces on real audio, and fix the top bottlenecks.

**Architecture:** A `Profiler` singleton in MetalMomCore wraps `OSSignposter` with three categories (bridge, engine, gpu). A `ProfilingRunner` executable target runs a standard analysis pipeline on real audio files. Signpost intervals appear in Instruments for timeline analysis.

**Tech Stack:** OSSignposter (OSLog framework), Instruments.app, Swift SPM executable target

---

### Task 1: Add Profiler Utility

**Files:**
- Create: `Sources/MetalMomCore/Dispatch/Profiler.swift`

**Step 1: Create the Profiler singleton**

```swift
import Foundation
import os

/// Lightweight profiling instrumentation using OSSignposter.
/// Always compiled in — signposts have near-zero overhead when Instruments is not attached.
public final class Profiler {
    public static let shared = Profiler()

    private let signposter: OSSignposter

    /// Named categories for grouping signpost intervals in Instruments.
    public enum Category: String {
        case bridge = "bridge"
        case engine = "engine"
        case gpu = "gpu"
    }

    private init() {
        self.signposter = OSSignposter(subsystem: "com.metalmom.engine", category: .pointsOfInterest)
    }

    /// Begin a signpost interval. Returns a state object to pass to `end()`.
    @inline(__always)
    public func begin(_ name: StaticString, category: Category = .engine) -> OSSignpostIntervalState {
        let id = signposter.makeSignpostID()
        return signposter.beginInterval(name, id: id)
    }

    /// End a signpost interval.
    @inline(__always)
    public func end(_ name: StaticString, _ state: OSSignpostIntervalState) {
        signposter.endInterval(name, state)
    }

    /// Emit a single signpost event (point, not interval).
    @inline(__always)
    public func event(_ name: StaticString) {
        signposter.emitEvent(name)
    }
}
```

**Step 2: Verify it compiles**

Run: `swift build 2>&1 | tail -5`
Expected: Build Succeeded

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/Profiler.swift
git commit -m "feat: add OSSignposter-based Profiler utility"
```

---

### Task 2: Instrument SmartDispatcher and GPU Synchronization

**Files:**
- Modify: `Sources/MetalMomCore/Dispatch/SmartDispatch.swift`
- Modify: `Sources/MetalMomCore/Dispatch/MetalBackend.swift`

**Step 1: Instrument SmartDispatcher.dispatch()**

In `Sources/MetalMomCore/Dispatch/SmartDispatch.swift`, wrap the `dispatch` method:

```swift
public func dispatch<Op: ComputeOperation>(_ op: Op, input: Op.Input, dataSize: Int) -> Op.Output {
    let decision = routingDecision(dataSize: dataSize, operationThreshold: Op.dispatchThreshold)
    let profiler = Profiler.shared
    switch decision {
    case .accelerate:
        let state = profiler.begin("CPU Compute")
        let result = op.executeCPU(input)
        profiler.end("CPU Compute", state)
        return result
    case .metal:
        let state = profiler.begin("GPU Compute")
        let result = op.executeGPU(input)
        profiler.end("GPU Compute", state)
        return result
    }
}
```

**Step 2: Instrument MetalBackend.makeCommandBuffer()**

In `Sources/MetalMomCore/Dispatch/MetalBackend.swift`, add a profiled command buffer factory:

Add a new method alongside `makeCommandBuffer()`:

```swift
/// Create a command buffer and log a signpost event for GPU dispatch tracking.
public func makeProfiledCommandBuffer() -> MTLCommandBuffer? {
    Profiler.shared.event("GPU CmdBuf Create")
    return commandQueue.makeCommandBuffer()
}
```

This is additive — existing code keeps using `makeCommandBuffer()`. The profiling runner and new instrumentation can call the profiled variant.

**Step 3: Verify it compiles**

Run: `swift build 2>&1 | tail -5`
Expected: Build Succeeded

**Step 4: Run existing tests to verify no regressions**

Run: `swift test 2>&1 | tail -10`
Expected: All tests pass

**Step 5: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/SmartDispatch.swift Sources/MetalMomCore/Dispatch/MetalBackend.swift
git commit -m "feat: instrument SmartDispatcher and MetalBackend with signposts"
```

---

### Task 3: Instrument Core Engine Operations

**Files:**
- Modify: `Sources/MetalMomCore/Spectral/STFT.swift`
- Modify: `Sources/MetalMomCore/Spectral/MelSpectrogram.swift`
- Modify: `Sources/MetalMomCore/Spectral/FusedMFCC.swift`
- Modify: `Sources/MetalMomCore/Features/Chroma.swift`
- Modify: `Sources/MetalMomCore/Rhythm/OnsetDetection.swift`
- Modify: `Sources/MetalMomCore/Rhythm/BeatTracker.swift`
- Modify: `Sources/MetalMomCore/Audio/AudioIO.swift`

Add signpost intervals to the entry point of each major operation. The pattern is the same for each — wrap the top-level compute function.

**Step 1: Instrument AudioIO.load()**

In `Sources/MetalMomCore/Audio/AudioIO.swift`, at the start and end of `load()`:

```swift
public static func load(...) throws -> Signal {
    let state = Profiler.shared.begin("AudioIO.load")
    defer { Profiler.shared.end("AudioIO.load", state) }
    // ... existing code unchanged ...
}
```

**Step 2: Instrument STFT**

In `Sources/MetalMomCore/Spectral/STFT.swift`, wrap `STFT.compute()` (the convenience static method at line 491):

```swift
public static func compute(...) -> Signal {
    let state = Profiler.shared.begin("STFT")
    defer { Profiler.shared.end("STFT", state) }
    // ... existing code unchanged ...
}
```

Also wrap `STFT.computePowerSpectrogram()` (line 227):

```swift
public static func computePowerSpectrogram(...) -> Signal {
    let state = Profiler.shared.begin("STFT.power")
    defer { Profiler.shared.end("STFT.power", state) }
    // ... existing code unchanged ...
}
```

**Step 3: Instrument MelSpectrogram.compute()**

In `Sources/MetalMomCore/Spectral/MelSpectrogram.swift`, wrap `compute()` (line 28):

```swift
public static func compute(...) -> Signal {
    let state = Profiler.shared.begin("MelSpectrogram")
    defer { Profiler.shared.end("MelSpectrogram", state) }
    // ... existing code unchanged ...
}
```

**Step 4: Instrument FusedMFCC**

In `Sources/MetalMomCore/Spectral/FusedMFCC.swift`, wrap both `compute()` (line 29) and `computeGPU()` (line 71):

```swift
public static func compute(...) -> Signal {
    let state = Profiler.shared.begin("MFCC")
    defer { Profiler.shared.end("MFCC", state) }
    // ... existing code unchanged ...
}

static func computeGPU(...) -> Signal? {
    let state = Profiler.shared.begin("MFCC.GPU")
    defer { Profiler.shared.end("MFCC.GPU", state) }
    // ... existing code unchanged ...
}
```

**Step 5: Instrument Chroma**

In `Sources/MetalMomCore/Features/Chroma.swift`, wrap the `stft()` static method:

```swift
public static func stft(...) -> Signal {
    let state = Profiler.shared.begin("Chroma")
    defer { Profiler.shared.end("Chroma", state) }
    // ... existing code unchanged ...
}
```

**Step 6: Instrument OnsetDetection**

In `Sources/MetalMomCore/Rhythm/OnsetDetection.swift`, wrap `onsetStrength()` (line 23):

```swift
public static func onsetStrength(...) -> Signal {
    let state = Profiler.shared.begin("OnsetStrength")
    defer { Profiler.shared.end("OnsetStrength", state) }
    // ... existing code unchanged ...
}
```

**Step 7: Instrument BeatTracker**

In `Sources/MetalMomCore/Rhythm/BeatTracker.swift`, wrap `beatTrack()` (line 24):

```swift
public static func beatTrack(...) -> (tempo: Float, beats: Signal) {
    let state = Profiler.shared.begin("BeatTrack")
    defer { Profiler.shared.end("BeatTrack", state) }
    // ... existing code unchanged ...
}
```

**Step 8: Verify everything compiles and tests pass**

Run: `swift build 2>&1 | tail -5`
Expected: Build Succeeded

Run: `swift test 2>&1 | tail -10`
Expected: All tests pass

**Step 9: Commit**

```bash
git add Sources/MetalMomCore/Spectral/STFT.swift Sources/MetalMomCore/Spectral/MelSpectrogram.swift Sources/MetalMomCore/Spectral/FusedMFCC.swift Sources/MetalMomCore/Features/Chroma.swift Sources/MetalMomCore/Rhythm/OnsetDetection.swift Sources/MetalMomCore/Rhythm/BeatTracker.swift Sources/MetalMomCore/Audio/AudioIO.swift
git commit -m "feat: add signpost instrumentation to core engine operations"
```

---

### Task 4: Add ProfilingRunner Executable Target

**Files:**
- Modify: `Package.swift`
- Create: `Sources/ProfilingRunner/main.swift`

**Step 1: Add executable target to Package.swift**

Add a new executable target that depends on MetalMomCore:

```swift
.executableTarget(
    name: "ProfilingRunner",
    dependencies: ["MetalMomCore"],
    path: "Sources/ProfilingRunner"
),
```

This goes in the `targets` array. It does NOT need to be in `products` — it's a dev-only tool.

**Step 2: Create the directory**

```bash
mkdir -p Sources/ProfilingRunner
```

**Step 3: Create main.swift with the profiling workload**

```swift
import Foundation
import MetalMomCore

/// ProfilingRunner — runs a standard audio analysis pipeline for profiling with Instruments.
///
/// Usage:
///   swift run ProfilingRunner <path-to-audio-file>
///
/// Attach Instruments (Time Profiler + os_signpost) to the process to see
/// signpost intervals for each pipeline stage.

@main
struct ProfilingRunner {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: swift run ProfilingRunner <audio-file>")
            print("  e.g.: swift run ProfilingRunner ~/Desktop/moth_30s.m4a")
            Foundation.exit(1)
        }

        let filePath = (args[1] as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: filePath) else {
            print("ERROR: File not found: \(filePath)")
            Foundation.exit(1)
        }

        print("MetalMom Profiling Runner")
        print("=========================")
        print("File: \(filePath)")
        print("Metal: \(MetalBackend.shared != nil ? "available" : "unavailable")")
        if let chip = MetalBackend.shared?.chipProfile {
            print("GPU: \(chip.gpuFamily), ~\(chip.estimatedCoreCount) cores")
        }
        print()

        // --- Load Audio ---
        let t0 = CFAbsoluteTimeGetCurrent()
        let signal = try AudioIO.load(path: filePath, sr: 22050, mono: true)
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print("Load:           \(String(format: "%8.1f", tLoad * 1000)) ms  (\(signal.count) samples, \(signal.sampleRate) Hz)")

        // --- STFT ---
        let t1 = CFAbsoluteTimeGetCurrent()
        let stftMag = STFT.compute(signal: signal, nFFT: 2048, hopLength: 512)
        let tSTFT = CFAbsoluteTimeGetCurrent() - t1
        print("STFT:           \(String(format: "%8.1f", tSTFT * 1000)) ms  shape=\(stftMag.shape)")

        // --- Mel Spectrogram ---
        let t2 = CFAbsoluteTimeGetCurrent()
        let mel = MelSpectrogram.compute(signal: signal, nFFT: 2048, hopLength: 512)
        let tMel = CFAbsoluteTimeGetCurrent() - t2
        print("Mel:            \(String(format: "%8.1f", tMel * 1000)) ms  shape=\(mel.shape)")

        // --- MFCC (via FusedMFCC which routes GPU/CPU) ---
        let t3 = CFAbsoluteTimeGetCurrent()
        let mfcc = FusedMFCC.compute(signal: signal, nFFT: 2048, hopLength: 512, nMFCC: 13)
        let tMFCC = CFAbsoluteTimeGetCurrent() - t3
        print("MFCC:           \(String(format: "%8.1f", tMFCC * 1000)) ms  shape=\(mfcc.shape)")

        // --- Chroma ---
        let t4 = CFAbsoluteTimeGetCurrent()
        let chroma = Chroma.stft(signal: signal, nFFT: 2048, hopLength: 512)
        let tChroma = CFAbsoluteTimeGetCurrent() - t4
        print("Chroma:         \(String(format: "%8.1f", tChroma * 1000)) ms  shape=\(chroma.shape)")

        // --- Onset Strength ---
        let t5 = CFAbsoluteTimeGetCurrent()
        let onset = OnsetDetection.onsetStrength(signal: signal, nFFT: 2048, hopLength: 512)
        let tOnset = CFAbsoluteTimeGetCurrent() - t5
        print("Onset:          \(String(format: "%8.1f", tOnset * 1000)) ms  shape=\(onset.shape)")

        // --- Beat Tracking ---
        let t6 = CFAbsoluteTimeGetCurrent()
        let (tempo, beats) = BeatTracker.beatTrack(signal: signal, hopLength: 512)
        let tBeat = CFAbsoluteTimeGetCurrent() - t6
        print("Beat:           \(String(format: "%8.1f", tBeat * 1000)) ms  tempo=\(String(format: "%.1f", tempo)) BPM, \(beats.count) beats")

        // --- Total ---
        let total = tLoad + tSTFT + tMel + tMFCC + tChroma + tOnset + tBeat
        print()
        print("Total:          \(String(format: "%8.1f", total * 1000)) ms")
        print()
        print("Attach Instruments (Time Profiler + os_signpost) to see detailed breakdown.")
    }
}
```

**Step 4: Verify it builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build Succeeded

**Step 5: Run with the short audio file to verify end-to-end**

Run: `swift run ProfilingRunner ~/Desktop/moth_30s.m4a`
Expected: Prints timing table for all 7 stages with no errors

**Step 6: Run with the full song**

Run: `swift run ProfilingRunner ~/Desktop/inamorata.m4a`
Expected: Prints timing table — this is the baseline for optimization

**Step 7: Commit**

```bash
git add Package.swift Sources/ProfilingRunner/main.swift
git commit -m "feat: add ProfilingRunner executable for Instruments profiling"
```

---

### Task 5: Capture Baseline and Identify Bottlenecks

This task is interactive — it requires running Instruments and analyzing the results.

**Step 1: Run ProfilingRunner and capture console output**

Run: `swift build -c release && swift run -c release ProfilingRunner ~/Desktop/inamorata.m4a`

Save the timing output — this is the baseline.

**Step 2: Profile with Instruments (user action)**

Open Instruments.app, create a new trace with:
- **Time Profiler** template
- **os_signpost** instrument (add manually if not present)

Target: the ProfilingRunner binary (set to the release build at `.build/release/ProfilingRunner`)
Arguments: `~/Desktop/inamorata.m4a`

Record a trace. The signpost intervals will show up as named regions in the timeline.

**Step 3: Analyze and document findings**

Create `docs/profiling-report.md` with:
- Baseline wall-clock times from Step 1
- Top 3 bottlenecks identified from the Instruments trace
- For each bottleneck: what's slow, why, and proposed fix
- Screenshots optional but signpost interval names and durations are required

**Step 4: Commit the report**

```bash
git add docs/profiling-report.md
git commit -m "docs: add baseline profiling report with bottleneck analysis"
```

---

### Task 6: Fix Bottleneck #1 (determined by profiling)

The specific fix depends on what the Instruments trace reveals. This task is a placeholder for the first optimization.

**Likely candidates based on architectural analysis:**

**If GPU sync stalls dominate:**
- Identify compound operations that make multiple sequential `cmdBuf.commit() + waitUntilCompleted()` calls
- Batch multiple GPU operations into a single command buffer where possible
- Files likely affected: `Sources/MetalMomCore/Dispatch/MetalShaders.swift`

**If buffer allocation overhead dominates:**
- Add a simple MTLBuffer pool to MetalBackend
- Reuse buffers of the same size instead of allocating new ones each call
- Files likely affected: `Sources/MetalMomCore/Dispatch/MetalBackend.swift`

**If STFT CPU framing dominates:**
- The framing/windowing loop in STFT.executeCPU() processes frames sequentially
- Could use Accelerate's `vDSP_vmul` across all frames via stride tricks
- Files likely affected: `Sources/MetalMomCore/Spectral/STFT.swift`

**Step 1: Implement the fix**
**Step 2: Run tests:** `swift test 2>&1 | tail -10`
**Step 3: Re-run profiling:** `swift run -c release ProfilingRunner ~/Desktop/inamorata.m4a`
**Step 4: Document improvement in profiling-report.md**
**Step 5: Commit**

---

### Task 7: Fix Bottleneck #2 (determined by profiling)

Same structure as Task 6. Apply the second-priority fix identified in the profiling report.

**Step 1: Implement the fix**
**Step 2: Run tests:** `swift test 2>&1 | tail -10`
**Step 3: Re-run profiling:** `swift run -c release ProfilingRunner ~/Desktop/inamorata.m4a`
**Step 4: Document improvement in profiling-report.md**
**Step 5: Commit**

---

### Task 8: Fix Bottleneck #3 (determined by profiling)

Same structure as Task 6. Apply the third-priority fix identified in the profiling report.

**Step 1: Implement the fix**
**Step 2: Run tests:** `swift test 2>&1 | tail -10`
**Step 3: Re-run profiling:** `swift run -c release ProfilingRunner ~/Desktop/inamorata.m4a`
**Step 4: Document improvement in profiling-report.md**
**Step 5: Commit**

---

### Task 9: Final Profiling and Summary

**Step 1: Run the full benchmark suite**

Run: `swift run -c release ProfilingRunner ~/Desktop/inamorata.m4a`
Compare against baseline from Task 5.

**Step 2: Run the Python benchmarks for regression check**

Run: `swift build -c release && ./scripts/build_dylib.sh && .venv/bin/python benchmarks/run_benchmarks.py`

**Step 3: Update the profiling report with final results**

Add a "Results" section to `docs/profiling-report.md`:
- Before/after timing table
- Speedup per operation
- Overall pipeline speedup

**Step 4: Run the full test suite**

Run: `swift test 2>&1 | tail -10` and `.venv/bin/pytest Tests/ -x -q 2>&1 | tail -10`
Expected: All tests pass

**Step 5: Commit**

```bash
git add docs/profiling-report.md
git commit -m "docs: update profiling report with optimization results"
```
