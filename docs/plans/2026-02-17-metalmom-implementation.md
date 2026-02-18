# MetalMom Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a GPU-accelerated audio/music analysis library on Apple Metal with feature parity across librosa and madmom, end-to-end from Swift core to Python API.

**Architecture:** Three SPM targets (MetalMomCBridge → MetalMomCore → MetalMomBridge) with Smart Dispatch protocol introduced from Phase 1 (CPU-only initially, Metal added in Phase 10). Exported via C ABI (`@_cdecl`), called from Python via cffi with minimal-copy NumPy interop. Incremental compat shims for librosa and madmom built alongside each feature.

**Tech Stack:** Swift, Metal (MSL/MPSGraph), Accelerate (vDSP/BLAS), CoreML, Python 3.11+, cffi, NumPy

**Design Doc:** `docs/plans/2026-02-17-metalmom-design.md`

---

## Phase Overview

| Phase | What | Proves |
|-------|------|--------|
| 1 | Vertical Slice: STFT end-to-end | Architecture works: 3 SPM targets, dispatch protocol, Swift → C → Python → NumPy |
| 2 | Core Spectral & Features | Mel, MFCC, chroma, spectral descriptors + compat shims |
| 3 | Audio I/O | Load, stream, resample, signal generation |
| 4 | Rhythm & Onset | Eval metrics first, then onset detection, beat tracking, tempo |
| 5 | Pitch & Effects | YIN, pYIN, piptrack, tuning, HPSS, time stretch, pitch shift, trim, split |
| 6 | ML Inference Engine | CoreML engine, decoders, all 9 model conversions (spike first), ensemble + chunking |
| 7 | Neural Features (madmom) | Neural beat/onset/key/chord/piano |
| 8 | Remaining Features | CQT, decompose, segment, sequence, filters, conversions, display |
| 9 | Compatibility Shim Audit | Completeness audit, signature verification, madmom compat |
| 10 | Metal GPU Backend | Metal backend, GPU STFT, shaders, threshold calibration, pipeline fusion |
| 11 | Distribution & CI | Wheels, GitHub Actions (CPU-only CI), API parity check |

**Task count by phase:** 1(13) 2(13) 3(5) 4(8) 5(10) 6(17) 7(8) 8(19) 9(7) 10(8) 11(7) = **115 tasks**

---

## Phase 1: Vertical Slice — STFT End-to-End

This phase proves the entire architecture works: three SPM targets compile, Smart Dispatch protocol routes to Accelerate, Swift computes STFT → exports via C ABI from MetalMomBridge → Python calls via cffi → result arrives as a NumPy array → parity test passes against librosa. Every subsequent feature follows this same pattern. Metal smoke test proves GPU is available.

### Task 1.1: Project Scaffolding — Swift Package (Three Targets)

**Files:**
- Create: `Package.swift`
- Create: `Sources/MetalMomCBridge/include/metalmom.h` (placeholder)
- Create: `Sources/MetalMomCore/MetalMomCore.swift` (placeholder)
- Create: `Sources/MetalMomBridge/Bridge.swift` (placeholder)
- Create: `Tests/MetalMomTests/MetalMomTests.swift`

**Step 1: Create Package.swift**

Three targets with clean dependency chain: MetalMomCBridge (C types) → MetalMomCore (Swift engine) → MetalMomBridge (@_cdecl exports). The dylib product is built from MetalMomBridge.

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalMom",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
        .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
    ],
    targets: [
        .target(
            name: "MetalMomCBridge",
            dependencies: [],
            path: "Sources/MetalMomCBridge",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MetalMomCore",
            dependencies: ["MetalMomCBridge"],
            path: "Sources/MetalMomCore"
        ),
        .target(
            name: "MetalMomBridge",
            dependencies: ["MetalMomCore"],
            path: "Sources/MetalMomBridge"
        ),
        .testTarget(
            name: "MetalMomTests",
            dependencies: ["MetalMomCore"],
            path: "Tests/MetalMomTests"
        ),
    ]
)
```

**Step 2: Create placeholder files**

`Sources/MetalMomCBridge/include/metalmom.h`:
```c
#ifndef METALMOM_H
#define METALMOM_H

#include <stdint.h>

#define MM_OK 0
#define MM_ERR_INVALID_INPUT -1
#define MM_ERR_METAL_UNAVAILABLE -2
#define MM_ERR_ALLOC_FAILED -3

#endif /* METALMOM_H */
```

`Sources/MetalMomCore/MetalMomCore.swift`:
```swift
import Foundation

public enum MetalMom {
    public static let version = "0.1.0"
}
```

`Sources/MetalMomBridge/Bridge.swift`:
```swift
import Foundation
import MetalMomCore
```

`Tests/MetalMomTests/MetalMomTests.swift`:
```swift
import XCTest
@testable import MetalMomCore

final class MetalMomTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(MetalMom.version, "0.1.0")
    }
}
```

**Step 3: Build and test**

Run: `swift build`
Expected: BUILD SUCCEEDED

Run: `swift test`
Expected: Test Suite 'All tests' passed

**Step 4: Commit**

```bash
git add Package.swift Sources/ Tests/
git commit -m "feat: scaffold Swift package with 3 SPM targets"
```

---

### Task 1.2: Project Scaffolding — Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `python/metalmom/__init__.py`
- Create: `python/metalmom/_native.py` (placeholder)
- Create: `python/metalmom/_buffer.py` (placeholder)
- Create: `python/metalmom/core.py` (placeholder)
- Create: `Tests/test_import.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "metalmom"
version = "0.1.0"
description = "GPU-accelerated audio/music analysis on Apple Metal"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "numpy>=1.24",
    "cffi>=1.16",
    "soundfile>=0.12",
]

[project.optional-dependencies]
display = ["matplotlib>=3.7"]
eval = ["mir_eval>=0.7"]
dev = ["pytest>=7.0", "librosa>=0.11.0"]

[tool.hatch.build.targets.wheel]
packages = ["python/metalmom"]

[tool.pytest.ini_options]
testpaths = ["Tests"]
```

**Step 2: Create Python placeholder files**

`python/metalmom/__init__.py`:
```python
"""MetalMom: GPU-accelerated audio/music analysis on Apple Metal."""

__version__ = "0.1.0"
```

`python/metalmom/_native.py`, `python/metalmom/_buffer.py`, `python/metalmom/core.py`: empty docstring placeholders.

**Step 3: Write the test**

`Tests/test_import.py`:
```python
def test_import():
    import metalmom
    assert metalmom.__version__ == "0.1.0"
```

**Step 4: Create venv, install, and test**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Run: `.venv/bin/pytest Tests/test_import.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add pyproject.toml python/ Tests/test_import.py
git commit -m "feat: scaffold Python package with pyproject.toml"
```

---

### Task 1.3: Signal Type (Pinned Buffer Storage)

**Files:**
- Create: `Sources/MetalMomCore/Audio/Signal.swift`
- Create: `Tests/MetalMomTests/SignalTests.swift`

**Step 1: Write the failing test**

`Tests/MetalMomTests/SignalTests.swift`:
```swift
import XCTest
@testable import MetalMomCore

final class SignalTests: XCTestCase {
    func testSignalFromArray() {
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let signal = Signal(data: data, sampleRate: 22050)
        XCTAssertEqual(signal.count, 4)
        XCTAssertEqual(signal.sampleRate, 22050)
        XCTAssertEqual(signal[0], 1.0)
        XCTAssertEqual(signal[3], 4.0)
    }

    func testSignalShape() {
        let data: [Float] = [1.0, 2.0, 3.0]
        let signal = Signal(data: data, sampleRate: 44100)
        XCTAssertEqual(signal.shape, [3])
    }

    func testSignal2D() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let signal = Signal(data: data, shape: [2, 3], sampleRate: 22050)
        XCTAssertEqual(signal.shape, [2, 3])
        XCTAssertEqual(signal.count, 6)
    }

    func testSignalDataPointerIsStable() {
        let data: [Float] = [1.0, 2.0, 3.0]
        let signal = Signal(data: data, sampleRate: 22050)
        // Pointer must be stable across multiple accesses (pinned storage)
        let ptr1 = signal.dataPointer
        let ptr2 = signal.dataPointer
        XCTAssertEqual(ptr1, ptr2)
        XCTAssertEqual(ptr1[0], 1.0)
        XCTAssertEqual(ptr1[2], 3.0)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter SignalTests`
Expected: FAIL — `Signal` not defined

**Step 3: Write implementation (UnsafeMutableBufferPointer storage)**

`Sources/MetalMomCore/Audio/Signal.swift`:
```swift
import Foundation

/// Core data type wrapping audio data with shape metadata.
/// Uses manually-allocated UnsafeMutableBufferPointer for stable pointer addresses
/// safe to pass to Accelerate, Metal, and across the C ABI.
public final class Signal {
    private let storage: UnsafeMutableBufferPointer<Float>
    public let shape: [Int]
    public let sampleRate: Int

    public var count: Int { storage.count }

    /// Create a 1D signal from a Float array (copies into pinned storage).
    public init(data: [Float], sampleRate: Int) {
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = [data.count]
        self.sampleRate = sampleRate
    }

    /// Create an N-dimensional signal from a flat Float array with explicit shape.
    public init(data: [Float], shape: [Int], sampleRate: Int) {
        precondition(data.count == shape.reduce(1, *),
                     "Data count \(data.count) doesn't match shape \(shape)")
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = shape
        self.sampleRate = sampleRate
    }

    /// Create from pre-allocated buffer (takes ownership, caller must not free).
    public init(taking buffer: UnsafeMutableBufferPointer<Float>, shape: [Int], sampleRate: Int) {
        precondition(buffer.count == shape.reduce(1, *))
        self.storage = buffer
        self.shape = shape
        self.sampleRate = sampleRate
    }

    deinit {
        storage.baseAddress?.deinitialize(count: storage.count)
        storage.baseAddress?.deallocate()
    }

    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    /// Stable pointer to underlying data. Safe because storage is manually allocated.
    public var dataPointer: UnsafePointer<Float> {
        UnsafePointer(storage.baseAddress!)
    }

    /// Mutable pointer to underlying data.
    public var mutableDataPointer: UnsafeMutablePointer<Float> {
        storage.baseAddress!
    }

    /// Access underlying storage for Accelerate operations.
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(storage))
    }

    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(storage)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter SignalTests`
Expected: All tests passed

**Step 5: Commit**

```bash
git add Sources/MetalMomCore/Audio/Signal.swift Tests/MetalMomTests/SignalTests.swift
git commit -m "feat: add Signal core data type with pinned buffer storage"
```

---

### Task 1.4: Window Functions

Same as original plan — no changes needed. Creates `Sources/MetalMomCore/Filters/Windows.swift` with Hann window.

Commit: `feat: add Hann window function`

---

### Task 1.5: ComputeBackend Protocol and AccelerateBackend

**Files:**
- Create: `Sources/MetalMomCore/Dispatch/ComputeBackend.swift`
- Create: `Sources/MetalMomCore/Dispatch/AccelerateBackend.swift`
- Create: `Sources/MetalMomCore/Dispatch/SmartDispatch.swift`
- Create: `Tests/MetalMomTests/DispatchTests.swift`

**Step 1: Write the failing test**

`Tests/MetalMomTests/DispatchTests.swift`:
```swift
import XCTest
@testable import MetalMomCore

final class DispatchTests: XCTestCase {
    func testDispatcherDefaultsToAccelerate() {
        let dispatcher = SmartDispatcher()
        XCTAssertEqual(dispatcher.activeBackend, .accelerate)
    }

    func testDispatcherRoutesSmallDataToCPU() {
        let dispatcher = SmartDispatcher()
        // With threshold = Int.max (GPU not available), all work goes to CPU
        let decision = dispatcher.routingDecision(dataSize: 1000, operationThreshold: Int.max)
        XCTAssertEqual(decision, .accelerate)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter DispatchTests`
Expected: FAIL — types not defined

**Step 3: Write implementation**

`Sources/MetalMomCore/Dispatch/ComputeBackend.swift`:
```swift
import Foundation

/// Identifies which compute backend to use.
public enum BackendType: Equatable {
    case accelerate
    case metal
}

/// Protocol for compute operations that support both CPU and GPU paths.
public protocol ComputeOperation {
    associatedtype Input
    associatedtype Output

    /// Data size threshold above which GPU is preferred. Int.max = always CPU.
    static var dispatchThreshold: Int { get }

    func executeCPU(_ input: Input) -> Output

    /// GPU path. During Phases 1-9 this calls fatalError().
    /// Phase 10 fills in real implementations.
    func executeGPU(_ input: Input) -> Output
}

/// Default GPU implementation — fatalError until Phase 10.
extension ComputeOperation {
    public func executeGPU(_ input: Input) -> Output {
        fatalError("\(type(of: self)).executeGPU not yet implemented — Phase 10")
    }
}
```

`Sources/MetalMomCore/Dispatch/AccelerateBackend.swift`:
```swift
import Foundation

/// Marker for the Accelerate (CPU) backend. Holds no state —
/// Accelerate functions are all stateless and thread-safe.
public final class AccelerateBackend {
    public static let shared = AccelerateBackend()
    private init() {}
}
```

`Sources/MetalMomCore/Dispatch/SmartDispatch.swift`:
```swift
import Foundation

/// Routes compute operations to the optimal backend based on data size.
/// During Phases 1-9, always routes to Accelerate (CPU).
public final class SmartDispatcher {
    public let activeBackend: BackendType

    public init() {
        // Phase 10 will add Metal availability check here
        self.activeBackend = .accelerate
    }

    /// Determine which backend to use for a given data size and operation threshold.
    public func routingDecision(dataSize: Int, operationThreshold: Int) -> BackendType {
        guard activeBackend == .metal else { return .accelerate }
        return dataSize >= operationThreshold ? .metal : .accelerate
    }

    /// Execute a compute operation, routing to the appropriate backend.
    public func dispatch<Op: ComputeOperation>(_ op: Op, input: Op.Input, dataSize: Int) -> Op.Output {
        let decision = routingDecision(dataSize: dataSize, operationThreshold: Op.dispatchThreshold)
        switch decision {
        case .accelerate:
            return op.executeCPU(input)
        case .metal:
            return op.executeGPU(input)
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter DispatchTests`
Expected: All tests passed

**Step 5: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/ Tests/MetalMomTests/DispatchTests.swift
git commit -m "feat: add ComputeBackend protocol, AccelerateBackend, and SmartDispatcher"
```

---

### Task 1.6: STFT as ComputeOperation (Accelerate)

**Files:**
- Create: `Sources/MetalMomCore/Spectral/STFT.swift`
- Create: `Tests/MetalMomTests/STFTTests.swift`

Same STFT implementation as original plan, but structured as a `ComputeOperation` with `executeCPU()` containing the vDSP logic and `dispatchThreshold = Int.max`. The `executeGPU()` uses the default `fatalError()` from the protocol extension.

The `STFT.compute()` convenience static method creates the operation, gets the shared `SmartDispatcher`, and calls `dispatch()`.

Test the STFT output shape, sine wave peak bin, and skip invertibility test (Phase 2).

Commit: `feat: add forward STFT via Accelerate/vDSP as ComputeOperation`

---

### Task 1.7: C ABI Bridge — Context and STFT

**Files:**
- Modify: `Sources/MetalMomCBridge/include/metalmom.h` — add MMBuffer, MMSTFTParams, function declarations
- Create: `Sources/MetalMomBridge/Bridge.swift` — @_cdecl exports

**Critical change from original plan:** Bridge code goes in `Sources/MetalMomBridge/Bridge.swift` (NOT MetalMomCBridge). MetalMomCBridge contains only the C header with type definitions. MetalMomBridge imports MetalMomCore and provides the @_cdecl exports.

`Sources/MetalMomCBridge/include/metalmom.h`:
```c
#ifndef METALMOM_H
#define METALMOM_H

#include <stdint.h>

/* Status codes */
#define MM_OK 0
#define MM_ERR_INVALID_INPUT -1
#define MM_ERR_METAL_UNAVAILABLE -2
#define MM_ERR_ALLOC_FAILED -3

/* Buffer type: holds data + shape for NumPy interop */
typedef struct {
    float* data;
    int64_t shape[8];   /* max 8 dimensions */
    int32_t ndim;
    int32_t dtype;      /* 0=float32 */
    int64_t count;      /* total element count */
} MMBuffer;

/* STFT parameters */
typedef struct {
    int32_t n_fft;
    int32_t hop_length;
    int32_t win_length;
    int32_t center;     /* bool: 1=true, 0=false */
} MMSTFTParams;

/* Opaque context handle — NOT thread-safe. Create one per thread. */
typedef void* mm_context;

/* Lifecycle */
mm_context mm_init(void);
void mm_destroy(mm_context ctx);

/* STFT */
int32_t mm_stft(mm_context ctx, const float* signal_data, int64_t signal_length,
                int32_t sample_rate, const MMSTFTParams* params, MMBuffer* out);

/* Memory */
void mm_buffer_free(MMBuffer* buf);

#endif /* METALMOM_H */
```

`Sources/MetalMomBridge/Bridge.swift` — same @_cdecl implementation as original plan but in MetalMomBridge target. `MMContextInternal` holds a `SmartDispatcher` instance.

Run: `swift build`
Expected: BUILD SUCCEEDED

Commit: `feat: add C ABI bridge with context and STFT export`

---

### Task 1.8: Build the Dynamic Library

**Files:**
- Create: `scripts/build_dylib.sh`

Updated script using `find -L` and correct dylib name (MetalMomBridge):

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building libmetalmom.dylib..."
cd "$PROJECT_DIR"

swift build -c release

# Get the build directory (avoids find issues with symlinks)
BIN_PATH=$(swift build -c release --show-bin-path)
DYLIB_PATH="$BIN_PATH/libMetalMomBridge.dylib"

if [ ! -f "$DYLIB_PATH" ]; then
    echo "ERROR: Could not find built dylib at $DYLIB_PATH"
    echo "Looking with find -L as fallback..."
    DYLIB_PATH=$(find -L .build/release -name "libMetalMomBridge.dylib" -type f | head -1)
    if [ -z "$DYLIB_PATH" ]; then
        echo "ERROR: Could not find built dylib"
        exit 1
    fi
fi

# Copy to python package
mkdir -p python/metalmom/_lib
cp "$DYLIB_PATH" python/metalmom/_lib/libmetalmom.dylib
cp Sources/MetalMomCBridge/include/metalmom.h python/metalmom/_lib/metalmom.h

echo "Copied dylib and header to python/metalmom/_lib/"
echo "Done."
```

Run: `chmod +x scripts/build_dylib.sh && ./scripts/build_dylib.sh`
Expected: `libmetalmom.dylib` and `metalmom.h` in `python/metalmom/_lib/`

Commit: `feat: add dylib build script`

---

### Task 1.9: Python cffi Bindings

Same as original plan — `_native.py` and `_buffer.py`. The `_buffer.py` docstring should say "Minimal-copy MMBuffer -> NumPy wrapper" (not zero-copy).

Commit: `feat: add Python cffi bindings and buffer interop`

---

### Task 1.10: Python Public API — core.stft()

Same as original plan.

Commit: `feat: add Python core.stft() public API`

---

### Task 1.11: First Parity Test — STFT vs librosa

Same as original plan — golden file generator, parity tests with Tier 1 tolerances (rtol=1e-5, atol=1e-6).

Commit: `feat: add STFT parity tests against librosa golden files`

---

### Task 1.12: Metal Smoke Test

**Files:**
- Create: `Tests/MetalMomTests/MetalSmokeTests.swift`

Verify the Metal pipeline is available on this machine. This doesn't add any GPU compute — just proves `MTLDevice` and `MTLCommandQueue` initialize. Tagged with `.metal` so CI skips it.

```swift
import XCTest
import Metal

final class MetalSmokeTests: XCTestCase {
    func testMetalDeviceAvailable() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available (CI runner?)")
        }
        XCTAssertFalse(device.name.isEmpty)
        print("Metal device: \(device.name)")

        let queue = device.makeCommandQueue()
        XCTAssertNotNil(queue, "Failed to create command queue")
    }

    func testGPUFamilyDetection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        // All Apple Silicon supports at least family apple3
        let supportsApple3 = device.supportsFamily(.apple3)
        XCTAssertTrue(supportsApple3, "Expected Apple GPU family 3+")
        print("GPU core count: estimated via family support")
    }
}
```

Run: `swift test --filter MetalSmokeTests`
Expected: PASSED on Apple Silicon, SKIPPED on CI/VM

Commit: `test: add Metal GPU smoke test`

---

### Task 1.13: CLAUDE.md

Updated to reflect three-target structure, dispatch layer, and minimal-copy:

```markdown
# MetalMom — Project Conventions

## What is this?

GPU-accelerated audio/music analysis library. Swift core + Metal/Accelerate, Python bindings via cffi.

## Build Commands

- Swift: `swift build`
- Swift tests: `swift test`
- Release dylib: `swift build -c release && ./scripts/build_dylib.sh`
- Python tests: `.venv/bin/pytest Tests/ -v` (must build dylib first)
- Venv: `.venv/` (created by `python3 -m venv .venv && pip install -e ".[dev]"`)

## Architecture

See `docs/plans/2026-02-17-metalmom-design.md` for full design.

Three SPM targets: MetalMomCBridge (C types) → MetalMomCore (Swift engine) → MetalMomBridge (@_cdecl exports)

- `Sources/MetalMomCBridge/` — C header only (MMBuffer, status codes, param structs)
- `Sources/MetalMomCore/` — Swift engine (Audio, Spectral, Features, Rhythm, Dispatch, etc.)
- `Sources/MetalMomBridge/` — @_cdecl exported functions
- `python/metalmom/` — Python package (cffi bindings, public API, compat shims)
- `Tests/` — Swift XCTest + Python pytest

## Conventions

- Every C bridge function returns int32 status code (0 = MM_OK)
- Every compute operation conforms to ComputeOperation protocol (CPU + GPU paths)
- Signal uses UnsafeMutableBufferPointer for stable pointer addresses
- Python copies MMBuffer data into NumPy, then frees C-side (minimal-copy pattern)
- Parity tests compare against librosa/madmom golden files
- New features need: Swift ComputeOperation, XCTest, C bridge export, Python wrapper, parity test, compat shim
- mm_context is NOT thread-safe — one per thread
```

Commit: `docs: add CLAUDE.md project conventions`

---

## Phase 2: Core Spectral & Features

> From here forward, each task follows the established pattern: Swift ComputeOperation → XCTest → C bridge in MetalMomBridge → Python wrapper → Tier 1 parity test → librosa compat shim → commit. Steps are listed at the task level, since the vertical slice in Phase 1 established the pattern.
>
> **Every task includes its compat shim.** When adding `feature.mfcc()`, also add `compat/librosa/feature.py:mfcc()` in the same commit.

### Task 2.1: Complex Signal Support

- Modify: `Sources/MetalMomCore/Audio/Signal.swift` — add `SignalDType` enum (`.float32`, `.complex64`), add interleaved real/imag storage variant
- Tests: complex creation, interleaved access, pointer stability
- Commit: `feat: add complex signal support`

### Task 2.2: Full Complex STFT

- Modify: `Sources/MetalMomCore/Spectral/STFT.swift` — return complex Signal, add `computeComplex()` method
- Tests: complex output, phase preservation
- Commit: `feat: return complex STFT output`

### Task 2.3: Inverse STFT (iSTFT)

- Modify: `Sources/MetalMomCore/Spectral/STFT.swift` — proper overlap-add iSTFT from complex STFT
- Unskip: invertibility test
- C bridge: `mm_istft` in MetalMomBridge + metalmom.h
- Python: `core.istft()`
- Parity test (Tier 1): iSTFT round-trip
- Compat shim: `compat/librosa/core.py:istft()`
- Commit: `feat: add inverse STFT with overlap-add`

### Task 2.4: Magnitude Scaling — dB Conversion

- Create: `Sources/MetalMomCore/Spectral/Scaling.swift` — `amplitude_to_db`, `power_to_db`, `db_to_amplitude`, `db_to_power`
- As ComputeOperation (dispatchThreshold = Int.max)
- Tests, bridge, Python, Tier 1 parity, compat shim
- Commit: `feat: add dB scaling functions`

### Task 2.5: Mel Filterbank

- Create: `Sources/MetalMomCore/Filters/FilterBank.swift` — `mel()` filterbank generation
- Create: `Sources/MetalMomCore/Convert/Units.swift` — `hzToMel`, `melToHz`
- Tests, Tier 1 parity for filterbank matrix shape and values
- Commit: `feat: add mel filterbank and Hz/mel conversion`

### Task 2.6: Mel Spectrogram

- Create: `Sources/MetalMomCore/Spectral/MelSpectrogram.swift` — STFT magnitude → mel filterbank → mel spectrogram (as ComputeOperation)
- Bridge: `mm_mel_spectrogram`, Python: `feature.melspectrogram()`
- Parity test, compat shim
- Commit: `feat: add mel spectrogram`

### Task 2.7: MFCC

- Create: `Sources/MetalMomCore/Features/MFCC.swift` — DCT-II on log mel spectrogram (as ComputeOperation)
- Bridge: `mm_mfcc`, Python: `feature.mfcc()`
- Parity test (Tier 1: rtol=1e-4, atol=1e-5), compat shim
- Commit: `feat: add MFCC extraction`

### Task 2.8: Chroma (STFT-based)

- Create: `Sources/MetalMomCore/Features/Chroma.swift` — chroma_stft
- Bridge, Python, parity (Tier 1: rtol=1e-4, atol=1e-4), compat shim
- Commit: `feat: add STFT-based chroma features`

### Task 2.9: Spectral Descriptors

- Create: `Sources/MetalMomCore/Features/SpectralDescriptors.swift` — centroid, bandwidth, contrast, rolloff, flatness
- Bridge, Python, parity tests, compat shims
- Commit: `feat: add spectral descriptor features`

### Task 2.10: RMS and Zero-Crossing Rate

- Create: `Sources/MetalMomCore/Features/RMS.swift`, `ZeroCrossing.swift`
- Bridge, Python, parity, compat shims
- Commit: `feat: add RMS and zero-crossing rate`

### Task 2.11: Tonnetz

- Create: `Sources/MetalMomCore/Features/Tonnetz.swift` — 6D tonal centroid from chroma
- Bridge, Python, parity, compat shim
- Commit: `feat: add tonnetz features`

### Task 2.12: Delta Features

- Create: `Sources/MetalMomCore/Features/Delta.swift` — Savitzky-Golay delta, stack_memory
- Bridge, Python, parity, compat shim
- Commit: `feat: add delta and stack_memory feature manipulation`

### Task 2.13: Poly Features

- Create: `Sources/MetalMomCore/Features/PolyFeatures.swift` — polynomial fit features
- Bridge, Python, parity, compat shim
- Commit: `feat: add polynomial features`

---

## Phase 3: Audio I/O

### Task 3.1: Audio Loading (Swift/AVFoundation)

- Create: `Sources/MetalMomCore/Audio/AudioIO.swift` — load audio file, decode to float, resample, mono
- Bridge (`mm_load`), Python (`core.load()`), compat shim
- Commit: `feat: add audio loading via AVFoundation`

### Task 3.2: Resampling

- Create: `Sources/MetalMomCore/Audio/Resample.swift` — band-limited sinc interpolation via vDSP
- Bridge, Python (`core.resample()`), Tier 2 parity (round-trip SNR > 40 dB), compat shim
- Commit: `feat: add audio resampling`

### Task 3.3: Signal Generation

- Create: `Sources/MetalMomCore/Audio/SignalGen.swift` — tone, chirp, clicks
- Bridge, Python, compat shim
- Commit: `feat: add signal generation (tone, chirp, clicks)`

### Task 3.4: Duration and Sample Rate

- Modify: `Sources/MetalMomCore/Audio/AudioIO.swift` — `getDuration`, `getSampleRate`
- Bridge, Python, compat shim
- Commit: `feat: add duration and sample rate utilities`

### Task 3.5: Streaming Audio Load

- Modify: `Sources/MetalMomCore/Audio/AudioIO.swift` — `stream()` generator-style loading
- Bridge, Python (`core.stream()`), compat shim
- Commit: `feat: add streaming audio load`

---

## Phase 4: Rhythm & Onset

> Evaluation metrics come first in this phase — they're needed for debugging the rhythm features that follow.

### Task 4.1: Evaluation Metrics — Onset

- Create: `Sources/MetalMomCore/Eval/OnsetEval.swift` — P/R/F with tolerance window
- Bridge, Python (`evaluate.py`), compat shim
- Commit: `feat: add onset evaluation metrics (P/R/F)`

### Task 4.2: Evaluation Metrics — Beat, Tempo, Chord

- Create: `Sources/MetalMomCore/Eval/BeatEval.swift` — F-measure, CML, AML, P-score, Cemgil
- Create: `Sources/MetalMomCore/Eval/TempoEval.swift`
- Create: `Sources/MetalMomCore/Eval/ChordEval.swift`
- Bridge, Python, compat shims
- Commit: `feat: add beat, tempo, and chord evaluation metrics`

### Task 4.3: Onset Strength Envelope

- Create: `Sources/MetalMomCore/Rhythm/OnsetDetection.swift` — spectral flux onset strength
- Bridge, Python (`onset.onset_strength()`), Tier 1 parity, compat shim
- Commit: `feat: add onset strength envelope`

### Task 4.4: Onset Detection (Peak Picking)

- Modify: `Sources/MetalMomCore/Rhythm/OnsetDetection.swift` — peak picking with backtracking
- Bridge, Python (`onset.onset_detect()`), compat shim
- Commit: `feat: add onset detection with peak picking`

### Task 4.5: Beat Tracking (Dynamic Programming)

- Create: `Sources/MetalMomCore/Rhythm/BeatTracker.swift` — Ellis 2007 DP beat tracker
- Bridge, Python (`beat.beat_track()`), Tier 3 parity (F-measure), compat shim
- Commit: `feat: add DP beat tracking`

### Task 4.6: Tempo Estimation

- Create: `Sources/MetalMomCore/Rhythm/TempoEstimation.swift` — ACF-based tempo
- Bridge, Python (`feature.tempo()`), compat shim
- Commit: `feat: add tempo estimation`

### Task 4.7: Tempogram

- Create: `Sources/MetalMomCore/Rhythm/Tempogram.swift` — local autocorrelation tempogram, fourier tempogram
- Bridge, Python, compat shim
- Commit: `feat: add tempogram features`

### Task 4.8: PLP (Predominant Local Pulse)

- Modify: `Sources/MetalMomCore/Rhythm/BeatTracker.swift` — add PLP
- Bridge, Python (`beat.plp()`), compat shim
- Commit: `feat: add predominant local pulse estimation`

---

## Phase 5: Pitch & Effects

### Task 5.1: YIN Fundamental Frequency Estimation

- Create: `Sources/MetalMomCore/Pitch/YIN.swift`
- Bridge, Python (`pitch.yin()`), Tier 1 parity, compat shim
- Commit: `feat: add YIN pitch estimation`

### Task 5.2: pYIN Probabilistic Pitch

- Modify: `Sources/MetalMomCore/Pitch/YIN.swift` — add pYIN with Viterbi
- Bridge, Python (`pitch.pyin()`), compat shim
- Commit: `feat: add pYIN probabilistic pitch estimation`

### Task 5.3: Piptrack (Parabolic Interpolation)

- Create: `Sources/MetalMomCore/Pitch/Piptrack.swift`
- Bridge, Python (`pitch.piptrack()`), compat shim
- Commit: `feat: add parabolic interpolation pitch tracking`

### Task 5.4: Tuning Estimation

- Create: `Sources/MetalMomCore/Pitch/Tuning.swift`
- Bridge, Python, compat shim
- Commit: `feat: add tuning estimation`

### Task 5.5: Harmonic-Percussive Source Separation (HPSS)

- Create: `Sources/MetalMomCore/Harmony/HPSS.swift` — median filtering on spectrogram
- Bridge, Python (`effects.hpss()`, `effects.harmonic()`, `effects.percussive()`)
- Tier 2 parity (energy ratio H/(H+P) within 5%), compat shims
- Commit: `feat: add HPSS`

### Task 5.6: Time Stretching

- Create: `Sources/MetalMomCore/Effects/TimeStretch.swift` — phase vocoder
- Bridge, Python (`effects.time_stretch()`), compat shim
- Commit: `feat: add time stretching via phase vocoder`

### Task 5.7: Pitch Shifting

- Create: `Sources/MetalMomCore/Effects/PitchShift.swift` — time stretch + resample
- Bridge, Python (`effects.pitch_shift()`), compat shim
- Commit: `feat: add pitch shifting`

### Task 5.8: Trim

- Create: `Sources/MetalMomCore/Effects/Trim.swift` — silence trimming
- Bridge, Python (`effects.trim()`), compat shim
- Commit: `feat: add silence trimming`

### Task 5.9: Split

- Create: `Sources/MetalMomCore/Effects/Split.swift` — non-silent interval detection
- Bridge, Python (`effects.split()`), compat shim
- Commit: `feat: add non-silent interval splitting`

### Task 5.10: Preemphasis / Deemphasis

- Create: `Sources/MetalMomCore/Effects/Preemphasis.swift` — first-order IIR filter
- Bridge, Python (`effects.preemphasis()`, `effects.deemphasis()`), compat shims
- Commit: `feat: add preemphasis and deemphasis filters`

---

## Phase 6: ML Inference Engine

> Expanded from original plan. Spike task (6.6) de-risks the model conversion approach before committing to all 9 model families. All 9 models get explicit conversion tasks. Ensemble orchestration and chunked inference are cross-cutting infrastructure.

### Task 6.1: CoreML Inference Engine

- Create: `Sources/MetalMomCore/ML/InferenceEngine.swift` — load .mlpackage, run inference, return Signal
- Tests with a trivial test model
- Commit: `feat: add CoreML inference engine`

### Task 6.2: Model Registry

- Create: `Sources/MetalMomCore/ML/ModelRegistry.swift` — discover bundled models, cache loaded models
- Tests
- Commit: `feat: add model registry for pre-trained models`

### Task 6.3: HMM / Viterbi Decoding

- Create: `Sources/MetalMomCore/ML/HMM.swift` — Viterbi, forward-backward, transition matrices
- Known-answer Viterbi tests
- Commit: `feat: add HMM with Viterbi decoding`

### Task 6.4: CRF Decoding

- Create: `Sources/MetalMomCore/ML/CRF.swift`
- Tests
- Commit: `feat: add CRF decoding`

### Task 6.5: GMM Evaluation

- Create: `Sources/MetalMomCore/ML/GMM.swift`
- Tests
- Commit: `feat: add Gaussian Mixture Model evaluation`

### Task 6.6: Spike — Convert 1 RNNBeatProcessor Model

> **This is the risk mitigation task.** Prove the entire conversion pipeline works for one model before converting all 9 families. If this fails, we pivot (e.g., ONNX Runtime, re-training).

- Create: `models/conversion/convert_beat_rnn.py` — load madmom pkl, reconstruct as PyTorch BiLSTM, map weights, export via coremltools
- Create: `models/conversion/validate_conversion.py` — run both madmom and CoreML on test audio, compare activations
- Document any weight layout mismatches, preprocessing coupling issues, or CoreML constraints encountered
- **Gate:** If conversion fails, stop and reassess the ML strategy before proceeding
- Commit: `feat: spike — convert first RNNBeatProcessor model to CoreML`

### Task 6.7: Convert Remaining 7 Beat Ensemble Models

- Modify: `models/conversion/convert_beat_rnn.py` — loop over all 8 ensemble models
- Validate each
- Commit: `feat: convert all 8 RNNBeatProcessor ensemble models`

### Task 6.8: Convert RNNOnsetProcessor (8 ensemble models)

- Create: `models/conversion/convert_onset_rnn.py`
- Validate
- Commit: `feat: convert RNNOnsetProcessor ensemble to CoreML`

### Task 6.9: Convert CNNOnsetProcessor

- Create: `models/conversion/convert_onset_cnn.py`
- Validate
- Commit: `feat: convert CNNOnsetProcessor to CoreML`

### Task 6.10: Convert RNNDownBeatProcessor

- Create: `models/conversion/convert_downbeat_rnn.py`
- Validate
- Commit: `feat: convert RNNDownBeatProcessor to CoreML`

### Task 6.11: Convert CNNKeyRecognitionProcessor

- Create: `models/conversion/convert_key_cnn.py`
- Validate
- Commit: `feat: convert CNNKeyRecognitionProcessor to CoreML`

### Task 6.12: Convert CNNChordFeatureProcessor

- Create: `models/conversion/convert_chord_cnn.py`
- Validate
- Commit: `feat: convert CNNChordFeatureProcessor to CoreML`

### Task 6.13: Convert DeepChromaProcessor

- Create: `models/conversion/convert_deep_chroma.py`
- Validate
- Commit: `feat: convert DeepChromaProcessor to CoreML`

### Task 6.14: Convert RNNPianoNoteProcessor

- Create: `models/conversion/convert_piano_rnn.py`
- Validate
- Commit: `feat: convert RNNPianoNoteProcessor to CoreML`

### Task 6.15: Convert RNNBarProcessor

- Create: `models/conversion/convert_bar_rnn.py`
- Validate
- Commit: `feat: convert RNNBarProcessor to CoreML`

### Task 6.16: Ensemble Inference Orchestration

- Create: `Sources/MetalMomCore/ML/EnsembleRunner.swift` — load N models, parallel prediction, averaging
- Tests with beat ensemble
- Commit: `feat: add ensemble inference runner`

### Task 6.17: Chunked Inference for Long Audio

- Create: `Sources/MetalMomCore/ML/ChunkedInference.swift` — split into overlapping chunks, run inference, merge with overlap-add or max-pooling at boundaries
- Tests with sequences exceeding model input length
- Commit: `feat: add chunked inference for long audio`

---

## Phase 7: Neural Features (madmom Parity)

> Tier 3 parity testing: compare against ground truth annotations using MIR evaluation metrics, not element-wise comparison against madmom output.

### Task 7.1: Neural Beat Tracking (RNN + DBN)

- Create: `Sources/MetalMomCore/Rhythm/NeuralBeatTracker.swift` — CoreML ensemble inference → DBN/HMM decoding
- Bridge, Python, Tier 3 parity (beat F-measure), compat shim (madmom)
- Commit: `feat: add neural beat tracking (RNN + DBN)`

### Task 7.2: Neural Onset Detection (CNN + RNN)

- Create: `Sources/MetalMomCore/Rhythm/NeuralOnsetDetector.swift`
- Bridge, Python, Tier 3 parity (onset P/R/F), compat shim
- Commit: `feat: add neural onset detection`

### Task 7.3: Spectral Onset Methods (SuperFlux, Complex Flux)

- Modify: `Sources/MetalMomCore/Rhythm/OnsetDetection.swift` — add superflux, complex_flux, HFC, KL divergence
- Tests, Tier 1 parity
- Commit: `feat: add SuperFlux and complex flux onset detection`

### Task 7.4: Downbeat Detection

- Create: `Sources/MetalMomCore/Rhythm/Downbeat.swift`
- Uses RNNDownBeatProcessor model from Task 6.10
- Bridge, Python, Tier 3 parity, compat shim
- Commit: `feat: add downbeat detection`

### Task 7.5: Comb Filter Tempo Estimation

- Create: `Sources/MetalMomCore/Filters/CombFilter.swift`
- Modify: `Sources/MetalMomCore/Rhythm/TempoEstimation.swift` — add comb filter method
- Tests
- Commit: `feat: add comb filter tempo estimation`

### Task 7.6: Key Detection (CNN)

- Create: `Sources/MetalMomCore/Harmony/KeyDetection.swift`
- Uses CNNKeyRecognitionProcessor model from Task 6.11
- Bridge, Python, Tier 3 parity (key accuracy), compat shim
- Commit: `feat: add CNN key detection`

### Task 7.7: Chord Recognition (Deep Chroma + CRF)

- Create: `Sources/MetalMomCore/Harmony/ChordRecognition.swift`
- Uses DeepChromaProcessor (6.13) + CNNChordFeatureProcessor (6.12)
- Bridge, Python, Tier 3 parity (chord accuracy), compat shim
- Commit: `feat: add chord recognition`

### Task 7.8: Piano Transcription (RNN + ADSR-HMM)

- Create: `Sources/MetalMomCore/Pitch/PianoTranscription.swift`
- Uses RNNPianoNoteProcessor model from Task 6.14
- Bridge, Python, Tier 3 parity (note F-measure), compat shim
- Commit: `feat: add polyphonic piano transcription`

---

## Phase 8: Remaining Features

### Task 8.1: CQT / VQT / Hybrid CQT

- Create: `Sources/MetalMomCore/Spectral/CQT.swift`
- Bridge, Python, Tier 2 parity (same peak frequencies, energy within 1e-3), compat shim
- Commit: `feat: add CQT, VQT, and hybrid CQT`

### Task 8.2: Reassigned Spectrogram

- Create: `Sources/MetalMomCore/Spectral/Reassigned.swift`
- Bridge, Python, compat shim
- Commit: `feat: add reassigned spectrogram`

### Task 8.3: Phase Vocoder and Griffin-Lim

- Create: `Sources/MetalMomCore/Spectral/PhaseVocoder.swift` — phase_vocoder, griffinlim, griffinlim_cqt
- Bridge, Python, Tier 2 parity (Griffin-Lim SNR > 15 dB), compat shim
- Commit: `feat: add phase vocoder and Griffin-Lim`

### Task 8.4: PCEN and Weighting Curves

- Modify: `Sources/MetalMomCore/Spectral/Scaling.swift` — add PCEN, A/B/C/D weighting
- Tests, Tier 1 parity, compat shim
- Commit: `feat: add PCEN and frequency weighting curves`

### Task 8.5: Chroma Variants (CQT, CENS, VQT, Deep)

- Modify: `Sources/MetalMomCore/Features/Chroma.swift` — add CQT/CENS/VQT chroma + deep chroma
- Tests, parity, compat shims
- Commit: `feat: add chroma variants (CQT, CENS, VQT, deep)`

### Task 8.6: Feature Inversion (mel_to_audio, mfcc_to_mel, etc.)

- Create: `Sources/MetalMomCore/Features/Inversion.swift`
- Bridge, Python, Tier 2 parity (reconstruction SNR > 20 dB), compat shim
- Commit: `feat: add feature inversion functions`

### Task 8.7: NMF Decomposition

- Create: `Sources/MetalMomCore/Decompose/NMF.swift`
- Bridge, Python, Tier 2 parity (reconstruction error within 10% of librosa), compat shim
- Commit: `feat: add non-negative matrix factorization`

### Task 8.8: Nearest-Neighbor Filter

- Create: `Sources/MetalMomCore/Decompose/NNFilter.swift`
- Bridge, Python, compat shim
- Commit: `feat: add nearest-neighbor spectrogram filter`

### Task 8.9: Recurrence Matrix, Cross-Similarity, and RQA

- Create: `Sources/MetalMomCore/Segment/Recurrence.swift` — includes recurrence quantification analysis
- Bridge, Python, compat shim
- Commit: `feat: add recurrence matrix, cross-similarity, and RQA`

### Task 8.10: DTW (Dynamic Time Warping)

- Create: `Sources/MetalMomCore/Segment/DTW.swift`
- Bridge, Python, compat shim
- Commit: `feat: add dynamic time warping`

### Task 8.11: Agglomerative Segmentation

- Create: `Sources/MetalMomCore/Segment/Clustering.swift`
- Bridge, Python, compat shim
- Commit: `feat: add agglomerative temporal segmentation`

### Task 8.12: Viterbi Decoding (Python API)

- Python: `sequence.viterbi()`, `sequence.viterbi_discriminative()`, `sequence.viterbi_binary()`
- Bridge to Swift HMM module
- Tests, parity, compat shim
- Commit: `feat: add Viterbi sequence decoding Python API`

### Task 8.13: Unit Conversion Functions

- Modify: `Sources/MetalMomCore/Convert/Units.swift` — all Hz/mel/midi/note/oct/time/frame/sample conversions
- Round-trip tests, bridge, Python (`convert.py`), compat shim
- Commit: `feat: add full unit conversion suite`

### Task 8.14: Additional Window Functions

- Modify: `Sources/MetalMomCore/Filters/Windows.swift` — Hamming, Blackman, Bartlett, Kaiser, etc.
- Tests
- Commit: `feat: add Hamming, Blackman, and Kaiser windows`

### Task 8.15: Precompute Semitone Bandpass Filter Coefficients

- Create: `scripts/generate_filter_coefficients.py` — use scipy.signal.ellip() for all semitones (C1-B8) at sr=22050/44100/48000, export as JSON
- Create: `Sources/MetalMomCore/Filters/SemitoneBandpass.swift` — load pre-computed b/a coefficients, apply via vDSP_deq22 biquad cascade
- Tests, bridge, Python
- Commit: `feat: add semitone bandpass filterbank with pre-computed coefficients`

### Task 8.16: Display Module

- Create: `python/metalmom/display.py` — `specshow()`, `waveshow()` (pure Python/matplotlib)
- Tests, compat shim
- Commit: `feat: add specshow and waveshow display functions`

### Task 8.17: madmom Signal Framing Compat

- Add madmom-specific STFT parameter handling (frame sizes, hop sizes, filterbank construction matching madmom conventions)
- Tests against madmom golden outputs
- Commit: `feat: add madmom-compatible signal framing`

### Task 8.18: Filters Module (Python)

- Create: `python/metalmom/filters.py` — mel, chroma, constant_q, semitone filterbank Python wrappers
- Compat shim
- Commit: `feat: add Python filters module`

### Task 8.19: Remaining Compat Shim Functions

- Sweep through librosa and madmom APIs for any functions not yet shimmed
- Add missing shims discovered during sweep
- Commit: `feat: add remaining compat shim functions from sweep`

---

## Phase 9: Compatibility Shim Audit

> Compat shims were built incrementally in Phases 2-8. This phase is a **completeness audit**, not a from-scratch build. Focus is on signature verification, gap filling, and the madmom class-based API.

### Task 9.1: librosa Signature Verification Tests

- Create: `Tests/test_compat_signatures.py` — automated `inspect.signature()` comparison against real librosa
- Run, fix any mismatches
- Commit: `test: add automated librosa signature compatibility checks`

### Task 9.2: librosa Completeness Audit

- Script to walk every public function in librosa, check if compat shim exists
- Fill gaps
- Commit: `feat: fill remaining librosa compat gaps from audit`

### Task 9.3: madmom Compat — Audio Module

- Create: `python/metalmom/compat/madmom/audio/signal.py`, `stft.py`, `spectrogram.py`, `filters.py`, `chroma.py`
- Tests
- Commit: `feat: add madmom compat shim — audio module`

### Task 9.4: madmom Compat — Features Module

- Create: `python/metalmom/compat/madmom/features/beats.py`, `onsets.py`, `tempo.py`, `key.py`, `chords.py`, `notes.py`, `downbeats.py`
- These wrap the class-based Processor API (RNNBeatProcessor, DBNBeatTrackingProcessor, etc.)
- Tests
- Commit: `feat: add madmom compat shim — features module`

### Task 9.5: madmom Compat — Evaluation Module

- Create: `python/metalmom/compat/madmom/evaluation/` — onset, beat, tempo eval
- Tests
- Commit: `feat: add madmom compat shim — evaluation module`

### Task 9.6: madmom Signature Verification Tests

- Create: `Tests/test_compat_madmom_signatures.py`
- Commit: `test: add automated madmom signature compatibility checks`

### Task 9.7: End-to-End Compat Integration Tests

- Create: `Tests/test_compat_e2e.py` — run common librosa and madmom workflows through compat shims
- Commit: `test: add end-to-end compat integration tests`

---

## Phase 10: Metal GPU Backend

> The ComputeBackend protocol and SmartDispatcher exist since Phase 1. All operations have `executeCPU()` implementations and `fatalError()` in `executeGPU()`. This phase fills in the GPU paths and sets real dispatch thresholds. Search for `fatalError.*Phase 10` to find all stubs.

### Task 10.1: Metal Device Initialization and Chip Profiling

- Create: `Sources/MetalMomCore/Dispatch/MetalBackend.swift` — `MTLDevice`, `MTLCommandQueue` setup, shader library loading
- Create: `Sources/MetalMomCore/Dispatch/ChipProfile.swift` — GPU family detection, core count, pre-baked threshold profiles for M1/M2/M3/M4
- Modify `SmartDispatcher` to use Metal when available
- Tests (tagged `.metal`)
- Commit: `feat: add Metal device initialization and chip profiling`

### Task 10.2: Metal STFT (MPSGraph FFT)

- Implement `executeGPU()` for STFT operation using MPSGraph FFT
- Tests: GPU STFT produces same output as CPU STFT (tagged `.metal`)
- Set STFT `dispatchThreshold` based on ChipProfile
- Commit: `feat: add Metal GPU STFT via MPSGraph`

### Task 10.3: Metal Elementwise Shaders

- Create: `Sources/MetalMomShaders/Elementwise.metal` — log, exp, power, abs, dB conversion
- Swift wrapper to load and dispatch shaders
- Tests (tagged `.metal`)
- Commit: `feat: add Metal elementwise compute shaders`

### Task 10.4: Metal Matrix Operations (Mel Filterbank)

- Wire mel filterbank application through MPS `MPSMatrixMultiplication`
- Tests: parity with CPU mel spectrogram (tagged `.metal`)
- Commit: `feat: add Metal mel filterbank via MPS matrix multiply`

### Task 10.5: Metal Reduction Shaders

- Create: `Sources/MetalMomShaders/Reduction.metal` — parallel sum, max, mean
- Tests (tagged `.metal`)
- Commit: `feat: add Metal parallel reduction shaders`

### Task 10.6: Metal Convolution Shaders

- Create: `Sources/MetalMomShaders/Convolution.metal` — 1D/2D convolution
- Tests (tagged `.metal`)
- Commit: `feat: add Metal convolution shaders`

### Task 10.7: Dispatch Threshold Calibration Script

- Create: `scripts/calibrate_thresholds.py` — benchmark each operation at various sizes on current hardware, find CPU/GPU crossover
- Output: threshold values for ChipProfile
- Commit: `feat: add dispatch threshold calibration script`

### Task 10.8: Pipeline Fusion — MFCC

- Fused MFCC pipeline: entire STFT→mel→log→DCT chain in single dispatch decision and command buffer
- Benchmark: fused vs per-op dispatch
- Commit: `feat: add fused MFCC pipeline with single dispatch`

---

## Phase 11: Distribution & CI

> CI uses CPU-only testing. Metal tests are tagged and run locally or on self-hosted runners.

### Task 11.1: Wheel Build Script

- Create: `scripts/build_wheel.sh` — build dylib, copy to package, build wheel
- Adjust pyproject.toml for wheel bundling
- Test: `pip install dist/metalmom-*.whl` in clean venv
- Commit: `feat: add wheel build script`

### Task 11.2: GitHub Actions — CI (CPU-Only)

- Create: `.github/workflows/ci.yml` — build Swift, build dylib, run Swift tests (excluding .metal tags), run Python tests
- Metal tests explicitly excluded via test filter
- Commit: `ci: add GitHub Actions CI pipeline (CPU-only)`

### Task 11.3: GitHub Actions — Parity Tests

- Create: `.github/workflows/parity.yml` — weekly side-by-side comparison
- Commit: `ci: add weekly parity test pipeline`

### Task 11.4: GitHub Actions — API Drift Detection

- Create: `Tests/parity/check_api_surface.py` — introspect librosa/madmom API
- Create: `scripts/ci/file_api_drift_issues.py` — auto-file GitHub issues
- Create: `.github/workflows/api-parity.yml`
- Commit: `ci: add weekly API drift detection with auto-issue filing`

### Task 11.5: GitHub Actions — Release

- Create: `.github/workflows/release.yml` — build wheels for py3.11/3.12/3.13, publish to PyPI
- Commit: `ci: add release pipeline`

### Task 11.6: README and Licenses

- Create: `README.md`
- Create: `LICENSE` (MIT)
- Create: `LICENSE-MODELS` (CC BY-NC-SA 4.0)
- Commit: `docs: add README and license files`

### Task 11.7: .gitattributes for LFS

- Create: `.gitattributes` — `*.mlpackage filter=lfs`
- Run: `git lfs install && git lfs track "*.mlpackage"`
- Commit: `chore: configure Git LFS for CoreML models`

---

## Execution Notes

- **Phase 1 is the critical path.** It establishes the 3-target architecture, dispatch protocol, and end-to-end pipeline.
- **Phases 2-9 can partially parallelize** — e.g., Phase 6 (ML engine) is independent of Phase 5 (pitch/effects).
- **Phase 10 (Metal GPU) intentionally comes late.** Get everything working on Accelerate first, then add GPU acceleration. The protocol-first approach (Phase 1) means Phase 10 is additive, not a rewrite.
- **Compat shims are incremental.** Each feature task includes its corresponding shim. Phase 9 is an audit, not a build.
- **Model conversion has a spike gate.** Task 6.6 proves the approach works. If it fails, reassess before converting all 9 model families.
- **Each commit should leave the project in a buildable, testable state.**
- **Run `swift test && .venv/bin/pytest Tests/ -v` after every task** to catch regressions early.
- **CI is CPU-only.** Metal tests are tagged `.metal` and run locally on Apple Silicon.
