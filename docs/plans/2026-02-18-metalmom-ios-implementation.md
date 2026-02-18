# MetalMom iOS Port ("The Minivan") Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship MetalMom as a multi-platform Swift library supporting iOS 17+ via SPM and XCFramework, with CoreML models hosted on Hugging Face.

**Architecture:** Monorepo approach — the existing Swift core is already 95% iOS-compatible. Changes are primarily build/distribution plumbing, Sendable conformances for Swift concurrency, and a new model download utility. No core engine changes needed.

**Tech Stack:** Swift 5.9+, SPM, Metal, Accelerate, CoreML, AVFoundation, XCFramework, GitHub Actions, Hugging Face Hub

---

### Task 1: Add Static Library Product to Package.swift

**Files:**
- Modify: `Package.swift`

**Step 1: Add MetalMomBridgeStatic product**

In `Package.swift`, add a static library product alongside the existing dynamic one:

```swift
products: [
    .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
    .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
    .library(name: "MetalMomBridgeStatic", type: .static, targets: ["MetalMomBridge"]),
],
```

**Step 2: Verify both products build**

Run: `swift build`
Expected: BUILD SUCCEEDED (both dynamic and static products resolve)

**Step 3: Commit**

```bash
git add Package.swift
git commit -m "feat: add static MetalMomBridge product for iOS embedding"
```

---

### Task 2: Guard ProfilingRunner for macOS Only

**Files:**
- Modify: `Sources/ProfilingRunner/main.swift`

**Step 1: Wrap entire file with platform guard**

Replace the contents of `Sources/ProfilingRunner/main.swift` with:

```swift
#if os(macOS)
import Foundation
import MetalMomCore

@main
struct ProfilingRunner {
    static func main() throws {
        // ... existing code unchanged (lines 7-77) ...
    }
}
#else
// ProfilingRunner is macOS-only (requires command-line arguments and tilde expansion).
@main
struct ProfilingRunner {
    static func main() {
        print("ProfilingRunner is only available on macOS.")
    }
}
#endif
```

Keep all existing code inside the `#if os(macOS)` block unchanged.

**Step 2: Build and verify**

Run: `swift build`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add Sources/ProfilingRunner/main.swift
git commit -m "feat: guard ProfilingRunner for macOS only"
```

---

### Task 3: Add Sendable Conformance — Trivial Types

Add `Sendable` markers to types that are already thread-safe.

**Files:**
- Modify: `Sources/MetalMomCore/Dispatch/SmartDispatch.swift:5`
- Modify: `Sources/MetalMomCore/Dispatch/AccelerateBackend.swift:5`
- Modify: `Sources/MetalMomCore/Dispatch/Profiler.swift:6`
- Modify: `Sources/MetalMomCore/Dispatch/ChipProfile.swift:5`
- Modify: `Sources/MetalMomCore/Harmony/KeyDetection.swift` (KeyResult struct)

**Step 1: Mark classes as Sendable**

`SmartDispatch.swift` line 5 — change:
```swift
public final class SmartDispatcher {
```
to:
```swift
public final class SmartDispatcher: Sendable {
```

`AccelerateBackend.swift` line 5 — change:
```swift
public final class AccelerateBackend {
```
to:
```swift
public final class AccelerateBackend: Sendable {
```

`Profiler.swift` line 6 — change:
```swift
public final class Profiler {
```
to:
```swift
public final class Profiler: @unchecked Sendable {
```
(`@unchecked` because `OSSignposter` may not formally conform but is thread-safe)

`ChipProfile.swift` line 5 — change:
```swift
public struct ChipProfile {
```
to:
```swift
public struct ChipProfile: Sendable {
```

`ChipProfile.swift` line 13 — change:
```swift
public enum GPUFamily: Comparable {
```
to:
```swift
public enum GPUFamily: Comparable, Sendable {
```

`ChipProfile.swift` line 65 — change:
```swift
public enum OperationType {
```
to:
```swift
public enum OperationType: Sendable {
```

For `KeyResult` in `KeyDetection.swift`, find the struct definition and add `Sendable`:
```swift
public struct KeyResult: Sendable {
```

**Step 2: Build and test**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/ Sources/MetalMomCore/Harmony/KeyDetection.swift
git commit -m "feat: add Sendable conformance to trivial types"
```

---

### Task 4: Add Sendable Conformance — Error Enums

**Files:**
- Modify: `Sources/MetalMomCore/ML/InferenceEngine.swift:7`
- Modify: `Sources/MetalMomCore/ML/EnsembleRunner.swift:8`
- Modify: `Sources/MetalMomCore/ML/ChunkedInference.swift:16`
- Modify: `Sources/MetalMomCore/ML/ModelRegistry.swift:6`
- Modify: `Sources/MetalMomCore/Audio/AudioIO.swift` (AudioIOError enum)

**Step 1: Add Sendable to each error enum**

`InferenceEngine.swift` line 7 — change:
```swift
public enum InferenceError: Error, Equatable {
```
to:
```swift
public enum InferenceError: Error, Equatable, Sendable {
```

`EnsembleRunner.swift` line 8 — change:
```swift
public enum EnsembleError: Error, Equatable {
```
to:
```swift
public enum EnsembleError: Error, Equatable, Sendable {
```

`ChunkedInference.swift` line 16 — change:
```swift
public enum ChunkedInferenceError: Error, Equatable {
```
to:
```swift
public enum ChunkedInferenceError: Error, Equatable, Sendable {
```

`ModelRegistry.swift` line 6 — change:
```swift
public enum ModelRegistryError: Error, Equatable {
```
to:
```swift
public enum ModelRegistryError: Error, Equatable, Sendable {
```

`AudioIO.swift` — find `AudioIOError` enum and add `Sendable`:
```swift
public enum AudioIOError: Error, LocalizedError, Sendable {
```

Also add `Sendable` to other value enums:
- `MergeStrategy` in `ChunkedInference.swift` line 5
- `CombineStrategy` in `EnsembleRunner.swift` line 25
- `SignalDType` in `Signal.swift` line 5

**Step 2: Build and test**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/ML/ Sources/MetalMomCore/Audio/
git commit -m "feat: add Sendable conformance to error enums and value types"
```

---

### Task 5: Add Sendable Conformance — ML Classes

**Files:**
- Modify: `Sources/MetalMomCore/ML/ModelRegistry.swift:31`
- Modify: `Sources/MetalMomCore/ML/InferenceEngine.swift:32`
- Modify: `Sources/MetalMomCore/ML/EnsembleRunner.swift:22`

**Step 1: Mark ML classes as @unchecked Sendable**

`ModelRegistry.swift` line 31 — change:
```swift
public final class ModelRegistry {
```
to:
```swift
public final class ModelRegistry: @unchecked Sendable {
```
Justification: Already thread-safe via `NSLock` on all public APIs.

`InferenceEngine.swift` line 32 — change:
```swift
public final class InferenceEngine {
```
to:
```swift
public final class InferenceEngine: @unchecked Sendable {
```
Justification: Immutable stored properties. `MLModel` is internally thread-safe.

`EnsembleRunner.swift` line 22 — change:
```swift
public final class EnsembleRunner {
```
to:
```swift
public final class EnsembleRunner: @unchecked Sendable {
```
Justification: Immutable stored properties. All engines are now Sendable.

**Step 2: Build and test**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/ML/
git commit -m "feat: add @unchecked Sendable to ML classes"
```

---

### Task 6: Thread-Safe MetalBackend Lazy Property + Sendable

**Files:**
- Modify: `Sources/MetalMomCore/Dispatch/MetalBackend.swift`

**Step 1: Write a test for concurrent MetalBackend access**

Create a test in `Tests/MetalMomTests/` (or add to existing dispatch tests) that verifies concurrent `defaultLibrary` access doesn't crash:

```swift
func testConcurrentDefaultLibraryAccess() throws {
    guard let backend = MetalBackend.shared else {
        throw XCTSkip("Metal unavailable")
    }
    let group = DispatchGroup()
    for _ in 0..<100 {
        group.enter()
        DispatchQueue.global().async {
            _ = backend.defaultLibrary
            group.leave()
        }
    }
    group.wait()
}
```

**Step 2: Run test to verify it passes (or crashes without fix)**

Run: `swift test --filter testConcurrentDefaultLibraryAccess`

**Step 3: Add NSLock to MetalBackend**

In `MetalBackend.swift`, add a lock and synchronize the lazy property:

After line 18 (`private var _defaultLibrary: MTLLibrary?`), add:
```swift
private let libraryLock = NSLock()
```

Replace lines 23-28 (the `defaultLibrary` computed property) with:
```swift
public var defaultLibrary: MTLLibrary? {
    libraryLock.lock()
    defer { libraryLock.unlock() }
    if _defaultLibrary == nil {
        _defaultLibrary = device.makeDefaultLibrary()
    }
    return _defaultLibrary
}
```

Also change line 6 to add Sendable:
```swift
public final class MetalBackend: @unchecked Sendable {
```

The `lazy var shaders` on line 39 also needs synchronization. Replace with:
```swift
private var _shaders: MetalShaders?
private var _shadersInitialized = false
private let shadersLock = NSLock()

public var shaders: MetalShaders? {
    shadersLock.lock()
    defer { shadersLock.unlock() }
    if !_shadersInitialized {
        _shaders = MetalShaders(device: device)
        _shadersInitialized = true
    }
    return _shaders
}
```

**Step 4: Run all tests**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 5: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/MetalBackend.swift Tests/
git commit -m "feat: thread-safe MetalBackend lazy properties + Sendable"
```

---

### Task 7: Signal @unchecked Sendable

**Files:**
- Modify: `Sources/MetalMomCore/Audio/Signal.swift:13`

**Step 1: Add @unchecked Sendable to Signal**

`Signal.swift` line 13 — change:
```swift
public final class Signal {
```
to:
```swift
/// - Important: Thread-safety contract — do not mutate a Signal while it is being read
///   on another thread. Compute operations return new Signals and do not share mutable
///   state, so this naturally holds in typical usage.
public final class Signal: @unchecked Sendable {
```

**Step 2: Build and test**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/Audio/Signal.swift
git commit -m "feat: mark Signal as @unchecked Sendable with usage contract"
```

---

### Task 8: Fix Test Fixture Paths for iOS Compatibility

**Files:**
- Modify: `Tests/MetalMomTests/ModelRegistryTests.swift:74,82`
- Modify: `Tests/MetalMomTests/InferenceEngineTests.swift:39,47`

**Step 1: Replace /tmp/ paths in ModelRegistryTests.swift**

Line 74 — change:
```swift
let bogus = URL(fileURLWithPath: "/tmp/nonexistent_models_dir")
```
to:
```swift
let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_models_dir")
```

Line 82 — change:
```swift
let bogus = URL(fileURLWithPath: "/tmp/nonexistent_models_dir_\(UUID().uuidString)")
```
to:
```swift
let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_models_dir_\(UUID().uuidString)")
```

**Step 2: Replace /tmp/ paths in InferenceEngineTests.swift**

Line 39 — change:
```swift
let bogus = URL(fileURLWithPath: "/tmp/nonexistent_model.mlmodelc")
```
to:
```swift
let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_model.mlmodelc")
```

Line 47 — change:
```swift
let bogus = URL(fileURLWithPath: "/tmp/nonexistent_model.mlmodel")
```
to:
```swift
let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_model.mlmodel")
```

**Step 3: Run tests to verify nothing broke**

Run: `swift test --filter ModelRegistryTests && swift test --filter InferenceEngineTests`
Expected: All tests pass

**Step 4: Commit**

```bash
git add Tests/MetalMomTests/ModelRegistryTests.swift Tests/MetalMomTests/InferenceEngineTests.swift
git commit -m "fix: use FileManager.temporaryDirectory instead of /tmp/ in tests"
```

---

### Task 9: Extend ChipProfile for iOS GPU Families

**Files:**
- Modify: `Sources/MetalMomCore/Dispatch/ChipProfile.swift`
- Test: `Tests/MetalMomTests/` (add or extend ChipProfile tests)

**Step 1: Write a test for A-series GPU detection**

```swift
func testChipProfileGPUFamilyDetection() throws {
    guard let backend = MetalBackend.shared else {
        throw XCTSkip("Metal unavailable")
    }
    let profile = backend.chipProfile
    // On any Apple Silicon device, GPU family should not be .unknown
    XCTAssertNotEqual(profile.gpuFamily, .unknown,
        "Should detect a known GPU family on Apple Silicon")
    XCTAssertGreaterThan(profile.estimatedCoreCount, 0)
}
```

**Step 2: Add A-series GPU families to ChipProfile**

In `ChipProfile.swift`, extend the `GPUFamily` enum (line 13):

```swift
public enum GPUFamily: Comparable, Sendable {
    case apple5   // A12 Bionic (iPhone XS/XR) — iOS 17 minimum
    case apple6   // A13 Bionic (iPhone 11)
    case apple7   // A14 / M1 family
    case apple8   // A15/A16 / M2 family
    case apple9   // A17 Pro / M3 / M4 family
    case unknown
}
```

Update the `init(device:)` detection (lines 20-31) to probe from highest to lowest:

```swift
public init(device: MTLDevice) {
    if device.supportsFamily(.apple9) {
        self.gpuFamily = .apple9
    } else if device.supportsFamily(.apple8) {
        self.gpuFamily = .apple8
    } else if device.supportsFamily(.apple7) {
        self.gpuFamily = .apple7
    } else if device.supportsFamily(.apple6) {
        self.gpuFamily = .apple6
    } else if device.supportsFamily(.apple5) {
        self.gpuFamily = .apple5
    } else {
        self.gpuFamily = .unknown
    }

    switch self.gpuFamily {
    case .apple5:  self.estimatedCoreCount = 4   // A12 (iPhone XS)
    case .apple6:  self.estimatedCoreCount = 4   // A13 (iPhone 11)
    case .apple7:  self.estimatedCoreCount = 8   // A14 / M1 base
    case .apple8:  self.estimatedCoreCount = 10  // A15/A16 / M2 base
    case .apple9:  self.estimatedCoreCount = 10  // A17 Pro / M3 / M4 base
    case .unknown: self.estimatedCoreCount = 4
    }

    self.maxBufferLength = device.maxBufferLength
    self.hasNonUniformThreadgroups = device.supportsFamily(.apple4)
}
```

Update `threshold(for:)` to handle A-series GPUs with higher thresholds (smaller GPUs need larger data to benefit from GPU dispatch):

```swift
public func threshold(for operation: OperationType) -> Int {
    let isMobile = gpuFamily <= .apple6  // A12/A13 have fewer GPU cores
    switch operation {
    case .stft:
        return isMobile ? 32768 : (gpuFamily >= .apple9 ? 8192 : 16384)
    case .matmul:
        return isMobile ? 16384 : (gpuFamily >= .apple9 ? 4096 : 8192)
    case .elementwise:
        return isMobile ? 262144 : (gpuFamily >= .apple9 ? 65536 : 131072)
    case .reduction:
        return isMobile ? 131072 : (gpuFamily >= .apple9 ? 32768 : 65536)
    case .convolution:
        return isMobile ? 32768 : (gpuFamily >= .apple9 ? 8192 : 16384)
    }
}
```

**Step 3: Run all tests**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 4: Commit**

```bash
git add Sources/MetalMomCore/Dispatch/ChipProfile.swift Tests/
git commit -m "feat: extend ChipProfile with A-series iOS GPU families"
```

---

### Task 10: Add iOS Simulator CI Job

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add iOS build-and-test job**

Add a new job to `.github/workflows/ci.yml`:

```yaml
  build-and-test-ios:
    name: Build & Test (iOS Simulator)
    runs-on: macos-15

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Select Xcode 16
        run: sudo xcode-select -s /Applications/Xcode_16.app

      - name: Build for iOS Simulator
        run: |
          xcodebuild build \
            -scheme MetalMomCore \
            -destination 'platform=iOS Simulator,name=iPhone 16' \
            -skipPackagePluginValidation

      - name: Test on iOS Simulator
        run: |
          xcodebuild test \
            -scheme MetalMomTests \
            -destination 'platform=iOS Simulator,name=iPhone 16' \
            -skipPackagePluginValidation
```

**Step 2: Verify the existing macOS job still works**

Run locally: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add iOS Simulator build and test job"
```

---

### Task 11: Create XCFramework Build Script

**Files:**
- Create: `scripts/build_xcframework.sh`

**Step 1: Write the build script**

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/xcframework"

echo "Building MetalMom XCFramework..."
cd "$PROJECT_DIR"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Archive for iOS device
echo "Archiving for iOS device (arm64)..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS" \
    -archivePath "$BUILD_DIR/MetalMomCore-iOS.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Archive for iOS Simulator
echo "Archiving for iOS Simulator..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS Simulator" \
    -archivePath "$BUILD_DIR/MetalMomCore-iOSSimulator.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Archive for macOS
echo "Archiving for macOS..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=macOS" \
    -archivePath "$BUILD_DIR/MetalMomCore-macOS.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Create XCFramework
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
    -archive "$BUILD_DIR/MetalMomCore-iOS.xcarchive" -framework MetalMomCore.framework \
    -archive "$BUILD_DIR/MetalMomCore-iOSSimulator.xcarchive" -framework MetalMomCore.framework \
    -archive "$BUILD_DIR/MetalMomCore-macOS.xcarchive" -framework MetalMomCore.framework \
    -output "$BUILD_DIR/MetalMomCore.xcframework"

echo ""
echo "XCFramework created at: $BUILD_DIR/MetalMomCore.xcframework"
echo "Done."
```

**Step 2: Make executable and test locally**

Run: `chmod +x scripts/build_xcframework.sh && ./scripts/build_xcframework.sh`
Expected: XCFramework created at `.build/xcframework/MetalMomCore.xcframework`

Note: This may require Xcode project generation or may work directly with SPM in Xcode 16. If `xcodebuild` can't find the scheme, fall back to `swift package generate-xcodeproj` first, or use `swift build` per-platform and assemble manually. Adjust the script as needed based on what works.

**Step 3: Commit**

```bash
git add scripts/build_xcframework.sh
git commit -m "feat: add XCFramework build script for multi-platform distribution"
```

---

### Task 12: Create ModelDownloader Utility

**Files:**
- Create: `Sources/MetalMomCore/ML/ModelDownloader.swift`
- Test: `Tests/MetalMomTests/ModelDownloaderTests.swift`

**Step 1: Write failing tests**

```swift
import XCTest
@testable import MetalMomCore

final class ModelDownloaderTests: XCTestCase {

    func testModelFamilyRawValues() {
        // Verify all model families have valid raw values for URL construction
        for family in ModelFamily.allCases {
            XCTAssertFalse(family.rawValue.isEmpty,
                "Model family \(family) should have a non-empty raw value")
        }
    }

    func testDefaultCacheDirectory() {
        let downloader = ModelDownloader.shared
        let cacheDir = downloader.cacheDirectory
        XCTAssertTrue(cacheDir.path.contains("MetalMom"),
            "Cache directory should be in MetalMom namespace")
    }

    func testIsCachedReturnsFalseForMissingModel() {
        let downloader = ModelDownloader.shared
        XCTAssertFalse(downloader.isCached(.rnnBeatProcessor),
            "Should return false for model not yet downloaded")
    }

    func testClearCacheDoesNotThrow() {
        let downloader = ModelDownloader.shared
        XCTAssertNoThrow(try downloader.clearCache())
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter ModelDownloaderTests`
Expected: FAIL — `ModelDownloader` not defined

**Step 3: Implement ModelDownloader**

```swift
import Foundation

/// Families of pre-trained CoreML models available for download.
public enum ModelFamily: String, CaseIterable, Sendable {
    case rnnBeatProcessor = "rnn_beat_processor"
    case rnnOnsetProcessor = "rnn_onset_processor"
    case rnnDownbeatProcessor = "rnn_downbeat_processor"
    case cnnOnsetDetector = "cnn_onset_detector"
    case combFilterTempoEstimator = "comb_filter_tempo"
    case cnnChordRecognizer = "cnn_chord_recognizer"
    case keyDetector = "key_detector"
    case pianoTranscriber = "piano_transcriber"
    case spectralOnsetProcessor = "spectral_onset_processor"
}

/// Downloads and caches CoreML models from Hugging Face Hub.
///
/// Core models (beat tracking, onset, downbeat, tempo, onset detector) are
/// bundled with the package. Extended models are downloaded on demand.
///
/// Thread-safe: all public API is protected by an internal lock.
public final class ModelDownloader: @unchecked Sendable {
    public static let shared = ModelDownloader()

    /// Base URL for the Hugging Face model repository.
    public var repositoryURL: URL = URL(string: "https://huggingface.co/zkeown/metalmom-coreml-models/resolve/main/")!

    private let lock = NSLock()

    /// Local cache directory for downloaded models.
    public var cacheDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("MetalMom/Models")
    }

    private init() {}

    /// Check if a model family is already cached locally.
    public func isCached(_ family: ModelFamily) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let modelDir = cacheDirectory.appendingPathComponent("\(family.rawValue).mlmodelc")
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir) && isDir.boolValue
    }

    /// Download a model family from HF Hub, compile if needed, and cache locally.
    ///
    /// - Parameters:
    ///   - family: The model family to download.
    ///   - progress: Optional progress callback (0.0 to 1.0).
    /// - Returns: URL to the compiled `.mlmodelc` bundle.
    public func download(_ family: ModelFamily,
                         progress: ((Double) -> Void)? = nil) async throws -> URL {
        let cachedURL = cacheDirectory.appendingPathComponent("\(family.rawValue).mlmodelc")

        // Return cached if available
        if isCached(family) {
            return cachedURL
        }

        // Ensure cache directory exists
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        // Download from HF
        let remoteURL = repositoryURL
            .appendingPathComponent("extended")
            .appendingPathComponent("\(family.rawValue).mlpackage.zip")

        let (tempURL, _) = try await URLSession.shared.download(from: remoteURL)

        // Unzip and compile
        let unzipDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: unzipDir) }

        // Use Process to unzip (or implement zip extraction)
        let unzippedModel = unzipDir.appendingPathComponent("\(family.rawValue).mlpackage")

        // Move downloaded file
        let destZip = unzipDir.appendingPathComponent("model.zip")
        try FileManager.default.moveItem(at: tempURL, to: destZip)

        // Compile the model
        let compiledURL = try await compileModel(at: unzippedModel)

        // Move to cache
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            try FileManager.default.removeItem(at: cachedURL)
        }
        try FileManager.default.moveItem(at: compiledURL, to: cachedURL)

        return cachedURL
    }

    /// Clear all cached models.
    public func clearCache() throws {
        lock.lock()
        defer { lock.unlock() }
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
        }
    }

    // MARK: - Private

    private func compileModel(at url: URL) async throws -> URL {
        // CoreML compiles .mlpackage to .mlmodelc on device
        let compiled = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let compiledURL = try MLModel.compileModel(at: url)
                    continuation.resume(returning: compiledURL)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        return compiled
    }
}
```

Note: The `download` method's zip extraction step will need refinement during implementation — iOS doesn't have a `Process` API for `unzip`. Use Apple's `Compression` framework or a lightweight zip library. The exact implementation depends on how models are packaged on HF. The structure above captures the API contract.

**Step 4: Run tests**

Run: `swift test --filter ModelDownloaderTests`
Expected: All tests pass

**Step 5: Commit**

```bash
git add Sources/MetalMomCore/ML/ModelDownloader.swift Tests/MetalMomTests/ModelDownloaderTests.swift
git commit -m "feat: add ModelDownloader for HF-hosted CoreML models"
```

---

### Task 13: Add ModelRegistry iOS Convenience Initializer

**Files:**
- Modify: `Sources/MetalMomCore/ML/ModelRegistry.swift`

**Step 1: Add iOS convenience method**

Add at the bottom of `ModelRegistry.swift`, before the closing brace:

```swift
#if os(iOS) || os(visionOS)
extension ModelRegistry {
    /// Configure with default iOS model locations.
    ///
    /// Scans the app bundle's `MetalMom-Models` directory (for bundled core models)
    /// and the `ModelDownloader` cache directory (for downloaded models).
    public func configureDefaultiOS() {
        // Check for bundled models in the app bundle
        if let bundlePath = Bundle.main.path(forResource: "MetalMom-Models", ofType: nil) {
            configure(modelsDirectory: URL(fileURLWithPath: bundlePath))
        }
        // Also check the download cache
        let cacheDir = ModelDownloader.shared.cacheDirectory
        if FileManager.default.fileExists(atPath: cacheDir.path) {
            // If we already configured with bundle, we need a multi-directory approach
            // For v1, prioritize downloaded models (they may be newer)
            configure(modelsDirectory: cacheDir)
        }
    }
}
#endif
```

**Step 2: Build and test**

Run: `swift build && swift test`
Expected: BUILD SUCCEEDED, all tests pass (iOS extension not compiled on macOS, that's fine)

**Step 3: Commit**

```bash
git add Sources/MetalMomCore/ML/ModelRegistry.swift
git commit -m "feat: add iOS convenience initializer for ModelRegistry"
```

---

### Task 14: Create Hugging Face Model Repository

**Files:** None (external setup)

**Step 1: Create the HF repo**

Using HF CLI or web UI, create `zkeown/metalmom-coreml-models` with:
- README.md (model card describing the models)
- `metadata.json` with version and compatibility info
- `core/` directory with bundled models
- `extended/` directory with on-demand models

**Step 2: Upload core models**

Upload the 5 core model families from `models/converted/`:
- rnn_beat_processor
- rnn_onset_processor
- rnn_downbeat_processor
- cnn_onset_detector
- comb_filter_tempo

**Step 3: Upload extended models**

Upload remaining 62 model families from `models/converted/`.

**Step 4: Verify downloads work**

Test that `curl` can fetch a model from the repo:
```bash
curl -L https://huggingface.co/zkeown/metalmom-coreml-models/resolve/main/core/rnn_beat_processor.mlmodelc.zip -o /tmp/test_model.zip
```

---

### Task 15: Verify Full iOS Simulator Build Locally

**Files:** None (verification only)

**Step 1: Build MetalMomCore for iOS Simulator**

```bash
xcodebuild build \
    -scheme MetalMomCore \
    -destination 'platform=iOS Simulator,name=iPhone 16' \
    -skipPackagePluginValidation
```

Expected: BUILD SUCCEEDED

**Step 2: Build MetalMomBridge (static) for iOS Simulator**

```bash
xcodebuild build \
    -scheme MetalMomBridgeStatic \
    -destination 'platform=iOS Simulator,name=iPhone 16' \
    -skipPackagePluginValidation
```

Expected: BUILD SUCCEEDED

**Step 3: Run tests on iOS Simulator**

```bash
xcodebuild test \
    -scheme MetalMomTests \
    -destination 'platform=iOS Simulator,name=iPhone 16' \
    -skipPackagePluginValidation
```

Expected: All tests pass (Metal tests may skip if simulator doesn't support Metal on your machine)

**Step 4: Verify macOS still works**

```bash
swift build && swift test
```

Expected: BUILD SUCCEEDED, all tests pass

**Step 5: Commit any fixes discovered during verification**

```bash
git add -A
git commit -m "fix: address issues discovered during iOS simulator verification"
```

---

### Task 16: Update README and Documentation

**Files:**
- Modify: `README.md` (add iOS section)
- Update design doc if needed

**Step 1: Add iOS section to README**

Add a section covering:
- iOS 17+ support via SPM
- How to add MetalMom to an Xcode project
- XCFramework availability for non-SPM projects
- Model download setup (bundled vs. on-demand)
- Threading model (Sendable types, single-context-per-thread for C bridge)

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add iOS support documentation to README"
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Static library product in Package.swift | 5 min |
| 2 | Guard ProfilingRunner for macOS | 5 min |
| 3 | Sendable — trivial types | 10 min |
| 4 | Sendable — error enums | 10 min |
| 5 | Sendable — ML classes | 5 min |
| 6 | Thread-safe MetalBackend + Sendable | 20 min |
| 7 | Signal @unchecked Sendable | 5 min |
| 8 | Fix /tmp/ test fixture paths | 10 min |
| 9 | ChipProfile A-series GPU families | 20 min |
| 10 | iOS Simulator CI job | 10 min |
| 11 | XCFramework build script | 30 min |
| 12 | ModelDownloader utility | 45 min |
| 13 | ModelRegistry iOS convenience | 10 min |
| 14 | HF model repository setup | 30 min |
| 15 | Full iOS simulator verification | 20 min |
| 16 | README and docs update | 15 min |

**Total estimated: ~4 hours of implementation work.**
