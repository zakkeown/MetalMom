# MetalMom iOS Port — "The Minivan"

**Date:** 2026-02-18
**Status:** Approved

## Goal

Ship MetalMom as a multi-platform library supporting both macOS (existing) and iOS 17+. Distribute via Swift Package Manager (primary) and pre-built XCFramework (secondary). File-based audio analysis only for v1; architect for real-time audio capture later.

## Approach

Monorepo — all iOS support lives in the existing MetalMom repository. The Swift core (`MetalMomCore`) is already 95% iOS-compatible. No platform-conditional code exists today. The work is primarily build/distribution plumbing plus `Sendable` conformance for Swift concurrency.

## Portability Assessment

| Component | iOS Status | Notes |
|-----------|-----------|-------|
| Metal shaders | Ready | Runtime compilation via `makeLibrary(source:)` |
| Accelerate/vDSP | Ready | All functions available on iOS 17+ |
| CoreML | Ready | Full iOS 17+ support |
| AVFoundation audio I/O | Ready | High-level APIs only |
| MPS/MPSGraph | Ready | iOS 17+ |
| C bridge (`@_cdecl`) | Needs static build | Currently dynamic-only |
| File I/O | Needs sandbox awareness | No `/tmp` or `~` paths |
| Python bindings | N/A on iOS | Swift core is cleanly separable |

## 1. Package.swift Changes

Add a static library product for iOS consumers alongside the existing dynamic product for macOS/Python:

```swift
products: [
    .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
    .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
    .library(name: "MetalMomBridgeStatic", type: .static, targets: ["MetalMomBridge"]),
]
```

iOS consumers import `MetalMomCore` directly (Swift-native API) or `MetalMomBridgeStatic` (C exports for ObjC/C++ interop). macOS/Python continues using the dynamic `MetalMomBridge`.

Guard `ProfilingRunner` with `#if os(macOS)` so it compiles to a no-op on iOS.

## 2. XCFramework Build

New script `scripts/build_xcframework.sh` using `swift-create-xcframework` (or its maintained fork) to produce a universal XCFramework:

- iOS device (arm64)
- iOS simulator (arm64 + x86_64)
- macOS (arm64 + x86_64)

Fallback: raw `xcodebuild archive` + `xcodebuild -create-xcframework` commands.

XCFramework attached to GitHub Releases as a binary artifact. SPM consumers use source; non-SPM consumers embed the XCFramework.

## 3. CoreML Model Distribution

### Bundled Core Models (~5-10)

The most commonly used model families ship as SPM resources inside the package:

- RNN beat processor
- RNN onset processor
- RNN downbeat processor
- CNN onset detector
- Comb filter tempo estimator

### On-Demand Models via Hugging Face (~57+)

Remaining models hosted on `zkeown/metalmom-coreml-models` HF repo:

```
zkeown/metalmom-coreml-models/
├── README.md
├── core/                    # Mirrored bundled models
├── extended/                # On-demand models
└── metadata.json            # Version, size, compatibility
```

### ModelDownloader API

```swift
public class ModelDownloader {
    public static let shared = ModelDownloader()

    /// Download a model family, compile, and cache locally
    public func download(_ family: ModelFamily,
                         progress: ((Double) -> Void)? = nil) async throws -> URL

    /// Check if a model is cached
    public func isCached(_ family: ModelFamily) -> Bool

    /// Clear cached models
    public func clearCache() throws
}
```

Storage: `Application Support/MetalMom/Models/` on iOS. Compiled `.mlmodelc` bundles cached separately. Cache respects iOS storage pressure (models are re-downloadable).

## 4. Code Changes

Minimal changes required — the core engine needs zero modifications.

### 4a. ModelRegistry iOS convenience

```swift
#if os(iOS)
extension ModelRegistry {
    public static func defaultiOS() -> ModelRegistry {
        // Discovers models in Bundle.main + Application Support cache
    }
}
#endif
```

### 4b. ProfilingRunner guard

Wrap `Sources/ProfilingRunner/main.swift` with `#if os(macOS)`.

### 4c. Test fixture paths

Replace hardcoded `/tmp/` references with `FileManager.default.temporaryDirectory` across test files.

### 4d. Documentation

Document sandbox path requirements for iOS consumers in AudioIO and ModelRegistry APIs.

## 5. Sendable Conformance

Required for Swift concurrency on iOS. Categorized by effort:

### Trivial (add conformance marker)

| Type | Justification |
|------|--------------|
| `SmartDispatcher` | Immutable after init |
| `AccelerateBackend` | Stateless singleton |
| `Profiler` | Holds only `OSSignposter` (Sendable) |
| `ChipProfile` | Pure value type |
| `KeyResult` | Pure value type |
| All public enums (20+) | Value types |

### Easy (`@unchecked Sendable`)

| Type | Justification |
|------|--------------|
| `ModelRegistry` | Already protected by `NSLock` |
| `InferenceEngine` | Immutable `MLModel` (CoreML is thread-safe) |
| `EnsembleRunner` | Immutable array of engines |

### Medium (small fix + `@unchecked Sendable`)

| Type | Fix Required |
|------|-------------|
| `MetalBackend` | Synchronize lazy `_defaultLibrary` init with `NSLock` |

### Signal (`@unchecked Sendable` with usage contract)

`Signal` wraps `UnsafeMutableBufferPointer<Float>` for stable memory addresses (Accelerate/Metal requirement). Mark as `@unchecked Sendable` with documented contract:

> Do not mutate a Signal while it is being read on another thread. Compute operations return new Signals and do not share mutable state, so this naturally holds in typical usage.

This avoids an architecture rewrite. Compute operations are self-contained — they allocate output Signals internally.

## 6. CI & Testing

### New GitHub Actions job

```yaml
test-ios:
  runs-on: macos-15
  steps:
    - uses: actions/checkout@v4
    - name: Build for iOS Simulator
      run: xcodebuild build -scheme MetalMomCore \
           -destination 'platform=iOS Simulator,name=iPhone 16'
    - name: Test on iOS Simulator
      run: xcodebuild test -scheme MetalMomTests \
           -destination 'platform=iOS Simulator,name=iPhone 16'
```

Metal is available on iOS simulator (Apple Silicon runners). CoreML works on simulator. Audio fixture loading via `Bundle.main` works. Python tests remain macOS-only.

### Test resources

Add audio fixtures and model files as test resources in Package.swift so they're available in the iOS simulator test bundle.

## 7. Architecture for Future Real-Time Audio

Design file-based APIs so real-time can be added without breaking changes:

- Compute operations accept `Signal` (contiguous float buffer) — a real-time capture engine would feed captured buffers as Signals
- No streaming/callback architecture in v1 — that's a future addition
- `@MainActor` annotations deferred to real-time phase

## Non-Goals for v1

- Real-time microphone capture
- SwiftUI demo app (separate repo if needed)
- CocoaPods/Carthage distribution (XCFramework covers non-SPM)
- iOS-specific Metal optimizations (existing GPU backend is sufficient)
- Actor-based Signal redesign (documented `@unchecked Sendable` is sufficient)
