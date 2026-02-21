# MetalMomCBridge: Convert to systemLibrary Target

**Date:** 2026-02-20
**Status:** Approved

## Problem

MetalMomCBridge is a header-only SPM C target (`Sources/MetalMomCBridge/include/metalmom.h`). SPM requires at least one compilable source file in regular `.target()` C targets, so builds fail after a clean. The current workaround — manually creating an `empty.c` in DerivedData — is lost on every `swift package clean` or DerivedData wipe.

## Solution

Convert MetalMomCBridge from a regular `.target()` to a `.systemLibrary()` target with an explicit `module.modulemap`. System library targets don't require compilable sources — they exist solely to expose headers via a modulemap.

## Design

### Directory layout

```
Sources/MetalMomCBridge/
  module.modulemap    ← NEW
  include/
    metalmom.h        ← unchanged
```

### module.modulemap

```
module MetalMomCBridge {
    header "include/metalmom.h"
    export *
}
```

No `[system]` attribute — this is our own code and we want full compiler diagnostics.

### Package.swift

```swift
// Before:
.target(
    name: "MetalMomCBridge",
    dependencies: [],
    path: "Sources/MetalMomCBridge",
    publicHeadersPath: "include"
),

// After:
.systemLibrary(
    name: "MetalMomCBridge",
    path: "Sources/MetalMomCBridge"
),
```

### What stays the same

- `import MetalMomCBridge` in all Swift files
- Header path (`Sources/MetalMomCBridge/include/metalmom.h`) — build scripts unaffected
- Dependency graph: MetalMomCore depends on MetalMomCBridge

### Migration path for future C code

If `.c` implementation files are needed later, convert back to a regular `.target()` with `publicHeadersPath: "include"` and add the `.c` files. The modulemap can optionally be kept (SPM respects explicit modulemaps in regular C targets) or removed.

## Verification

1. `swift build` succeeds
2. `swift test` passes (existing XCTests exercise the bridge)
3. `swift build -c release` succeeds (dylib path)
4. `import MetalMomCBridge` resolves in MetalMomCore and MetalMomBridge
