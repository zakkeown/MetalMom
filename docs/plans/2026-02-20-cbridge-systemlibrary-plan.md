# MetalMomCBridge systemLibrary Conversion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert MetalMomCBridge from a regular `.target()` to `.systemLibrary()` so builds survive clean without a DerivedData workaround.

**Architecture:** Add an explicit `module.modulemap` to expose `metalmom.h`, then change the target type in Package.swift. Two files touched, nothing else moves.

**Tech Stack:** Swift Package Manager, Clang module maps

---

### Task 1: Create the module.modulemap

**Files:**
- Create: `Sources/MetalMomCBridge/module.modulemap`

**Step 1: Create the modulemap file**

```
module MetalMomCBridge {
    header "include/metalmom.h"
    export *
}
```

No `[system]` attribute — we want full compiler diagnostics on our own headers.

**Step 2: Verify the file exists alongside include/**

Run: `ls Sources/MetalMomCBridge/`
Expected: `include  module.modulemap`

**Step 3: Commit**

```bash
git add Sources/MetalMomCBridge/module.modulemap
git commit -m "feat: add explicit module.modulemap for MetalMomCBridge"
```

---

### Task 2: Convert target type in Package.swift

**Files:**
- Modify: `Package.swift:13-18`

**Step 1: Change the target declaration**

Replace:
```swift
.target(
    name: "MetalMomCBridge",
    dependencies: [],
    path: "Sources/MetalMomCBridge",
    publicHeadersPath: "include"
),
```

With:
```swift
.systemLibrary(
    name: "MetalMomCBridge",
    path: "Sources/MetalMomCBridge"
),
```

**Step 2: Commit**

```bash
git add Package.swift
git commit -m "feat: convert MetalMomCBridge to systemLibrary target"
```

---

### Task 3: Verify the build

**Step 1: Clean build from scratch**

Run: `swift package clean && swift build`
Expected: Build succeeds with no errors.

**Step 2: Run tests**

Run: `swift test`
Expected: All existing tests pass. `import MetalMomCBridge` resolves in MetalMomCore and MetalMomBridge.

**Step 3: Release build**

Run: `swift build -c release`
Expected: Release build succeeds (dylib path works).

**Step 4: Verify build scripts reference still valid**

Run: `ls Sources/MetalMomCBridge/include/metalmom.h`
Expected: File exists at the same path referenced by `scripts/build_dylib.sh` and `scripts/build_xcframework.sh`.

---

### Task 4: Final commit (squash if preferred)

**Step 1: Verify clean state**

Run: `git status`
Expected: Clean working tree (all changes committed in Tasks 1-2).

If Tasks 1-2 were committed separately and a squash is preferred:
```bash
git rebase -i HEAD~2
```
Squash into: `fix: convert MetalMomCBridge to systemLibrary target to survive clean builds`
