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
