# MetalMom Design Document

**Date:** 2026-02-17
**Revised:** 2026-02-17 (stress test review)
**Status:** Approved

## Overview

MetalMom is a GPU-accelerated audio/music analysis library built on Apple Metal, targeting feature and accuracy parity with both librosa (v0.11.0) and madmom (v0.16.1). It provides a native Swift core engine, a Python package with minimal-copy NumPy interop, and drop-in compatibility shims for both librosa and madmom APIs.

### Goals

- Full feature parity with librosa and madmom combined
- Accuracy parity or improvement over both libraries
- Significant performance improvement via Metal GPU acceleration
- Drop-in Python API compatibility (one-line import change)
- Parity testing from day one

### Non-Goals (for initial release)

- Cross-platform support (Apple Silicon only, no apologies)
- Real-time streaming (architected for it, but not in v1)
- Training new ML models (inference only with converted madmom weights)

## Architecture

Metal-first with smart dispatch. Every compute operation routes to its fastest backend automatically — Metal GPU for large workloads, Accelerate/vDSP for small ones. Callers never specify which backend.

```
+-----------------------------------------------------------+
|                      Python Layer                         |
|  +----------------+  +----------------+  +-------------+  |
|  | metalmom API   |  | compat.librosa |  | compat.madmom| |
|  | (native)       |  | (drop-in)      |  | (drop-in)   | |
|  +-------+--------+  +-------+--------+  +------+------+ |
|          +--------------------+--------------------+      |
|                               | cffi                      |
+-------------------------------+--------------------------+
|                       C ABI Bridge                        |
|              (libmetalmom.dylib via MetalMomBridge)       |
+-------------------------------+--------------------------+
|                    Swift Core Engine                       |
|  +--------------------------------------------------------+|
|  |              Smart Dispatch Layer                      ||
|  |   (routes to optimal backend per operation + size)     ||
|  +----------------------------+---------------------------+|
|  |     Metal Backend          |   Accelerate Backend      ||
|  |  - MPSGraph (FFT, matmul) |  - vDSP (FFT, filter)    ||
|  |  - Custom MSL shaders     |  - BLAS/LAPACK (matrix)   ||
|  |  - Metal 4 (ML inference) |                           ||
|  +----------------------------+---------------------------+|
+-----------------------------------------------------------+
```

### Key Principles

- **Three Python entry points**: MetalMom native API, librosa compat shim, madmom compat shim. All call through the same C ABI into the same Swift engine.
- **Smart Dispatch is transparent**: each operation registers size thresholds determined by benchmarking. Below threshold = Accelerate. Above = Metal. No user configuration needed.
- **Unified memory makes dispatch cheap**: no data copying between CPU and GPU on Apple Silicon. The dispatch decision adds nanoseconds of overhead.
- **Data stays in the engine**: chained operations (e.g., load -> STFT -> mel -> MFCC) keep intermediate results as internal buffers. Only the final result crosses the C ABI back to Python.
- **Pipeline fusion**: dispatch decisions happen once per pipeline, not per operation. Avoids command buffer overhead from bouncing between backends.

### Concurrency Model

`mm_context` is **not thread-safe**. Each context holds its own Metal command queue and engine state. Rules:

- One context per thread. Python users create one `MetalMomLib()` per thread if they need parallelism.
- Multiple contexts coexist safely — they share the Metal device (singleton) but have independent command queues.
- cffi releases the Python GIL during C calls, so multiple threads can execute Metal work concurrently via separate contexts.
- Future: `mm_context_create_concurrent()` for internal dispatch queue parallelism.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| GPU Compute | Metal (MSL shaders, MPSGraph FFT, MPS matrix ops) |
| CPU Compute | Accelerate (vDSP, BLAS/LAPACK) |
| ML Inference | CoreML (auto-routes to ANE/GPU/CPU) |
| Core Language | Swift |
| GPU Shaders | Metal Shading Language (MSL, C++14-based) |
| Python Bridge | C ABI via @_cdecl, Python cffi |
| Audio I/O | AVFoundation (Swift side), soundfile (Python side) |

## Smart Dispatch Layer

### How It Works

The dispatch layer is introduced in Phase 1 with the Accelerate (CPU) backend only. All operations are written to the `ComputeOperation` protocol from day one. The Metal backend is added in Phase 10 — filling in GPU paths without restructuring existing code.

Every compute operation conforms to a `ComputeOperation` protocol declaring both a GPU and CPU implementation plus a size threshold:

```swift
protocol ComputeOperation {
    associatedtype Input
    associatedtype Output
    static var dispatchThreshold: Int { get }
    func executeGPU(_ input: Input, device: MTLDevice) -> Output
    func executeCPU(_ input: Input) -> Output
}
```

The dispatcher routes based on data size:

```swift
func dispatch<Op: ComputeOperation>(_ op: Op, input: Op.Input, size: Int) -> Op.Output {
    if size >= Op.dispatchThreshold {
        return op.executeGPU(input, device: metalDevice)
    } else {
        return op.executeCPU(input)
    }
}
```

During Phases 1-9, every operation's `executeGPU()` calls `fatalError("GPU not yet implemented")` and `dispatchThreshold` is set to `Int.max`, so all work routes to CPU. Phase 10 fills in GPU implementations and sets real thresholds. Searching for `fatalError` in GPU methods provides a clear completeness checklist.

### Expected Crossover Points

| Operation | CPU->GPU Crossover | Rationale |
|-----------|-------------------|-----------|
| FFT | ~512-1024 points | GPU dispatch overhead dominates below this |
| STFT (batched) | ~64 frames | Batch size matters more than FFT size |
| Mel filterbank | ~128 bands x 256 frames | Matrix multiply — GPU wins at scale |
| Element-wise (log, dB) | ~10K elements | Trivial per-element — GPU overhead dominates small arrays |
| Convolution (1D) | ~4K samples x 64 kernel | GPU convolution shines with large kernels |
| Matrix multiply | ~256x256 | MPS matmul is heavily optimized |
| Neural network inference | Always GPU | CoreML routes to Neural Engine/GPU automatically |

### Pipeline Fusion

For chained operations, the dispatch decision happens once for the entire pipeline:

```swift
func mfcc(signal: Signal, nMFCC: Int = 13) -> Signal {
    // If total data size warrants GPU, run entire pipeline on Metal:
    //   STFT -> mel filterbank -> log -> DCT (all in one command buffer)
    // Otherwise run entire pipeline on Accelerate:
    //   vDSP.FFT -> matrix multiply -> vForce.log -> vDSP.DCT
}
```

### Chip-Family Profiles

Thresholds differ between M1, M2, M3, M4. A `ChipProfile` auto-detects at initialization:

```swift
struct ChipProfile {
    let family: GPUFamily
    let gpuCoreCount: Int
    let thresholds: [String: Int]
}
```

Pre-baked profiles for known chips, with a one-time calibration fallback for unknown future hardware.

## Swift Package Structure

Three SPM targets with a clean dependency chain:

```
MetalMomCBridge (C types) → MetalMomCore (Swift engine) → MetalMomBridge (@_cdecl exports)
```

- **MetalMomCBridge**: C header only — `MMBuffer`, status codes, parameter structs. No Swift logic. No dependencies.
- **MetalMomCore**: The Swift engine — Signal, STFT, features, ML, dispatch. Depends on MetalMomCBridge for C types.
- **MetalMomBridge**: `@_cdecl` exported functions that call into MetalMomCore. Depends on MetalMomCore. The dylib product is built from this target.

```swift
// Package.swift
let package = Package(
    name: "MetalMom",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalMomCore", targets: ["MetalMomCore"]),
        .library(name: "MetalMomBridge", type: .dynamic, targets: ["MetalMomBridge"]),
    ],
    targets: [
        .target(name: "MetalMomCBridge", dependencies: [],
                path: "Sources/MetalMomCBridge", publicHeadersPath: "include"),
        .target(name: "MetalMomCore", dependencies: ["MetalMomCBridge"],
                path: "Sources/MetalMomCore"),
        .target(name: "MetalMomBridge", dependencies: ["MetalMomCore"],
                path: "Sources/MetalMomBridge"),
        .testTarget(name: "MetalMomTests", dependencies: ["MetalMomCore"],
                    path: "Tests/MetalMomTests"),
    ]
)
```

## Swift Core Module Structure

```
Sources/
├── MetalMomCore/
│   ├── Dispatch/
│   │   ├── ComputeBackend.swift          # Protocol: Metal & Accelerate conform to
│   │   ├── AccelerateBackend.swift        # CPU backend (sole backend in Phases 1-9)
│   │   ├── SmartDispatch.swift           # Size-based routing logic
│   │   └── Thresholds.swift              # Per-operation crossover points
│   │
│   ├── Audio/
│   │   ├── AudioIO.swift                 # Load, stream, decode (AVFoundation)
│   │   ├── Signal.swift                  # Core signal type (pinned buffer or MTLBuffer)
│   │   ├── Resample.swift                # Band-limited sinc interpolation
│   │   └── SignalGen.swift               # Tone, chirp, click generation
│   │
│   ├── Spectral/
│   │   ├── STFT.swift                    # Forward/inverse STFT
│   │   ├── CQT.swift                     # Constant-Q / Variable-Q / Hybrid CQT
│   │   ├── MelSpectrogram.swift          # Mel filterbank + spectrogram
│   │   ├── Reassigned.swift              # Reassigned spectrogram
│   │   ├── PhaseVocoder.swift            # Phase vocoder, Griffin-Lim
│   │   └── Scaling.swift                 # dB conversion, PCEN, weighting curves
│   │
│   ├── Features/
│   │   ├── MFCC.swift                    # Mel-frequency cepstral coefficients
│   │   ├── Chroma.swift                  # STFT/CQT/CENS/VQT + deep chroma
│   │   ├── SpectralDescriptors.swift     # Centroid, bandwidth, contrast, rolloff, flatness
│   │   ├── Tonnetz.swift                 # Tonal centroid features
│   │   ├── PolyFeatures.swift            # Polynomial features
│   │   ├── RMS.swift                     # Root-mean-square energy
│   │   ├── ZeroCrossing.swift            # Zero-crossing rate
│   │   └── Delta.swift                   # Delta/delta-delta, stack_memory
│   │
│   ├── Rhythm/
│   │   ├── BeatTracker.swift             # DP beat tracking (librosa-style)
│   │   ├── NeuralBeatTracker.swift       # RNN+DBN beat tracking (madmom-style)
│   │   ├── OnsetDetection.swift          # Spectral flux, superflux, complex flux
│   │   ├── NeuralOnsetDetector.swift     # CNN/RNN onset detection
│   │   ├── TempoEstimation.swift         # ACF, comb filter, DBN tempo
│   │   ├── Downbeat.swift                # Downbeat/bar tracking
│   │   └── Tempogram.swift               # Local tempo features
│   │
│   ├── Pitch/
│   │   ├── YIN.swift                     # YIN and pYIN F0 estimation
│   │   ├── Piptrack.swift                # Parabolic interpolation pitch tracking
│   │   ├── PianoTranscription.swift      # Polyphonic piano (ADSR-HMM)
│   │   └── Tuning.swift                  # Tuning estimation
│   │
│   ├── Harmony/
│   │   ├── KeyDetection.swift            # CNN-based key recognition
│   │   ├── ChordRecognition.swift        # Deep chroma + CRF chord recognition
│   │   └── HPSS.swift                    # Harmonic-percussive source separation
│   │
│   ├── ML/
│   │   ├── InferenceEngine.swift         # CoreML model runner
│   │   ├── EnsembleRunner.swift          # Parallel ensemble inference + averaging
│   │   ├── ChunkedInference.swift        # Sequence chunking for long audio
│   │   ├── ModelRegistry.swift           # Pre-trained model management
│   │   ├── HMM.swift                     # Viterbi decoding, DBN approximation
│   │   ├── CRF.swift                     # Conditional Random Field decoding
│   │   └── GMM.swift                     # Gaussian Mixture Models
│   │
│   ├── Decompose/
│   │   ├── NMF.swift                     # Non-negative matrix factorization
│   │   └── NNFilter.swift                # Nearest-neighbor filtering
│   │
│   ├── Segment/
│   │   ├── Recurrence.swift              # Self-similarity, cross-similarity, RQA
│   │   ├── DTW.swift                     # Dynamic time warping
│   │   └── Clustering.swift              # Agglomerative segmentation
│   │
│   ├── Filters/
│   │   ├── FilterBank.swift              # Mel, log, rectangular, semitone bandpass
│   │   ├── CombFilter.swift              # Feed-forward/backward comb filters
│   │   ├── SemitoneBandpass.swift         # Pre-computed elliptic IIR coefficients
│   │   └── Windows.swift                 # Hann, Hamming, Blackman, etc.
│   │
│   ├── Effects/
│   │   ├── TimeStretch.swift             # Phase-vocoder time stretching
│   │   ├── PitchShift.swift              # Time-stretch + resample
│   │   ├── Trim.swift                    # Silence trim
│   │   ├── Split.swift                   # Non-silent interval detection
│   │   └── Preemphasis.swift             # Preemphasis / deemphasis filters
│   │
│   ├── Convert/
│   │   └── Units.swift                   # Hz/mel/midi/note/oct/time/frame/sample conversions
│   │
│   └── Eval/
│       ├── OnsetEval.swift               # P/R/F with tolerance window
│       ├── BeatEval.swift                # F-measure, CML, AML, P-score, Cemgil
│       ├── TempoEval.swift               # Tempo accuracy
│       └── ChordEval.swift               # Chord recognition accuracy
│
├── MetalMomShaders/
│   ├── FFT.metal                         # Custom FFT kernels
│   ├── SpectralOps.metal                 # Magnitude, phase, filterbank application
│   ├── Reduction.metal                   # Parallel reductions (sum, max, mean)
│   ├── Convolution.metal                 # 1D/2D convolution kernels
│   └── Elementwise.metal                 # Log, exp, power, dB conversion
│
├── MetalMomBridge/
│   └── Bridge.swift                      # @_cdecl exported Swift functions
│
└── MetalMomCBridge/
    └── include/
        └── metalmom.h                    # Public C API header (types + function decls)
```

### Signal Type

`Signal` is the core data type. It uses manually-allocated `UnsafeMutableBufferPointer<Float>` storage, providing stable pointer addresses that are safe to pass to Accelerate, Metal, and across the C ABI. Future: an MTLBuffer variant for GPU-resident data.

```swift
public final class Signal {
    private let storage: UnsafeMutableBufferPointer<Float>
    public let shape: [Int]
    public let sampleRate: Int

    public var count: Int { storage.count }

    public var dataPointer: UnsafePointer<Float> {
        UnsafePointer(storage.baseAddress!)
        // Safe: storage is manually allocated, never moves
    }

    public var mutableDataPointer: UnsafeMutablePointer<Float> {
        storage.baseAddress!
    }

    /// Create from a Swift array (copies data into pinned storage).
    public init(data: [Float], shape: [Int], sampleRate: Int) {
        precondition(data.count == shape.reduce(1, *))
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = shape
        self.sampleRate = sampleRate
    }

    deinit {
        storage.baseAddress?.deinitialize(count: storage.count)
        storage.baseAddress?.deallocate()
    }
}
```

Pointer stability is guaranteed because the buffer is manually allocated (not backed by a Swift Array that could reallocate). This is the standard pattern for audio libraries that need to hand pointers to C APIs, Accelerate, and Metal.

## ML Inference Engine

### Strategy

Replace madmom's hand-rolled Python/Cython neural network engine with CoreML. CoreML automatically routes inference to the optimal hardware: Apple Neural Engine (ANE), Metal GPU, or CPU.

### Model Conversion Pipeline

1. Load madmom `.pkl` weights in Python (numpy arrays)
2. Reconstruct architecture in PyTorch (matching madmom layer specs exactly)
3. Load weights into PyTorch model
4. Export via `coremltools` to `.mlpackage`
5. Validate: run both madmom original and CoreML on test corpus, compare within tolerance

**Risk mitigation**: A spike task converts a single RNNBeatProcessor model first, proving the BiLSTM weight mapping, custom preprocessing, and CoreML export pipeline all work before committing to all 9 model families. Known risks:

- madmom's BiLSTMs are custom implementations — weight layout may differ from standard PyTorch LSTMs
- madmom couples preprocessing (signal framing, filterbanks, normalization) with inference — these must be matched exactly in Swift
- CoreML has sequence length constraints that may require chunked inference for long audio

### Model Inventory

| madmom Model | Architecture | Purpose | Priority |
|---|---|---|---|
| RNNBeatProcessor (x8 ensemble) | 3-layer BiLSTM, 25 units | Beat activation | P0 |
| RNNOnsetProcessor (x8 ensemble) | BiLSTM | Onset activation | P0 |
| CNNOnsetProcessor | CNN | Onset activation | P0 |
| RNNDownBeatProcessor | BiLSTM | Downbeat activation | P1 |
| CNNKeyRecognitionProcessor | CNN ensemble | Key detection | P1 |
| CNNChordFeatureProcessor | CNN | Chord features | P1 |
| DeepChromaProcessor | DNN | Chroma extraction | P1 |
| RNNPianoNoteProcessor | RNN | Piano note activation | P2 |
| RNNBarProcessor | GRU | Bar/downbeat activation | P2 |

### Ensemble Inference

Several madmom models (Beat, Onset) use ensembles of 8 independently-trained models whose outputs are averaged. MetalMom handles this via:

- `EnsembleRunner`: loads N CoreML models, runs inference in parallel via separate prediction requests, averages activations
- CoreML on Apple Silicon can pipeline multiple model inferences efficiently on the ANE

### Chunked Inference for Long Audio

madmom processes arbitrary-length sequences. CoreML models have fixed or bounded input dimensions. `ChunkedInference` handles this:

- Split input into overlapping chunks matching the model's expected sequence length
- Run inference on each chunk
- Merge outputs with overlap-add or max-pooling at chunk boundaries
- Configurable overlap size and merge strategy per model type

### Probabilistic Decoders

HMM, CRF, and GMM decoders are reimplemented in Swift (not neural networks — dynamic programming algorithms that run on neural network output). These run on CPU via Accelerate, as they operate on small data and are latency-sensitive.

### Model Licensing

madmom model weights are CC BY-NC-SA 4.0 (non-commercial). Two paths:
1. Convert and redistribute under same NC license for initial release
2. Train new models on open datasets for a fully permissive license (roadmap)

Ship with option 1, document clearly, roadmap option 2.

## C ABI Bridge

Swift functions exported with `@_cdecl` in the MetalMomBridge target. Python calls through cffi.

### Design Rules

- Every function takes a context (holds Metal device, command queue, chip profile)
- Every function returns `int` status code (0 = success, negative = error)
- Output via `MMBuffer*` out-param — caller owns memory, frees with `mm_buffer_free`
- Python copies `MMBuffer` data into a NumPy array, then frees the C-side buffer (see Data Flow below)
- Param structs are flat C structs — no pointers-to-pointers, no callbacks, no ObjC objects

### Data Flow (Python ↔ Swift)

The bridge uses a **minimal-copy** approach, not zero-copy. The actual flow:

1. Python passes input data pointer (NumPy array's `.ctypes.data`) to Swift via cffi
2. Swift copies input into a Signal, processes it, produces output Signal
3. Bridge allocates an MMBuffer and copies Signal data into it
4. Python copies MMBuffer data into a new NumPy array via `np.frombuffer` + `.copy()`
5. Python calls `mm_buffer_free()` to release the C-side allocation

Total: 2 copies of contiguous float32 arrays (Swift→MMBuffer, MMBuffer→NumPy). Both are fast memcpy operations — nanoseconds for typical spectrogram sizes. The compute dominates, not the copies.

### Core Types

```c
typedef struct {
    float* data;
    int64_t shape[8];   /* max 8 dimensions */
    int32_t ndim;
    int32_t dtype;      /* 0=float32, 1=float64, 2=complex64 */
    int64_t count;      /* total element count */
} MMBuffer;

typedef struct {
    int32_t n_fft;
    int32_t hop_length;
    int32_t win_length;
    int32_t window_type;
    int32_t center;     /* bool: 1=true, 0=false */
    int32_t pad_mode;
} MMSTFTParams;
```

## Python Package

### Structure

```
metalmom/
├── __init__.py                # Public native API
├── _native.py                 # cffi bindings to libmetalmom.dylib
├── _buffer.py                 # MMBuffer -> NumPy minimal-copy wrapper
├── core.py                    # load, stream, resample, stft, istft, cqt, ...
├── feature.py                 # mfcc, chroma, spectral_*, rms, zcr, tonnetz, ...
├── beat.py                    # beat_track, plp
├── onset.py                   # onset_detect, onset_strength
├── pitch.py                   # yin, pyin, piptrack
├── effects.py                 # time_stretch, pitch_shift, hpss, trim, split
├── decompose.py               # nmf, nn_filter, hpss (spectrogram-level)
├── segment.py                 # recurrence_matrix, dtw, agglomerative
├── sequence.py                # viterbi, rqa
├── convert.py                 # hz_to_mel, midi_to_note, frames_to_time, ...
├── display.py                 # specshow, waveshow (matplotlib)
├── filters.py                 # mel, chroma, constant_q filterbanks
├── evaluate.py                # onset/beat/tempo/chord eval metrics
│
├── compat/
│   ├── librosa/               # Drop-in: import metalmom.compat.librosa as librosa
│   └── madmom/                # Drop-in: import metalmom.compat.madmom as madmom
│
└── _lib/
    ├── libmetalmom.dylib
    ├── metalmom.h
    └── models/
```

### Compatibility Shims

Drop-in replacement via one import change:

```python
# Before:
import librosa
y, sr = librosa.load('song.mp3')

# After:
import metalmom.compat.librosa as librosa
y, sr = librosa.load('song.mp3')
```

Shims are pure Python — thin wrappers that accept the exact same function signatures (same param names, same defaults) and return the same types (NumPy arrays with same shapes/dtypes).

**Incremental development**: compat shim functions are added alongside each feature as it is built (not batched at the end). A final completeness audit and automated signature verification happens after all features are implemented.

### Python Version Support

Python 3.11, 3.12, 3.13. No legacy Python.

### Dependencies (minimal)

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "cffi>=1.16",
    "soundfile>=0.12",
]

[project.optional-dependencies]
display = ["matplotlib>=3.7"]
eval = ["mir_eval>=0.7"]
```

No scipy, no scikit-learn, no numba, no joblib. The heavy lifting happens in Swift.

## Testing Strategy

### Three-Tier Parity Testing

Not all operations can or should be tested the same way. Operations are assigned to one of three tiers based on the nature of the computation:

**Tier 1 — Element-wise parity** (tight rtol/atol, compare every value against golden files):

| Operation | rtol | atol | Rationale |
|-----------|------|------|-----------|
| STFT magnitude | 1e-5 | 1e-6 | Deterministic math, near-exact |
| iSTFT | 1e-5 | 1e-5 | Overlap-add reconstruction |
| Mel spectrogram | 1e-5 | 1e-6 | Matrix multiply, tight tolerance |
| MFCC | 1e-4 | 1e-5 | DCT accumulates small errors |
| Chroma (STFT) | 1e-4 | 1e-4 | Filterbank differences propagate |
| Spectral descriptors | 1e-5 | 1e-6 | Weighted sums, tight tolerance |
| RMS, ZCR | 1e-5 | 1e-6 | Simple aggregations |
| dB scaling | 1e-6 | 1e-7 | Log/exp, near-exact |
| Unit conversions | 1e-6 | 1e-7 | Closed-form math |
| Delta features | 1e-5 | 1e-6 | Savitzky-Golay, deterministic |
| Window functions | 1e-6 | 1e-7 | Trig, near-exact |
| Mel filterbank | 1e-6 | 1e-7 | Construction is deterministic |

**Tier 2 — Behavioral equivalence** (structural/statistical tests, not element-wise):

| Operation | Test Method | Rationale |
|-----------|------------|-----------|
| CQT / VQT | Same peak frequencies, energy within 1e-3 | Kernel construction differs from librosa |
| Resampling | Round-trip SNR > 40 dB | Sinc implementations vary |
| Phase vocoder | Temporal structure preserved, SNR > 20 dB | Phase accumulation diverges over time |
| Feature inversion | Reconstruction SNR > 20 dB | Inherently lossy |
| Griffin-Lim | SNR > 15 dB after N iterations | Iterative, convergence varies |
| HPSS | Energy ratio H/(H+P) within 5% | Median filter implementations differ |
| NMF | Reconstruction error within 10% of librosa's | Iterative, convergence varies |
| Neural inference | Activation correlation > 0.99 | CoreML may use float16 internally |

**Tier 3 — Task-level accuracy** (MIR evaluation metrics against ground truth annotations, not against madmom's output):

| Task | Metric | Target |
|------|--------|--------|
| Beat tracking | F-measure (0.07s tolerance) | >= madmom's published results |
| Onset detection | P/R/F (0.025s tolerance) | >= madmom's published results |
| Tempo estimation | Accuracy1 / Accuracy2 | Within 4% or octave error |
| Key detection | Weighted accuracy | >= madmom's published results |
| Chord recognition | Weighted chord symbol recall | >= madmom's published results |
| Piano transcription | Note-level F-measure | >= madmom's published results |
| Downbeat detection | F-measure | >= madmom's published results |

### Golden Output Tests (every commit, fast)

Run librosa + madmom on a corpus of test audio files, save outputs as `.npy` ground truth. MetalMom Tier 1 and Tier 2 tests compare against these golden files.

### Side-by-Side CI (weekly + pre-release)

CI job installs librosa, madmom (3.10-compat fork), and MetalMom. Runs all three on same inputs, compares outputs. Generates compatibility report: pass/fail per function, max/mean deviation.

### Unit Tests (every commit)

- Python unit tests: dispatch routing, signal management, invertibility properties, known-answer tests
- Swift XCTest suite: kernel correctness, dispatch logic, Signal type
- Signature compatibility tests: automated verification that shim functions match original signatures exactly
- Benchmark suite: wall-clock speedup vs librosa/madmom, tracked over time for regression detection

### Dynamic API Parity Check (CI, weekly)

A CI job that pulls the latest librosa release, introspects its entire public API (function names, signatures, parameter names, defaults, return types), and compares against our `metalmom.compat.librosa` shim. Any drift — new functions, changed signatures, added parameters — automatically files a GitHub issue with the specific discrepancy.

```yaml
# .github/workflows/api-parity.yml
name: API Parity Check
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6am
  workflow_dispatch:

jobs:
  check-librosa-api:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install latest librosa
        run: pip install librosa --upgrade
      - name: Install MetalMom
        run: pip install .
      - name: Run API parity check
        run: python tests/parity/check_api_surface.py --target librosa --output drift_report.json
      - name: File issues for drift
        if: steps.check.outputs.has_drift == 'true'
        run: python scripts/ci/file_api_drift_issues.py --report drift_report.json
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The checker walks every public module, class, and function in librosa, extracts `inspect.signature()`, and diffs against our shim. For each discrepancy it files a labeled issue (`api-drift`, `compat-librosa`) with the function name, what changed, and the librosa version that introduced it. Same process for madmom (against the compat fork).

### CI and Metal Testing

GitHub Actions macOS runners do not have reliable Metal GPU compute support. CI strategy:

- **GitHub Actions**: runs `swift test` and `pytest` using the Accelerate (CPU) backend only. All logic is tested.
- **Metal tests tagged**: `@Tag(.metal)` in Swift XCTest, `@pytest.mark.metal` in Python. Skipped in CI.
- **Local testing**: Metal tests run on developer machines with Apple Silicon.
- **Optional**: self-hosted M-series runner for nightly Metal test suite.

Since Smart Dispatch routes to the same algorithms on both backends, CPU parity tests validate correctness. Metal tests only need to verify that the GPU path produces identical results to the CPU path.

## Distribution

### pip (Python users)

Prebuilt wheels with bundled `libmetalmom.dylib` and CoreML models:

```
metalmom-0.1.0-cp311-cp311-macosx_14_0_arm64.whl
metalmom-0.1.0-cp312-cp312-macosx_14_0_arm64.whl
metalmom-0.1.0-cp313-cp313-macosx_14_0_arm64.whl
```

One command: `pip install metalmom`

Estimated wheel size: ~30-50MB (dylib + CoreML models).

### Swift Package Manager (Swift-native users)

```swift
.package(url: "https://github.com/<org>/MetalMom.git", from: "0.1.0")
```

### Minimum Requirements

- macOS 14.0+ Sonoma
- Apple Silicon (M1 or later)
- Python 3.11+
- Xcode 15+ (build only)

## Repo Layout

```
MetalMom/
├── README.md
├── LICENSE                       # MIT for code
├── LICENSE-MODELS                # CC BY-NC-SA 4.0 for converted models
├── CLAUDE.md                     # Project conventions
├── Package.swift                 # SPM manifest (3 targets)
├── pyproject.toml                # Python build config
├── Sources/
│   ├── MetalMomCBridge/          # C types only
│   ├── MetalMomCore/             # Swift engine
│   ├── MetalMomBridge/           # @_cdecl exports
│   └── MetalMomShaders/          # Metal shading language kernels
├── python/metalmom/              # Python package + compat shims
├── models/
│   ├── converted/                # CoreML .mlpackage (Git LFS)
│   └── conversion/               # madmom -> CoreML conversion scripts
├── Tests/
│   ├── MetalMomTests/            # Swift XCTest
│   ├── golden/                   # Golden output parity tests (Python)
│   ├── unit/                     # Python unit tests
│   ├── parity/                   # Side-by-side CI comparison
│   └── benchmarks/               # Performance tracking
├── docs/plans/                   # Design documents
├── scripts/
│   ├── build_dylib.sh            # Build + copy dylib
│   ├── generate_filter_coefficients.py  # Pre-compute IIR coefficients via scipy
│   └── calibrate_thresholds.py   # Benchmark for dispatch crossover points
└── .github/workflows/            # CI/CD pipelines
```

## Feature Parity Scope

### From librosa (all of these):

- Audio I/O: load, stream, resample, duration, sample rate detection
- Spectral: STFT/iSTFT, CQT/iCQT/VQT/hybrid CQT, reassigned spectrogram, phase vocoder, Griffin-Lim
- Features: mel spectrogram, MFCC, chroma (STFT/CQT/CENS/VQT), spectral centroid/bandwidth/contrast/rolloff/flatness, RMS, zero-crossing rate, tonnetz, poly features
- Feature manipulation: delta, stack_memory
- Feature inversion: mel_to_stft, mel_to_audio, mfcc_to_mel, mfcc_to_audio
- Rhythm: beat_track (DP), onset_detect, onset_strength, tempo, tempogram, PLP
- Pitch: YIN, pYIN, piptrack, tuning estimation
- Effects: time_stretch, pitch_shift, HPSS, trim, split, preemphasis, deemphasis
- Decompose: NMF, nearest-neighbor filter
- Segment: recurrence matrix, cross-similarity, DTW, agglomerative clustering
- Sequence: Viterbi decoding, RQA
- Filters: mel, chroma, constant-Q, semitone filterbanks, window functions
- Display: specshow, waveshow
- Conversions: all Hz/mel/midi/note/octave/time/frame/sample conversions
- Scaling: amplitude_to_db, power_to_db, PCEN, A/B/C/D weighting

### From madmom (all of these):

- Neural beat tracking: RNN (BiLSTM) + DBN/HMM + CRF decoding
- Neural onset detection: BiLSTM, CNN, spectral methods (superflux, complex flux)
- Downbeat/bar detection: RNN + DBN/HMM, pattern tracking with GMM
- Tempo estimation: comb filter, ACF, DBN methods
- Key detection: CNN ensemble
- Chord recognition: deep chroma + CRF, CNN + CRF
- Piano transcription: RNN + ADSR-HMM
- Deep chroma: DNN-based harmonically-aware chroma
- Evaluation: onset/beat/tempo/chord/key/note metrics
- Processor pipeline architecture
- Comb filterbanks, semitone bandpass spectrograms
- Signal framing with madmom-specific parameters

### Semitone Bandpass Filterbank

madmom's semitone bandpass filterbank uses IIR elliptic (Cauer) filters. Elliptic filter design requires solving Jacobi elliptic integrals — complex math implemented in scipy's Fortran backend.

Our approach: **pre-compute filter coefficients** using a scipy script (`scripts/generate_filter_coefficients.py`) for all semitones (C1-B8) at standard sample rates (22050, 44100, 48000). Ship the b/a coefficients as constant tables in Swift. The IIR filtering itself (applying pre-designed coefficients) is trivial via `vDSP_deq22` (biquad cascade). This gives exact parity with madmom's filters without porting the filter design math.

### MetalMom-native additions (future roadmap):

- Real-time / streaming analysis
- Modern ML integration (PyTorch/CoreML model feeding)
- Batch corpus processing
- Thread-safe concurrent context
