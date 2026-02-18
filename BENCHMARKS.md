# Benchmarks

All benchmarks run on Apple M4 Max (14 CPU cores, ~10 GPU cores), macOS 15, Python 3.14, MetalMom 0.1.0.

## MetalMom vs librosa

Synthetic 440 Hz sine waves at 44100 Hz. Mean of 5 timed iterations after 1 warmup.

### 30-second signal

| Operation | MetalMom (ms) | librosa (ms) | Speedup |
|---|---:|---:|---:|
| STFT (n_fft=2048) | 7.9 | 12.7 | 1.6x |
| Mel spectrogram | 7.1 | 19.6 | 2.7x |
| MFCC | 7.3 | 19.8 | 2.7x |
| CQT | 11.8 | 27.5 | 2.3x |
| Onset strength | 7.6 | 20.6 | 2.7x |
| Beat tracking | 8.3 | 22.6 | 2.7x |

### 5-second signal

| Operation | MetalMom (ms) | librosa (ms) | Speedup |
|---|---:|---:|---:|
| STFT (n_fft=2048) | 3.5 | 2.3 | 0.6x |
| Mel spectrogram | 1.6 | 3.9 | 2.5x |
| MFCC | 1.8 | 4.5 | 2.5x |
| CQT | 8.1 | 9.9 | 1.2x |
| Onset strength | 1.9 | 4.1 | 2.2x |
| Beat tracking | 1.8 | 4.2 | 2.4x |

### 1-second signal

| Operation | MetalMom (ms) | librosa (ms) | Speedup |
|---|---:|---:|---:|
| STFT (n_fft=2048) | 4.9 | 0.6 | 0.1x |
| Mel spectrogram | 0.9 | 1.4 | 1.6x |
| MFCC | 1.0 | 1.4 | 1.4x |
| CQT | 7.6 | 7.8 | 1.0x |
| Onset strength | 0.8 | 1.4 | 1.7x |
| Beat tracking | 1.1 | 1.6 | 1.5x |

**Notes:**
- On very short signals (1s), the cffi bridge + Swift runtime overhead dominates, making STFT slower than librosa's pure-NumPy path. This crossover disappears above ~5 seconds.
- MetalMom's advantage grows with signal length because the Accelerate/Metal backends amortize fixed overhead.
- CQT shows the smallest speedup because both implementations are FFT-bound with similar algorithmic complexity.

## Full Analysis Pipeline

End-to-end timing of a complete analysis pipeline on real audio files (AAC decode + resample to 22050 Hz). Release build, single run.

Pipeline: **Load -> STFT -> Mel spectrogram -> MFCC -> Chroma -> Onset strength -> Beat tracking**

### ~11-minute song (14.8M samples at 22050 Hz)

| Step | Time (ms) | % of Total |
|---|---:|---:|
| Load (AAC decode + resample) | 637 | 47.8% |
| STFT (n_fft=2048, hop=512) | 139 | 10.4% |
| Mel spectrogram (128 bands) | 136 | 10.2% |
| MFCC (20 coefficients) | 63 | 4.7% |
| Chroma (12 bins) | 81 | 6.1% |
| Onset strength | 142 | 10.7% |
| Beat tracking (Ellis DP) | 136 | 10.2% |
| **Total** | **1,333** | |

### 30-second excerpt (661K samples at 22050 Hz)

| Step | Time (ms) | % of Total |
|---|---:|---:|
| Load (AAC decode + resample) | 49 | 31.4% |
| STFT | 51 | 32.7% |
| Mel spectrogram | 8 | 5.1% |
| MFCC | 16 | 10.3% |
| Chroma | 11 | 7.1% |
| Onset strength | 8 | 5.1% |
| Beat tracking | 13 | 8.3% |
| **Total** | **156** | |

## Optimization History

The pipeline was profiled and optimized on 2026-02-18. Three fixes reduced total time by 7.6x on full songs.

### Fix 1: Audio loading (13x speedup)

**Before:** Manual per-sample linear interpolation for resampling + manual mono downmix loop. O(n) with no SIMD.

**After:** Single `AVAudioConverter` pass handles sample rate conversion and channel downmix with hardware acceleration.

| File length | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| ~11 min | 8,502 | 637 | 13.3x |
| 30 sec | 2,084 | 49 | 42.5x |

### Fix 2: Beat tracker autocorrelation (O(n^2) -> O(n log n))

**Before:** Direct autocorrelation via nested loops, O(n^2).

**After:** FFT-based autocorrelation: `IFFT(|FFT(x)|^2)` via `vDSP_fft_zrip`, O(n log n).

### Fix 3: Beat tracker DP inner loop

**Before:** `log()` called per iteration in the dynamic programming inner loop (~635K calls for an 11-minute song).

**After:** Precomputed `logTable[i] = -log(Float(i))` before the DP loop.

### Before vs After (full pipeline, release build)

| Step | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| Load | 8,502 | 637 | 13.3x |
| STFT | 107 | 139 | -- |
| Mel spectrogram | 189 | 136 | 1.4x |
| MFCC | 180 | 63 | 2.9x |
| Chroma | 98 | 81 | 1.2x |
| Onset strength | 440 | 142 | 3.1x |
| Beat tracking | 580 | 136 | 4.3x |
| **Total** | **10,096** | **1,333** | **7.6x** |

The pipeline shifted from I/O-bound (load was 84% of time) to compute-bound (load is now 48%, with compute distributed across spectral and rhythm operations).

## Backend Dispatch

MetalMom uses smart dispatch to route operations to the fastest backend based on input size:

| Backend | Used for |
|---|---|
| Metal shaders | Elementwise ops, reductions, 1D convolution (large inputs) |
| MPSGraph | FFT/IFFT, fused MFCC pipeline |
| MPS | Matrix multiply (mel filterbank) |
| CoreML / ANE | Neural network inference (beat, onset, key, chord, piano) |
| Accelerate (vDSP) | FFT, windowing, autocorrelation, small inputs |
| Accelerate (BLAS) | Matrix multiply fallback |

Small inputs stay on CPU to avoid Metal command buffer overhead. The crossover threshold varies by operation (typically 4K-64K elements).

## Reproducing

### Synthetic benchmarks (MetalMom vs librosa)

```bash
# Build the dylib first
swift build -c release && ./scripts/build_dylib.sh

# Run benchmarks
.venv/bin/python benchmarks/run_benchmarks.py

# Compare against a previous run
.venv/bin/python benchmarks/run_benchmarks.py --compare benchmarks/results/bench_YYYYMMDD_HHMMSS.json
```

Results are saved to `benchmarks/results/` as JSON.

### Pipeline profiling (real audio)

```bash
# Build and run the ProfilingRunner
swift build -c release
swift run -c release ProfilingRunner /path/to/audio.m4a
```

Requires macOS 14+ with an Apple Silicon GPU for Metal acceleration.
