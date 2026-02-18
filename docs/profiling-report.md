# MetalMom Profiling Report

**Date:** 2026-02-18
**Hardware:** M3 Max, apple9 GPU family, ~10 cores
**Test file:** `inamorata.m4a` (24MB AAC, ~11 min, 14,779,086 samples at 22050 Hz)

## Baseline Results

| Step | Time (ms) | % of Total |
|------|----------:|-----------:|
| Load (AudioIO) | 8,502 | 84.2% |
| STFT | 107 | 1.1% |
| Mel spectrogram | 189 | 1.9% |
| MFCC (FusedMFCC) | 180 | 1.8% |
| Chroma | 98 | 1.0% |
| Onset strength | 440 | 4.4% |
| Beat tracking | 580 | 5.7% |
| **Total** | **10,096** | |

**Short file baseline** (`moth_30s.m4a`, 30s excerpt):

| Step | Time (ms) | Notes |
|------|----------:|-------|
| Load | 2,084 | Dominates here too |
| STFT | 136 | Slower than full song (GPU warmup) |
| Mel | 73 | |
| MFCC | 304 | GPU shader compilation on first call |
| Chroma | 19 | |
| Onset | 43 | |
| Beat | 48 | |
| **Total** | **2,706** | |

## Top 3 Bottlenecks

### Bottleneck #1: AudioIO.load() — 8,502ms (84%)

The audio file loading pipeline consumes 84% of total time. This includes:
- AAC decoding via AVAudioFile
- Mono downmix (averaging channels)
- Resampling via **linear interpolation** (the file's native sample rate to 22050 Hz)

The resampling implementation in `AudioIO.swift:107` uses a manual loop with linear interpolation — no SIMD, no Accelerate, O(n) with per-sample floating point math. For a 14.8M sample file, this is extremely expensive.

**Fix:** Replace linear interpolation resampling with `vDSP_desamp` or `AVAudioConverter` which use hardware-accelerated sample rate conversion. Alternatively, use `AVAudioFile`'s built-in format conversion to decode directly to the target sample rate.

### Bottleneck #2: BeatTracker.beatTrack() — 580ms (5.7%)

Beat tracking scales superlinearly with audio length:
- 30s: 48ms
- 11min (22x longer): 580ms (12x more time)

This is expected from the Ellis 2007 DP algorithm which has O(n * period) complexity. The onset strength computation inside `beatTrack` also recomputes a full mel spectrogram internally.

**Fix:** The DP itself is inherently sequential, but the redundant mel spectrogram computation can be avoided by caching. For now, this is acceptable at 580ms.

### Bottleneck #3: OnsetDetection.onsetStrength() — 440ms (4.4%)

Each call recomputes a full mel spectrogram from scratch. When beat tracking calls onset strength, this is a second mel spectrogram computation (the user may have already computed one). The internal pipeline: mel spectrogram → power_to_db → spectral flux → half-wave rectify → aggregate.

**Fix:** The mel spectrogram inside onset strength is the dominant cost. No quick fix without API changes (accepting pre-computed mel spectrograms). Lower priority.

## Optimization Plan

1. **Fix #1: AudioIO.load() resampling** — Replace linear interpolation with AVAudioConverter or vDSP-based resampling. Expected: 5-10x speedup on load.
2. **Fix #2: MFCC GPU warmup** — The first FusedMFCC call is 304ms (30s file) but 180ms on full song. MPSGraph shader compilation is a one-time cost. Add a warmup call or lazy-compile shaders on context init.
3. **Fix #3: Beat tracking DP** — Profile the DP inner loop for vectorization opportunities with Accelerate.

## Optimization Results

### Fix #1: AudioIO.load() — AVAudioConverter (commit `d631e14`)

Replaced manual per-sample linear interpolation resampling + manual mono downmix with a single `AVAudioConverter` pass. AVAudioConverter performs hardware-accelerated sample rate conversion and channel downmix in one operation.

**Changes:** `Sources/MetalMomCore/Audio/AudioIO.swift` — deleted `resample()` method entirely, replaced with `AVAudioConverter`-based path.

### Fix #2: BeatTracker autocorrelation — FFT-based O(n log n) (commit `c627b4d`)

Replaced O(n^2) direct autocorrelation with FFT-based implementation using `vDSP_fft_zrip`. The autocorrelation is computed as IFFT(|FFT(x)|^2), reducing complexity from O(n^2) to O(n log n).

**Changes:** `Sources/MetalMomCore/Rhythm/BeatTracker.swift` — rewrote `autocorrelation()` using vDSP FFT routines with proper handling of packed DC/Nyquist format.

### Fix #3: BeatTracker DP — Precomputed log table (commit `c627b4d`)

Added a precomputed log lookup table in `dpBeatTrack()` to eliminate ~635K per-iteration `log()` transcendental function calls in the DP inner loop.

**Changes:** `Sources/MetalMomCore/Rhythm/BeatTracker.swift` — precompute `logTable[i] = -log(Float(i))` before DP loop.

### Before vs After (Release Build)

**Full song** (`inamorata.m4a`, ~11 min):

| Step | Before (ms) | After (ms) | Speedup |
|------|------------:|----------:|--------:|
| Load (AudioIO) | 8,502 | 637 | 13.3x |
| STFT | 107 | 139 | 0.8x |
| Mel spectrogram | 189 | 136 | 1.4x |
| MFCC (FusedMFCC) | 180 | 63 | 2.9x |
| Chroma | 98 | 81 | 1.2x |
| Onset strength | 440 | 142 | 3.1x |
| Beat tracking | 580 | 136 | 4.3x |
| **Total** | **10,096** | **1,333** | **7.6x** |

**30s excerpt** (`moth_30s.m4a`):

| Step | Before (ms) | After (ms) | Speedup |
|------|------------:|----------:|--------:|
| Load | 2,084 | 49 | 42.5x |
| STFT | 136 | 51 | 2.7x |
| Mel | 73 | 8 | 9.1x |
| MFCC | 304 | 16 | 19.0x |
| Chroma | 19 | 11 | 1.7x |
| Onset | 43 | 8 | 5.4x |
| Beat | 48 | 13 | 3.7x |
| **Total** | **2,706** | **156** | **17.3x** |

### Summary

- **AudioIO.load()** was the dominant bottleneck at 84% of total time. AVAudioConverter eliminated the manual resampling loop for a 13-42x speedup depending on file length.
- **BeatTracker** benefited from both FFT-based autocorrelation (O(n^2) → O(n log n)) and precomputed log tables, yielding 4.3x speedup on long files.
- **Overall pipeline**: 7.6x faster on full songs, 17.3x faster on short excerpts.
- The pipeline is now compute-bound rather than I/O-bound. Load is 48% of total time (down from 84%), with the remaining time distributed across spectral/rhythm operations.
