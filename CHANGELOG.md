# Changelog

All notable changes to MetalMom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-18

### Added

**Core Audio**
- Audio I/O: `load`, `resample`, `stream`, `get_duration`, `get_samplerate` via AVFoundation
- STFT/ISTFT with GPU acceleration via Metal
- Reassigned spectrogram
- Signal generation: `tone`, `chirp`, `clicks`

**Spectral Features**
- Mel spectrogram, MFCC, inverse MFCC
- Chroma: STFT, CQT, CENS, VQT variants
- Spectral moments: centroid, bandwidth, contrast, rolloff, flatness
- RMS, zero crossing rate, tonnetz, poly features
- Delta and stack memory
- Tempogram, Fourier tempogram
- CQT, VQT, hybrid CQT transforms
- PCEN, A/B/C/D frequency weighting

**Onset & Rhythm**
- Onset detection and onset strength (single and multi-band)
- Beat tracking, tempo estimation, predominant local pulse (PLP)

**Pitch & Effects**
- Pitch estimation: YIN, pYIN, piptrack, tuning estimation
- HPSS (harmonic-percussive source separation)
- Time stretch, pitch shift, trim, split
- Pre/de-emphasis, phase vocoder, Griffin-Lim (STFT and CQT)

**Decomposition & Segmentation**
- NMF decomposition, neural network filtering
- DTW, RQA, recurrence/cross-similarity matrices
- Agglomerative clustering
- Viterbi decoding (standard, discriminative, binary)

**Harmony & Transcription**
- Key detection, chord recognition
- Piano transcription

**Neural Features (madmom-compatible)**
- 67 CoreML models converted from madmom (9 model families)
- RNN beat/onset/downbeat processors
- CNN key recognition, deep chroma chord recognition
- RNN piano note processor
- Dynamic Bayesian Network tracking processors

**Infrastructure**
- Filterbank generation: mel, chroma, constant-Q, semitone
- Unit conversions: Hz/mel/MIDI/note/frames/time/samples
- Window functions
- Display: specshow, waveshow (matplotlib)
- Evaluation metrics via mir_eval wrappers

**Compatibility**
- librosa compatibility shim: 120+ functions with matching signatures
- madmom compatibility shim: full processor pipeline API

**GPU Backend**
- Metal shader dispatch: elementwise, reduction, convolution, thresholds
- MPSGraph: STFT/ISTFT, fused MFCC pipeline
- MPS: matrix multiply (mel filterbank)
- Smart Dispatch: automatic CPU/GPU routing based on input size
- Per-chip crossover thresholds (M1-M4)

**Quality**
- 1,062 Swift XCTests + 1,233 Python pytest tests = 2,295 total
- Three-tier parity testing against librosa (element-wise, behavioral, task-level)
- PEP 561 type stubs for all 18 public modules
- Benchmark suite with librosa comparison

**Distribution**
- Wheel build script
- GitHub Actions CI (CPU-only, weekly parity tests, API drift detection)
- Dual license: MIT (source) + CC-BY-NC-SA 4.0 (model weights)
