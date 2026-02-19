# Changelog

All notable changes to MetalMom will be documented in this file.

## [0.1.0-rc1] - 2026-02-19

### Added

- add metalmom.models module for HF model downloads
- add HF Hub model upload script
- add version bump script
- add iOS convenience initializer for ModelRegistry
- add ModelDownloader for HF-hosted CoreML models
- add XCFramework build script for multi-platform distribution
- extend ChipProfile with A-series iOS GPU families
- mark Signal as @unchecked Sendable with usage contract
- thread-safe MetalBackend lazy properties + Sendable
- add @unchecked Sendable to ML classes
- add Sendable conformance to error enums and value types
- add Sendable conformance to trivial types
- guard ProfilingRunner for macOS only
- add static MetalMomBridge product for iOS embedding
- add launch configurations for Debug and Release ProfilingRunner
- add ProfilingRunner executable for Instruments profiling
- add signpost instrumentation to core engine operations
- instrument SmartDispatcher and MetalBackend with signposts
- add OSSignposter-based Profiler utility
- add benchmark suite with JSON output and librosa comparison
- add PEP 561 type stubs for all public modules
- add stress test suite (115 tests) and fix 4 bugs
- wire up deep chroma DNN model (replaces CQT stub)
- add Metal GPU backend with smart dispatch
- add display, filters, madmom compat, and compat test suite
- add wheel build script
- add madmom compat shim — evaluation module
- fill remaining librosa compat gaps from audit
- add semitone bandpass filterbank with pre-computed coefficients
- add Hamming, Blackman, and Kaiser windows
- add full unit conversion suite
- add Viterbi sequence decoding Python API
- add agglomerative temporal segmentation
- add dynamic time warping
- add recurrence matrix, cross-similarity, and RQA
- add nearest-neighbor spectrogram filter
- add non-negative matrix factorization
- add feature inversion functions
- add chroma variants (CQT, CENS, VQT, deep)
- add PCEN and frequency weighting curves
- add phase vocoder and Griffin-Lim
- add reassigned spectrogram
- add CQT, VQT, and hybrid CQT
- add polyphonic piano transcription
- add chord recognition
- add CNN key detection
- add comb filter tempo estimation
- add downbeat detection
- add SuperFlux, complex flux, HFC, and KL divergence onset detection
- add neural onset detection
- add neural beat tracking (RNN + DBN)
- add chunked inference for long audio
- add ensemble inference runner
- convert all madmom neural network models to CoreML
- spike -- convert first RNNBeatProcessor model to CoreML
- add Gaussian Mixture Model evaluation
- add CRF decoding
- add HMM with Viterbi decoding
- add model registry for pre-trained models
- add CoreML inference engine
- add preemphasis and deemphasis filters
- add non-silent interval splitting
- add silence trimming
- add pitch shifting
- add time stretching via phase vocoder
- add HPSS
- add tuning estimation
- add parabolic interpolation pitch tracking
- add pYIN probabilistic pitch estimation
- add YIN pitch estimation
- add predominant local pulse estimation
- add tempogram features
- add tempo estimation
- add DP beat tracking
- add onset detection with peak picking
- add onset strength envelope (Phase 4, Task 4.3)
- add beat, tempo, and chord evaluation metrics (Phase 4, Task 4.2)
- add onset evaluation metrics (P/R/F) (Phase 4, Task 4.1)
- add streaming audio load (Phase 3, Task 3.5)
- add duration and sample rate utilities (Task 3.4)
- add signal generation (tone, chirp, clicks) (Phase 3, Task 3.3)
- add high-quality audio resampling (Phase 3, Task 3.2)
- add audio loading via AVFoundation (Phase 3, Task 3.1)
- add polynomial features (poly_features)
- add tonnetz and delta feature manipulation
- add RMS and zero-crossing rate
- add spectral descriptor features
- add STFT-based chroma features
- add MFCC extraction
- add mel spectrogram
- add mel filterbank and Hz/mel conversion
- add dB scaling functions
- add inverse STFT with overlap-add reconstruction
- return complex STFT output
- add complex signal support
- add STFT parity tests against librosa golden files
- add Python core.stft() public API
- add Python cffi bindings and buffer interop
- add dylib build script
- add C ABI bridge with context and STFT export
- add forward STFT via Accelerate/vDSP as ComputeOperation
- add ComputeBackend protocol, AccelerateBackend, and SmartDispatcher
- add Hann window function with vDSP acceleration
- add Signal core data type with pinned buffer storage
- scaffold Python package with pyproject.toml
- scaffold Swift package with 3 SPM targets

### CI

- full release pipeline with XCFramework, sigstore, git-cliff, RC/TestPyPI
- auto-file GitHub issues on parity test failure
- add git-cliff changelog configuration
- add iOS Simulator build and test job
- add release pipeline
- add weekly API drift detection with auto-issue filing
- add weekly parity test pipeline
- add GitHub Actions CI pipeline (CPU-only)

### Documentation

- add release checklist
- deployment and infrastructure implementation plan
- deployment and infrastructure design
- update project site styling and homepage copy
- add iOS support section to README
- swap Syne for Outfit display font
- add iOS port implementation plan (16 tasks)
- fix API Reference links to point to ReadTheDocs
- add documentation links section to README
- fix nav consistency and add migration cross-links
- add iOS port design doc ("The Minivan")
- add tutorials, benchmarks, architecture, changelog, and deploy workflow
- add Getting Started, librosa and madmom migration guides
- add personality to homepage copy
- add project site scaffold with homepage
- fix autodoc import mocking for ReadTheDocs builds
- add API module reference pages for all 20 modules
- add Sphinx infrastructure for API reference
- add documentation implementation plan (17 tasks)
- add documentation design for public release
- add BENCHMARKS.md with detailed performance data
- add optimization results to profiling report
- add baseline profiling report with bottleneck analysis
- add performance profiling implementation plan
- add performance profiling deep-dive design
- add polish sprint implementation plan
- add polish sprint design (griffinlim fix, type stubs, benchmarks)
- add README and license files
- add CLAUDE.md project conventions

### Fixed

- address code review findings
- wheel platform tag to cp311-abi3-macosx_14_0_arm64
- numpy scalar conversion and iOS CI runner
- install metalmom package in CI, remove iOS runtime download
- CI failures — skip Metal tests, fix iOS Simulator destination
- ProfilingRunner iOS compat + CI adjustments
- use FileManager.temporaryDirectory instead of /tmp/ in tests
- add mm_griffinlim_cqt to cffi cdef (closes last xfail)
- improve MFCC and dB scaling numerical precision
- close remaining implementation plan gaps

### Performance

- FFT-based autocorrelation + precomputed log table in BeatTracker
- replace manual resampling with AVAudioConverter (hardware-accelerated)

### Testing

- remove stale xfail markers from e2e compat tests
- replace placeholder skip with real FramedSignal test
- add end-to-end compat integration tests
- add automated madmom signature compatibility checks
- add Metal GPU smoke test

### Chore

- add .worktrees/ to gitignore
- add .gitattributes for LFS and design docs
- add .gitignore and remove tracked .pyc file

