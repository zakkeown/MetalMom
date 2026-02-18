# MetalMom

**GPU-accelerated audio and music analysis for Python on Apple Silicon.**

[![CI](https://github.com/zakkeown/MetalMom/actions/workflows/ci.yml/badge.svg)](https://github.com/zakkeown/MetalMom/actions/workflows/ci.yml)

MetalMom is a drop-in replacement for [librosa](https://librosa.org/) and
[madmom](https://madmom.readthedocs.io/) that runs on Apple Metal GPUs and
Accelerate (vDSP/BNNS). It provides the same Python API you already know,
backed by a Swift/Metal engine that is typically 5-50x faster than the
CPU-only originals.

## Features

- **Full librosa compatibility** -- 120+ functions with matching signatures
- **Full madmom compatibility** -- neural beat/onset/tempo/key/chord/piano transcription
- **GPU-accelerated** -- Metal shaders for FFT, mel, MFCC, convolution, reductions
- **CPU fallback** -- Accelerate (vDSP/BLAS) when no GPU is available or for small inputs
- **Smart dispatch** -- automatically routes to GPU or CPU based on input size
- **CoreML neural inference** -- 67 converted madmom models running via CoreML/ANE
- **2,295 tests** -- 1,062 Swift XCTests + 1,233 Python pytest tests with librosa parity checks
- **2-3x faster than librosa** on 30s signals, [up to 7.6x on full pipelines](docs/site/benchmarks.html)

## Documentation

| | |
|---|---|
| **[Project Site](https://zakkeown.github.io/MetalMom/)** | Getting started, tutorials, benchmarks, architecture |
| **[API Reference](https://metalmom.readthedocs.io/)** | Full module documentation (Sphinx/autodoc) |
| **[librosa Migration](docs/site/migration-librosa.html)** | Side-by-side code comparisons, 120-function coverage table |
| **[madmom Migration](docs/site/migration-madmom.html)** | Processor mapping, CoreML model details |
| **[Tutorials](docs/site/tutorials/beat-tracking.html)** | Beat tracking, chord analysis, batch processing, visualization |
| **[Benchmarks](docs/site/benchmarks.html)** | Performance data, pipeline profiling, optimization history |

## Installation

Install from a pre-built wheel (macOS only):

```bash
pip install metalmom-0.1.0-py3-none-any.whl
```

Optional extras:

```bash
pip install "metalmom[display]"   # matplotlib for specshow/waveshow
pip install "metalmom[eval]"      # mir_eval for evaluation metrics
pip install "metalmom[dev]"       # pytest + librosa for development
```

## Quick Start

```python
import metalmom

# Load audio
y, sr = metalmom.load("song.wav", sr=22050)

# STFT (GPU-accelerated)
S = metalmom.stft(y, n_fft=2048, hop_length=512)

# Mel spectrogram
mel = metalmom.feature.melspectrogram(y=y, sr=sr)

# MFCCs
mfcc = metalmom.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Beat tracking
tempo, beats = metalmom.beat.beat_track(y=y, sr=sr)

# Onset detection
onsets = metalmom.onset.onset_detect(y=y, sr=sr)

# Chroma
chroma = metalmom.feature.chroma_stft(y=y, sr=sr)

# Pitch (YIN)
f0, voiced, confidence = metalmom.pitch.yin(y, sr=sr)
```

### Drop-in librosa Replacement

```python
from metalmom.compat import librosa

# Use exactly like librosa -- same function signatures, same return types
y, sr = librosa.load("song.wav")
S = librosa.stft(y)
mel = librosa.feature.melspectrogram(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
```

### Neural Features (madmom-compatible)

```python
import metalmom

# Neural beat tracking (CoreML inference)
beats = metalmom.beat.neural_beat_track(y=y, sr=sr)

# Key detection
key, scale, confidence = metalmom.key.detect(y=y, sr=sr)

# Chord recognition
chords = metalmom.chord.recognize(y=y, sr=sr)

# Piano transcription
notes = metalmom.transcribe.pianoroll(y=y, sr=sr)
```

## Architecture

```
Python                        Swift / Metal
------                        -------------
metalmom.core          -->    MetalMomBridge (@_cdecl)
metalmom.feature                  |
metalmom.beat                MetalMomCore (engine)
metalmom.onset               /          \
metalmom.pitch          Metal GPU    Accelerate CPU
metalmom.effects        (shaders)    (vDSP/BLAS)
  ...                        \          /
                          Smart Dispatch
                              |
cffi dlopen  <----------  libmetalmom.dylib
```

**SPM targets:**

| Target | Role |
|---|---|
| `MetalMomCBridge` | C types (`MMBuffer`, status codes) shared between Swift and Python |
| `MetalMomCore` | Swift engine: spectral, features, rhythm, pitch, ML, effects, I/O |
| `MetalMomBridge` | `@_cdecl` exports that Python calls via cffi |

**GPU acceleration:**

| Backend | Operations |
|---|---|
| Metal shaders | Elementwise ops, reductions, 1D convolution |
| MPSGraph | FFT/IFFT, fused MFCC pipeline |
| MPS | Matrix multiply (mel filterbank application) |
| CoreML | Neural network inference (beat, onset, key, chord, piano) |
| Accelerate | vDSP (FFT, windowing), BLAS (matmul), BNNS (fallback) |

## API Coverage

| Domain | Functions | Notes |
|---|---|---|
| Core I/O | `load`, `stream`, `get_duration`, `get_samplerate` | AVFoundation backend |
| Spectral | `stft`, `istft`, `cqt`, `vqt`, `hybrid_cqt`, `reassigned_spectrogram` | GPU-accelerated |
| Features | `melspectrogram`, `mfcc`, `chroma_*`, `spectral_*`, `tonnetz`, `tempogram` | Fused GPU MFCC |
| Onset | `onset_detect`, `onset_strength`, SuperFlux, ComplexFlux, HFC, KL, neural | 6 detection functions |
| Beat/Tempo | `beat_track`, `plp`, `neural_beat_track`, `tempo`, comb filter | ACF + neural |
| Pitch | `yin`, `pyin`, `piptrack` | Probabilistic pYIN |
| Effects | `hpss`, `time_stretch`, `pitch_shift` | Phase vocoder based |
| Decompose | `nmf`, `nn_filter`, `recurrence_matrix` | Non-negative MF |
| Harmony | `key.detect`, `chord.recognize` | CoreML neural |
| Transcription | `pianoroll`, `notes` | Piano transcription |
| Filters | `mel`, `chroma`, `constant_q`, `semitone`, `window` | Filterbank generation |
| Display | `specshow`, `waveshow` | matplotlib integration |
| Sequence | `dtw`, `rqa`, `viterbi` | Alignment & decoding |
| Evaluation | `beat`, `onset`, `chord`, `melody` | mir_eval wrappers |

## Development

### Prerequisites

- macOS 14+ (Sonoma) or iOS 17+
- Xcode 15+ (Swift 5.9+)
- Python 3.11+

### Build from Source

```bash
# Clone
git clone https://github.com/zakkeown/MetalMom.git
cd MetalMom

# Swift build and test
swift build
swift test

# Build the native dylib
swift build -c release
./scripts/build_dylib.sh

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy cffi soundfile pytest librosa

# Run Python tests
pytest Tests/ -v

# Build a wheel
./scripts/build_wheel.sh
```

## License

The MetalMom library source code is licensed under the [MIT License](LICENSE).

The converted neural network model weights in `models/converted/` are derived
from [madmom](https://github.com/CPJKU/madmom) and are licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE-MODELS).
See the original madmom project for commercial licensing inquiries.
