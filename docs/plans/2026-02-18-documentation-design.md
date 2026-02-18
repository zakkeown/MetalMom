# MetalMom Documentation Design

**Date:** 2026-02-18
**Status:** Approved

## Goal

Prepare MetalMom for public consumption with two documentation properties: a custom project site for discovery/adoption and an auto-generated API reference for daily use.

## Target Audience

1. **Python MIR researchers** migrating from librosa/madmom who want a faster drop-in replacement on Mac
2. **Audio/DSP developers** building audio apps who want GPU-accelerated analysis

## Architecture: Dual-Layer

| Layer | Purpose | Tech | Hosting | URL |
|-------|---------|------|---------|-----|
| Project site | Landing page, tutorials, migration guides, benchmarks, architecture | Static HTML/CSS/JS | GitHub Pages | TBD |
| API reference | Auto-generated function/class reference from docstrings | Sphinx + MyST + autodoc | ReadTheDocs | metalmom.readthedocs.io |

Cross-linking: project site links to ReadTheDocs for API details; ReadTheDocs links back for tutorials.

### Source Layout

```
docs/
  site/               # Project site (GitHub Pages)
    index.html
    getting-started/
    migration/
    tutorials/
    benchmarks/
    architecture/
    css/
    js/
  api/                # Sphinx source (ReadTheDocs)
    conf.py
    index.md
    modules/          # Auto-generated API reference pages
```

## Layer 1: Project Site (GitHub Pages)

### Homepage
- Hero section: tagline, value prop ("Drop-in librosa replacement, 2-3x faster on Apple Silicon")
- Feature cards: GPU acceleration, librosa compat, madmom neural features, type-safe
- Side-by-side code snippet (librosa vs MetalMom — identical API)
- Install command (`pip install metalmom`)
- Links to Getting Started, API Docs, GitHub

### Getting Started
- Installation: wheel, extras (`display`, `eval`, `dev`), requirements (macOS + Apple Silicon)
- First analysis: load audio, compute STFT, plot mel spectrogram
- GPU acceleration: Smart Dispatch explanation (automatic, no config)
- Expectations: which operations are GPU-accelerated, which are CPU

### Migration Guides

**Coming from librosa:**
- Side-by-side code comparisons for common workflows (load, stft, melspectrogram, mfcc, beat_track, onset_detect, chroma)
- Differences and gotchas
- Import alias pattern: `import metalmom as librosa`

**Coming from madmom:**
- Neural feature equivalents (RNNBeatProcessor, RNNOnsetProcessor, etc.)
- Model loading differences
- Processor pipeline mapping

### Tutorials / Cookbook
- Beat tracking a song end-to-end
- Building a chord analyzer
- Batch processing a music library
- Comparing MetalMom vs librosa output (parity verification)
- Using with matplotlib for visualization

### Benchmarks
- Content from existing BENCHMARKS.md
- Speedup tables/charts
- Hardware requirements and expectations
- Optimization history highlights

### Architecture Overview
- Condensed from design doc
- Diagram: Python -> cffi -> Swift/Metal -> Accelerate/CoreML
- Smart Dispatch: when GPU vs CPU, crossover points
- Why this architecture: minimal-copy, GIL release, single-threaded context

## Layer 2: API Reference (Sphinx + ReadTheDocs)

### Module Structure

```
API Reference
  metalmom.core            # load, stft, istft, resample, get_duration
  metalmom.feature         # melspectrogram, mfcc, chroma_stft, spectral_*
  metalmom.onset           # onset_detect, onset_strength, onset_strength_multi
  metalmom.beat            # beat_track, tempo, plp
  metalmom.pitch           # pyin, yin, piptrack
  metalmom.effects         # harmonic, percussive, hpss, trim, split, preemphasis
  metalmom.decompose       # nn_filter, nmf
  metalmom.harmony         # chroma_cqt, chroma_cens, key, chords
  metalmom.transcription   # piano_transcription
  metalmom.filters         # mel, chroma, constant_q, semitone_filterbank
  metalmom.display         # specshow, waveshow
  metalmom.sequence        # viterbi, dtw, rqa, agglomerative
  metalmom.convert         # hz_to_mel, mel_to_hz, note_to_hz
  metalmom.evaluation      # mir_eval wrappers
  metalmom.neural          # madmom-compatible neural feature processors
  metalmom.compat.librosa  # compat shim details
  metalmom.compat.madmom   # compat shim details
```

### Sphinx Configuration
- `myst_parser` for Markdown narrative pages
- `sphinx.ext.autodoc` + `sphinx.ext.autosummary` for API extraction
- `sphinx.ext.napoleon` for NumPy-style docstring parsing
- `sphinx.ext.intersphinx` for cross-links to NumPy, librosa docs
- `pydata-sphinx-theme` (same theme librosa uses)

### ReadTheDocs Configuration
- `.readthedocs.yaml` in repo root
- Builds on push to `main`
- Versioned docs for tagged releases

## Supporting Files

- **CHANGELOG.md**: Version history, maintained going forward
- **Updated README.md**: Streamlined, links to both doc properties

## Out of Scope

- Contributing guide (not priority for target audience)
- Troubleshooting/FAQ (add reactively as issues come in)
- Video tutorials
- Jupyter notebook tutorial versions (can add later)
