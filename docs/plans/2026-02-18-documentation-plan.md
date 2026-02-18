# MetalMom Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build dual-layer documentation — a custom project site (GitHub Pages) and auto-generated API reference (Sphinx/ReadTheDocs) — to prepare MetalMom for public consumption.

**Architecture:** Two independent doc properties. The project site is static HTML/CSS/JS deployed via GitHub Pages. The API reference uses Sphinx with autodoc to extract from existing NumPy-style docstrings, hosted on ReadTheDocs. Cross-linked.

**Tech Stack:** Sphinx, MyST-Parser, pydata-sphinx-theme, sphinx.ext.napoleon, sphinx.ext.autodoc, sphinx.ext.intersphinx. Static HTML/CSS/JS for project site. GitHub Actions for Pages deployment.

**Design doc:** `docs/plans/2026-02-18-documentation-design.md`

---

## Part A: Sphinx / ReadTheDocs API Reference

### Task 1: Sphinx Infrastructure Setup

**Files:**
- Create: `docs/api/conf.py`
- Create: `docs/api/index.md`
- Create: `docs/api/requirements.txt`
- Create: `.readthedocs.yaml`

**Step 1: Create Sphinx requirements file**

Create `docs/api/requirements.txt`:

```
sphinx>=7.0
myst-parser>=2.0
pydata-sphinx-theme>=0.15
sphinx-copybutton>=0.5
```

**Step 2: Create Sphinx conf.py**

Create `docs/api/conf.py`:

```python
"""Sphinx configuration for MetalMom API reference."""

import os
import sys

# Add the Python package to sys.path
sys.path.insert(0, os.path.abspath("../../python"))

# -- Project information --
project = "MetalMom"
copyright = "2026, Zak Keown"
author = "Zak Keown"
release = "0.1.0"

# -- General configuration --
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# Mock native imports that require the dylib (not available on ReadTheDocs)
autodoc_mock_imports = ["cffi", "metalmom._native", "metalmom._buffer"]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output --
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/zakkeown/MetalMom",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
}
html_title = "MetalMom API Reference"
html_short_title = "MetalMom"

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]
```

**Step 3: Create API index page**

Create `docs/api/index.md`:

```markdown
# MetalMom API Reference

GPU-accelerated audio/music analysis on Apple Metal.

```{toctree}
:maxdepth: 2
:caption: API Reference

modules/core
modules/feature
modules/onset
modules/beat
modules/pitch
modules/effects
modules/cqt
modules/decompose
modules/key
modules/chord
modules/transcribe
modules/filters
modules/display
modules/segment
modules/sequence
modules/convert
modules/evaluate
modules/neural
```

## Compatibility Layers

```{toctree}
:maxdepth: 1
:caption: Compatibility

modules/compat_librosa
modules/compat_madmom
```
```

**Step 4: Create ReadTheDocs config**

Create `.readthedocs.yaml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/api/conf.py

python:
  install:
    - requirements: docs/api/requirements.txt
```

**Step 5: Verify Sphinx builds locally**

Run:
```bash
cd /Users/zakkeown/Code/MetalMom
.venv/bin/pip install sphinx myst-parser pydata-sphinx-theme sphinx-copybutton
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html
```

Expected: Build completes with warnings (mocked imports) but no errors. HTML output in `docs/api/_build/html/`.

**Step 6: Add _build to .gitignore**

Append to `.gitignore`:
```
docs/api/_build/
```

**Step 7: Commit**

```bash
git add docs/api/conf.py docs/api/index.md docs/api/requirements.txt .readthedocs.yaml .gitignore
git commit -m "docs: add Sphinx infrastructure for API reference"
```

---

### Task 2: API Module Reference Pages

**Files:**
- Create: `docs/api/modules/core.md`
- Create: `docs/api/modules/feature.md`
- Create: `docs/api/modules/onset.md`
- Create: `docs/api/modules/beat.md`
- Create: `docs/api/modules/pitch.md`
- Create: `docs/api/modules/effects.md`
- Create: `docs/api/modules/cqt.md`
- Create: `docs/api/modules/decompose.md`
- Create: `docs/api/modules/key.md`
- Create: `docs/api/modules/chord.md`
- Create: `docs/api/modules/transcribe.md`
- Create: `docs/api/modules/filters.md`
- Create: `docs/api/modules/display.md`
- Create: `docs/api/modules/segment.md`
- Create: `docs/api/modules/sequence.md`
- Create: `docs/api/modules/convert.md`
- Create: `docs/api/modules/evaluate.md`
- Create: `docs/api/modules/neural.md`
- Create: `docs/api/modules/compat_librosa.md`
- Create: `docs/api/modules/compat_madmom.md`

Each module page follows this template (example for `core.md`):

```markdown
# metalmom.core

Audio I/O, transforms, and signal generation.

```{eval-rst}
.. automodule:: metalmom.core
   :members:
   :undoc-members: False
   :show-inheritance:
```
```

**Step 1: Create all 20 module reference pages**

Create each file in `docs/api/modules/` using the template above. Module descriptions:

| File | Module | Description |
|------|--------|-------------|
| `core.md` | `metalmom.core` | Audio I/O, transforms, and signal generation |
| `feature.md` | `metalmom.feature` | Spectral features: mel, MFCC, chroma, spectral moments |
| `onset.md` | `metalmom.onset` | Onset detection and onset strength computation |
| `beat.md` | `metalmom.beat` | Beat tracking and tempo estimation |
| `pitch.md` | `metalmom.pitch` | Pitch estimation (YIN, pYIN, piptrack) |
| `effects.md` | `metalmom.effects` | Audio effects: HPSS, time stretch, pitch shift, trim |
| `cqt.md` | `metalmom.cqt` | Constant-Q and Variable-Q transforms |
| `decompose.md` | `metalmom.decompose` | NMF and neural network filtering |
| `key.md` | `metalmom.key` | Musical key detection |
| `chord.md` | `metalmom.chord` | Chord recognition |
| `transcribe.md` | `metalmom.transcribe` | Piano transcription |
| `filters.md` | `metalmom.filters` | Filterbank generation (mel, chroma, CQT) |
| `display.md` | `metalmom.display` | Visualization (specshow, waveshow) |
| `segment.md` | `metalmom.segment` | Segmentation: DTW, RQA, agglomerative clustering |
| `sequence.md` | `metalmom.sequence` | Viterbi decoding and sequence models |
| `convert.md` | `metalmom.convert` | Unit conversions (Hz, MIDI, notes, frames, time) |
| `evaluate.md` | `metalmom.evaluate` | Evaluation metrics (mir_eval wrappers) |
| `neural.md` | `metalmom.neural` | madmom-compatible neural feature processors |
| `compat_librosa.md` | `metalmom.compat.librosa` | librosa compatibility shim |
| `compat_madmom.md` | `metalmom.compat.madmom` | madmom compatibility shim |

**Step 2: Build and verify**

Run:
```bash
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html
```

Expected: All 20 module pages rendered with function signatures and docstrings.

**Step 3: Commit**

```bash
git add docs/api/modules/
git commit -m "docs: add API module reference pages for all 20 modules"
```

---

### Task 3: Fix Autodoc Import Issues

The native cffi module cannot be imported on ReadTheDocs (no Swift dylib). This task ensures Sphinx can still extract docstrings.

**Files:**
- Modify: `docs/api/conf.py` (if mocking needs adjustment)
- Possibly modify: `python/metalmom/_native.py` (add doc-build guard)

**Step 1: Attempt a full Sphinx build and note any import errors**

Run:
```bash
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html 2>&1 | grep -i "error\|warning\|failed"
```

**Step 2: Fix any import chain issues**

The `autodoc_mock_imports` in conf.py mocks `cffi`, `metalmom._native`, and `metalmom._buffer`. If other modules fail to import because they depend on these at module level, add them to the mock list.

Common pattern: if `metalmom.core` does `from ._native import lib, ffi` at the top level, the mock should handle it. If not, add a doc-build guard to `_native.py`:

```python
import os

if os.environ.get("METALMOM_DOC_BUILD"):
    ffi = None
    lib = None
else:
    import cffi
    ffi = cffi.FFI()
    # ... rest of native setup
```

And update `.readthedocs.yaml` to set `METALMOM_DOC_BUILD=1`.

**Step 3: Verify clean build**

Run:
```bash
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html
```

Expected: Build completes. Open `docs/api/_build/html/index.html` in browser — all module pages have function listings with parameter docs.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "docs: fix autodoc import mocking for ReadTheDocs builds"
```

---

## Part B: Project Site (GitHub Pages)

### Task 4: Project Site Scaffold

**Files:**
- Create: `docs/site/index.html`
- Create: `docs/site/css/style.css`
- Create: `docs/site/js/main.js`

**Step 1: Create the CSS stylesheet**

Create `docs/site/css/style.css` — a clean, modern design system:

- Color palette: dark hero section (#0a0a0a background, white text), light content sections (#fafafa)
- Accent color: electric blue (#3b82f6) for links and CTAs
- Typography: system font stack (-apple-system, BlinkMacSystemFont, "Segoe UI", ...)
- Code blocks: dark background (#1e1e2e), syntax-highlighted with CSS classes
- Responsive: mobile-first, max-width container (1200px)
- Feature cards: grid layout, subtle border, hover shadow
- Navigation: sticky header, links to all sections

**Step 2: Create the homepage HTML**

Create `docs/site/index.html` with:

- `<head>`: meta tags, Open Graph, favicon, stylesheet link
- Navigation bar: MetalMom logo/name, links (Getting Started, Migration, Tutorials, Benchmarks, Architecture, API Docs)
- Hero section: heading "MetalMom", subheading "GPU-accelerated audio analysis for Python on Apple Silicon", install command, CTA buttons (Get Started, API Reference)
- Side-by-side code comparison: librosa code vs MetalMom code (identical)
- Feature cards (4): GPU Acceleration, librosa Compatible, Neural Features, Type Safe
- Install section: `pip install metalmom`
- Footer: GitHub link, license info, ReadTheDocs link

**Step 3: Create minimal JS**

Create `docs/site/js/main.js`:
- Copy-to-clipboard for code blocks
- Smooth scroll for nav links
- Mobile nav toggle

**Step 4: Verify locally**

Open `docs/site/index.html` in browser. Verify layout, responsiveness, links.

**Step 5: Commit**

```bash
git add docs/site/
git commit -m "docs: add project site scaffold with homepage"
```

---

### Task 5: Getting Started Page

**Files:**
- Create: `docs/site/getting-started.html`

**Content sections:**

1. **Requirements** — macOS 13+, Apple Silicon (M1+), Python 3.11+
2. **Installation** — `pip install metalmom` with extras (`[display]`, `[eval]`, `[dev]`)
3. **First Analysis** — Complete code example:
   ```python
   import metalmom

   # Load audio
   y, sr = metalmom.load("song.mp3")

   # Compute mel spectrogram
   S = metalmom.melspectrogram(y=y, sr=sr)
   S_db = metalmom.amplitude_to_db(S, ref=1.0)

   # Plot it
   import matplotlib.pyplot as plt
   metalmom.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
   plt.colorbar(format='%+2.0f dB')
   plt.show()
   ```
4. **GPU Acceleration** — Smart Dispatch explanation: automatic, no configuration. Table of which operations use Metal vs Accelerate vs CPU.
5. **Next Steps** — Links to Migration Guides, Tutorials, API Reference

**Step 1: Create the page using the site template structure from Task 4**

Reuse the same nav, header, footer, and CSS. Content in a single-column readable layout (max-width 720px for text content).

**Step 2: Verify locally**

Open in browser. Check code blocks render correctly, links work.

**Step 3: Commit**

```bash
git add docs/site/getting-started.html
git commit -m "docs: add Getting Started page"
```

---

### Task 6: Migration Guide — Coming from librosa

**Files:**
- Create: `docs/site/migration-librosa.html`

**Content sections:**

1. **TL;DR** — `import metalmom as librosa` works for most code. Or use the compat shim: `from metalmom.compat import librosa`.

2. **Side-by-side comparisons** — For each common workflow, show librosa code on the left and MetalMom equivalent on the right. Cover:
   - `librosa.load()` → `metalmom.load()`
   - `librosa.stft()` → `metalmom.stft()`
   - `librosa.feature.melspectrogram()` → `metalmom.melspectrogram()`
   - `librosa.feature.mfcc()` → `metalmom.mfcc()`
   - `librosa.beat.beat_track()` → `metalmom.beat_track()`
   - `librosa.onset.onset_detect()` → `metalmom.onset_detect()`
   - `librosa.feature.chroma_stft()` → `metalmom.chroma_stft()`
   - `librosa.effects.hpss()` → `metalmom.hpss()`

3. **API Differences** — Where MetalMom deviates:
   - Flat namespace: `metalmom.melspectrogram()` not `metalmom.feature.melspectrogram()` (both work via compat)
   - macOS-only (no Linux/Windows)
   - Returns float32 always (librosa can return float64)
   - File format support via AVFoundation (broader than soundfile)

4. **Compat Shim Details** — How `from metalmom.compat import librosa` works, coverage table (120 functions).

5. **Parity Guarantees** — Link to parity tests, tolerances used.

**Step 1: Create the page**

Use two-column code comparison layout (CSS grid or flexbox). Each comparison is a card with librosa on the left, MetalMom on the right.

**Step 2: Verify locally**

**Step 3: Commit**

```bash
git add docs/site/migration-librosa.html
git commit -m "docs: add librosa migration guide"
```

---

### Task 7: Migration Guide — Coming from madmom

**Files:**
- Create: `docs/site/migration-madmom.html`

**Content sections:**

1. **TL;DR** — `from metalmom.compat import madmom` gives you the familiar Processor API backed by CoreML models.

2. **Processor Mapping** — Table mapping madmom classes to MetalMom equivalents:
   - `madmom.features.beats.RNNBeatProcessor` → `metalmom.compat.madmom.features.beats.RNNBeatProcessor`
   - `madmom.features.beats.DBNBeatTrackingProcessor` → same path in MetalMom
   - `madmom.features.onsets.RNNOnsetProcessor` → same
   - `madmom.features.tempo.TempoEstimationProcessor` → same
   - `madmom.features.downbeats.RNNDownBeatProcessor` → same
   - `madmom.features.key.CNNKeyRecognitionProcessor` → same
   - `madmom.features.chords.DeepChromaChordRecognitionProcessor` → same
   - `madmom.features.notes.RNNPianoNoteProcessor` → same

3. **Code Comparison** — Complete beat tracking pipeline, madmom vs MetalMom side-by-side.

4. **Key Differences** — CoreML backend (not PyTorch), macOS-only, 67 pre-converted models, no training support.

5. **Model Details** — How models were converted (NeuralNetworkBuilder, SafeUnpickler), model weight license (CC-BY-NC-SA 4.0).

**Step 1: Create the page**

**Step 2: Verify locally**

**Step 3: Commit**

```bash
git add docs/site/migration-madmom.html
git commit -m "docs: add madmom migration guide"
```

---

### Task 8: Tutorials — Beat Tracking a Song

**Files:**
- Create: `docs/site/tutorials/beat-tracking.html`

**Content:** Complete walkthrough:

1. Load an audio file
2. Compute beat positions with `metalmom.beat_track()`
3. Visualize beats on a waveform
4. Export beat times
5. Advanced: use neural beat tracking via `metalmom.compat.madmom`
6. Compare CPU vs GPU timing

Include complete, runnable code at each step. Show expected output (described, not screenshots).

**Step 1: Create the page**

**Step 2: Commit**

```bash
git add docs/site/tutorials/
git commit -m "docs: add beat tracking tutorial"
```

---

### Task 9: Tutorials — Chord Analysis & Batch Processing

**Files:**
- Create: `docs/site/tutorials/chord-analysis.html`
- Create: `docs/site/tutorials/batch-processing.html`

**Chord Analysis content:**
1. Load audio
2. Compute chroma features
3. Run chord detection
4. Visualize chord progression over time
5. Export chord annotations

**Batch Processing content:**
1. Scan a directory for audio files
2. Process each file (extract features)
3. Save results to CSV/JSON
4. Performance tips (one context per thread, batch sizing)

**Step 1: Create both pages**

**Step 2: Commit**

```bash
git add docs/site/tutorials/
git commit -m "docs: add chord analysis and batch processing tutorials"
```

---

### Task 10: Tutorials — Parity Verification & Visualization

**Files:**
- Create: `docs/site/tutorials/parity-verification.html`
- Create: `docs/site/tutorials/visualization.html`

**Parity Verification content:**
1. Run the same analysis in librosa and MetalMom
2. Compare outputs element-wise
3. Understand tolerances (rtol/atol)
4. When to expect exact match vs approximate match

**Visualization content:**
1. Install display extras: `pip install metalmom[display]`
2. `specshow()` — spectrograms, mel, chroma, CQT
3. `waveshow()` — waveforms with onsets/beats overlaid
4. Customizing plots (colormaps, axis labels, time/frequency units)

**Step 1: Create both pages**

**Step 2: Commit**

```bash
git add docs/site/tutorials/
git commit -m "docs: add parity verification and visualization tutorials"
```

---

### Task 11: Benchmarks Page

**Files:**
- Create: `docs/site/benchmarks.html`

**Content:** Adapted from existing `BENCHMARKS.md`:

1. **Test Setup** — Hardware, OS, Python version, signal lengths
2. **MetalMom vs librosa** — Styled comparison table with speedup column. Cover: STFT, melspectrogram, MFCC, chroma, onset_strength, beat_track. Three signal lengths.
3. **Full Pipeline** — Real audio file benchmarks (11-min song, 30-sec excerpt)
4. **Optimization History** — The 3 major fixes with before/after
5. **Backend Dispatch** — When Metal kicks in vs Accelerate vs CPU
6. **Reproduce** — How to run the benchmark suite yourself

Style the tables as visually appealing HTML tables with alternating row colors and highlighted speedup values.

**Step 1: Create the page**

**Step 2: Commit**

```bash
git add docs/site/benchmarks.html
git commit -m "docs: add benchmarks page"
```

---

### Task 12: Architecture Overview Page

**Files:**
- Create: `docs/site/architecture.html`

**Content:** Condensed from design doc:

1. **System Diagram** — ASCII or SVG: Python → cffi → Swift (MetalMomBridge → MetalMomCore → MetalMomCBridge) → Metal/Accelerate/CoreML
2. **Three SPM Targets** — What each does, dependency direction
3. **Smart Dispatch** — Protocol-based CPU/GPU routing, crossover points table, how to force a backend
4. **Memory Model** — Minimal-copy pattern: Swift→MMBuffer copy, MMBuffer→NumPy copy, free C-side
5. **GIL Release** — cffi releases GIL during native calls, enables threading
6. **CoreML Integration** — 67 converted models, NeuralNetworkBuilder, inference pipeline

**Step 1: Create the page**

**Step 2: Commit**

```bash
git add docs/site/architecture.html
git commit -m "docs: add architecture overview page"
```

---

## Part C: Supporting Files & Deployment

### Task 13: Navigation & Cross-Linking

**Files:**
- Modify: all `docs/site/*.html` pages

**Step 1: Ensure consistent navigation**

All pages should share the same nav bar linking to:
- Getting Started
- Migration (dropdown: librosa, madmom)
- Tutorials (dropdown: all 5 tutorials)
- Benchmarks
- Architecture
- API Reference (external link to ReadTheDocs)
- GitHub (external link)

**Step 2: Add footer to all pages**

Footer with: GitHub link, license, "Built with Metal" tagline, link to ReadTheDocs.

**Step 3: Add cross-links within content**

- Getting Started → links to Migration Guides and Tutorials
- Migration guides → links to API Reference for function details
- Tutorials → links to relevant API Reference sections
- Benchmarks → link to reproduce instructions
- Architecture → link to full design doc on GitHub

**Step 4: Commit**

```bash
git add docs/site/
git commit -m "docs: add consistent navigation and cross-linking"
```

---

### Task 14: CHANGELOG

**Files:**
- Create: `CHANGELOG.md`

**Content:**

```markdown
# Changelog

All notable changes to MetalMom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-18

### Added
- Core audio I/O: load, resample, stream, get_duration, get_samplerate
- STFT/ISTFT with GPU acceleration via Metal
- Spectral features: melspectrogram, MFCC, chroma (STFT/CQT/CENS/VQT), spectral moments
- Onset detection: onset_detect, onset_strength, onset_strength_multi
- Beat tracking: beat_track, tempo, PLP
- Pitch estimation: YIN, pYIN, piptrack
- Audio effects: HPSS, time_stretch, pitch_shift, trim, split, phase vocoder, Griffin-Lim
- CQT/VQT/hybrid CQT transforms
- NMF decomposition and NN filtering
- Key detection, chord recognition, piano transcription
- Segmentation: DTW, RQA, agglomerative clustering, recurrence matrix
- Viterbi decoding (standard, discriminative, binary)
- Unit conversions: Hz/mel/MIDI/note/frames/time
- Filterbank generation: mel, chroma, constant-Q, semitone
- Display: specshow, waveshow (matplotlib)
- Evaluation metrics via mir_eval wrappers
- librosa compatibility shim (120 functions)
- madmom compatibility shim (67 CoreML models, 9 processor families)
- Metal GPU backend with smart dispatch (STFT, elementwise, matmul, reduction, convolution, fused MFCC)
- PEP 561 type stubs for all 18 public modules
- Benchmark suite with librosa comparison
```

**Step 1: Create the file**

**Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG for v0.1.0"
```

---

### Task 15: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add documentation links**

Add a "Documentation" section near the top of the README (after the badges/description, before Quick Start):

```markdown
## Documentation

- **[Getting Started](https://zakkeown.github.io/MetalMom/getting-started.html)** — Installation and first analysis
- **[API Reference](https://metalmom.readthedocs.io)** — Full function reference
- **[Migration from librosa](https://zakkeown.github.io/MetalMom/migration-librosa.html)** — Side-by-side code comparisons
- **[Migration from madmom](https://zakkeown.github.io/MetalMom/migration-madmom.html)** — Processor mapping guide
- **[Tutorials](https://zakkeown.github.io/MetalMom/tutorials/beat-tracking.html)** — Step-by-step guides
- **[Benchmarks](https://zakkeown.github.io/MetalMom/benchmarks.html)** — Performance data
```

**Step 2: Streamline README content**

The README currently duplicates content that will now live in dedicated pages. Keep:
- Project description and badges
- Documentation links (new)
- Quick Start (condensed to 3-4 examples max)
- Architecture overview (condensed to diagram + one paragraph)
- Development section (build commands)
- License

Remove or condense:
- Detailed API coverage table (→ API Reference)
- Extended code examples (→ Tutorials)
- Detailed backend tables (→ Architecture page)

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: streamline README with links to documentation site"
```

---

### Task 16: GitHub Pages Deployment

**Files:**
- Create: `.github/workflows/docs.yml`

**Step 1: Create GitHub Actions workflow**

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/site/**'

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/site
      - id: deployment
        uses: actions/deploy-pages@v4
```

**Step 2: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Pages deployment for project site"
```

---

### Task 17: Final Review & Polish

**Step 1: Full Sphinx build — verify clean**

```bash
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html -W
```

The `-W` flag turns warnings into errors. Fix any remaining issues.

**Step 2: Open project site locally — visual review**

Open `docs/site/index.html` in browser. Click through every page. Check:
- All links work (internal and cross-links to ReadTheDocs)
- Code examples are syntactically correct
- Mobile responsive layout
- No broken images or missing CSS

**Step 3: Verify cross-linking**

- Project site links to ReadTheDocs (placeholder URLs until RTD is set up)
- README links to GitHub Pages (placeholder URLs until Pages is set up)

**Step 4: Final commit**

```bash
git add -u
git commit -m "docs: final review polish and link fixes"
```
