# Model Conversion Validation Suite — Design

**Date:** 2026-02-19
**Status:** Approved

## Problem

65 madmom models were converted to CoreML format (.mlmodel) across 8 architecture
types. The conversion pipeline extracts weights via SafeUnpickler and builds CoreML
models via NeuralNetworkBuilder. However, only 1 of 65 models (beats_blstm_1) has
any forward-pass validation, and even that was never tested via actual CoreML
inference. All Phase 7 neural feature tests use synthetic activation arrays — they
validate the decoders (DBN, peak picking, CRF) but never run the converted models.

**Current confidence by family:**

| Family | Architecture | Models | Validation Status | Confidence |
|--------|-------------|--------|-------------------|------------|
| beats (BiLSTM) | 3x BiLSTM + Dense | 8 | 1 model numpy-validated | Medium |
| beats (LSTM) | 3x LSTM + Dense | 8 | None | Low |
| downbeats (BiLSTM) | BiLSTM + Dense | 8 | None | Low |
| downbeats (BiGRU) | BiGRU + Dense | 12 | None | Low |
| onsets (BiRNN) | BiRNN + Dense | 16 | None | Low |
| onsets (RNN) | RNN + Dense | 8 | None | Low |
| onsets (CNN) | Conv+BN+Pool+Dense | 1 | None | Low |
| key (CNN) | Conv+BN+Pool+Dense | 1 | None | Low |
| chords (CNN) | CNN + embedding | 1 | None | Low |
| chroma (DNN) | Dense layers | 1 | None | Low |
| notes (BiRNN) | BiRNN + Dense | 1 | None | Low |
| chords (CRF) | .npz params | 2 | N/A (not neural) | Medium |

## Goals

1. **Numerical parity**: Prove each converted CoreML model produces the same
   activations as a numpy reimplementation of the original madmom forward pass,
   given identical input.
2. **Task-level plausibility**: Prove the full pipeline (audio → features → model →
   decoder → task output) produces reasonable results on synthesized audio.
3. **Regression safety**: Establish golden baselines so future changes to the
   inference engine can be validated.

## Approach

Full numpy forward pass for all 8 architecture types, golden files generated
on-demand (not committed), CoreML comparison via both Python (C bridge) and
Swift (XCTest).

## Design

### Component 1: Numpy Forward Pass Engine

**File:** `models/conversion/numpy_forward.py`

Implements forward passes for all 8 neural architecture types, matching madmom's
exact computation.

**Already implemented** (in `validate_conversion.py`, to be extracted):
- `lstm_forward()` — full peephole LSTM (input/forget/output gates with peephole weights)
- `bilstm_forward()` — bidirectional wrapper (forward + reversed backward + concat)
- `dense_forward()` — matmul + bias + sigmoid

**New implementations needed:**

| Function | Gate Equations | Notes |
|----------|---------------|-------|
| `gru_forward()` | update: σ(W_z·x + R_z·h + b_z), reset: σ(W_r·x + R_r·h + b_r), candidate: tanh(W·x + R·(r⊙h) + b) | No peephole. Gate order differs from LSTM. |
| `bigru_forward()` | Forward + reversed backward GRU + concat | Same bidirectional pattern as BiLSTM |
| `rnn_forward()` | h = tanh(W·x + R·h + b) | Simplest recurrent unit |
| `birnn_forward()` | Forward + reversed backward RNN + concat | Same bidirectional pattern |
| `conv2d_forward()` | Standard convolution + optional bias | Match CoreML's NCHW layout |
| `batchnorm_forward()` | (x - μ) / √(σ² + ε) · γ + β | Use running mean/var from training |
| `maxpool_forward()` | Max over kernel window | Match CoreML pool params |
| `cnn_model_forward()` | Chain of conv → bn → activation → pool → dense | Architecture-specific layer ordering |
| `dnn_model_forward()` | Chain of dense → activation | Supports relu, elu, sigmoid, softmax, tanh |

All functions load weights from .pkl files via SafeUnpickler (no madmom import needed).

### Component 2: Golden File Generator

**File:** `models/conversion/generate_golden.py`

For each of the 65 .mlmodel files:
1. Load the corresponding .pkl model via SafeUnpickler
2. Classify architecture type
3. Generate deterministic input using `np.random.seed(42)` + appropriate shape
4. Run numpy forward pass
5. Return (input_array, output_array) pair

**Input shapes by architecture:**

| Architecture | Input Shape | Rationale |
|---|---|---|
| BiLSTM/LSTM (beats) | `(100, 266)` | 100 frames of 266-dim spectral features |
| BiLSTM (downbeats) | `(100, input_dim)` | Detected from pkl weight matrix shape |
| BiGRU (downbeats) | `(100, input_dim)` | Detected from pkl |
| BiRNN (onsets/notes) | `(100, input_dim)` | Detected from pkl |
| RNN (onsets) | `(100, input_dim)` | Detected from pkl |
| CNN (onsets/key/chords) | Architecture-specific | 2D input; shape from pkl conv weights |
| DNN (chroma) | `(100, input_dim)` | Frame-by-frame feedforward |

Golden files are **generated on-demand** during test runs (not committed to repo).
Reproducible via fixed random seed.

### Component 3: Python Parity Tests

**File:** `Tests/test_model_parity.py`

For each model:
1. Generate golden (input, expected_output) via numpy forward pass engine
2. Load the .mlmodel via MetalMom's C bridge (mm_predict or model registry)
3. Feed the same input through CoreML inference
4. Compare: `max(abs(coreml_output - numpy_output)) < 1e-4`

**Test organization:**
- Parameterized by model name (65 test cases)
- Grouped by family for readability
- Skipped if .mlmodel files not present (CI without models)
- Skipped if .pkl source files not present (can't generate golden)

**Tolerance:** `1e-4` (float32 precision; CoreML may use different accumulation
order or mixed precision on Apple Neural Engine/GPU).

### Component 4: Swift Inference Smoke Tests

**File:** `Tests/MetalMomTests/ModelInferenceTests.swift`

Lighter validation — confirms CoreML loads and produces valid output:
1. Load each `.mlmodelc` via `InferenceEngine`
2. Feed deterministic input (seeded, same as golden generator)
3. Verify:
   - Output shape matches expected dimensions
   - Output values in valid range (e.g., [0,1] for sigmoid, valid probabilities)
   - Deterministic (two runs produce identical output)
   - No NaN/Inf values
4. Does NOT compare against golden numpy output (that's the Python test's job)

**Test organization:**
- Parameterized by model name
- Requires compiled .mlmodelc fixtures (from `coremlcompiler compile`)
- Skipped if models not present

### Component 5: End-to-End Smoke Tests

**File:** `Tests/test_e2e_models.py`

Full pipeline validation with synthesized audio and loose task-level thresholds.

| Task | Synthesized Signal | Ground Truth | Pass Criteria |
|---|---|---|---|
| Beat tracking | Click track at 120 BPM | Beats every 0.5s | Detected BPM within 10% of 120 |
| Onset detection | Sine bursts at known intervals | Onset times | ≥50% onsets detected within 50ms |
| Downbeat tracking | Click track with accented 4/4 | Downbeat every 2s | ≥50% downbeats detected within 100ms |
| Key detection | C major scale tones | Key = C major | Correct key or relative minor |
| Chord recognition | C-F-G triads in sequence | Chord labels | ≥1 correct chord detected |
| Piano transcription | A4 sine (440 Hz) | Note = A4 | A4 detected in output |
| Chroma extraction | Single pitch | Dominant chroma bin | Correct bin has highest energy |

**Thresholds are deliberately loose.** These models were trained on real music,
not synthesized signals. The goal is "does the full pipeline produce plausible
results" — not exact match. The numerical parity tests (Component 3) provide
the rigorous validation.

## Architecture Validation Risk Assessment

| Architecture | Conversion Complexity | Key Risk | Priority |
|---|---|---|---|
| **CNN** (onsets, key, chords) | High — conv/bn/pool/reshape chains | Shape mismatches between CoreML layers | P0 |
| **BiGRU** (downbeats) | Medium — no native CoreML BiGRU, uses two GRU + concat | Gate order mapping (update→z, reset→r) | P0 |
| **BiRNN** (onsets, notes) | Medium — same pattern as BiGRU but simpler | Bidirectional concat ordering | P1 |
| **LSTM** (beats) | Low — reuse existing lstm_forward | Same as validated BiLSTM, just unidirectional | P2 |
| **RNN** (onsets) | Low — simplest recurrent | Minimal risk | P2 |
| **DNN** (chroma) | Low — just dense layers | Activation function mapping | P2 |
| **BiLSTM** (beats, downbeats) | Low — already validated for 1 model | Extend to all 16 models | P2 |

**CNN and BiGRU are highest priority** — they have the most complex conversion
logic and zero existing validation.

## Non-Goals

- Validating CRF .npz files (not neural inference; already used correctly in Python decoders)
- Comparing against actual madmom runtime (can't import on Python 3.14)
- GPU vs CPU inference parity (separate concern, already handled by Metal dispatch tests)
- Performance benchmarking of model inference (separate from correctness)

## Dependencies

- `.pkl` source files from madmom installation (for golden generation)
- `.mlmodel` files in `models/converted/` (for CoreML inference)
- Built dylib (for Python C bridge tests)
- coremltools (for protobuf spec inspection, if needed)
