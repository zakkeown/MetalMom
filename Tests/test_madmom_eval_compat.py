"""Tests for the madmom evaluation compatibility shim.

Verifies that madmom-compatible evaluation classes backed by MetalMom
produce correct metrics for known inputs.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# OnsetEvaluation tests
# ---------------------------------------------------------------------------

class TestOnsetEvaluation:
    """Tests for madmom.evaluation.onsets.OnsetEvaluation."""

    def test_perfect_detections(self):
        """Perfect detections should yield fmeasure = 1.0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [0.5, 1.0, 1.5, 2.0]
        detections = [0.5, 1.0, 1.5, 2.0]
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.fmeasure == pytest.approx(1.0, abs=1e-6)
        assert ev.precision == pytest.approx(1.0, abs=1e-6)
        assert ev.recall == pytest.approx(1.0, abs=1e-6)

    def test_no_detections(self):
        """No detections should yield fmeasure = 0.0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [0.5, 1.0, 1.5, 2.0]
        detections = []
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.fmeasure == pytest.approx(0.0, abs=1e-6)
        assert ev.precision == pytest.approx(0.0, abs=1e-6)
        assert ev.recall == pytest.approx(0.0, abs=1e-6)

    def test_no_annotations(self):
        """No annotations with detections: precision=0, recall=0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = []
        detections = [0.5, 1.0]
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.fmeasure == pytest.approx(0.0, abs=1e-6)

    def test_both_empty(self):
        """Both empty: everything should be 0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        ev = OnsetEvaluation([], [], window=0.05)
        assert ev.fmeasure == pytest.approx(0.0, abs=1e-6)
        assert ev.precision == pytest.approx(0.0, abs=1e-6)
        assert ev.recall == pytest.approx(0.0, abs=1e-6)

    def test_properties_exist(self):
        """All expected properties should be accessible."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [1.0, 2.0, 3.0]
        detections = [1.0, 2.0, 3.0]
        ev = OnsetEvaluation(detections, annotations)
        assert isinstance(ev.fmeasure, float)
        assert isinstance(ev.precision, float)
        assert isinstance(ev.recall, float)
        assert isinstance(ev.num_tp, int)
        assert isinstance(ev.num_fp, int)
        assert isinstance(ev.num_fn, int)

    def test_num_tp_perfect(self):
        """Perfect detections: TP = n, FP = 0, FN = 0."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [1.0, 2.0, 3.0]
        detections = [1.0, 2.0, 3.0]
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.num_tp == 3
        assert ev.num_fp == 0
        assert ev.num_fn == 0

    def test_num_fp_extra_detections(self):
        """Extra detections should count as false positives."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [1.0]
        detections = [1.0, 5.0, 6.0]
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.num_tp == 1
        assert ev.num_fp == 2
        assert ev.num_fn == 0

    def test_num_fn_missed_annotations(self):
        """Missed annotations should count as false negatives."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [1.0, 2.0, 3.0]
        detections = [1.0]
        ev = OnsetEvaluation(detections, annotations, window=0.05)
        assert ev.num_tp == 1
        assert ev.num_fp == 0
        assert ev.num_fn == 2

    def test_window_parameter(self):
        """Larger window should match nearby detections."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        annotations = [1.0, 2.0]
        # Detections slightly off
        detections = [1.04, 2.04]

        # Tight window: should miss
        ev_tight = OnsetEvaluation(detections, annotations, window=0.03)
        # Loose window: should match
        ev_loose = OnsetEvaluation(detections, annotations, window=0.05)

        assert ev_loose.fmeasure >= ev_tight.fmeasure

    def test_default_window(self):
        """Default window should be 0.025 (madmom convention)."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        # Detections within 0.025 of annotations
        annotations = [1.0, 2.0]
        detections = [1.02, 2.02]
        ev = OnsetEvaluation(detections, annotations)
        assert ev.fmeasure == pytest.approx(1.0, abs=1e-6)

    def test_repr(self):
        """OnsetEvaluation should have a readable repr."""
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        ev = OnsetEvaluation([1.0], [1.0])
        r = repr(ev)
        assert 'OnsetEvaluation' in r
        assert 'fmeasure' in r


# ---------------------------------------------------------------------------
# BeatEvaluation tests
# ---------------------------------------------------------------------------

class TestBeatEvaluation:
    """Tests for madmom.evaluation.beats.BeatEvaluation."""

    def test_perfect_beats(self):
        """Perfect beat detections should yield high fmeasure."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        # Beats at 120 BPM for 4 seconds
        annotations = np.arange(0.5, 4.0, 0.5)
        detections = annotations.copy()
        ev = BeatEvaluation(detections, annotations)
        assert ev.fmeasure == pytest.approx(1.0, abs=1e-4)

    def test_properties_exist(self):
        """All expected properties should be accessible."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        annotations = np.arange(0.5, 4.0, 0.5)
        detections = annotations.copy()
        ev = BeatEvaluation(detections, annotations)
        assert isinstance(ev.fmeasure, float)
        assert isinstance(ev.cemgil, float)
        assert isinstance(ev.p_score, float)
        assert isinstance(ev.cmlc, float)
        assert isinstance(ev.cmlt, float)
        assert isinstance(ev.amlc, float)
        assert isinstance(ev.amlt, float)

    def test_no_detections(self):
        """No detections should yield fmeasure = 0."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        annotations = np.arange(0.5, 4.0, 0.5)
        detections = []
        ev = BeatEvaluation(detections, annotations)
        assert ev.fmeasure == pytest.approx(0.0, abs=1e-6)

    def test_fmeasure_window_param(self):
        """fmeasure_window parameter should be accepted."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        annotations = np.arange(0.5, 4.0, 0.5)
        detections = annotations + 0.05  # 50ms offset
        ev = BeatEvaluation(detections, annotations, fmeasure_window=0.07)
        assert ev.fmeasure > 0.0

    def test_repr(self):
        """BeatEvaluation should have a readable repr."""
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        ev = BeatEvaluation([1.0, 2.0], [1.0, 2.0])
        r = repr(ev)
        assert 'BeatEvaluation' in r


# ---------------------------------------------------------------------------
# TempoEvaluation tests
# ---------------------------------------------------------------------------

class TestTempoEvaluation:
    """Tests for madmom.evaluation.tempo.TempoEvaluation."""

    def test_correct_tempo(self):
        """Correct tempo should give acc1 = 1.0 and acc2 = 1.0."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(120.0, 120.0)
        assert ev.acc1 == pytest.approx(1.0, abs=1e-6)
        assert ev.acc2 == pytest.approx(1.0, abs=1e-6)

    def test_wrong_tempo(self):
        """Completely wrong tempo should give acc1 = 0 and acc2 = 0."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(120.0, 80.0)
        assert ev.acc1 == pytest.approx(0.0, abs=1e-6)
        assert ev.acc2 == pytest.approx(0.0, abs=1e-6)

    def test_double_tempo(self):
        """Double tempo should give acc1 = 0 but acc2 = 1.0."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(240.0, 120.0)
        assert ev.acc1 == pytest.approx(0.0, abs=1e-6)
        assert ev.acc2 == pytest.approx(1.0, abs=1e-6)

    def test_half_tempo(self):
        """Half tempo should give acc1 = 0 but acc2 = 1.0."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(60.0, 120.0)
        assert ev.acc1 == pytest.approx(0.0, abs=1e-6)
        assert ev.acc2 == pytest.approx(1.0, abs=1e-6)

    def test_array_input(self):
        """Should accept array input (uses first element)."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation([120.0, 0.9], [120.0, 0.0])
        assert ev.acc1 == pytest.approx(1.0, abs=1e-6)

    def test_properties_exist(self):
        """acc1 and acc2 properties should exist and be floats."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(120.0, 120.0)
        assert isinstance(ev.acc1, float)
        assert isinstance(ev.acc2, float)

    def test_tolerance_parameter(self):
        """Tolerance parameter should affect matching."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        # 5% off: 120 * 1.05 = 126
        ev_tight = TempoEvaluation(126.0, 120.0, tolerance=0.04)
        ev_loose = TempoEvaluation(126.0, 120.0, tolerance=0.06)
        assert ev_tight.acc1 == pytest.approx(0.0, abs=1e-6)
        assert ev_loose.acc1 == pytest.approx(1.0, abs=1e-6)

    def test_repr(self):
        """TempoEvaluation should have a readable repr."""
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        ev = TempoEvaluation(120.0, 120.0)
        r = repr(ev)
        assert 'TempoEvaluation' in r


# ---------------------------------------------------------------------------
# Import / Package structure tests
# ---------------------------------------------------------------------------

class TestEvaluationImports:
    """Verify the evaluation compat package structure."""

    def test_import_onset_evaluation(self):
        from metalmom.compat.madmom.evaluation.onsets import OnsetEvaluation
        assert OnsetEvaluation is not None

    def test_import_beat_evaluation(self):
        from metalmom.compat.madmom.evaluation.beats import BeatEvaluation
        assert BeatEvaluation is not None

    def test_import_tempo_evaluation(self):
        from metalmom.compat.madmom.evaluation.tempo import TempoEvaluation
        assert TempoEvaluation is not None

    def test_import_via_compat_path(self):
        """Should be importable via metalmom.compat.madmom.evaluation."""
        from metalmom.compat import madmom
        assert hasattr(madmom, 'evaluation')
        assert hasattr(madmom.evaluation, 'onsets')
        assert hasattr(madmom.evaluation, 'beats')
        assert hasattr(madmom.evaluation, 'tempo')

    def test_evaluation_submodule_from_init(self):
        """Evaluation submodules should be accessible from package."""
        from metalmom.compat.madmom import evaluation
        assert hasattr(evaluation, 'onsets')
        assert hasattr(evaluation, 'beats')
        assert hasattr(evaluation, 'tempo')
