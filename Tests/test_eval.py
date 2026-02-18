"""Tests for beat, tempo, and chord evaluation metrics (Python bindings)."""

import numpy as np
import metalmom


# ── Beat evaluation ──────────────────────────────────────────────

class TestBeatEvaluate:
    def test_perfect_match(self):
        result = metalmom.beat_evaluate([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        assert abs(result['f_measure'] - 1.0) < 1e-4
        assert abs(result['cemgil'] - 1.0) < 1e-4
        assert result['p_score'] > 0.99

    def test_within_tolerance(self):
        result = metalmom.beat_evaluate([1.0, 2.0, 3.0], [1.01, 2.02, 3.01])
        assert abs(result['f_measure'] - 1.0) < 1e-4
        assert result['cemgil'] > 0.9

    def test_no_match(self):
        result = metalmom.beat_evaluate([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        assert result['f_measure'] < 0.01
        assert result['cemgil'] < 0.01
        assert result['p_score'] < 0.01

    def test_empty_reference(self):
        result = metalmom.beat_evaluate([], [1.0, 2.0])
        assert result['f_measure'] == 0
        assert result['cemgil'] == 0
        assert result['p_score'] == 0

    def test_empty_estimated(self):
        result = metalmom.beat_evaluate([1.0, 2.0], [])
        assert result['f_measure'] == 0
        assert result['cemgil'] == 0
        assert result['p_score'] == 0

    def test_both_empty(self):
        result = metalmom.beat_evaluate([], [])
        assert result['f_measure'] == 0

    def test_partial_match(self):
        result = metalmom.beat_evaluate([1.0, 2.0, 3.0, 4.0], [1.0, 2.0])
        assert 0 < result['f_measure'] < 1.0

    def test_custom_fmeasure_window(self):
        # 100ms offset, default 70ms window -> miss
        result1 = metalmom.beat_evaluate([1.0, 2.0, 3.0], [1.1, 2.1, 3.1],
                                         fmeasure_window=0.07)
        assert result1['f_measure'] == 0

        # 150ms window -> match
        result2 = metalmom.beat_evaluate([1.0, 2.0, 3.0], [1.1, 2.1, 3.1],
                                         fmeasure_window=0.15)
        assert abs(result2['f_measure'] - 1.0) < 1e-4

    def test_all_keys_present(self):
        result = metalmom.beat_evaluate([1.0, 2.0], [1.0, 2.0])
        expected_keys = {'f_measure', 'cemgil', 'p_score', 'cml_c', 'cml_t', 'aml_c', 'aml_t'}
        assert set(result.keys()) == expected_keys

    def test_aml_ge_cml(self):
        result = metalmom.beat_evaluate([1.0, 2.0, 3.0, 4.0], [1.0, 3.0])
        assert result['aml_c'] >= result['cml_c']


# ── Tempo evaluation ──────────────────────────────────────────────

class TestTempoEvaluate:
    def test_exact_match(self):
        result = metalmom.tempo_evaluate(120.0, 120.0)
        assert result['p_score'] == 1.0

    def test_within_tolerance(self):
        result = metalmom.tempo_evaluate(120.0, 128.0)
        assert result['p_score'] == 1.0

    def test_outside_tolerance(self):
        result = metalmom.tempo_evaluate(120.0, 132.0)
        assert result['p_score'] == 0.0

    def test_double_tempo(self):
        result = metalmom.tempo_evaluate(120.0, 240.0)
        assert result['p_score'] == 1.0

    def test_half_tempo(self):
        result = metalmom.tempo_evaluate(120.0, 60.0)
        assert result['p_score'] == 1.0

    def test_triple_tempo(self):
        result = metalmom.tempo_evaluate(120.0, 360.0)
        assert result['p_score'] == 1.0

    def test_third_tempo(self):
        result = metalmom.tempo_evaluate(120.0, 40.0)
        assert result['p_score'] == 1.0

    def test_no_match(self):
        result = metalmom.tempo_evaluate(120.0, 90.0)
        assert result['p_score'] == 0.0

    def test_zero_tempo(self):
        result = metalmom.tempo_evaluate(0.0, 120.0)
        assert result['p_score'] == 0.0

    def test_custom_tolerance(self):
        # ~8.3% off, should fail at 8% but pass at 10%
        result1 = metalmom.tempo_evaluate(120.0, 130.0, tolerance=0.08)
        assert result1['p_score'] == 0.0

        result2 = metalmom.tempo_evaluate(120.0, 130.0, tolerance=0.10)
        assert result2['p_score'] == 1.0


# ── Chord evaluation ──────────────────────────────────────────────

class TestChordAccuracy:
    def test_perfect(self):
        acc = metalmom.chord_accuracy([0, 1, 2, 3], [0, 1, 2, 3])
        assert abs(acc - 1.0) < 1e-5

    def test_half_correct(self):
        acc = metalmom.chord_accuracy([0, 1, 2, 3], [0, 1, 5, 6])
        assert abs(acc - 0.5) < 1e-5

    def test_no_correct(self):
        acc = metalmom.chord_accuracy([0, 1, 2], [3, 4, 5])
        assert abs(acc) < 1e-5

    def test_empty(self):
        acc = metalmom.chord_accuracy([], [])
        assert acc == 0.0

    def test_single_frame(self):
        assert abs(metalmom.chord_accuracy([5], [5]) - 1.0) < 1e-5
        assert abs(metalmom.chord_accuracy([5], [6])) < 1e-5

    def test_numpy_arrays(self):
        ref = np.array([0, 1, 2, 3], dtype=np.int32)
        est = np.array([0, 1, 2, 3], dtype=np.int32)
        acc = metalmom.chord_accuracy(ref, est)
        assert abs(acc - 1.0) < 1e-5
