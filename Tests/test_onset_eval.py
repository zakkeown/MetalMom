import numpy as np
import metalmom


def test_perfect_match():
    result = metalmom.onset_evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(result['precision'] - 1.0) < 1e-5
    assert abs(result['recall'] - 1.0) < 1e-5
    assert abs(result['f_measure'] - 1.0) < 1e-5


def test_within_tolerance():
    result = metalmom.onset_evaluate([1.0, 2.0, 3.0], [1.02, 2.03, 2.98])
    assert abs(result['f_measure'] - 1.0) < 1e-5


def test_missed_onsets():
    result = metalmom.onset_evaluate([1.0, 2.0, 3.0], [1.0])
    assert abs(result['precision'] - 1.0) < 1e-5
    assert abs(result['recall'] - 1.0/3.0) < 1e-5


def test_false_positives():
    result = metalmom.onset_evaluate([1.0], [1.0, 2.0, 3.0])
    assert abs(result['precision'] - 1.0/3.0) < 1e-5
    assert abs(result['recall'] - 1.0) < 1e-5


def test_no_match():
    result = metalmom.onset_evaluate([1.0, 2.0], [5.0, 6.0])
    assert result['f_measure'] == 0


def test_custom_window():
    result = metalmom.onset_evaluate([1.0], [1.08], window=0.1)
    assert abs(result['f_measure'] - 1.0) < 1e-5
