"""madmom.evaluation.onsets compatibility shim.

Provides OnsetEvaluation backed by MetalMom's onset_evaluate.
"""

import numpy as np
from metalmom.evaluate import onset_evaluate


class OnsetEvaluation:
    """madmom-compatible onset evaluation.

    Computes onset detection metrics on construction and exposes results
    as properties matching madmom's OnsetEvaluation API.

    Parameters
    ----------
    detections : array-like
        Detected onset times in seconds.
    annotations : array-like
        Ground-truth onset times in seconds.
    window : float
        Tolerance window in seconds. Default: 0.025 (madmom convention).
    **kwargs
        Ignored; accepted for madmom API compatibility.
    """

    def __init__(self, detections, annotations, window=0.025, **kwargs):
        detections = np.asarray(detections, dtype=np.float64).ravel()
        annotations = np.asarray(annotations, dtype=np.float64).ravel()

        # MetalMom's onset_evaluate expects (reference, estimated)
        result = onset_evaluate(annotations, detections, window=window)

        self._fmeasure = result.get('f_measure', 0.0)
        self._precision = result.get('precision', 0.0)
        self._recall = result.get('recall', 0.0)

        # Derive counts from precision/recall and array lengths
        # TP + FP = len(detections), TP + FN = len(annotations)
        # precision = TP / (TP + FP), recall = TP / (TP + FN)
        n_det = len(detections)
        n_ann = len(annotations)

        if n_det > 0 and self._precision > 0:
            self._num_tp = int(round(self._precision * n_det))
        elif n_ann == 0 and n_det == 0:
            self._num_tp = 0
        else:
            self._num_tp = 0

        self._num_fp = n_det - self._num_tp
        self._num_fn = n_ann - self._num_tp

    @property
    def fmeasure(self):
        """F-measure (F1 score)."""
        return self._fmeasure

    @property
    def precision(self):
        """Precision."""
        return self._precision

    @property
    def recall(self):
        """Recall."""
        return self._recall

    @property
    def num_tp(self):
        """Number of true positives."""
        return self._num_tp

    @property
    def num_fp(self):
        """Number of false positives."""
        return self._num_fp

    @property
    def num_fn(self):
        """Number of false negatives."""
        return self._num_fn

    def __repr__(self):
        return (
            f"OnsetEvaluation(fmeasure={self._fmeasure:.4f}, "
            f"precision={self._precision:.4f}, "
            f"recall={self._recall:.4f})"
        )
