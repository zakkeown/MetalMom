"""madmom.evaluation.beats compatibility shim.

Provides BeatEvaluation backed by MetalMom's beat_evaluate.
"""

import numpy as np
from metalmom.evaluate import beat_evaluate


class BeatEvaluation:
    """madmom-compatible beat evaluation.

    Computes beat tracking metrics on construction and exposes results
    as properties matching madmom's BeatEvaluation API.

    Parameters
    ----------
    detections : array-like
        Detected beat times in seconds.
    annotations : array-like
        Ground-truth beat times in seconds.
    fmeasure_window : float
        Tolerance window for F-measure in seconds. Default: 0.07.
    **kwargs
        Ignored; accepted for madmom API compatibility.
    """

    def __init__(self, detections, annotations, fmeasure_window=0.07,
                 **kwargs):
        detections = np.asarray(detections, dtype=np.float64).ravel()
        annotations = np.asarray(annotations, dtype=np.float64).ravel()

        # MetalMom's beat_evaluate expects (reference, estimated)
        result = beat_evaluate(annotations, detections,
                               fmeasure_window=fmeasure_window)

        self._fmeasure = result.get('f_measure', 0.0)
        self._cemgil = result.get('cemgil', 0.0)
        self._p_score = result.get('p_score', 0.0)
        self._cmlc = result.get('cml_c', 0.0)
        self._cmlt = result.get('cml_t', 0.0)
        self._amlc = result.get('aml_c', 0.0)
        self._amlt = result.get('aml_t', 0.0)

    @property
    def fmeasure(self):
        """F-measure (F1 score)."""
        return self._fmeasure

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        return self._cemgil

    @property
    def p_score(self):
        """P-score (information gain)."""
        return self._p_score

    @property
    def cmlc(self):
        """Correct metrical level, continuity required (CMLc)."""
        return self._cmlc

    @property
    def cmlt(self):
        """Correct metrical level, continuity not required (CMLt)."""
        return self._cmlt

    @property
    def amlc(self):
        """Allowed metrical levels, continuity required (AMLc)."""
        return self._amlc

    @property
    def amlt(self):
        """Allowed metrical levels, continuity not required (AMLt)."""
        return self._amlt

    def __repr__(self):
        return (
            f"BeatEvaluation(fmeasure={self._fmeasure:.4f}, "
            f"cemgil={self._cemgil:.4f}, "
            f"cmlc={self._cmlc:.4f})"
        )
