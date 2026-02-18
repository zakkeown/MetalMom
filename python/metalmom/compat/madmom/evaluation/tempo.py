"""madmom.evaluation.tempo compatibility shim.

Provides TempoEvaluation backed by MetalMom's tempo_evaluate.
"""

import numpy as np
from metalmom.evaluate import tempo_evaluate


class TempoEvaluation:
    """madmom-compatible tempo evaluation.

    Computes tempo estimation metrics on construction and exposes results
    as properties matching madmom's TempoEvaluation API.

    In madmom, detections and annotations can be scalars (single BPM)
    or arrays. This shim extracts the primary tempo from each and
    evaluates at two tolerance levels: strict (acc1) and allowing
    double/half tempo (acc2).

    Parameters
    ----------
    detections : float or array-like
        Estimated tempo in BPM. If array, the first element is used.
    annotations : float or array-like
        Reference tempo in BPM. If array, the first element is used.
    tolerance : float
        Relative tolerance. Default: 0.08 (8%).
    **kwargs
        Ignored; accepted for madmom API compatibility.
    """

    def __init__(self, detections, annotations, tolerance=0.08, **kwargs):
        # Extract primary tempo from scalar or array
        det_bpm = self._extract_tempo(detections)
        ann_bpm = self._extract_tempo(annotations)

        # acc1: strict match -- no octave errors allowed.
        # MetalMom's tempo_evaluate already allows double/half internally,
        # so we compute acc1 directly as a relative tolerance check.
        if ann_bpm > 0 and det_bpm > 0:
            ratio = det_bpm / ann_bpm
            self._acc1 = 1.0 if abs(ratio - 1.0) <= tolerance else 0.0
        else:
            self._acc1 = 0.0

        # acc2: allows double/half tempo (octave errors).
        # MetalMom's tempo_evaluate p_score already handles this.
        result = tempo_evaluate(ann_bpm, det_bpm, tolerance=tolerance)
        self._acc2 = result.get('p_score', 0.0)

    @staticmethod
    def _extract_tempo(value):
        """Extract primary tempo BPM from scalar or array."""
        value = np.asarray(value, dtype=np.float64).ravel()
        if len(value) == 0:
            return 0.0
        return float(value[0])

    @property
    def acc1(self):
        """Accuracy at tolerance level 1 (strict, no octave errors)."""
        return self._acc1

    @property
    def acc2(self):
        """Accuracy at tolerance level 2 (allowing double/half tempo)."""
        return self._acc2

    def __repr__(self):
        return (
            f"TempoEvaluation(acc1={self._acc1:.4f}, "
            f"acc2={self._acc2:.4f})"
        )
