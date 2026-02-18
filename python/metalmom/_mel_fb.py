"""Pure-Python mel filterbank (mirrors FilterBank.swift / Units.swift exactly).

Used when the user provides a pre-computed spectrogram ``S`` to
``melspectrogram()`` so we can apply the mel filterbank without
going through the C bridge.
"""

import math
import numpy as np


def _hz_to_mel(hz):
    """Slaney formula: matches Units.hzToMel in Swift."""
    f_sp = 200.0 / 3.0
    mel = hz / f_sp

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15.0
    logstep = math.log(6.4) / 27.0

    if hz >= min_log_hz:
        mel = min_log_mel + math.log(hz / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel):
    """Inverse Slaney formula: matches Units.melToHz in Swift."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    if mel < min_log_mel:
        return mel * f_sp
    else:
        return min_log_hz * math.exp((mel - min_log_mel) * logstep)


def _mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """Reimplement FilterBank.mel() in Python (mirrors Swift implementation).

    Returns shape ``(n_mels, n_fft // 2 + 1)`` float32.
    """
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = [
        _mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1))
        for i in range(n_mels + 2)
    ]

    fft_freqs = [k * sr / n_fft for k in range(n_freqs)]

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(n_mels):
        f_left = mel_points[m]
        f_center = mel_points[m + 1]
        f_right = mel_points[m + 2]

        for k in range(n_freqs):
            freq = fft_freqs[k]
            if freq >= f_left and freq <= f_center and f_center != f_left:
                weights[m, k] = (freq - f_left) / (f_center - f_left)
            elif freq > f_center and freq <= f_right and f_right != f_center:
                weights[m, k] = (f_right - freq) / (f_right - f_center)

        # Slaney normalisation
        enorm = 2.0 / (mel_points[m + 2] - mel_points[m])
        weights[m, :] *= enorm

    return weights
