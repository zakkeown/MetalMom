"""librosa compatibility shim.

Drop-in replacement for common librosa functions backed by MetalMom.

Usage::

    from metalmom.compat import librosa
    S = librosa.stft(y)
    y_hat = librosa.istft(S)
    mel = librosa.feature.melspectrogram(y=y, sr=22050)
"""

from .core import load, resample, stft, istft
from . import feature
