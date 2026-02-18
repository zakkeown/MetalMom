"""librosa compatibility shim.

Drop-in replacement for common librosa functions backed by MetalMom.

Usage::

    from metalmom.compat import librosa
    S = librosa.stft(y)
    y_hat = librosa.istft(S)
"""

from .core import stft, istft
