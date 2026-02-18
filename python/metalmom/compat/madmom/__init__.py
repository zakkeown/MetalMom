"""madmom compatibility shim.

Drop-in replacement for common madmom classes backed by MetalMom.

Usage::

    from metalmom.compat import madmom
    sig = madmom.audio.signal.Signal("audio.wav")
    frames = madmom.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
    spec = madmom.audio.spectrogram.Spectrogram(frames)
"""

from . import audio
from . import features
from . import evaluation
