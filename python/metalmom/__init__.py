"""MetalMom: GPU-accelerated audio/music analysis on Apple Metal."""

__version__ = "0.1.0"

from .core import stft, istft, db_to_amplitude, db_to_power
from .feature import (
    amplitude_to_db, power_to_db, melspectrogram, mfcc, chroma_stft,
    spectral_centroid, spectral_bandwidth, spectral_contrast,
    spectral_rolloff, spectral_flatness,
    rms, zero_crossing_rate,
)
