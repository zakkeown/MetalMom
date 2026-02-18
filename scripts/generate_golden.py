#!/usr/bin/env python3
"""Generate golden reference files from librosa for parity testing."""

import os
import numpy as np
import librosa

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Tests", "golden")
os.makedirs(GOLDEN_DIR, exist_ok=True)

# Test signal: 1 second of 440 Hz sine at 22050 Hz
sr = 22050
t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
signal_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Save test signal
np.save(os.path.join(GOLDEN_DIR, "signal_440hz_22050sr.npy"), signal_440)

# STFT with default params (n_fft=2048, hop_length=512, center=True)
stft_complex = librosa.stft(signal_440, n_fft=2048, hop_length=512, win_length=2048, center=True)
stft_magnitude = np.abs(stft_complex)
np.save(os.path.join(GOLDEN_DIR, "stft_440hz_default_magnitude.npy"), stft_magnitude)

# dB scaling golden files
# amplitude_to_db with ref=np.max (librosa default)
stft_amplitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
np.save(os.path.join(GOLDEN_DIR, "stft_440hz_amplitude_db.npy"), stft_amplitude_db)

# amplitude_to_db with ref=1.0
stft_amplitude_db_ref1 = librosa.amplitude_to_db(stft_magnitude, ref=1.0)
np.save(os.path.join(GOLDEN_DIR, "stft_440hz_amplitude_db_ref1.npy"), stft_amplitude_db_ref1)

# power_to_db with ref=np.max
stft_power = stft_magnitude ** 2
stft_power_db = librosa.power_to_db(stft_power, ref=np.max)
np.save(os.path.join(GOLDEN_DIR, "stft_440hz_power_db.npy"), stft_power_db)

# Save shape info for verification
print(f"Signal shape: {signal_440.shape}")
print(f"STFT shape: {stft_magnitude.shape}")
print(f"STFT dtype: {stft_magnitude.dtype}")
print(f"STFT max: {stft_magnitude.max():.6f}")
print(f"amplitude_to_db shape: {stft_amplitude_db.shape}, range: [{stft_amplitude_db.min():.2f}, {stft_amplitude_db.max():.2f}]")
print(f"power_to_db shape: {stft_power_db.shape}, range: [{stft_power_db.min():.2f}, {stft_power_db.max():.2f}]")
print(f"Golden files saved to {GOLDEN_DIR}")
