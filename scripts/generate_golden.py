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

# ── Mel filterbank golden files ──
# Default params: sr=22050, n_fft=2048, n_mels=128, htk=False (Slaney)
mel_fb_128_2048 = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)
np.save(os.path.join(GOLDEN_DIR, "mel_filterbank_128_2048.npy"), mel_fb_128_2048)

# Smaller variant: sr=22050, n_fft=1024, n_mels=40
mel_fb_40_1024 = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=40)
np.save(os.path.join(GOLDEN_DIR, "mel_filterbank_40_1024.npy"), mel_fb_40_1024)

# With custom fmin/fmax: sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000
mel_fb_custom = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000)
np.save(os.path.join(GOLDEN_DIR, "mel_filterbank_64_2048_300_8000.npy"), mel_fb_custom)

# ── Mel spectrogram golden files ──
# Default params: sr=22050, n_fft=2048, hop_length=512, n_mels=128, power=2.0
mel_spec_default = librosa.feature.melspectrogram(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "mel_spectrogram_440hz_default.npy"), mel_spec_default)

# With custom params: n_mels=40, n_fft=1024
mel_spec_40_1024 = librosa.feature.melspectrogram(
    y=signal_440, sr=22050, n_fft=1024, n_mels=40
)
np.save(os.path.join(GOLDEN_DIR, "mel_spectrogram_440hz_40_1024.npy"), mel_spec_40_1024)

# With custom fmin/fmax
mel_spec_custom = librosa.feature.melspectrogram(
    y=signal_440, sr=22050, n_fft=2048, n_mels=64, fmin=300, fmax=8000
)
np.save(os.path.join(GOLDEN_DIR, "mel_spectrogram_440hz_64_fmin300_fmax8000.npy"), mel_spec_custom)

# ── MFCC golden files ──
# Default params: n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128
mfcc_default = librosa.feature.mfcc(y=signal_440, sr=22050, n_mfcc=20)
np.save(os.path.join(GOLDEN_DIR, "mfcc_440hz_default.npy"), mfcc_default)

# Custom params: n_mfcc=13, n_fft=1024, n_mels=40
mfcc_13_1024 = librosa.feature.mfcc(y=signal_440, sr=22050, n_mfcc=13, n_fft=1024, n_mels=40)
np.save(os.path.join(GOLDEN_DIR, "mfcc_440hz_13_1024.npy"), mfcc_13_1024)

# Custom fmin/fmax: n_mfcc=20, n_fft=2048, n_mels=64, fmin=300, fmax=8000
mfcc_custom = librosa.feature.mfcc(
    y=signal_440, sr=22050, n_mfcc=20, n_fft=2048, n_mels=64, fmin=300, fmax=8000
)
np.save(os.path.join(GOLDEN_DIR, "mfcc_440hz_20_64_fmin300_fmax8000.npy"), mfcc_custom)

# ── Chroma STFT golden files ──
# No normalization: n_chroma=12, n_fft=2048, hop_length=512, norm=None
chroma_no_norm = librosa.feature.chroma_stft(y=signal_440, sr=22050, norm=None)
np.save(os.path.join(GOLDEN_DIR, "chroma_stft_440hz_no_norm.npy"), chroma_no_norm)

# librosa default (norm=inf): n_chroma=12, n_fft=2048, hop_length=512
chroma_default = librosa.feature.chroma_stft(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "chroma_stft_440hz_default.npy"), chroma_default)

# Custom params: n_fft=1024, norm=None
chroma_1024 = librosa.feature.chroma_stft(y=signal_440, sr=22050, n_fft=1024, norm=None)
np.save(os.path.join(GOLDEN_DIR, "chroma_stft_440hz_1024.npy"), chroma_1024)

# Save shape info for verification
print(f"Signal shape: {signal_440.shape}")
print(f"STFT shape: {stft_magnitude.shape}")
print(f"STFT dtype: {stft_magnitude.dtype}")
print(f"STFT max: {stft_magnitude.max():.6f}")
print(f"amplitude_to_db shape: {stft_amplitude_db.shape}, range: [{stft_amplitude_db.min():.2f}, {stft_amplitude_db.max():.2f}]")
print(f"power_to_db shape: {stft_power_db.shape}, range: [{stft_power_db.min():.2f}, {stft_power_db.max():.2f}]")
print(f"mel_fb_128_2048 shape: {mel_fb_128_2048.shape}, dtype: {mel_fb_128_2048.dtype}, range: [{mel_fb_128_2048.min():.6f}, {mel_fb_128_2048.max():.6f}]")
print(f"mel_fb_40_1024 shape: {mel_fb_40_1024.shape}")
print(f"mel_fb_custom shape: {mel_fb_custom.shape}")
print(f"mel_spec_default shape: {mel_spec_default.shape}, dtype: {mel_spec_default.dtype}, range: [{mel_spec_default.min():.6f}, {mel_spec_default.max():.6f}]")
print(f"mel_spec_40_1024 shape: {mel_spec_40_1024.shape}")
print(f"mel_spec_custom shape: {mel_spec_custom.shape}")
print(f"mfcc_default shape: {mfcc_default.shape}, dtype: {mfcc_default.dtype}, range: [{mfcc_default.min():.4f}, {mfcc_default.max():.4f}]")
print(f"mfcc_13_1024 shape: {mfcc_13_1024.shape}")
print(f"mfcc_custom shape: {mfcc_custom.shape}")
print(f"Golden files saved to {GOLDEN_DIR}")
