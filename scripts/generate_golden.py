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

# ── Spectral Descriptor golden files ──
# Spectral centroid: default params
sc_centroid = librosa.feature.spectral_centroid(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "spectral_centroid_440hz_default.npy"), sc_centroid)

# Spectral bandwidth: default params
sc_bandwidth = librosa.feature.spectral_bandwidth(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "spectral_bandwidth_440hz_default.npy"), sc_bandwidth)

# Spectral contrast: default params
sc_contrast = librosa.feature.spectral_contrast(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "spectral_contrast_440hz_default.npy"), sc_contrast)

# Spectral rolloff: default params
sc_rolloff = librosa.feature.spectral_rolloff(y=signal_440, sr=22050)
np.save(os.path.join(GOLDEN_DIR, "spectral_rolloff_440hz_default.npy"), sc_rolloff)

# Spectral flatness: default params
sc_flatness = librosa.feature.spectral_flatness(y=signal_440)
np.save(os.path.join(GOLDEN_DIR, "spectral_flatness_440hz_default.npy"), sc_flatness)

# ── RMS Energy golden files ──
# Default params: frame_length=2048, hop_length=512, center=True
rms_default = librosa.feature.rms(y=signal_440)
np.save(os.path.join(GOLDEN_DIR, "rms_440hz_default.npy"), rms_default)

# ── Zero-Crossing Rate golden files ──
# Default params: frame_length=2048, hop_length=512, center=True
zcr_default = librosa.feature.zero_crossing_rate(y=signal_440)
np.save(os.path.join(GOLDEN_DIR, "zcr_440hz_default.npy"), zcr_default)

# ── Tonnetz golden files ──
# Generate with chroma_stft for fair comparison (we use chroma_stft, not chroma_cqt)
chroma_for_tonnetz = librosa.feature.chroma_stft(y=signal_440, sr=22050)
tonnetz_default = librosa.feature.tonnetz(chroma=chroma_for_tonnetz)
np.save(os.path.join(GOLDEN_DIR, "tonnetz_440hz_default.npy"), tonnetz_default)

# ── Delta features golden files ──
# First, compute MFCC for delta input
mfcc_for_delta = librosa.feature.mfcc(y=signal_440, sr=22050, n_mfcc=20)
np.save(os.path.join(GOLDEN_DIR, "mfcc.npy"), mfcc_for_delta)

# Delta (first derivative) of MFCC
delta_default = librosa.feature.delta(mfcc_for_delta)
np.save(os.path.join(GOLDEN_DIR, "delta.npy"), delta_default)

# Delta-delta (second derivative) of MFCC
delta_delta = librosa.feature.delta(mfcc_for_delta, order=2)
np.save(os.path.join(GOLDEN_DIR, "delta_delta.npy"), delta_delta)

# Stack memory of MFCC
stack_mem = librosa.feature.stack_memory(mfcc_for_delta, n_steps=3)
np.save(os.path.join(GOLDEN_DIR, "stack_memory.npy"), stack_mem)

# ── Poly features golden files ──
# Compute from magnitude STFT
S_for_poly = np.abs(librosa.stft(signal_440, n_fft=2048, hop_length=512))
poly_default = librosa.feature.poly_features(S=S_for_poly, order=1)
np.save(os.path.join(GOLDEN_DIR, "poly_features_440hz_default.npy"), poly_default)

poly_order2 = librosa.feature.poly_features(S=S_for_poly, order=2)
np.save(os.path.join(GOLDEN_DIR, "poly_features_440hz_order2.npy"), poly_order2)

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
print(f"spectral_centroid shape: {sc_centroid.shape}, range: [{sc_centroid.min():.4f}, {sc_centroid.max():.4f}]")
print(f"spectral_bandwidth shape: {sc_bandwidth.shape}, range: [{sc_bandwidth.min():.4f}, {sc_bandwidth.max():.4f}]")
print(f"spectral_contrast shape: {sc_contrast.shape}, range: [{sc_contrast.min():.4f}, {sc_contrast.max():.4f}]")
print(f"spectral_rolloff shape: {sc_rolloff.shape}, range: [{sc_rolloff.min():.4f}, {sc_rolloff.max():.4f}]")
print(f"spectral_flatness shape: {sc_flatness.shape}, range: [{sc_flatness.min():.6f}, {sc_flatness.max():.6f}]")
print(f"rms_default shape: {rms_default.shape}, range: [{rms_default.min():.6f}, {rms_default.max():.6f}]")
print(f"zcr_default shape: {zcr_default.shape}, range: [{zcr_default.min():.6f}, {zcr_default.max():.6f}]")
print(f"tonnetz_default shape: {tonnetz_default.shape}, range: [{tonnetz_default.min():.6f}, {tonnetz_default.max():.6f}]")
print(f"mfcc_for_delta shape: {mfcc_for_delta.shape}, range: [{mfcc_for_delta.min():.4f}, {mfcc_for_delta.max():.4f}]")
print(f"delta_default shape: {delta_default.shape}, range: [{delta_default.min():.6f}, {delta_default.max():.6f}]")
print(f"delta_delta shape: {delta_delta.shape}, range: [{delta_delta.min():.6f}, {delta_delta.max():.6f}]")
print(f"stack_mem shape: {stack_mem.shape}")
print(f"poly_default shape: {poly_default.shape}, range: [{poly_default.min():.6f}, {poly_default.max():.6f}]")
print(f"poly_order2 shape: {poly_order2.shape}, range: [{poly_order2.min():.6f}, {poly_order2.max():.6f}]")
print(f"Golden files saved to {GOLDEN_DIR}")
