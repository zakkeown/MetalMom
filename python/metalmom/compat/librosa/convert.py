"""librosa.convert compatibility shim."""

from metalmom.convert import (
    hz_to_midi, midi_to_hz, hz_to_note, note_to_hz, midi_to_note, note_to_midi,
    times_to_frames, frames_to_time, times_to_samples, samples_to_time,
    frames_to_samples, samples_to_frames,
    fft_frequencies, mel_frequencies,
)

# Aliases: librosa uses time_to_frames / time_to_samples (singular)
# alongside times_to_frames / times_to_samples (plural)
time_to_frames = times_to_frames
time_to_samples = times_to_samples
