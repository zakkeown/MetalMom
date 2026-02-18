#!/usr/bin/env python3
"""Generate semitone bandpass filter coefficients for validation.

This script uses scipy.signal to design 2nd-order bandpass biquad filters
for all 96 semitones (C1=MIDI 24 through B8=MIDI 119) at sample rates
22050, 44100, and 48000 Hz.

The output is JSON containing the second-order sections (SOS) for each
semitone/sample-rate combination. This is used for reference and validation
against the runtime Swift implementation.

Usage:
    python scripts/generate_filter_coefficients.py [--output coefficients.json]
"""

import argparse
import json
import sys

import numpy as np


def midi_to_hz(midi):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def design_bandpass_biquad(center_freq, sr, Q=None):
    """Design a 2nd-order bandpass biquad using the Audio EQ Cookbook.

    Parameters
    ----------
    center_freq : float
        Center frequency in Hz.
    sr : int
        Sample rate in Hz.
    Q : float or None
        Quality factor. If None, uses semitone bandwidth Q.

    Returns
    -------
    dict
        Biquad coefficients: b0, b1, b2, a1, a2 (normalized by a0).
    """
    if Q is None:
        # Q for one semitone bandwidth: Q = 1 / (2 * sinh(ln(2)/2 * BW))
        # For BW = 1 semitone: Q ~ 1 / (2 * sinh(ln(2)/24))
        # Simpler: Q = f0 / bandwidth, bandwidth = f0 * (2^(1/12) - 1) * 2
        # But more precisely: for constant-Q one semitone:
        bw_ratio = 2.0 ** (1.0 / 12.0) - 2.0 ** (-1.0 / 12.0)
        Q = 1.0 / bw_ratio  # ~ 8.65

    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * Q)

    # Bandpass (constant skirt gain, peak gain = Q):
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    # Normalize
    return {
        "b0": b0 / a0,
        "b1": b1 / a0,
        "b2": b2 / a0,
        "a1": a1 / a0,
        "a2": a2 / a0,
    }


def generate_all_coefficients(sample_rates=None, midi_low=24, midi_high=119):
    """Generate bandpass biquad coefficients for all semitones and sample rates.

    Parameters
    ----------
    sample_rates : list of int
        Sample rates to generate for. Default: [22050, 44100, 48000].
    midi_low : int
        Lowest MIDI note (default 24 = C1).
    midi_high : int
        Highest MIDI note (default 119 = B8).

    Returns
    -------
    dict
        Nested dict: {sr: {midi: {b0, b1, b2, a1, a2}}}.
    """
    if sample_rates is None:
        sample_rates = [22050, 44100, 48000]

    result = {}
    for sr in sample_rates:
        nyquist = sr / 2.0
        sr_coeffs = {}
        for midi in range(midi_low, midi_high + 1):
            freq = midi_to_hz(midi)
            if freq >= nyquist * 0.95:
                # Skip frequencies too close to Nyquist
                continue
            coeffs = design_bandpass_biquad(freq, sr)
            sr_coeffs[str(midi)] = coeffs
        result[str(sr)] = sr_coeffs

    return result


def validate_coefficients(coefficients):
    """Basic validation that coefficients are finite and reasonable."""
    for sr, sr_coeffs in coefficients.items():
        for midi, coeffs in sr_coeffs.items():
            for key, val in coeffs.items():
                if not np.isfinite(val):
                    print(f"WARNING: Non-finite {key}={val} at sr={sr}, midi={midi}")
                    return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate semitone bandpass filter coefficients"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--midi-low", type=int, default=24,
        help="Lowest MIDI note (default 24 = C1)",
    )
    parser.add_argument(
        "--midi-high", type=int, default=119,
        help="Highest MIDI note (default 119 = B8)",
    )
    args = parser.parse_args()

    coefficients = generate_all_coefficients(
        midi_low=args.midi_low,
        midi_high=args.midi_high,
    )

    if not validate_coefficients(coefficients):
        print("ERROR: Coefficient validation failed", file=sys.stderr)
        sys.exit(1)

    # Summary
    for sr, sr_coeffs in coefficients.items():
        print(f"sr={sr}: {len(sr_coeffs)} semitones", file=sys.stderr)

    output = json.dumps(coefficients, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote coefficients to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
