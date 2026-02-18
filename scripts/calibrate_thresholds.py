#!/usr/bin/env python3
"""Run dispatch threshold calibration benchmarks.

Usage:
    python scripts/calibrate_thresholds.py

This runs the Swift threshold calibration tests and displays results.
The benchmarks compare CPU (Accelerate/vDSP) vs GPU (Metal) performance
for each operation type at various data sizes.  The crossover point where
GPU becomes faster indicates the optimal dispatch threshold.

Results can be used to update the hardcoded values in:
    Sources/MetalMomCore/Dispatch/ChipProfile.swift
"""

import subprocess
import sys
import os


def main():
    # Resolve project root (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("Running MetalMom dispatch threshold calibration...")
    print("=" * 60)

    result = subprocess.run(
        ["swift", "test", "--filter", "ThresholdCalibrationTests", "-v"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # Extract benchmark output (lines with | separators, headers, and profile info)
    output_lines = result.stdout.split("\n") + result.stderr.split("\n")
    for line in output_lines:
        stripped = line.strip()
        if any(
            marker in stripped
            for marker in [
                "===",
                "|",
                "---",
                "GPU Family",
                "threshold",
                "Estimated",
            ]
        ):
            print(stripped)

    if result.returncode != 0:
        print("\nCalibration failed!")
        if result.stderr:
            # Print last 500 chars of stderr for diagnostics
            print(result.stderr[-500:])
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Calibration complete.")
    print("\nTo update thresholds, edit:")
    print("  Sources/MetalMomCore/Dispatch/ChipProfile.swift")


if __name__ == "__main__":
    main()
