
"""Validate EEG frequency-band features using synthetic sine waves."""

import sys
from pathlib import Path

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from feature_selection.feature_extraction import (
    FEATURE_NAMES,
    FREQUENCY_BANDS,
    extract_features,
)

SFREQ = 256
DURATION_SECONDS = 10
AMPLITUDE = 1.0

TEST_FREQUENCIES = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 35.0,
}


def main():
    samples = int(SFREQ * DURATION_SECONDS)
    time = np.arange(samples) / SFREQ

    band_feature_indices = {
        name: FEATURE_NAMES.index(f"{name}_power")
        for name in FREQUENCY_BANDS
    }

    all_passed = True

    print("\nEEG FREQUENCY FEATURE VALIDATION")
    print("=" * 65)

    for expected_band, frequency in TEST_FREQUENCIES.items():
        signal = AMPLITUDE * np.sin(2 * np.pi * frequency * time)

        # Shape: (windows, channels, samples)
        windows = signal[np.newaxis, np.newaxis, :]

        features = extract_features(windows, sfreq=SFREQ)

        powers = {
            band: float(features[0, index])
            for band, index in band_feature_indices.items()
        }

        dominant_band = max(powers, key=powers.get)
        passed = dominant_band == expected_band

        all_passed = all_passed and passed

        print(f"\nInput frequency: {frequency:.1f} Hz")
        print(f"Expected band:   {expected_band}")
        print(f"Dominant band:   {dominant_band}")

        for band, power in powers.items():
            print(f"  {band:6s}: {power:.8e}")

        print(f"Result: {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 65)

    if all_passed:
        print("ALL FREQUENCY-BAND VALIDATION TESTS PASSED.")
    else:
        print("FREQUENCY-BAND VALIDATION FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
