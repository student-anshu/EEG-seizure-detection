"""Extended robustness checks for EEG frequency feature extraction.

Run from the EEG_Seizure_Detection project root:
    python .\\scripts\\validate_frequency_robustness.py

This script is diagnostic. It does not modify the feature extractor.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from feature_selection.feature_extraction import (
    FEATURE_NAMES,
    FREQUENCY_BANDS,
    extract_features,
)

SFREQ = 256
DURATION_SECONDS = 10
SAMPLES = SFREQ * DURATION_SECONDS

BAND_INDEX = {
    band: FEATURE_NAMES.index(f"{band}_power")
    for band in FREQUENCY_BANDS
}


def make_tone(frequency: float, amplitude: float = 1.0) -> np.ndarray:
    """Generate a pure sine wave at the configured sampling frequency."""
    time = np.arange(SAMPLES) / SFREQ
    return amplitude * np.sin(2 * np.pi * frequency * time)


def extract_band_powers(signal: np.ndarray) -> dict[str, float]:
    """Run the production feature extractor on one synthetic signal."""
    windows = signal[np.newaxis, np.newaxis, :]
    features = extract_features(windows, sfreq=SFREQ)
    return {
        band: float(features[0, index])
        for band, index in BAND_INDEX.items()
    }


def print_band_powers(powers: dict[str, float]) -> None:
    for band, power in powers.items():
        print(f"  {band:6s}: {power:.8e}")


def test_boundary_frequencies() -> None:
    print("\nTEST 1: FREQUENCIES NEAR BAND BOUNDARIES")
    print("=" * 65)

    frequencies = (
        0.75, 3.5,
        4.5, 7.5,
        8.5, 12.5,
        13.5, 29.5,
        30.5, 39.5,
    )

    for frequency in frequencies:
        powers = extract_band_powers(make_tone(frequency))
        dominant = max(powers, key=powers.get)
        print(f"\nInput: {frequency:.2f} Hz | Dominant band: {dominant}")
        print_band_powers(powers)

    print("\nNOTE: Boundary results are diagnostic, not strict PASS/FAIL.")
    print("Finite frequency resolution and spectral leakage can affect them.")


def test_mixed_frequency_signal() -> None:
    print("\n\nTEST 2: MIXED FREQUENCY SIGNAL")
    print("=" * 65)

    # Two components in distinct bands: 2 Hz (delta) and 10 Hz (alpha).
    signal = make_tone(2.0, 1.0) + make_tone(10.0, 0.7)
    powers = extract_band_powers(signal)
    print_band_powers(powers)

    passed = (
        powers["delta"] > 0.0
        and powers["alpha"] > 0.0
        and powers["delta"] > powers["theta"]
        and powers["alpha"] > powers["beta"]
    )
    print(f"\nMixed-signal check: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("Expected both delta and alpha power to be represented.")


def test_noisy_signal() -> None:
    print("\n\nTEST 3: SIGNAL WITH GAUSSIAN NOISE")
    print("=" * 65)

    rng = np.random.default_rng(42)
    clean_signal = make_tone(10.0)
    noisy_signal = clean_signal + rng.normal(0.0, 0.3, SAMPLES)

    powers = extract_band_powers(noisy_signal)
    print_band_powers(powers)

    finite = all(np.isfinite(value) for value in powers.values())
    dominant = max(powers, key=powers.get)
    passed = finite and dominant == "alpha"

    print(f"\nNoisy alpha signal check: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError("Noisy 10 Hz signal did not retain alpha dominance.")


def integrate_band_with_edges(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Integrate PSD while interpolating PSD values at both band edges."""
    inside = (frequencies > low) & (frequencies < high)
    selected_frequencies = np.concatenate(
        ([low], frequencies[inside], [high])
    )
    selected_psd = np.interp(selected_frequencies, frequencies, psd)
    return float(trapezoid(selected_psd, selected_frequencies))


def test_welch_resolution_and_band_edges() -> None:
    print("\n\nTEST 4: WELCH RESOLUTION AND BAND-EDGE INTEGRATION")
    print("=" * 65)

    signal = make_tone(2.0) + make_tone(10.0, 0.5)

    for segment_seconds in (1, 2):
        frequencies, psd = welch(
            signal,
            fs=SFREQ,
            nperseg=segment_seconds * SFREQ,
            detrend="constant",
            scaling="density",
        )

        resolution = SFREQ / (segment_seconds * SFREQ)
        print(f"\nWelch segment: {segment_seconds} second(s)")
        print(f"Frequency resolution: {resolution:.2f} Hz")

        for band, (low, high) in FREQUENCY_BANDS.items():
            mask = (frequencies >= low) & (frequencies < high)

            existing_power = (
                float(trapezoid(psd[mask], frequencies[mask]))
                if np.count_nonzero(mask) >= 2
                else 0.0
            )
            edge_corrected_power = integrate_band_with_edges(
                frequencies, psd, low, high
            )

            print(
                f"{band:6s} | existing={existing_power:.8e} "
                f"| edge-corrected={edge_corrected_power:.8e}"
            )


def main() -> None:
    test_boundary_frequencies()
    test_mixed_frequency_signal()
    test_noisy_signal()
    test_welch_resolution_and_band_edges()

    print("\n\nEXTENDED DIAGNOSTIC RUN COMPLETED.")
    print("Review results before changing feature_extraction.py.")


if __name__ == "__main__":
    main()
