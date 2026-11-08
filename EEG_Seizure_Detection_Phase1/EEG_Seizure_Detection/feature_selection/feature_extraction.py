"""EEG window-level feature extraction for classical ML models.

Input:
    EEG windows with shape (n_windows, n_channels, n_samples).

Output:
    Feature matrix with shape (n_windows, n_channels * n_features).

This module performs deterministic feature extraction only.
It does not learn from labels or fit any statistical transformer.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


FEATURE_NAMES = (
    "mean",
    "std",
    "variance",
    "rms",
    "line_length",
    "skewness",
    "kurtosis",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "spectral_entropy",
)

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

WELCH_SEGMENT_SECONDS = 2.0


def _validate_windows(windows: np.ndarray, sfreq: float) -> np.ndarray:
    """Validate EEG window shape, values and sampling frequency."""

    values = np.asarray(windows, dtype=np.float64)

    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError(
            "Expected non-empty EEG windows shaped "
            "(n_windows, n_channels, n_samples)."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("EEG windows contain NaN or infinite values.")

    if not np.isscalar(sfreq) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be finite and positive.")

    if float(sfreq) / 2.0 < 40.0:
        raise ValueError(
            "Sampling frequency must be at least 80 Hz "
            "to calculate all configured frequency bands."
        )

    return values


def _integrate_band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Integrate PSD over a band, interpolating PSD at both band edges.

    The frequency vector from Welch is uniformly spaced and spans 0 Hz
    through Nyquist. Interpolating at low/high avoids dropping the partial
    frequency intervals between the band edges and the nearest PSD bins.
    """

    if low < frequencies[0] or high > frequencies[-1] or low >= high:
        raise ValueError("Frequency band lies outside the available PSD range.")

    interior = (frequencies > low) & (frequencies < high)
    band_frequencies = np.concatenate(
        ([low], frequencies[interior], [high])
    )
    band_psd = np.interp(band_frequencies, frequencies, psd)

    return max(float(trapezoid(band_psd, band_frequencies)), 0.0)


def _calculate_spectral_features(
    signal: np.ndarray,
    sfreq: float,
) -> tuple[list[float], float]:
    """Calculate five band powers and normalized spectral entropy.

    Welch uses up to two-second segments. For shorter input signals, the
    segment is limited to the available number of samples.
    """

    nperseg = min(len(signal), max(1, int(round(WELCH_SEGMENT_SECONDS * sfreq))))

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )

    band_powers = [
        _integrate_band_power(frequencies, psd, low, high)
        for low, high in FREQUENCY_BANDS.values()
    ]

    # Spectral entropy is calculated over the retained 0.5-40 Hz range.
    retained = (frequencies >= 0.5) & (frequencies <= 40.0)
    retained_psd = np.maximum(psd[retained], 0.0)
    total_power = float(np.sum(retained_psd))

    if total_power <= np.finfo(np.float64).eps:
        entropy = 0.0
    else:
        probabilities = retained_psd / total_power
        probabilities = probabilities[probabilities > 0.0]
        entropy = (
            float(
                -np.sum(probabilities * np.log(probabilities))
                / np.log(len(retained_psd))
            )
            if len(retained_psd) > 1
            else 0.0
        )

    return band_powers, entropy


def extract_features(
    windows: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Convert EEG windows into deterministic channel-wise feature vectors.

    Parameters
    ----------
    windows:
        Array shaped (n_windows, n_channels, n_samples).
    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Float32 array shaped
        (n_windows, n_channels * len(FEATURE_NAMES)).

    Feature ordering:
        For each channel, all FEATURE_NAMES are stored consecutively.
    """

    values = _validate_windows(windows, sfreq)
    n_windows, n_channels, _ = values.shape

    output = np.empty(
        (n_windows, n_channels, len(FEATURE_NAMES)),
        dtype=np.float64,
    )

    for window_idx in range(n_windows):
        for channel_idx in range(n_channels):
            signal = values[window_idx, channel_idx]

            mean = float(np.mean(signal))
            std = float(np.std(signal))
            variance = float(np.var(signal))
            rms = float(np.sqrt(np.mean(signal ** 2)))
            line_length = float(np.mean(np.abs(np.diff(signal))))

            # Constant or numerically near-constant signals have zero
            # skewness and excess kurtosis by convention.
            if std <= np.finfo(np.float64).eps * max(
                1.0, float(np.max(np.abs(signal)))
            ):
                skewness = 0.0
                excess_kurtosis = 0.0
            else:
                centered = signal - mean
                skewness = float(np.mean(centered ** 3) / (std ** 3))
                excess_kurtosis = float(
                    np.mean(centered ** 4) / (std ** 4) - 3.0
                )

            band_powers, spectral_entropy = _calculate_spectral_features(
                signal,
                float(sfreq),
            )

            output[window_idx, channel_idx] = [
                mean,
                std,
                variance,
                rms,
                line_length,
                skewness,
                excess_kurtosis,
                *band_powers,
                spectral_entropy,
            ]

    if not np.all(np.isfinite(output)):
        raise ValueError("Feature extraction generated NaN or infinite values.")

    return output.reshape(n_windows, -1).astype(np.float32)
