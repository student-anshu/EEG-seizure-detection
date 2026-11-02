import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def bandpass_and_notch(
    data: np.ndarray,
    sfreq: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    notch_hz: float = 60.0,
) -> np.ndarray:
    """
    Apply zero-phase bandpass and optional mains-frequency notch filtering.

    Parameters
    ----------
    data:
        EEG array with shape (channels, samples).
    sfreq:
        Sampling frequency in Hz.
    low_hz, high_hz:
        Requested bandpass cutoffs in Hz. The high cutoff is capped below
        Nyquist when necessary.
    notch_hz:
        Notch frequency in Hz. Set to 0 to disable the notch filter. A notch
        at or above Nyquist is skipped.

    Returns
    -------
    np.ndarray
        Filtered EEG data with shape (channels, samples), dtype float32.

    Notes
    -----
    Forward-backward filtering avoids phase delay. It is an offline operation
    and is not suitable as-is for causal real-time streaming.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("EEG data must be a non-empty 2D array (channels, samples).")
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data contains NaN or infinite values.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a finite positive number.")
    if not np.isfinite(low_hz) or not np.isfinite(high_hz):
        raise ValueError("Bandpass cutoff frequencies must be finite.")
    if not np.isfinite(notch_hz) or notch_hz < 0:
        raise ValueError("Notch frequency must be finite and non-negative.")

    nyquist = sfreq / 2.0
    high = min(high_hz, nyquist * 0.95)
    if low_hz <= 0 or high_hz <= 0 or low_hz >= high:
        raise ValueError(
            "Invalid bandpass frequencies for this sampling rate: "
            f"low={low_hz:g} Hz, effective high={high:g} Hz, "
            f"Nyquist={nyquist:g} Hz."
        )

    bandpass_sos = butter(
        4, [low_hz, high], btype="bandpass", fs=sfreq, output="sos"
    )
    try:
        filtered = sosfiltfilt(bandpass_sos, data, axis=-1)
    except ValueError as exc:
        raise ValueError(
            "EEG recording is too short for zero-phase bandpass filtering. "
            "Provide more samples or handle the short recording explicitly."
        ) from exc

    if 0 < notch_hz < nyquist:
        b, a = iirnotch(notch_hz, Q=30, fs=sfreq)
        notch_sos = tf2sos(b, a)
        try:
            filtered = sosfiltfilt(notch_sos, filtered, axis=-1)
        except ValueError as exc:
            raise ValueError(
                "EEG recording is too short for zero-phase notch filtering."
            ) from exc

    return filtered.astype(np.float32)


def robust_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize each channel using its median and interquartile range.

    Input must have shape (channels, samples). Statistics are calculated over
    the complete supplied recording, so this is an offline, per-recording
    transform rather than a causal streaming transform.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("EEG data must be a non-empty 2D array (channels, samples).")
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data contains NaN or infinite values.")

    median = np.median(data, axis=-1, keepdims=True)
    q75 = np.percentile(data, 75, axis=-1, keepdims=True)
    q25 = np.percentile(data, 25, axis=-1, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-6)
    normalized = (data - median) / scale
    return normalized.astype(np.float32)
