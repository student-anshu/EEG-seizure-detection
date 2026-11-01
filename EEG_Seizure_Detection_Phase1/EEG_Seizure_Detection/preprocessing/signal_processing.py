import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

def bandpass_and_notch(data: np.ndarray, sfreq: float,
                       low_hz: float = 0.5, high_hz: float = 40.0,
                       notch_hz: float = 60.0) -> np.ndarray:
    """
    Filter array shaped (channels, samples).
    Notch is applied only when its frequency is below Nyquist.
    """
    data = np.asarray(data, dtype=np.float32)
    nyquist = sfreq / 2.0
    high = min(high_hz, nyquist * 0.95)
    if low_hz <= 0 or low_hz >= high:
        raise ValueError("Invalid bandpass frequencies for this sampling rate.")

    sos = butter(4, [low_hz, high], btype="bandpass",
                 fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, data, axis=-1)

    if 0 < notch_hz < nyquist:
        b, a = iirnotch(notch_hz, Q=30, fs=sfreq)
        filtered = filtfilt(b, a, filtered, axis=-1)

    return filtered.astype(np.float32)

def robust_normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel median/IQR normalization; protects against zero IQR."""
    median = np.median(data, axis=-1, keepdims=True)
    q75 = np.percentile(data, 75, axis=-1, keepdims=True)
    q25 = np.percentile(data, 25, axis=-1, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-6)
    return ((data - median) / scale).astype(np.float32)
