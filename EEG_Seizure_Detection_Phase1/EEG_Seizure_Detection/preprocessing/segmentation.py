import numpy as np


def make_windows(
    data: np.ndarray,
    sfreq: float,
    seizure_intervals: list[tuple[float, float]],
    window_seconds: float = 10,
    stride_seconds: float = 5,
):
    """
    Convert (channels, samples) into windows shaped (windows, channels, samples).

    A window is seizure-positive when its time interval overlaps any seizure
    annotation. Windows are half-open intervals [start, end), so a window
    ending exactly at seizure onset is not positive.
    """
    data = np.asarray(data)

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("EEG data must be a non-empty 2D array (channels, samples).")
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data contains NaN or infinite values.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a finite positive number.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be a finite positive number.")
    if not np.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError("Window stride must be a finite positive number.")

    window_size = int(round(window_seconds * sfreq))
    stride = int(round(stride_seconds * sfreq))
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window and stride must each correspond to at least one sample.")

    validated_intervals = []
    for index, interval in enumerate(seizure_intervals):
        if len(interval) != 2:
            raise ValueError(f"Seizure interval #{index + 1} must contain start and end.")
        start, end = interval
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(
                f"Invalid seizure interval #{index + 1}: {interval}. "
                "Expected finite values with 0 <= start < end."
            )
        validated_intervals.append((float(start), float(end)))

    windows, labels, starts = [], [], []
    total_samples = data.shape[1]

    for start_sample in range(0, total_samples - window_size + 1, stride):
        end_sample = start_sample + window_size
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        label = int(any(
            start_sec < seizure_end and end_sec > seizure_start
            for seizure_start, seizure_end in validated_intervals
        ))
        windows.append(data[:, start_sample:end_sample])
        labels.append(label)
        starts.append(start_sec)

    if not windows:
        return (
            np.empty((0, data.shape[0], window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(starts, dtype=np.float32),
    )
