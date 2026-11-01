import numpy as np

def make_windows(data: np.ndarray, sfreq: float,
                 seizure_intervals: list[tuple[float, float]],
                 window_seconds: int = 10, stride_seconds: int = 5):
    """
    Convert (channels, samples) into windows shaped (windows, channels, samples).
    A window is seizure-positive if it overlaps any annotated seizure interval.
    """
    window_size = int(window_seconds * sfreq)
    stride = int(stride_seconds * sfreq)
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive.")
    windows, labels, starts = [], [], []

    total_samples = data.shape[1]
    for start_sample in range(0, total_samples - window_size + 1, stride):
        end_sample = start_sample + window_size
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        label = int(any(
            start_sec < seizure_end and end_sec > seizure_start
            for seizure_start, seizure_end in seizure_intervals
        ))
        windows.append(data[:, start_sample:end_sample])
        labels.append(label)
        starts.append(start_sec)

    if not windows:
        return (np.empty((0, data.shape[0], window_size), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    return (np.stack(windows).astype(np.float32),
            np.asarray(labels, dtype=np.int64),
            np.asarray(starts, dtype=np.float32))
