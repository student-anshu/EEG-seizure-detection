import numpy as np
from sklearn.feature_selection import mutual_info_classif


def channel_features(windows: np.ndarray) -> np.ndarray:
    """
    Build per-channel features for channel ranking:
    mean absolute amplitude, standard deviation and line length.

    Input: (windows, channels, samples).
    Output: (windows, channels * 3).
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or any(size == 0 for size in windows.shape):
        raise ValueError(
            "Expected a non-empty 3D array (n_windows, n_channels, n_samples)."
        )
    if not np.all(np.isfinite(windows)):
        raise ValueError("Windows contain NaN or infinite values.")

    mean_abs = np.mean(np.abs(windows), axis=2)
    std = np.std(windows, axis=2)
    line_length = np.mean(np.abs(np.diff(windows, axis=2)), axis=2)
    return np.concatenate([mean_abs, std, line_length], axis=1).astype(np.float32)


def rank_channels(windows: np.ndarray, labels: np.ndarray):
    """
    Rank channels by mutual information.

    IMPORTANT: call this using training-fold windows only. Never include
    held-out patient windows when calculating channel scores.
    """
    windows = np.asarray(windows)
    labels = np.asarray(labels)

    if windows.ndim != 3 or windows.shape[0] == 0 or windows.shape[1] == 0:
        raise ValueError("Expected non-empty windows shaped (n_windows, n_channels, n_samples).")
    if labels.ndim != 1 or len(labels) != windows.shape[0]:
        raise ValueError("Labels must be 1D and have one value per window.")
    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain NaN or infinite values.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Channel ranking requires at least two label classes.")
    if not np.all(np.isin(np.unique(labels), [0, 1])):
        raise ValueError("Labels must be binary: 0 (normal) or 1 (seizure).")

    n_channels = windows.shape[1]
    features = channel_features(windows)
    mi = mutual_info_classif(features, labels.astype(int), random_state=42)
    scores = (
        mi[:n_channels]
        + mi[n_channels:2 * n_channels]
        + mi[2 * n_channels:]
    )
    return np.argsort(scores)[::-1], scores
