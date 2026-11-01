import numpy as np
from sklearn.feature_selection import mutual_info_classif

def channel_features(windows: np.ndarray) -> np.ndarray:
    """
    Build simple channel-wise features for initial channel ranking:
    mean absolute amplitude, standard deviation, and line length.
    Input: (windows, channels, samples).
    Output: (windows, channels * 3).
    """
    if windows.ndim != 3:
        raise ValueError("Expected windows with shape (n_windows, n_channels, n_samples).")
    mean_abs = np.mean(np.abs(windows), axis=2)
    std = np.std(windows, axis=2)
    line_length = np.mean(np.abs(np.diff(windows, axis=2)), axis=2)
    return np.concatenate([mean_abs, std, line_length], axis=1)

def rank_channels(windows: np.ndarray, labels: np.ndarray):
    """
    Rank channels using mutual information computed from training data only.
    Do not fit this on held-out test patients.
    """
    n_channels = windows.shape[1]
    features = channel_features(windows)
    mi = mutual_info_classif(features, labels, random_state=42)
    scores = mi[:n_channels] + mi[n_channels:2*n_channels] + mi[2*n_channels:]
    return np.argsort(scores)[::-1], scores
