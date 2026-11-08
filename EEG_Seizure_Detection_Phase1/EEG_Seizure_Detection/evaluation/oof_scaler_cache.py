"""Reusable per-recording channel statistics for patient-grouped OOF scaling.

Statistics are computed from the existing processed EEG windows. They contain
no labels and can be merged to fit a scaler without rereading all samples.
"""
from __future__ import annotations

import numpy as np


def channel_sufficient_statistics(
    windows: np.ndarray, chunk_size: int = 256
) -> dict[str, object]:
    """Return population count, mean and M2 per channel using bounded memory."""
    values = np.asarray(windows)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("windows must have non-empty (windows, channels, samples) shape")
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    state = None
    for start in range(0, len(values), int(chunk_size)):
        chunk = np.asarray(values[start:start + int(chunk_size)], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("windows contain non-finite values")
        flattened = chunk.transpose(1, 0, 2).reshape(chunk.shape[1], -1)
        mean = np.mean(flattened, axis=1)
        centered = flattened - mean[:, None]
        batch = {
            "count": int(flattened.shape[1]),
            "mean": mean,
            "m2": np.sum(centered * centered, axis=1),
        }
        state = batch if state is None else merge_channel_statistics(state, batch)
    return state


def merge_channel_statistics(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    """Merge two population-statistic states using parallel Welford."""
    n_a, n_b = int(left["count"]), int(right["count"])
    mean_a = np.asarray(left["mean"], dtype=np.float64)
    mean_b = np.asarray(right["mean"], dtype=np.float64)
    m2_a = np.asarray(left["m2"], dtype=np.float64)
    m2_b = np.asarray(right["m2"], dtype=np.float64)
    if n_a <= 0 or n_b <= 0:
        raise ValueError("statistics counts must be positive")
    if mean_a.ndim != 1 or mean_b.shape != mean_a.shape:
        raise ValueError("statistics channel dimensions do not match")
    if m2_a.shape != mean_a.shape or m2_b.shape != mean_a.shape:
        raise ValueError("statistics M2 dimensions do not match")
    if not all(np.isfinite(v).all() for v in (mean_a, mean_b, m2_a, m2_b)):
        raise ValueError("statistics contain non-finite values")
    if np.any(m2_a < 0) or np.any(m2_b < 0):
        raise ValueError("statistics M2 must be non-negative")
    total = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / total)
    m2 = m2_a + m2_b + delta * delta * (n_a * n_b / total)
    return {"count": total, "mean": mean, "m2": m2}


def finalize_channel_statistics(
    state: dict[str, object], epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and population-standard-deviation scale."""
    count = int(state["count"])
    mean = np.asarray(state["mean"], dtype=np.float64)
    m2 = np.asarray(state["m2"], dtype=np.float64)
    if count <= 0 or mean.ndim != 1 or m2.shape != mean.shape:
        raise ValueError("invalid channel statistics state")
    if not np.isfinite(mean).all() or not np.isfinite(m2).all() or np.any(m2 < 0):
        raise ValueError("invalid channel statistics values")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return mean.copy(), np.maximum(np.sqrt(np.maximum(m2 / count, 0.0)), epsilon)


def recompute_log_std_features(
    windows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """Recompute channel log_std exactly under a supplied fold scaler.

    Mirrors FoldChannelStandardizer.transform followed by the log_std portion
    of extract_robust_features, including float32 transform precision and the
    1e-12 log floor. Output shape is (n_windows, n_channels), float32.
    """
    values = np.asarray(windows)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("windows must have non-empty (windows, channels, samples) shape")
    if mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("mean/scale must match the window channel count")
    if not np.isfinite(values).all() or not np.isfinite(mean).all():
        raise ValueError("windows and mean must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scale must be finite and positive")
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    result = np.empty((len(values), values.shape[1]), dtype=np.float32)
    for start in range(0, len(values), int(chunk_size)):
        stop = min(start + int(chunk_size), len(values))
        standardized = ((np.asarray(values[start:stop], dtype=np.float64)
                         - mean[None, :, None]) / scale[None, :, None]).astype(np.float32)
        centered = standardized.astype(np.float64)
        centered -= np.mean(centered, axis=-1, keepdims=True)
        std = np.std(centered, axis=-1)
        result[start:stop] = np.log(np.maximum(std, 1e-12)).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("recomputed log_std contains non-finite values")
    return result



def window_channel_standard_deviations(
    windows: np.ndarray, chunk_size: int = 256
) -> np.ndarray:
    """Cache raw per-window/per-channel population std for fast fold log_std.

    Standardization is a per-channel affine transform. Apart from float32
    rounding, std( (window - fold_mean) / fold_scale ) equals raw_std/fold_scale.
    This compact cache avoids rereading/decompressing all EEG windows for each
    OOF fold. Output is float32 with shape (n_windows, n_channels).
    """
    values = np.asarray(windows)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("windows must have non-empty (windows, channels, samples) shape")
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    result = np.empty((len(values), values.shape[1]), dtype=np.float32)
    for start in range(0, len(values), int(chunk_size)):
        stop = min(start + int(chunk_size), len(values))
        chunk = np.asarray(values[start:stop], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("windows contain non-finite values")
        centered = chunk - np.mean(chunk, axis=-1, keepdims=True)
        result[start:stop] = np.std(centered, axis=-1).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise RuntimeError("invalid per-window channel standard deviations")
    return result


def log_std_from_raw_standard_deviations(
    raw_std: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply fold channel scales to cached raw std, returning robust log_std."""
    values = np.asarray(raw_std, dtype=np.float32)
    channel_scale = np.asarray(scale, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(channel_scale):
        raise ValueError("raw_std must be (windows, channels) and match scale")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("raw_std must be finite and non-negative")
    if not np.isfinite(channel_scale).all() or np.any(channel_scale <= 0):
        raise ValueError("scale must be finite and positive")
    result = np.log(np.maximum(
        values.astype(np.float64) / channel_scale[None, :], 1e-12
    )).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("fold log_std contains non-finite values")
    return result


def exact_log_std_fallback_mask(
    raw_std: np.ndarray,
    scale: np.ndarray,
    normalized_std_threshold: float = 1e-4,
) -> np.ndarray:
    """Flag windows whose tiny normalized variation may be altered by float32 rounding.

    This is a conservative routing rule, not a change to the feature definition.
    Flagged windows must be recomputed from the original EEG samples using the
    exact FoldChannelStandardizer float32 transform and extractor calculation.
    """
    values = np.asarray(raw_std, dtype=np.float64)
    channel_scale = np.asarray(scale, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(channel_scale):
        raise ValueError("raw_std must be (windows, channels) and match scale")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("raw_std must be finite and non-negative")
    if not np.isfinite(channel_scale).all() or np.any(channel_scale <= 0):
        raise ValueError("scale must be finite and positive")
    if (not np.isfinite(normalized_std_threshold)
            or normalized_std_threshold <= 0):
        raise ValueError("normalized_std_threshold must be finite and positive")
    normalized_std = values / channel_scale[None, :]
    return np.any(normalized_std <= normalized_std_threshold, axis=1)


def patient_grouped_channel_scale(
    patient_statistics: dict[str, dict[str, object]],
    heldout_patient: str,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit channel mean/scale using every patient except the held-out patient."""
    if heldout_patient not in patient_statistics:
        raise ValueError("held-out patient is missing from the statistics mapping")
    training_states = [
        state for patient, state in patient_statistics.items()
        if patient != heldout_patient
    ]
    if len(training_states) < 1:
        raise ValueError("at least one non-held-out training patient is required")
    merged = training_states[0]
    for state in training_states[1:]:
        merged = merge_channel_statistics(merged, state)
    return finalize_channel_statistics(merged, epsilon=epsilon)
