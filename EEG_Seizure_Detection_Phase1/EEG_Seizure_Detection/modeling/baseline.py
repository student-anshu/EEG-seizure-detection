"""Leakage-aware classical baseline utilities.

This module deliberately separates inner validation from outer evaluation.
It materializes feature rows for one split at a time; it never combines
recordings from the outer test patient with training data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_management.fold_data import load_recording_windows
from config import PREFERRED_CHANNELS
from data_management.patient_mapping import Recording
from feature_selection.feature_extraction import FEATURE_NAMES, extract_features
from preprocessing.normalization import (
    FoldChannelStandardizer, scaler_checkpoint_signature,
)


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    X: np.ndarray
    y: np.ndarray
    recording_paths: tuple[str, ...]


def fit_channel_standardizer(
    recordings: Sequence[Recording], processed_dir,
    checkpoint_dir: Path | None = None,
    checkpoint_context: dict | None = None,
) -> FoldChannelStandardizer:
    """Fit channel statistics on exactly the supplied recordings, streamed.

    When checkpoint_dir is provided, the running Welford accumulator is saved
    atomically after every complete recording and resumed only for an exact
    recording/source/context signature.
    """
    scaler = FoldChannelStandardizer()

    def batches(start_index=0):
        for index, recording in enumerate(recordings[start_index:], start=start_index + 1):
            item = load_recording_windows(processed_dir, recording)
            yield item.X
            del item
            if checkpoint_dir is None and (index % 25 == 0 or index == len(recordings)):
                print(
                    f"    channel-scaler fit: {index}/{len(recordings)} recordings",
                    flush=True,
                )

    if checkpoint_dir is None:
        return scaler.fit_batches(batches())

    if checkpoint_context is None:
        raise ValueError("checkpoint_context is required when checkpointing a scaler")
    signature = scaler_checkpoint_signature(
        recordings, processed_dir, checkpoint_context, scaler.epsilon
    )
    return scaler.fit_batches_checkpointed(
        batches,
        total_batches=len(recordings),
        checkpoint_dir=checkpoint_dir,
        signature=signature,
        progress_label="outer-training channel-scaler fit",
    )


def _feature_cache_signature(recordings, channel_scaler, processed_paths) -> str:
    """Fingerprint all inputs that determine the extracted feature values."""
    if channel_scaler.mean_ is None or channel_scaler.scale_ is None:
        raise RuntimeError("Channel scaler must be fitted before feature extraction")
    digest = hashlib.sha256()
    digest.update(b"eeg-feature-cache-v1")
    digest.update(np.asarray(channel_scaler.mean_, dtype=np.float64).tobytes())
    digest.update(np.asarray(channel_scaler.scale_, dtype=np.float64).tobytes())
    digest.update(json.dumps({
        "cache_version": 1,
        "processed_sources": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in processed_paths
        ],
        "epsilon": channel_scaler.epsilon,
        "channels": list(PREFERRED_CHANNELS),
        "features": list(FEATURE_NAMES),
        "recordings": [
            {"relative_path": item.relative_path, "patient_id": item.patient_id,
             "seizure_count": item.seizure_count}
            for item in recordings
        ],
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _write_cache_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def collect_features(
    recordings: Sequence[Recording], processed_dir,
    channel_scaler: FoldChannelStandardizer,
    cache_dir: Path | None = None,
    cache_key: str = "features",
) -> FeatureBatch:
    """Extract features with optional per-recording disk checkpoints.

    Cached X/y arrays are .npy memory maps. The manifest advances atomically
    only after both arrays are flushed, so an interrupted recording is safely
    recomputed on resume. Cache identity includes scaler and cohort metadata.
    """
    from data_management.fold_data import processed_path_for_recording

    if not recordings:
        raise ValueError("Cannot collect features from an empty recording list")

    total_windows = 0
    labels_by_recording = []
    processed_paths = []
    for recording in recordings:
        path = processed_path_for_recording(processed_dir, recording)
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed windows missing for {recording.relative_path}: {path}"
            )
        with np.load(path, allow_pickle=False) as saved:
            if "y" not in saved.files:
                raise ValueError(f"{path.name} is missing the y array")
            raw_labels = np.asarray(saved["y"])
        if raw_labels.ndim != 1 or len(raw_labels) == 0 or not np.isin(raw_labels, (0, 1)).all():
            raise ValueError(f"{path.name} has empty or non-binary labels")
        labels = raw_labels.astype(np.int8, copy=False)
        if bool(np.any(labels == 1)) != (recording.seizure_count > 0):
            raise ValueError(f"{path.name} labels disagree with the audit manifest")
        labels_by_recording.append(labels)
        processed_paths.append(path)
        total_windows += len(labels)

    feature_count = len(PREFERRED_CHANNELS) * len(FEATURE_NAMES)
    cache_path = None
    manifest_path = None
    X = y = None
    cursor = 0
    completed_recordings = 0

    if cache_dir is not None:
        cache_root = Path(cache_dir)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key)).strip("._") or "features"
        signature = _feature_cache_signature(recordings, channel_scaler, processed_paths)
        cache_path = cache_root / f"{safe_key}-{signature[:20]}"
        cache_path.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_path / "manifest.json"
        x_path, y_path = cache_path / "X.npy", cache_path / "y.npy"
        expected_shape = (total_windows, feature_count)

        if manifest_path.is_file() and x_path.is_file() and y_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (manifest.get("signature") == signature
                    and manifest.get("shape") == list(expected_shape)
                    and manifest.get("dtype") == "float32"
                    and manifest.get("label_dtype") == "int8"):
                    candidate_x = np.load(x_path, mmap_mode="r+", allow_pickle=False)
                    candidate_y = np.load(y_path, mmap_mode="r+", allow_pickle=False)
                    completed = int(manifest.get("completed_recordings", 0))
                    saved_cursor = int(manifest.get("cursor", 0))
                    expected_cursor = sum(len(labels_by_recording[i]) for i in range(completed))
                    if (candidate_x.shape == expected_shape and candidate_x.dtype == np.float32
                        and candidate_y.shape == (total_windows,) and candidate_y.dtype == np.int8
                        and 0 <= completed <= len(recordings) and saved_cursor == expected_cursor):
                        X, y = candidate_x, candidate_y
                        completed_recordings, cursor = completed, saved_cursor
            except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
                X = y = None

        if X is None:
            # Close partially opened mappings before replacing files (Windows compatibility).
            if "candidate_x" in locals():
                del candidate_x
            if "candidate_y" in locals():
                del candidate_y
            for stale in (x_path, y_path, manifest_path):
                if stale.exists():
                    stale.unlink()
            X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=expected_shape)
            y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8, shape=(total_windows,))
            _write_cache_manifest(manifest_path, {
                "signature": signature, "shape": list(expected_shape),
                "dtype": "float32", "label_dtype": "int8",
                "completed_recordings": 0, "cursor": 0, "complete": False,
            })
        else:
            print(f"    resuming feature cache: {completed_recordings}/{len(recordings)} recordings", flush=True)
    else:
        X = np.empty((total_windows, feature_count), dtype=np.float32)
        y = np.empty(total_windows, dtype=np.int8)

    for index, (recording, expected_y, path) in enumerate(
        zip(recordings, labels_by_recording, processed_paths), start=1
    ):
        if index <= completed_recordings:
            continue
        item = load_recording_windows(processed_dir, recording)
        if not np.array_equal(item.y, expected_y):
            raise RuntimeError(
                f"Labels changed between metadata and feature pass: {recording.relative_path}"
            )
        standardized = channel_scaler.transform(item.X)
        features = extract_features(standardized, item.sfreq)
        expected_shape = (len(item.y), feature_count)
        if features.shape != expected_shape or features.dtype != np.float32:
            raise RuntimeError(
                f"Invalid feature matrix for {recording.relative_path}: {features.shape}, {features.dtype}"
            )
        if not np.isfinite(features).all():
            raise RuntimeError(f"Non-finite features for {recording.relative_path}")
        next_cursor = cursor + len(item.y)
        X[cursor:next_cursor] = features
        y[cursor:next_cursor] = item.y
        cursor = next_cursor
        del item, standardized, features

        if cache_path is not None:
            X.flush()
            y.flush()
            _write_cache_manifest(manifest_path, {
                "signature": signature, "shape": list(X.shape),
                "dtype": "float32", "label_dtype": "int8",
                "completed_recordings": index, "cursor": cursor,
                "complete": index == len(recordings),
            })
        if index % 25 == 0 or index == len(recordings):
            print(f"    feature extraction: {index}/{len(recordings)} recordings", flush=True)

    if cursor != total_windows or not np.isin(y, (0, 1)).all():
        raise RuntimeError("Collected features and binary labels are inconsistent")
    if cache_path is not None:
        X.flush()
        y.flush()
        del X, y
        X = np.load(cache_path / "X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(cache_path / "y.npy", mmap_mode="r", allow_pickle=False)
    return FeatureBatch(X, y, ())


def build_baseline_estimators(random_state: int = 42) -> dict[str, Pipeline]:
    """Return fixed, documented baseline estimators (no outer-test tuning)."""
    return {
        "logistic_regression": Pipeline([
            ("feature_scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear", class_weight="balanced",
                max_iter=1000, random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("feature_scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=18, min_samples_leaf=5,
                class_weight="balanced_subsample", max_samples=0.5,
                n_jobs=-1, random_state=random_state,
            )),
        ]),
    }


def positive_scores(estimator, X: np.ndarray) -> np.ndarray:
    """Return a continuous positive-class score for threshold selection."""
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
    else:
        raise TypeError("Estimator must expose predict_proba or decision_function")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Estimator returned invalid positive-class scores")
    return scores


def choose_fbeta_threshold(
    y_true: np.ndarray, scores: np.ndarray, beta: float = 1.0,
) -> float:
    """Choose an F-beta-maximizing threshold using validation labels only.

    beta=1 is F1; beta=2 weights recall more heavily. This function must only
    receive inner-validation labels/scores, never outer-test labels.
    """
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number")
    if y.ndim != 1 or values.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Expected aligned one-dimensional binary labels and scores")
    if not np.isfinite(values).all() or not np.any(y == 1):
        raise ValueError("Threshold selection requires finite scores and positive labels")

    precision, recall, thresholds = precision_recall_curve(y, values)
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = float(beta) ** 2
    denominator = beta_squared * precision + recall
    fbeta_values = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    # thresholds are ascending; argmax preserves smallest-threshold tie breaking.
    best_index = int(np.argmax(fbeta_values))
    return float(thresholds[best_index])


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Backward-compatible F1 threshold selector."""
    return choose_fbeta_threshold(y_true, scores, beta=1.0)


def calculate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Calculate window metrics; AUROC is null for a single-class target."""
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or len(y) == 0:
        raise ValueError("Expected non-empty aligned one-dimensional arrays")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(values).all():
        raise ValueError("Labels must be binary and scores finite")
    predicted = values >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2.0, zero_division=0)),
        "auroc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, values)) if np.any(y == 1) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "window_count": int(len(y)),
        "positive_windows": int(np.sum(y == 1)),
    }


def fit_validate_model(
    estimator, fit_batch: FeatureBatch, validation_batch: FeatureBatch,
) -> tuple[object, float, dict]:
    """Fit on inner-fit patients and choose threshold on validation patients."""
    if len(np.unique(fit_batch.y)) != 2:
        raise ValueError("Inner-fit data must contain both classes")
    if len(np.unique(validation_batch.y)) != 2:
        raise ValueError("Validation data must contain both classes for this baseline")
    print("    fitting inner Logistic Regression...", flush=True)
    estimator.fit(fit_batch.X, fit_batch.y)
    print("    scoring inner-validation windows...", flush=True)
    scores = positive_scores(estimator, validation_batch.X)
    print("    selecting F1 threshold from validation scores...", flush=True)
    threshold = choose_f1_threshold(validation_batch.y, scores)
    return estimator, threshold, calculate_metrics(validation_batch.y, scores, threshold)


def refit_and_evaluate(
    estimator, outer_train_batch: FeatureBatch, outer_test_batch: FeatureBatch,
    validation_threshold: float,
) -> tuple[object, dict]:
    """Refit a fresh estimator on all outer-training patients, then test once.

    Callers must build both batches using a channel standardizer fitted on
    all outer-training patients. The threshold is frozen from inner validation;
    outer-test labels are used only by calculate_metrics after predictions.
    """
    if len(np.unique(outer_train_batch.y)) != 2:
        raise ValueError("Outer-training data must contain both classes")
    if not np.isfinite(validation_threshold):
        raise ValueError("Validation threshold must be finite")
    estimator.fit(outer_train_batch.X, outer_train_batch.y)
    test_scores = positive_scores(estimator, outer_test_batch.X)
    return estimator, calculate_metrics(
        outer_test_batch.y, test_scores, float(validation_threshold)
    )
