"""First-fold experiment for patient-robust EEG features.

Run from the EEG project root:
    python -u .\\scripts\\run_robust_feature_first_fold.py

This experiment deliberately evaluates the SAME fitted model on validation
and outer-test patients. It therefore avoids transferring a threshold from an
inner-fit model to a separately refitted model. Outer-test labels never enter
feature/model/threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR, PREFERRED_CHANNELS, RANDOM_STATE
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import load_recording_windows, processed_path_for_recording, prepare_lopo_fold
from data_management.patient_mapping import load_recording_manifest
from feature_selection.robust_feature_extraction import (
    ROBUST_FEATURE_NAMES, ROBUST_FEATURE_VERSION, extract_robust_features,
)
from modeling.baseline import calculate_metrics, choose_fbeta_threshold, positive_scores
from modeling.baseline import FeatureBatch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_signature(recordings, scaler) -> str:
    entries = []
    for recording in recordings:
        path = processed_path_for_recording(PROCESSED_DIR, recording)
        stat = path.stat()
        entries.append({
            "relative_path": recording.relative_path,
            "patient_id": recording.patient_id,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "feature_version": ROBUST_FEATURE_VERSION,
        "features": list(ROBUST_FEATURE_NAMES),
        "channels": list(PREFERRED_CHANNELS),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "recordings": entries,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_robust_batch(recordings, split_name: str, scaler, cache_root: Path) -> FeatureBatch:
    """Build/resume one split's robust features using recording-level commits."""
    if not recordings:
        raise ValueError(f"{split_name} split is empty")
    signature = _cache_signature(recordings, scaler)
    directory = cache_root / split_name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    x_path, y_path = directory / "X.npy", directory / "y.npy"

    lengths = []
    for recording in recordings:
        source = processed_path_for_recording(PROCESSED_DIR, recording)
        with np.load(source, allow_pickle=False) as saved:
            labels = np.asarray(saved["y"])
        if labels.ndim != 1 or not len(labels) or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"Invalid labels in {source.name}")
        if bool(np.any(labels == 1)) != (recording.seizure_count > 0):
            raise ValueError(f"Audit/label mismatch in {source.name}")
        lengths.append(len(labels))

    total_windows = int(sum(lengths))
    feature_count = len(PREFERRED_CHANNELS) * len(ROBUST_FEATURE_NAMES)
    completed, cursor = 0, 0
    X = y = None
    if manifest_path.is_file() and x_path.is_file() and y_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature and manifest.get("shape") == [total_windows, feature_count]:
                candidate_x = np.load(x_path, mmap_mode="r+", allow_pickle=False)
                candidate_y = np.load(y_path, mmap_mode="r+", allow_pickle=False)
                completed = int(manifest["completed_recordings"])
                cursor = int(manifest["cursor"])
                expected_cursor = sum(lengths[:completed])
                if (candidate_x.shape == (total_windows, feature_count)
                    and candidate_x.dtype == np.float32
                    and candidate_y.shape == (total_windows,)
                    and candidate_y.dtype == np.int8
                    and 0 <= completed <= len(recordings)
                    and cursor == expected_cursor):
                    X, y = candidate_x, candidate_y
                else:
                    completed, cursor = 0, 0
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            completed, cursor = 0, 0

    if X is None:
        # Release any rejected mappings before replacing cache files (Windows-safe).
        if "candidate_x" in locals():
            del candidate_x
        if "candidate_y" in locals():
            del candidate_y
        X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                      shape=(total_windows, feature_count))
        y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8,
                                      shape=(total_windows,))
        manifest = {
            "signature": signature, "shape": [total_windows, feature_count],
            "completed_recordings": 0, "cursor": 0, "complete": False,
        }
        _write_json(manifest_path, manifest)
    else:
        print(f"  Resuming {split_name} feature cache at {completed}/{len(recordings)} recordings", flush=True)

    for index in range(completed, len(recordings)):
        recording = recordings[index]
        item = load_recording_windows(PROCESSED_DIR, recording)
        next_cursor = cursor + len(item.y)
        if len(item.y) != lengths[index]:
            raise RuntimeError(f"Window count changed for {recording.relative_path}")

        # Bound temporary standardized EEG and Welch PSD memory. A recording is
        # committed to the resumable manifest only after every chunk is written.
        chunk_size = 256
        for chunk_start in range(0, len(item.y), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(item.y))
            standardized_chunk = scaler.transform(item.X[chunk_start:chunk_end])
            features = extract_robust_features(
                standardized_chunk, item.sfreq, chunk_size=chunk_size
            )
            expected_shape = (chunk_end - chunk_start, feature_count)
            if features.shape != expected_shape or not np.isfinite(features).all():
                raise RuntimeError(
                    f"Invalid robust feature chunk for {recording.relative_path} "
                    f"at windows {chunk_start}:{chunk_end}"
                )
            destination_start = cursor + chunk_start
            destination_end = cursor + chunk_end
            X[destination_start:destination_end] = features
            del standardized_chunk, features

        y[cursor:next_cursor] = item.y.astype(np.int8, copy=False)
        cursor = next_cursor
        X.flush()
        y.flush()
        _write_json(manifest_path, {
            "signature": signature, "shape": [total_windows, feature_count],
            "completed_recordings": index + 1, "cursor": cursor,
            "complete": index + 1 == len(recordings),
        })
        if (index + 1) % 25 == 0 or index + 1 == len(recordings):
            print(f"  {split_name}: {index + 1}/{len(recordings)} recordings", flush=True)
        del item

    if cursor != total_windows or not np.isin(y, (0, 1)).all():
        raise RuntimeError(f"{split_name} feature/label arrays failed integrity checks")
    X.flush()
    y.flush()
    return FeatureBatch(np.load(x_path, mmap_mode="r", allow_pickle=False),
                        np.load(y_path, mmap_mode="r", allow_pickle=False), ())


def _patient_validation_metrics(recordings, batch, scores, threshold):
    rows, cursor = [], 0
    for recording in recordings:
        source = processed_path_for_recording(PROCESSED_DIR, recording)
        with np.load(source, allow_pickle=False) as saved:
            count = len(saved["y"])
        end = cursor + count
        metrics = calculate_metrics(batch.y[cursor:end], scores[cursor:end], threshold)
        rows.append({"patient_id": recording.patient_id, **metrics})
        cursor = end
    if cursor != len(batch.y):
        raise RuntimeError("Patient-wise validation cursor does not match feature batch")
    return rows


def _models(seed: int):
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=10,
            class_weight="balanced_subsample", max_samples=0.25,
            n_jobs=-1, random_state=seed,
        ),
    }


def run(audit_path: Path, output_path: Path, seed: int, selected_models: list[str]):
    started = perf_counter()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment result: {output_path}. "
            "Choose a new --output path to preserve previous results."
        )
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    fold = build_lopo_folds(recordings)[0]
    if fold.fold_id != 1:
        raise RuntimeError("Expected deterministic first LOPO fold")
    print(f"Primary cohort: {len(recordings)} recordings; held-out: {fold.test_patient_id}", flush=True)

    cache_root = PROJECT_ROOT / "cache" / "robust_feature_first_fold_v1"
    prepared = prepare_lopo_fold(
        fold, PROCESSED_DIR, validation_fraction=0.2, random_state=seed,
        scaler_checkpoint_dir=cache_root / "scaler_checkpoint",
    )
    print("Extracting robust inner-fit features...", flush=True)
    fit_batch = collect_robust_batch(prepared.fit_recordings, "inner_fit",
                                     prepared.standardizer, cache_root)
    print("Extracting robust inner-validation features...", flush=True)
    validation_batch = collect_robust_batch(prepared.validation_recordings, "inner_validation",
                                            prepared.standardizer, cache_root)
    print("Extracting robust held-out test features...", flush=True)
    test_batch = collect_robust_batch(prepared.test_recordings, "outer_test",
                                      prepared.standardizer, cache_root)

    result = {
        "experiment": "patient_robust_feature_first_fold",
        "experiment_version": "robust_features_same_model_validation_test_v1",
        "feature_version": ROBUST_FEATURE_VERSION,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "cohort": {"recordings": len(recordings), "patients": len({r.patient_id for r in recordings})},
        "split": {
            "inner_fit_recordings": len(prepared.fit_recordings),
            "inner_validation_recordings": len(prepared.validation_recordings),
            "outer_test_recordings": len(prepared.test_recordings),
            "inner_fit_patients": sorted({r.patient_id for r in prepared.fit_recordings}),
            "inner_validation_patients": sorted({r.patient_id for r in prepared.validation_recordings}),
        },
        "features": {
            "channels": list(PREFERRED_CHANNELS),
            "per_channel": list(ROBUST_FEATURE_NAMES),
            "feature_count": len(PREFERRED_CHANNELS) * len(ROBUST_FEATURE_NAMES),
            "scaler_fit": "inner_fit_recordings_only",
        },
        "threshold_protocol": (
            "F2 threshold selected on inner validation; the exact same fitted model "
            "scores validation and outer test. No refit occurs between these splits."
        ),
        "models": {},
        "limitations": [
            "First LOPO fold only; not a complete 23-fold estimate.",
            "Model is trained on inner-fit patients only, not all outer-training patients.",
            "Window-level metrics only; event-level metrics are not yet included.",
        ],
    }

    score_root = output_path.parent / "robust_feature_first_fold_v1"
    score_root.mkdir(parents=True, exist_ok=True)
    for name in selected_models:
        print(f"Training {name}...", flush=True)
        model = _models(seed)[name]
        model.fit(fit_batch.X, fit_batch.y)
        validation_scores = positive_scores(model, validation_batch.X)
        threshold = choose_fbeta_threshold(validation_batch.y, validation_scores, beta=2.0)
        validation_metrics = calculate_metrics(validation_batch.y, validation_scores, threshold)
        test_scores = positive_scores(model, test_batch.X)
        test_metrics = calculate_metrics(test_batch.y, test_scores, threshold)
        patient_rows = _patient_validation_metrics(
            prepared.validation_recordings, validation_batch, validation_scores, threshold
        )
        model_dir = score_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_dir / "validation_scores.npz",
                            y_true=np.asarray(validation_batch.y), scores=validation_scores)
        np.savez_compressed(model_dir / "outer_test_scores.npz",
                            y_true=np.asarray(test_batch.y), scores=test_scores)
        joblib.dump(model, model_dir / "model.joblib")
        result["models"][name] = {
            "threshold": float(threshold),
            "threshold_objective": "inner-validation F2 (beta=2)",
            "validation_metrics": validation_metrics,
            "validation_patient_metrics": patient_rows,
            "outer_test_metrics": test_metrics,
            "model_artifacts": str(model_dir.relative_to(PROJECT_ROOT)),
        }
        print(f"{name}: validation F2={validation_metrics['f2']:.5f}; "
              f"outer-test F2={test_metrics['f2']:.5f}; "
              f"outer-test sensitivity={test_metrics['sensitivity']:.3%}", flush=True)
        del model, validation_scores, test_scores

    result["runtime_seconds"] = round(perf_counter() - started, 3)
    _write_json(output_path, result)
    print(f"Saved experiment result: {output_path}", flush=True)
    print("No outer-test labels were used for model or threshold selection.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
        default=Path(__file__).resolve().parents[3] / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--output", type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "robust_feature_first_fold_v1.json")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--models", nargs="+",
        choices=["logistic_regression", "random_forest"],
        default=["logistic_regression"])
    args = parser.parse_args()
    run(args.audit, args.output, args.seed, args.models)


if __name__ == "__main__":
    main()
