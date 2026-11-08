"""Run the first nested LOPO Logistic Regression experiment.

Run from the EEG_Seizure_Detection project directory:
    python -u .\\scripts\\run_first_baseline_fold.py

This runs only fold 1 (patient_01 held out). It is an engineering validation
run, not the final 23-fold evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_SCHEMA_VERSION = 2
THRESHOLD_STRATEGY_VERSION = "validation_f1_and_f2_v1"
EXPERIMENT_VERSION = "nested_lopo_lr_f1_f2_v1"

from config import PROCESSED_DIR, PREFERRED_CHANNELS, RANDOM_STATE
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import prepare_lopo_fold
from data_management.validation_split import build_patient_validation_split
from data_management.patient_mapping import load_recording_manifest
from feature_selection.feature_extraction import FEATURE_NAMES
from modeling.experiment_checkpoints import ModelStageCheckpointStore
from data_management.fold_data import processed_path_for_recording
from modeling.baseline import (
    build_baseline_estimators,
    calculate_metrics,
    collect_features,
    fit_channel_standardizer,
    positive_scores,
    choose_fbeta_threshold,
)


def _split_summary(recordings):
    return {
        "recordings": len(recordings),
        "patients": sorted({item.patient_id for item in recordings}),
    }


def _write_result(output_path: Path, result: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def _check_existing_completed_result(output_path: Path) -> None:
    """Prevent accidental retraining when a completed result lacks a checkpoint."""
    if not output_path.is_file():
        return

    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing output cannot be safely inspected: {output_path}. "
            "Preserve and inspect this file before running again."
        ) from exc

    if not isinstance(existing, dict):
        return

    if (
        existing.get("experiment") == "first_nested_lopo_logistic_regression"
        and existing.get("status") == "completed"
        and existing.get("fold_id") == 1
        and existing.get("held_out_patient") == "patient_01"
    ):
        raise RuntimeError(
            "A completed first-fold result already exists, but no compatible "
            "durable checkpoint was found. To prevent accidental retraining, "
            f"the runner has stopped without modifying the result: {output_path}. "
            "Inspect or archive the existing result before intentionally "
            "starting a new experiment with a different --output path."
        )


def _is_compatible_completed_result(result: dict, fold) -> bool:
    """Reject stale or incomplete summaries before reusing durable results."""
    if not isinstance(result, dict):
        return False
    if (
        result.get("experiment") != "first_nested_lopo_logistic_regression"
        or result.get("experiment_version") != EXPERIMENT_VERSION
        or result.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or result.get("threshold_strategy_version") != THRESHOLD_STRATEGY_VERSION
        or result.get("status") != "completed"
        or result.get("fold_id") != fold.fold_id
        or result.get("held_out_patient") != fold.test_patient_id
    ):
        return False

    selection = result.get("threshold_selection")
    if not isinstance(selection, dict):
        return False
    if (
        selection.get("method") != "maximize validation F1"
        or selection.get("outer_test_used_for_selection") is not False
        or selection.get("f2_beta") != 2.0
        or selection.get("comparison_strategy")
        != "F2 threshold is selected independently on inner validation only"
    ):
        return False

    try:
        f1_threshold = float(selection["validation_threshold"])
        f2_threshold = float(selection["f2_validation_threshold"])
    except (KeyError, TypeError, ValueError):
        return False
    if not np.isfinite([f1_threshold, f2_threshold]).all():
        return False

    metric_keys = {
        "threshold", "sensitivity", "specificity", "precision", "f1", "f2",
        "tn", "fp", "fn", "tp", "window_count", "positive_windows",
    }
    metric_thresholds = {
        "validation_metrics": f1_threshold,
        "outer_test_metrics": f1_threshold,
        "validation_metrics_f2": f2_threshold,
        "outer_test_metrics_f2": f2_threshold,
    }
    for name, expected_threshold in metric_thresholds.items():
        metrics = result.get(name)
        if not isinstance(metrics, dict) or not metric_keys.issubset(metrics):
            return False
        try:
            actual_threshold = float(metrics["threshold"])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(actual_threshold) or not np.isclose(
            actual_threshold, expected_threshold, rtol=0.0, atol=1e-12
        ):
            return False
    return True


def _recording_fingerprint(recordings):
    values = []
    for item in recordings:
        source = processed_path_for_recording(PROCESSED_DIR, item)
        stat = source.stat()
        values.append({
            "relative_path": item.relative_path,
            "patient_id": item.patient_id,
            "seizure_count": item.seizure_count,
            "processed_path": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return values


def _build_checkpoint_context(fold, inner_split, seed: int) -> dict:
    """Build the exact data/model signature used by durable and stage checkpoints."""
    return {
        "experiment": "first_nested_lopo_logistic_regression",
        "experiment_version": EXPERIMENT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "threshold_strategy_version": THRESHOLD_STRATEGY_VERSION,
        "checkpoint_context_schema_version": 2,
        "checkpoint_version": 1,
        "feature_pipeline_version": 1,
        "channel_normalization_version": 1,
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "random_state": int(seed),
        "model": {
            "name": "LogisticRegression", "solver": "liblinear",
            "class_weight": "balanced", "max_iter": 1000,
        },
        "inner_fit": _recording_fingerprint(inner_split.train_recordings),
        "inner_validation": _recording_fingerprint(inner_split.validation_recordings),
        "outer_train": _recording_fingerprint(fold.train_recordings),
        "outer_test": _recording_fingerprint(fold.test_recordings),
        "features": {
            "channels": list(PREFERRED_CHANNELS),
            "feature_names": list(FEATURE_NAMES),
        },
    }


def run_first_fold(audit_path: Path, output_path: Path, seed: int = RANDOM_STATE):
    started = perf_counter()
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    folds = build_lopo_folds(recordings)
    fold = folds[0]
    if fold.fold_id != 1:
        raise RuntimeError("Expected the first deterministic LOPO fold")

    print(f"Primary cohort: {len(recordings)} recordings", flush=True)
    print(f"Outer fold {fold.fold_id}: held-out patient {fold.test_patient_id}", flush=True)
    cache_root = PROJECT_ROOT / "cache" / "first_fold"
    persistent_checkpoint_root = (
        PROJECT_ROOT / "results" / "baselines" / "checkpoints"
        / "first_fold_patient_01_logistic_regression_f1_f2_v1"
    )
    # Build the deterministic patient split without fitting/loading the scaler.
    inner_split = build_patient_validation_split(
        fold.train_recordings, validation_fraction=0.2, random_state=seed
    )
    checkpoint_context = _build_checkpoint_context(fold, inner_split, seed)
    # Check durable completed results before scaler fitting or feature extraction.
    completed_store = ModelStageCheckpointStore(
        persistent_checkpoint_root, checkpoint_context
    )
    completed_result = completed_store.load_json("fold_result")
    if completed_result is not None and completed_result.get("status") == "completed":
        if not _is_compatible_completed_result(completed_result, fold):
            raise RuntimeError(
                "A completed durable result has an incompatible or incomplete "
                "experiment/result schema. Preserve it; do not reuse it as the "
                "current F1/F2 experiment."
            )
        # A completed result is only reusable when both score artifacts are
        # present and signature/checksum-valid.
        if (completed_store.load_scores("inner_validation_scores") is None
                or completed_store.load_scores("outer_test_scores") is None):
            raise RuntimeError(
                "Completed fold result is missing durable score artifacts. "
                "Preserve it and run a deliberate new experiment with a new "
                "--output path; do not silently treat summary metrics as recoverable."
            )
        _write_result(output_path, completed_result)
        print(f"Recovered completed fold result: {output_path}", flush=True)
        return completed_result

    # A legacy output JSON is not a trusted checkpoint because it has no
    # context signature or artifact manifest. Stop rather than retrain silently.
    _check_existing_completed_result(output_path)

    stage_store = ModelStageCheckpointStore(
        cache_root / "model_stages" / "fold_1_logistic_regression_f1_f2_v1",
        checkpoint_context,
    )
    print("Preparing inner fit/validation split and fit-only channel scaler...", flush=True)
    prepared = prepare_lopo_fold(
        fold, PROCESSED_DIR, validation_fraction=0.2, random_state=seed,
        scaler_checkpoint_dir=cache_root / "scalers" / "fold_1_inner_fit",
    )
    print("Extracting/resuming inner-fit features...", flush=True)
    fit_batch = collect_features(
        prepared.fit_recordings, PROCESSED_DIR, prepared.standardizer,
        cache_dir=cache_root, cache_key="inner_fit",
    )
    print("Extracting/resuming inner-validation features...", flush=True)
    validation_batch = collect_features(
        prepared.validation_recordings, PROCESSED_DIR, prepared.standardizer,
        cache_dir=cache_root, cache_key="inner_validation",
    )

    inner_model = stage_store.load_estimator("inner_model_fit")
    if inner_model is None:
        print("Fitting inner Logistic Regression (an interrupted fit restarts)...", flush=True)
        inner_model = build_baseline_estimators(random_state=seed)["logistic_regression"]
        inner_model.fit(fit_batch.X, fit_batch.y)
        stage_store.save_estimator("inner_model_fit", inner_model)
    else:
        print("Reusing completed inner Logistic Regression fit checkpoint.", flush=True)

    validation_scores = stage_store.load_scores("inner_validation_scores")
    if validation_scores is None:
        print("Scoring inner-validation windows...", flush=True)
        validation_y = np.asarray(validation_batch.y, dtype=np.int8)
        validation_values = positive_scores(inner_model, validation_batch.X)
        stage_store.save_scores("inner_validation_scores", validation_y, validation_values)
    else:
        validation_y, validation_values = validation_scores
        if not np.array_equal(validation_y, validation_batch.y):
            raise RuntimeError("Cached validation labels do not match current validation split")
        print("Reusing completed validation-score checkpoint.", flush=True)
    # Keep a durable copy outside the disposable working cache.
    completed_store.save_scores("inner_validation_scores", validation_y, validation_values)

    threshold_payload = stage_store.load_json("inner_validation_threshold")
    if threshold_payload is None:
        from modeling.baseline import choose_f1_threshold
        threshold = choose_f1_threshold(validation_batch.y, validation_values)
        validation_metrics = calculate_metrics(validation_batch.y, validation_values, threshold)
        stage_store.save_json("inner_validation_threshold", {
            "threshold": float(threshold), "metrics": validation_metrics,
            "selection": "maximize validation F1",
        })
    else:
        threshold = float(threshold_payload["threshold"])
        validation_metrics = threshold_payload["metrics"]

    f2_threshold_payload = stage_store.load_json("inner_validation_f2_threshold")
    if f2_threshold_payload is None:
        f2_threshold = choose_fbeta_threshold(validation_y, validation_values, beta=2.0)
        validation_metrics_f2 = calculate_metrics(validation_y, validation_values, f2_threshold)
        stage_store.save_json("inner_validation_f2_threshold", {
            "threshold": float(f2_threshold),
            "metrics": validation_metrics_f2,
            "selection": "maximize validation F2 (beta=2)",
            "beta": 2.0,
        })
    else:
        f2_threshold = float(f2_threshold_payload["threshold"])
        validation_metrics_f2 = f2_threshold_payload["metrics"]

    print(f"Validation F1 threshold selected: {threshold:.8g}", flush=True)
    print(f"Validation F2 threshold selected: {f2_threshold:.8g}", flush=True)
    print(
        "Validation metrics: "
        + json.dumps(validation_metrics, sort_keys=True),
        flush=True,
    )

    # Free the inner feature matrices before allocating outer-train matrices.
    del fit_batch, validation_batch, inner_model
    gc.collect()

    print("Refitting channel scaler on all outer-training patients...", flush=True)
    outer_scaler = fit_channel_standardizer(
        fold.train_recordings,
        PROCESSED_DIR,
        checkpoint_dir=cache_root / "scalers" / "fold_1_outer_train",
        checkpoint_context={
            "stage": "outer_train_channel_scaler",
            "fold_id": fold.fold_id,
            "outer_test_patient": fold.test_patient_id,
            "outer_train_patients": sorted({item.patient_id for item in fold.train_recordings}),
            "random_state": int(seed),
        },
    )

    print("Extracting all outer-training features...", flush=True)
    outer_train_batch = collect_features(
        fold.train_recordings, PROCESSED_DIR, outer_scaler,
        cache_dir=cache_root, cache_key="outer_train",
    )
    print("Extracting/resuming held-out test-patient features...", flush=True)
    outer_test_batch = collect_features(
        fold.test_recordings, PROCESSED_DIR, outer_scaler,
        cache_dir=cache_root, cache_key="outer_test",
    )

    print("Training final Logistic Regression on all outer-training patients...", flush=True)
    final_model = stage_store.load_estimator("final_model_fit")
    if final_model is None:
        print("Fitting final Logistic Regression (an interrupted fit restarts)...", flush=True)
        final_model = build_baseline_estimators(random_state=seed)["logistic_regression"]
        final_model.fit(outer_train_batch.X, outer_train_batch.y)
        stage_store.save_estimator("final_model_fit", final_model)
    else:
        print("Reusing completed final-model fit checkpoint.", flush=True)

    print("Scoring held-out patient...", flush=True)
    test_score_checkpoint = stage_store.load_scores("outer_test_scores")
    if test_score_checkpoint is None:
        test_scores = positive_scores(final_model, outer_test_batch.X)
        stage_store.save_scores("outer_test_scores", outer_test_batch.y, test_scores)
    else:
        test_y, test_scores = test_score_checkpoint
        if not np.array_equal(test_y, outer_test_batch.y):
            raise RuntimeError("Cached test labels do not match current held-out patient")
        print("Reusing completed held-out score checkpoint.", flush=True)
    completed_store.save_scores("outer_test_scores", outer_test_batch.y, test_scores)
    test_metrics = calculate_metrics(outer_test_batch.y, test_scores, threshold)
    validation_metrics_f2 = calculate_metrics(validation_y, validation_values, f2_threshold)
    test_metrics_f2 = calculate_metrics(outer_test_batch.y, test_scores, f2_threshold)

    result = {
        "experiment": "first_nested_lopo_logistic_regression",
        "experiment_version": EXPERIMENT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "threshold_strategy_version": THRESHOLD_STRATEGY_VERSION,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "primary_cohort": {
            "recordings": len(recordings),
            "patients": len({item.patient_id for item in recordings}),
        },
        "inner_fit": _split_summary(prepared.fit_recordings),
        "inner_validation": _split_summary(prepared.validation_recordings),
        "outer_train": _split_summary(fold.train_recordings),
        "outer_test": _split_summary(fold.test_recordings),
        "model": {
            "name": "LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": seed,
        },
        "preprocessing": {
            "sampling_rate_hz": 256,
            "window_seconds": 10,
            "stride_seconds": 5,
            "channels": list(PREFERRED_CHANNELS),
            "features_per_channel": list(FEATURE_NAMES),
            "feature_count": len(PREFERRED_CHANNELS) * len(FEATURE_NAMES),
            "channel_scaler_fit": "inner_fit_only_for_threshold_selection_then_all_outer_train_for_refit",
            "feature_scaler": "StandardScaler fitted within each LogisticRegression pipeline",
        },
        "threshold_selection": {
            "method": "maximize validation F1",
            "validation_threshold": float(threshold),
            "outer_test_used_for_selection": False,
            "comparison_strategy": "F2 threshold is selected independently on inner validation only",
            "f2_beta": 2.0,
            "f2_validation_threshold": float(f2_threshold),
        },
        "validation_metrics": validation_metrics,
        "outer_test_metrics": test_metrics,
        "validation_metrics_f2": validation_metrics_f2,
        "outer_test_metrics_f2": test_metrics_f2,
        "runtime_seconds": round(perf_counter() - started, 3),
        "limitations": [
            "This is only the first LOPO fold, not the complete 23-fold evaluation.",
            "The validation-selected decision threshold is transferred to the refitted model.",
            "Metrics are window-level; event-level aggregation is not implemented in this milestone.",
        ],
    }

    stage_store.save_json("fold_result", result)
    completed_store.save_json("fold_result", result)
    _write_result(output_path, result)
    print("\nFIRST-FOLD EXPERIMENT COMPLETED", flush=True)
    print(f"Saved results: {output_path}", flush=True)
    print("Outer-test metrics (held-out patient; window-level):", flush=True)
    print(json.dumps(test_metrics, indent=2, sort_keys=True), flush=True)
    print("This result is not the final 23-fold performance estimate.", flush=True)

    # Feature caches are only needed to resume an incomplete run. Preserve them
    # on any exception/interruption; remove them only after the result is saved.
    # Release read-only memory maps before deleting their backing files
    # (required on Windows).
    del outer_train_batch, outer_test_batch, final_model, test_scores
    gc.collect()
    if cache_root.exists():
        shutil.rmtree(cache_root)
        print(f"Cleaned completed fold feature cache: {cache_root}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "docs" / "full_annotation_audit.csv",
        help="Path to docs/full_annotation_audit.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "first_fold_patient_01_f1_f2_v1.json",
        help="JSON output path for this experiment",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    run_first_fold(args.audit, args.output, args.seed)


if __name__ == "__main__":
    main()
