"""Evaluate saved robust-feature first-fold scores at seizure-event level.

Run from the EEG project directory:
    python -m scripts.evaluate_robust_first_fold_events --audit ..\\..\\docs\\full_annotation_audit.csv

This script loads saved score arrays only. It does not load or fit the model,
re-extract features, or modify training artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import PROCESSED_DIR, RANDOM_STATE, WINDOW_SECONDS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split
from evaluation.recording_inventory import load_recording_durations
from scripts.evaluate_first_fold_events import _evaluate_threshold


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Score artifact not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"y_true", "scores"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        y_true = np.asarray(saved["y_true"], dtype=np.int8)
        scores = np.asarray(saved["scores"], dtype=np.float64)
    if y_true.ndim != 1 or scores.shape != y_true.shape or not len(y_true):
        raise ValueError(f"{path} has invalid score/label shapes")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError(f"{path} contains labels outside {{0, 1}}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{path} contains non-finite or non-probability scores")
    return y_true, scores


def evaluate_robust_first_fold_events(
    audit_path: Path,
    result_path: Path,
    score_dir: Path,
    output_path: Path,
    processed_dir: Path = PROCESSED_DIR,
    inventory_path: Path | None = None,
) -> dict:
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    folds = build_lopo_folds(recordings)
    fold = folds[0]
    inner_split = build_patient_validation_split(
        fold.train_recordings, validation_fraction=0.2, random_state=RANDOM_STATE
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"Robust experiment result not found: {result_path}")
    experiment = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        experiment.get("status") != "completed"
        or experiment.get("fold_id") != fold.fold_id
        or experiment.get("held_out_patient") != fold.test_patient_id
    ):
        raise ValueError("Robust result does not match the expected completed first fold")
    model_result = experiment.get("models", {}).get("logistic_regression")
    if not isinstance(model_result, dict):
        raise ValueError("Robust result has no completed logistic_regression model")
    threshold = float(model_result["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Robust inner-validation threshold is invalid")

    validation_y, validation_scores = _load_scores(score_dir / "validation_scores.npz")
    test_y, test_scores = _load_scores(score_dir / "outer_test_scores.npz")
    if inventory_path is None:
        inventory_path = Path(__file__).resolve().parents[3] / "docs" / "signal_inventory.csv"
    duration_by_path, duration_audit = load_recording_durations(inventory_path, recordings)

    validation = _evaluate_threshold(
        inner_split.validation_recordings, validation_y, validation_scores,
        processed_dir, threshold, "robust_inner_validation", duration_by_path,
    )
    outer_test = _evaluate_threshold(
        fold.test_recordings, test_y, test_scores,
        processed_dir, threshold, "robust_outer_test", duration_by_path,
    )
    result = {
        "experiment": "robust_feature_first_fold_event_evaluation",
        "version": "saved_robust_score_event_metrics_v1_exact_inventory_duration",
        "status": "completed",
        "source_experiment_result": str(result_path),
        "score_artifact_directory": str(score_dir),
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "model": "logistic_regression",
        "feature_version": experiment.get("feature_version"),
        "model_retrained": False,
        "features_reextracted": False,
        "threshold": threshold,
        "threshold_source": "robust experiment inner-validation F2 selection",
        "score_artifacts": {
            "validation": str(score_dir / "validation_scores.npz"),
            "outer_test": str(score_dir / "outer_test_scores.npz"),
        },
        "recording_duration_availability": "exact_inventory_duration_for_all_primary_recordings",
        "duration_source_audit": duration_audit,
        "alarm_policy": {
            "window_seconds": WINDOW_SECONDS,
            "merge_gap_seconds": 0.0,
            "positive_window_rule": "score >= robust inner-validation-selected threshold",
            "event_match": "any half-open interval overlap",
        },
        "validation": validation,
        "outer_test": outer_test,
        "limitations": [
            "Only the first LOPO fold is evaluated; this is not a 23-patient result.",
            "The threshold was selected on inner validation and was not tuned using outer-test labels.",
            "This evaluates the saved inner-fit model; it is not a refit on all outer-training patients.",
            "Event alarms use the same unsmoothed positive-window policy as the legacy comparison.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--result", type=Path,
        default=Path("results/baselines/robust_feature_first_fold_v1.json"),
    )
    parser.add_argument(
        "--score-dir", type=Path,
        default=Path("results/baselines/robust_feature_first_fold_v1/logistic_regression"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/baselines/robust_feature_first_fold_event_metrics_v1.json"),
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--inventory", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_robust_first_fold_events(
        args.audit, args.result, args.score_dir, args.output,
        processed_dir=args.processed_dir, inventory_path=args.inventory,
    )
    print(json.dumps({
        "status": result["status"],
        "model_retrained": result["model_retrained"],
        "features_reextracted": result["features_reextracted"],
        "output": str(args.output),
        "validation": result["validation"]["aggregate"],
        "outer_test": result["outer_test"]["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
