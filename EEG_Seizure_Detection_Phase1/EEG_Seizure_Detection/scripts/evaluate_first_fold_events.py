"""Evaluate saved first-fold scores at seizure-event level without retraining.

Run from the EEG_Seizure_Detection project directory:
    python -m scripts.evaluate_first_fold_events --audit ..\\..\\docs\\full_annotation_audit.csv

The script loads checksum-validated durable score checkpoints and existing
processed NPZ timestamps. It never fits a model or changes source artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from config import PROCESSED_DIR, RANDOM_STATE, WINDOW_SECONDS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import processed_path_for_recording
from data_management.patient_mapping import Recording, load_recording_manifest
from data_management.validation_split import build_patient_validation_split
from evaluation.event_metrics import (
    TimeInterval,
    aggregate_event_metrics,
    evaluate_recording_events,
    parse_seizure_intervals,
)
from evaluation.recording_inventory import load_recording_durations
from modeling.experiment_checkpoints import ModelStageCheckpointStore
from scripts.run_first_baseline_fold import (
    _build_checkpoint_context,
    _write_result,
)


def _add_event_score_diagnostics(
    events: list[dict],
    starts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    window_seconds: float,
    context_seconds: float = 30.0,
) -> None:
    """Attach score summaries for windows overlapping each event and its context."""
    for event in events:
        event_start = float(event["seizure_start_seconds"])
        event_end = float(event["seizure_end_seconds"])
        overlap = (starts < event_end) & (starts + window_seconds > event_start)
        context = (
            (starts < event_end + context_seconds)
            & (starts + window_seconds > event_start - context_seconds)
        )
        event_scores = scores[overlap]
        context_scores = scores[context]
        if event_scores.size:
            top_index = int(np.argmax(event_scores))
            event["overlapping_window_count"] = int(event_scores.size)
            event["max_overlapping_window_score"] = float(event_scores[top_index])
            event["median_overlapping_window_score"] = float(np.median(event_scores))
            event["top_scoring_window_start_seconds"] = float(starts[overlap][top_index])
            event["max_score_minus_threshold"] = float(event_scores[top_index] - threshold)
        else:
            event["overlapping_window_count"] = 0
            event["max_overlapping_window_score"] = None
            event["median_overlapping_window_score"] = None
            event["top_scoring_window_start_seconds"] = None
            event["max_score_minus_threshold"] = None
        event["context_seconds_each_side"] = float(context_seconds)
        event["context_window_count"] = int(context_scores.size)
        event["max_context_window_score"] = (
            float(np.max(context_scores)) if context_scores.size else None
        )


def _score_slices(
    recordings: Sequence[Recording],
    y_scores: np.ndarray,
    score_values: np.ndarray,
    processed_dir: Path,
    threshold: float,
    split_name: str,
    duration_by_path: dict[str, float] | None = None,
) -> list[dict]:
    expected_total = 0
    lengths: list[int] = []
    arrays: list[tuple[Recording, np.ndarray, np.ndarray]] = []
    for recording in recordings:
        path = processed_path_for_recording(processed_dir, recording)
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed recording: {path}")
        with np.load(path, allow_pickle=False) as saved:
            required = {"y", "starts"}
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"{path.name} missing arrays: {sorted(missing)}")
            labels = np.asarray(saved["y"], dtype=np.int8)
            starts = np.asarray(saved["starts"], dtype=np.float64)
        if labels.ndim != 1 or starts.shape != labels.shape or len(labels) == 0:
            raise ValueError(f"{path.name} has invalid label/timestamp alignment")
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(starts).all():
            raise ValueError(f"{path.name} contains invalid labels or timestamps")
        lengths.append(len(labels))
        expected_total += len(labels)
        arrays.append((recording, labels, starts))

    if len(y_scores) != expected_total or len(score_values) != expected_total:
        raise ValueError(
            f"{split_name} score artifact length mismatch: expected {expected_total}, "
            f"got labels={len(y_scores)}, scores={len(score_values)}"
        )

    output: list[dict] = []
    cursor = 0
    for recording, labels, starts in arrays:
        next_cursor = cursor + len(labels)
        score_labels = y_scores[cursor:next_cursor]
        scores = score_values[cursor:next_cursor]
        if not np.array_equal(score_labels, labels):
            raise ValueError(
                f"{split_name} score labels do not align with {recording.relative_path}; "
                "refusing to guess score-to-recording boundaries"
            )
        predictions = (scores >= threshold).astype(np.int8)
        seizures = parse_seizure_intervals(recording.seizure_intervals_seconds)
        duration = duration_by_path[recording.relative_path] if duration_by_path is not None else None
        if duration is not None:
            seizures = tuple(
                TimeInterval(max(0.0, item.start), min(duration, item.end))
                for item in seizures
                if min(duration, item.end) > max(0.0, item.start)
            )
        metrics = evaluate_recording_events(
            starts,
            predictions,
            seizures,
            recording_duration_seconds=duration,
            window_seconds=WINDOW_SECONDS,
        )
        _add_event_score_diagnostics(
            metrics["events"], starts, scores, threshold, WINDOW_SECONDS
        )
        output.append({
            "relative_path": recording.relative_path,
            "subject": recording.subject,
            "patient_id": recording.patient_id,
            "threshold": float(threshold),
            "window_count": int(len(labels)),
            "positive_windows": int(np.sum(labels == 1)),
            "predicted_positive_windows": int(np.sum(predictions == 1)),
            **metrics,
        })
        cursor = next_cursor

    if cursor != expected_total:
        raise RuntimeError("Internal score slicing cursor mismatch")
    return output


def _patient_summaries(recording_results: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in recording_results:
        grouped.setdefault(item["patient_id"], []).append(item)
    summaries = []
    for patient_id, items in sorted(grouped.items()):
        event_count = sum(item["event_count"] for item in items)
        detected = sum(item["detected_events"] for item in items)
        delays = [
            (item["mean_detection_delay_seconds"], item["detected_events"])
            for item in items
            if item["mean_detection_delay_seconds"] is not None
            and item["detected_events"] > 0
        ]
        delay_total = sum(weight for _, weight in delays)
        summaries.append({
            "patient_id": patient_id,
            "recording_count": len(items),
            "event_count": event_count,
            "detected_events": detected,
            "missed_events": sum(item["missed_events"] for item in items),
            "event_sensitivity": detected / event_count if event_count else None,
            "mean_detection_delay_seconds": (
                sum(value * weight for value, weight in delays) / delay_total
                if delay_total else None
            ),
            "false_alarm_count": sum(item["false_alarm_count"] for item in items),
            "non_seizure_duration_seconds": (
                sum(float(item["non_seizure_duration_seconds"]) for item in items)
                if all(item.get("non_seizure_duration_seconds") is not None for item in items)
                else None
            ),
            "false_alarms_per_hour": None,
        })
        exposure = summaries[-1]["non_seizure_duration_seconds"]
        if exposure is not None and exposure > 0:
            summaries[-1]["false_alarms_per_hour"] = (
                summaries[-1]["false_alarm_count"] / (exposure / 3600.0)
            )
    return summaries


def _evaluate_threshold(
    recordings: Sequence[Recording],
    y_true: np.ndarray,
    scores: np.ndarray,
    processed_dir: Path,
    threshold: float,
    split_name: str,
    duration_by_path: dict[str, float] | None = None,
) -> dict:
    by_recording = _score_slices(
        recordings, y_true, scores, processed_dir, threshold, split_name, duration_by_path
    )
    return {
        "threshold": float(threshold),
        "aggregate": aggregate_event_metrics(by_recording),
        "patient_summaries": _patient_summaries(by_recording),
        "recordings": by_recording,
    }


def evaluate_first_fold_events(
    audit_path: Path,
    result_path: Path,
    output_path: Path,
    processed_dir: Path = PROCESSED_DIR,
    checkpoint_dir: Path | None = None,
    inventory_path: Path | None = None,
) -> dict:
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    folds = build_lopo_folds(recordings)
    fold = folds[0]
    inner_split = build_patient_validation_split(
        fold.train_recordings, validation_fraction=0.2, random_state=RANDOM_STATE
    )
    context = _build_checkpoint_context(fold, inner_split, RANDOM_STATE)
    root = checkpoint_dir or (
        result_path.parent / "checkpoints"
        / "first_fold_patient_01_logistic_regression_f1_f2_v1"
    )
    store = ModelStageCheckpointStore(root, context)
    validation_scores = store.load_scores("inner_validation_scores")
    outer_scores = store.load_scores("outer_test_scores")
    if validation_scores is None or outer_scores is None:
        raise RuntimeError(
            "Required durable score artifacts are missing, corrupt, or have an "
            "incompatible signature. No model training was attempted."
        )
    if not result_path.is_file():
        raise FileNotFoundError(f"First-fold result JSON not found: {result_path}")
    baseline = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "completed"
        or baseline.get("fold_id") != fold.fold_id
        or baseline.get("held_out_patient") != fold.test_patient_id
    ):
        raise ValueError("Baseline result does not match the expected completed first fold")

    selection = baseline["threshold_selection"]
    thresholds = {
        "validation_f1": float(selection["validation_threshold"]),
        "validation_f2": float(selection["f2_validation_threshold"]),
    }
    validation_y, validation_values = validation_scores
    test_y, test_values = outer_scores
    if inventory_path is None:
        repository_root = Path(__file__).resolve().parents[3]
        inventory_path = repository_root / "docs" / "signal_inventory.csv"
    duration_by_path, duration_audit = load_recording_durations(inventory_path, recordings)

    results = {
        "experiment": "first_fold_event_evaluation",
        "version": "saved_score_event_metrics_v2_exact_inventory_duration",
        "status": "completed",
        "source_baseline_result": str(result_path),
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "model_retrained": False,
        "score_artifacts_checksum_validated": True,
        "recording_duration_availability": "exact_inventory_duration_for_all_primary_recordings",
        "duration_source_audit": duration_audit,
        "duration_policy": (
            "Exact duration is joined by normalized relative_path and cross-checked "
            "against subject and EDF filename. Non-seizure exposure subtracts the "
            "union of seizure annotations clipped to recording bounds."
        ),
        "alarm_policy": {
            "window_seconds": WINDOW_SECONDS,
            "merge_gap_seconds": 0.0,
            "positive_window_rule": "score >= inner-validation-selected threshold",
            "event_match": "any half-open interval overlap",
        },
        "thresholds": thresholds,
        "validation": {
            "f1_threshold": _evaluate_threshold(
                inner_split.validation_recordings, validation_y, validation_values,
                processed_dir, thresholds["validation_f1"], "inner_validation", duration_by_path,
            ),
            "f2_threshold": _evaluate_threshold(
                inner_split.validation_recordings, validation_y, validation_values,
                processed_dir, thresholds["validation_f2"], "inner_validation", duration_by_path,
            ),
        },
        "outer_test": {
            "f1_threshold": _evaluate_threshold(
                fold.test_recordings, test_y, test_values,
                processed_dir, thresholds["validation_f1"], "outer_test", duration_by_path,
            ),
            "f2_threshold": _evaluate_threshold(
                fold.test_recordings, test_y, test_values,
                processed_dir, thresholds["validation_f2"], "outer_test", duration_by_path,
            ),
        },
        "limitations": [
            "Only the first LOPO fold is evaluated; this is not a 23-patient result.",
            "Thresholds were selected on inner validation and were not tuned using outer-test labels.",
        ],
    }
    _write_result(output_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate first-fold event metrics from saved scores; no retraining"
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/baselines/first_fold_patient_01_f1_f2_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/first_fold_patient_01_event_metrics_v2.json"),
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_first_fold_events(
        args.audit, args.result, args.output,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        inventory_path=args.inventory,
    )
    print(json.dumps({
        "status": result["status"],
        "model_retrained": result["model_retrained"],
        "output": str(args.output),
        "validation_f1": result["validation"]["f1_threshold"]["aggregate"],
        "outer_test_f1": result["outer_test"]["f1_threshold"]["aggregate"],
        "outer_test_f2": result["outer_test"]["f2_threshold"]["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
