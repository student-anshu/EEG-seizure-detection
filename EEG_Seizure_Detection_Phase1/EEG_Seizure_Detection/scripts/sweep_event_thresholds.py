"""Validation-only event-threshold sweep over saved first-fold scores.

No model fitting, feature extraction, or outer-test threshold selection occurs.
Run from the EEG_Seizure_Detection directory:
    python -m scripts.sweep_event_thresholds --audit ..\\..\\docs\\full_annotation_audit.csv
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
from evaluation.event_metrics import aggregate_event_metrics, evaluate_recording_events, parse_seizure_intervals
from modeling.experiment_checkpoints import ModelStageCheckpointStore
from scripts.run_first_baseline_fold import _build_checkpoint_context, _write_result
from scripts.evaluate_first_fold_events import _patient_summaries

SWEEP_VERSION = "validation_event_f2_macro_patient_v1"
QUANTILE_GRID = (0.95, 1.0, 121)


def event_f2_proxy(detected_events: int, event_count: int, false_alarm_count: int) -> float:
    """F2 using event recall and event-vs-false-alarm-episode precision.

    This is explicitly a proxy: without exact recording exposure, it is not
    false alarms/hour and must not be presented as a clinical operating metric.
    """
    if event_count <= 0:
        return 0.0
    recall = detected_events / event_count
    precision = (
        detected_events / (detected_events + false_alarm_count)
        if detected_events + false_alarm_count else 0.0
    )
    denominator = 4.0 * precision + recall
    return (5.0 * precision * recall / denominator) if denominator else 0.0


def _load_recording_score_data(
    recordings: Sequence[Recording],
    y_scores: np.ndarray,
    score_values: np.ndarray,
    processed_dir: Path,
    split_name: str,
) -> list[dict]:
    """Load NPZ labels/timestamps once and verify saved-score alignment."""
    total = 0
    loaded = []
    for recording in recordings:
        path = processed_path_for_recording(processed_dir, recording)
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed recording: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if not {"y", "starts"}.issubset(saved.files):
                raise ValueError(f"{path.name} must contain y and starts")
            labels = np.asarray(saved["y"], dtype=np.int8)
            starts = np.asarray(saved["starts"], dtype=np.float64)
        if labels.ndim != 1 or starts.shape != labels.shape or not len(labels):
            raise ValueError(f"{path.name} has invalid label/timestamp alignment")
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(starts).all():
            raise ValueError(f"{path.name} has invalid labels or timestamps")
        loaded.append((recording, labels, starts))
        total += len(labels)

    if len(y_scores) != total or len(score_values) != total:
        raise ValueError(
            f"{split_name} score lengths mismatch: expected {total}, "
            f"got labels={len(y_scores)}, scores={len(score_values)}"
        )

    output = []
    cursor = 0
    for recording, labels, starts in loaded:
        end = cursor + len(labels)
        if not np.array_equal(y_scores[cursor:end], labels):
            raise ValueError(
                f"{split_name} score labels do not align with "
                f"{recording.relative_path}; refusing to continue"
            )
        scores = np.asarray(score_values[cursor:end], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError(f"{split_name} scores contain non-finite values")
        output.append({
            "recording": recording,
            "labels": labels,
            "starts": starts,
            "scores": scores,
            "seizures": parse_seizure_intervals(recording.seizure_intervals_seconds),
        })
        cursor = end
    return output


def _evaluate_loaded(
    data: Sequence[dict], threshold: float, *, include_recordings: bool = False
) -> dict:
    rows = []
    for item in data:
        predictions = (item["scores"] >= threshold).astype(np.int8)
        metrics = evaluate_recording_events(
            item["starts"], predictions, item["seizures"],
            recording_duration_seconds=None, window_seconds=WINDOW_SECONDS,
        )
        recording = item["recording"]
        metrics.pop("alarms", None)
        rows.append({
            "relative_path": recording.relative_path,
            "subject": recording.subject,
            "patient_id": recording.patient_id,
            **metrics,
        })

    patients = _patient_summaries(rows)
    seizure_patients = [p for p in patients if p["event_count"] > 0]
    for patient in seizure_patients:
        patient["event_f2_proxy"] = event_f2_proxy(
            patient["detected_events"], patient["event_count"],
            patient["false_alarm_count"],
        )
    macro_f2 = (
        float(np.mean([p["event_f2_proxy"] for p in seizure_patients]))
        if seizure_patients else 0.0
    )
    macro_sensitivity = (
        float(np.mean([p["event_sensitivity"] for p in seizure_patients]))
        if seizure_patients else 0.0
    )
    aggregate = aggregate_event_metrics(rows)
    return {
        "aggregate": aggregate,
        "patient_summaries": patients,
        "macro_patient_event_f2_proxy": macro_f2,
        "macro_patient_event_sensitivity": macro_sensitivity,
        "seizure_bearing_patient_count": len(seizure_patients),
        "recordings": rows if include_recordings else None,
    }


def _candidate_thresholds(scores: np.ndarray, baseline_thresholds: Sequence[float]) -> list[float]:
    low_q, high_q, count = QUANTILE_GRID
    quantiles = np.linspace(low_q, high_q, count)
    candidates = np.quantile(scores, quantiles).tolist()
    candidates.extend(float(value) for value in baseline_thresholds)
    return sorted({float(value) for value in candidates if np.isfinite(value)})


def _select_candidate(candidate_rows: Sequence[dict]) -> dict:
    if not candidate_rows:
        raise ValueError("No threshold candidates were evaluated")
    # Deterministic tie-breaks: higher macro sensitivity, fewer false alarms,
    # then higher threshold. All objective values come from inner validation.
    return max(
        candidate_rows,
        key=lambda row: (
            row["macro_patient_event_f2_proxy"],
            row["macro_patient_event_sensitivity"],
            -row["aggregate"]["false_alarm_count"],
            row["threshold"],
        ),
    )


def run_sweep(
    audit_path: Path,
    baseline_path: Path,
    output_path: Path,
    processed_dir: Path = PROCESSED_DIR,
    checkpoint_dir: Path | None = None,
) -> dict:
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    fold = build_lopo_folds(recordings)[0]
    split = build_patient_validation_split(
        fold.train_recordings, validation_fraction=0.2, random_state=RANDOM_STATE
    )
    context = _build_checkpoint_context(fold, split, RANDOM_STATE)
    root = checkpoint_dir or (
        baseline_path.parent / "checkpoints"
        / "first_fold_patient_01_logistic_regression_f1_f2_v1"
    )
    store = ModelStageCheckpointStore(root, context)
    validation_scores = store.load_scores("inner_validation_scores")
    test_scores = store.load_scores("outer_test_scores")
    if validation_scores is None or test_scores is None:
        raise RuntimeError("Checksum-valid saved validation/test scores are required; no training was attempted")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline result not found: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or baseline.get("held_out_patient") != fold.test_patient_id:
        raise ValueError("Baseline result does not match the completed first LOPO fold")

    val_y, val_scores = validation_scores
    test_y, test_values = test_scores
    val_data = _load_recording_score_data(
        split.validation_recordings, val_y, val_scores, processed_dir, "inner_validation"
    )
    test_data = _load_recording_score_data(
        fold.test_recordings, test_y, test_values, processed_dir, "outer_test"
    )

    selection = baseline["threshold_selection"]
    existing = {
        "validation_f1": float(selection["validation_threshold"]),
        "validation_f2": float(selection["f2_validation_threshold"]),
    }
    thresholds = _candidate_thresholds(val_scores, list(existing.values()))
    candidate_rows = []
    for index, threshold in enumerate(thresholds, start=1):
        evaluated = _evaluate_loaded(val_data, threshold)
        candidate_rows.append({
            "rank_candidate": index,
            "threshold": threshold,
            "aggregate": evaluated["aggregate"],
            "macro_patient_event_f2_proxy": evaluated["macro_patient_event_f2_proxy"],
            "macro_patient_event_sensitivity": evaluated["macro_patient_event_sensitivity"],
            "seizure_bearing_patient_count": evaluated["seizure_bearing_patient_count"],
        })

    selected = _select_candidate(candidate_rows)
    selected_threshold = float(selected["threshold"])
    # The outer-test set is evaluated once, after selection is frozen.
    outer_selected = _evaluate_loaded(
        test_data, selected_threshold, include_recordings=True
    )
    outer_existing_f1 = _evaluate_loaded(test_data, existing["validation_f1"])
    outer_existing_f2 = _evaluate_loaded(test_data, existing["validation_f2"])
    validation_selected_detail = _evaluate_loaded(
        val_data, selected_threshold, include_recordings=True
    )

    result = {
        "experiment": "first_fold_validation_event_threshold_sweep",
        "version": SWEEP_VERSION,
        "status": "completed",
        "model_retrained": False,
        "score_artifacts_checksum_validated": True,
        "fold_id": fold.fold_id,
        "held_out_patient": fold.test_patient_id,
        "threshold_selection_split": "inner_validation_only",
        "outer_test_used_for_selection": False,
        "objective": (
            "Maximize macro-average patient event F2 proxy on seizure-bearing "
            "inner-validation patients. Event precision proxy = detected seizure "
            "events / (detected seizure events + false-alarm episodes)."
        ),
        "limitations": [
            "False-alarm episodes are reported as counts; false alarms/hour is unavailable because exact recording durations are absent.",
            "The event F2 proxy is an exploratory threshold-selection objective, not a clinical false-alarm-rate metric.",
            "Only the first LOPO fold is evaluated; the result is not a 23-patient estimate.",
            "Threshold selection uses only the three seizure-bearing inner-validation patients in this fold.",
        ],
        "candidate_generation": {
            "method": "score quantiles from 0.95 through 1.00, plus existing validation F1/F2 thresholds",
            "candidate_count": len(thresholds),
        },
        "existing_thresholds": existing,
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": validation_selected_detail,
        "selected_validation_objective": {
            "macro_patient_event_f2_proxy": selected["macro_patient_event_f2_proxy"],
            "macro_patient_event_sensitivity": selected["macro_patient_event_sensitivity"],
            "false_alarm_count": selected["aggregate"]["false_alarm_count"],
        },
        "validation_candidate_sweep": candidate_rows,
        "outer_test_comparison_after_freeze": {
            "selected_validation_threshold": outer_selected,
            "existing_validation_f1_threshold": outer_existing_f1,
            "existing_validation_f2_threshold": outer_existing_f2,
        },
    }
    _write_result(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep event thresholds using saved inner-validation scores only")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("results/baselines/first_fold_patient_01_f1_f2_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines/first_fold_event_threshold_sweep_v1.json"))
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_sweep(
        args.audit, args.baseline, args.output,
        processed_dir=args.processed_dir, checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "model_retrained": result["model_retrained"],
        "candidate_count": result["candidate_generation"]["candidate_count"],
        "existing_thresholds": result["existing_thresholds"],
        "selected_threshold": result["selected_threshold"],
        "selected_validation_objective": result["selected_validation_objective"],
        "outer_test_used_for_selection": result["outer_test_used_for_selection"],
        "outer_test_selected": result["outer_test_comparison_after_freeze"]["selected_validation_threshold"]["aggregate"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
