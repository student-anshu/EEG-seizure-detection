"""Evaluate all 22 saved patient-held-out OOF score folds at seizure-event level.

This is read-only post-processing: no model fitting, feature extraction,
threshold tuning, or access to the outer-test patient_01.
Run from EEG_Seizure_Detection:
    python -m scripts.evaluate_inner_oof_events --audit ..\\..\\docs\\full_annotation_audit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import PROCESSED_DIR, WINDOW_SECONDS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import processed_path_for_recording
from data_management.patient_mapping import Recording, load_recording_manifest
from evaluation.event_metrics import (
    aggregate_event_metrics,
    evaluate_recording_events,
    parse_seizure_intervals,
    TimeInterval,
)
from evaluation.recording_inventory import load_recording_durations


def validate_recording_offsets(
    offsets: list[dict], recordings: list[Recording], score_count: int
) -> None:
    """Require exact ordered identity and contiguous score slices."""
    if len(offsets) != len(recordings):
        raise ValueError(
            f"OOF recording-offset count mismatch: {len(offsets)} != {len(recordings)}"
        )
    cursor = 0
    for offset, recording in zip(offsets, recordings):
        if offset.get("relative_path") != recording.relative_path:
            raise ValueError(
                "OOF recording order/identity mismatch: "
                f"{offset.get('relative_path')} != {recording.relative_path}"
            )
        start, end = int(offset.get("start", -1)), int(offset.get("end", -1))
        if start != cursor or end <= start:
            raise ValueError(
                f"Non-contiguous or invalid OOF score offsets for {recording.relative_path}"
            )
        cursor = end
    if cursor != score_count:
        raise ValueError(
            f"OOF offsets cover {cursor} scores, but artifact contains {score_count}"
        )


def _patient_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append(row)
    result = []
    for patient_id, items in sorted(grouped.items()):
        events = [event for item in items for event in item["events"]]
        delays = [
            float(event["detection_delay_seconds"])
            for event in events if event["detected"]
        ]
        exposure_values = [item["non_seizure_duration_seconds"] for item in items]
        exposure = (
            float(sum(exposure_values))
            if all(value is not None for value in exposure_values) else None
        )
        false_alarms = sum(int(item["false_alarm_count"]) for item in items)
        event_count = sum(int(item["event_count"]) for item in items)
        detected = sum(int(item["detected_events"]) for item in items)
        result.append({
            "patient_id": patient_id,
            "recording_count": len(items),
            "event_count": event_count,
            "detected_events": detected,
            "missed_events": event_count - detected,
            "event_sensitivity": detected / event_count if event_count else None,
            "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
            "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
            "false_alarm_count": false_alarms,
            "non_seizure_duration_seconds": exposure,
            "false_alarms_per_hour": (
                false_alarms / (exposure / 3600.0)
                if exposure is not None and exposure > 0 else None
            ),
        })
    return result


def _evaluate_threshold(
    recordings: list[Recording],
    fold_data: dict[str, tuple[np.ndarray, np.ndarray, list[dict]]],
    duration_by_path: dict[str, float],
    processed_dir: Path,
    threshold: float,
    merge_gap_seconds: float,
) -> dict:
    rows = []
    for recording in recordings:
        scores, labels, offset = fold_data[recording.relative_path]
        path = processed_path_for_recording(processed_dir, recording)
        with np.load(path, allow_pickle=False) as saved:
            required = {"y", "starts", "subject", "source_file"}
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"{path.name} missing arrays: {sorted(missing)}")
            y = np.asarray(saved["y"], dtype=np.int8)
            starts = np.asarray(saved["starts"], dtype=np.float64)
            subject = str(saved["subject"])
            source_file = str(saved["source_file"])
        if subject != recording.subject or source_file != recording.edf_file:
            raise ValueError(f"Processed recording identity mismatch: {recording.relative_path}")
        if y.ndim != 1 or starts.shape != y.shape or len(y) != len(scores):
            raise ValueError(f"Processed window alignment mismatch: {recording.relative_path}")
        if not np.array_equal(labels, y):
            raise ValueError(f"OOF score labels disagree with processed labels: {recording.relative_path}")
        if not np.isfinite(starts).all() or (len(starts) > 1 and np.any(np.diff(starts) <= 0)):
            raise ValueError(f"Invalid window timestamps: {recording.relative_path}")

        duration = float(duration_by_path[recording.relative_path])
        seizures = parse_seizure_intervals(recording.seizure_intervals_seconds)
        # Clip annotations to true recording bounds; exposure subtracts their union.
        seizures = tuple(
            TimeInterval(max(0.0, event.start), min(duration, event.end))
            for event in seizures
            if min(duration, event.end) > max(0.0, event.start)
        )
        predictions = (scores >= threshold).astype(np.int8)
        metrics = evaluate_recording_events(
            starts, predictions, seizures,
            recording_duration_seconds=duration,
            window_seconds=WINDOW_SECONDS,
            merge_gap_seconds=merge_gap_seconds,
        )
        rows.append({
            "relative_path": recording.relative_path,
            "subject": recording.subject,
            "patient_id": recording.patient_id,
            "threshold": float(threshold),
            "window_count": int(len(y)),
            "positive_windows": int(np.sum(y == 1)),
            "predicted_positive_windows": int(np.sum(predictions == 1)),
            **metrics,
        })

    aggregate = aggregate_event_metrics(rows)
    all_delays = [
        float(event["detection_delay_seconds"])
        for row in rows for event in row["events"] if event["detected"]
    ]
    aggregate["median_detection_delay_seconds"] = (
        float(np.median(all_delays)) if all_delays else None
    )
    return {
        "threshold": float(threshold),
        "merge_gap_seconds": float(merge_gap_seconds),
        "aggregate": aggregate,
        "patient_summaries": _patient_summary(rows),
        "recordings": rows,
    }


def evaluate_inner_oof_events(
    audit_path: Path,
    inventory_path: Path,
    output_path: Path,
    score_root: Path,
    processed_dir: Path = PROCESSED_DIR,
    thresholds: list[float] | None = None,
    merge_gap_seconds: float = 0.0,
) -> dict:
    manifest = filter_primary_cohort(load_recording_manifest(audit_path))
    outer = build_lopo_folds(manifest)[0]
    if outer.test_patient_id != "patient_01":
        raise RuntimeError("Unexpected outer-test patient; refusing OOF event evaluation")
    if any(item.patient_id == "patient_01" for item in outer.train_recordings):
        raise RuntimeError("Outer-test patient_01 leaked into OOF cohort")
    recordings = list(outer.train_recordings)
    expected_patients = sorted({item.patient_id for item in recordings})
    if len(expected_patients) != 22 or len(recordings) != 608:
        raise RuntimeError("Unexpected OOF cohort dimensions")

    report_path = score_root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"OOF report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("patient_fold_count") != 22
        or report.get("outer_test_patient_excluded") != "patient_01"
        or report.get("window_count") != 652077
    ):
        raise ValueError("OOF report does not match the verified 22-fold inner cohort")
    report_folds = {item["patient_id"]: item for item in report.get("patient_folds", [])}
    if set(report_folds) != set(expected_patients):
        raise ValueError("OOF report patient-fold identities do not match the audit cohort")

    duration_by_path, duration_audit = load_recording_durations(inventory_path, manifest)
    fold_data: dict[str, tuple[np.ndarray, np.ndarray, list[dict]]] = {}
    fold_diagnostics = []
    fold_dir = score_root / "folds"

    for patient_id in expected_patients:
        artifact = fold_dir / f"{patient_id}.npz"
        if not artifact.is_file():
            raise FileNotFoundError(f"OOF score artifact missing: {artifact}")
        with np.load(artifact, allow_pickle=False) as saved:
            required = {"signature", "y_true", "scores", "recording_offsets"}
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"{artifact.name} missing arrays: {sorted(missing)}")
            signature = str(saved["signature"].item())
            y_true = np.asarray(saved["y_true"], dtype=np.int8)
            scores = np.asarray(saved["scores"], dtype=np.float64)
            offsets = json.loads(str(saved["recording_offsets"].item()))
        patient_recordings = [item for item in recordings if item.patient_id == patient_id]
        if not signature:
            raise ValueError(f"Empty OOF artifact signature: {artifact.name}")
        if y_true.ndim != 1 or scores.shape != y_true.shape or not len(scores):
            raise ValueError(f"Invalid OOF score/label arrays: {artifact.name}")
        if not np.isin(y_true, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError(f"Invalid labels or non-finite scores: {artifact.name}")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Non-probability OOF scores: {artifact.name}")
        validate_recording_offsets(offsets, patient_recordings, len(scores))
        if len(scores) != int(report_folds[patient_id]["window_count"]):
            raise ValueError(f"OOF fold length disagrees with report: {patient_id}")
        if int(np.sum(y_true == 1)) != int(report_folds[patient_id]["positive_windows"]):
            raise ValueError(f"OOF fold positive count disagrees with report: {patient_id}")

        for offset, recording in zip(offsets, patient_recordings):
            start, end = int(offset["start"]), int(offset["end"])
            fold_data[recording.relative_path] = (
                scores[start:end].copy(), y_true[start:end].copy(), offset
            )
        fold_diagnostics.append({
            "patient_id": patient_id,
            "recording_count": len(patient_recordings),
            "window_count": len(scores),
            "positive_windows": int(np.sum(y_true == 1)),
            "auroc_at_fold": report_folds[patient_id]["metrics_at_probability_0_5"].get("auroc"),
            "auprc_at_fold": report_folds[patient_id]["metrics_at_probability_0_5"].get("auprc"),
            "score_artifact": str(artifact),
            "signature": signature,
        })

    if set(fold_data) != {item.relative_path for item in recordings}:
        raise RuntimeError("OOF recording score coverage is incomplete")
    if thresholds is None:
        provisional = float(report["threshold"]["value"])
        thresholds = [0.5, provisional]
    normalized_thresholds = []
    for value in thresholds:
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid probability threshold: {value}")
        if not any(np.isclose(value, existing, rtol=0.0, atol=1e-12)
                   for existing in normalized_thresholds):
            normalized_thresholds.append(value)
    if not np.isfinite(merge_gap_seconds) or merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds must be finite and non-negative")

    evaluations = [
        _evaluate_threshold(
            recordings, fold_data, duration_by_path, processed_dir,
            threshold, merge_gap_seconds,
        )
        for threshold in normalized_thresholds
    ]
    result = {
        "experiment": "patient_grouped_inner_oof_event_evaluation",
        "version": "inner_oof_event_metrics_v1",
        "status": "completed",
        "model_retrained": False,
        "features_reextracted": False,
        "thresholds_selected": False,
        "outer_test_patient_loaded": False,
        "outer_test_patient_excluded": "patient_01",
        "patient_fold_count": 22,
        "recording_count": len(recordings),
        "window_count": int(report["window_count"]),
        "positive_windows": int(report["positive_windows"]),
        "score_source": str(score_root),
        "threshold_provenance": {
            "0.5": "fixed reference threshold",
            "pooled_oof_f2": "provisional pooled window-F2 threshold selected on these same OOF labels; descriptive only and not an unbiased event-performance estimate",
        },
        "thresholds_evaluated": normalized_thresholds,
        "alarm_policy": {
            "window_seconds": WINDOW_SECONDS,
            "merge_gap_seconds": float(merge_gap_seconds),
            "positive_window_rule": "score >= threshold",
            "event_match": "any half-open interval overlap",
            "false_alarm_rule": "alarm episode overlaps no annotated seizure",
            "false_alarm_rate_denominator": "exact recording duration minus union of clipped annotated seizure intervals",
        },
        "duration_source_audit": duration_audit,
        "fold_diagnostics": fold_diagnostics,
        "evaluations": evaluations,
        "limitations": [
            "These are inner OOF development results, not outer-test performance.",
            "The pooled OOF F2 threshold was selected using these same OOF labels; its event metrics are descriptive and optimistic for threshold selection.",
            "Probabilities come from 22 different fold models; cross-fold calibration comparability has not yet been established.",
            "No threshold or alarm policy was selected by this evaluator.",
            "This academic research prototype is not a medical device and is not for diagnosis or treatment decisions.",
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
        "--inventory", type=Path,
        default=Path(__file__).resolve().parents[3] / "docs" / "signal_inventory.csv",
    )
    parser.add_argument(
        "--score-root", type=Path, default=Path("cache/oof_patient_scores_v1"),
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/baselines/inner_oof_event_metrics_v1.json"),
    )
    parser.add_argument(
        "--threshold", type=float, action="append", dest="thresholds",
        help="Probability threshold to evaluate; repeat to evaluate multiple values. Defaults to 0.5 and the provisional pooled OOF F2 threshold.",
    )
    parser.add_argument("--merge-gap-seconds", type=float, default=0.0)
    args = parser.parse_args()
    result = evaluate_inner_oof_events(
        args.audit, args.inventory, args.output, args.score_root,
        processed_dir=args.processed_dir, thresholds=args.thresholds,
        merge_gap_seconds=args.merge_gap_seconds,
    )
    print(json.dumps({
        "status": result["status"],
        "model_retrained": result["model_retrained"],
        "outer_test_patient_loaded": result["outer_test_patient_loaded"],
        "output": str(args.output),
        "thresholds": [
            {
                "threshold": item["threshold"],
                "aggregate": item["aggregate"],
                "patients": len(item["patient_summaries"]),
            }
            for item in result["evaluations"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
