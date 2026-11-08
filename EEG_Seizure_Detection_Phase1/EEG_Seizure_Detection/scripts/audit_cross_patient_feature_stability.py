"""Descriptive cross-patient robust-feature distribution and stability audit.

Uses only the existing first-outer-fold inner-fit/inner-validation robust
feature caches and the already completed 22-patient OOF report. It does not
re-extract features, fit models, tune thresholds, or open outer-test patient_01.

Run from the EEG project root:
    python -m scripts.audit_cross_patient_feature_stability --audit ..\..\docs\full_annotation_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PREFERRED_CHANNELS, RANDOM_STATE
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import processed_path_for_recording
from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split
from feature_selection.robust_feature_extraction import ROBUST_FEATURE_NAMES

FEATURES_PER_CHANNEL = len(ROBUST_FEATURE_NAMES)
FEATURE_COUNT = len(PREFERRED_CHANNELS) * FEATURES_PER_CHANNEL
EXPECTED_INNER_PATIENTS = 22
ROBUST_ROOT = ROOT / "cache" / "robust_feature_first_fold_v1"
OOF_ROOT = ROOT / "cache" / "oof_patient_scores_v1"
DEFAULT_OUTPUT = ROOT / "results" / "diagnostics" / "cross_patient_feature_stability_v1"


def summarize_vector(values: np.ndarray) -> dict:
    """Return descriptive quantiles for a finite, non-empty 1-D vector."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("values must be a non-empty finite 1-D array")
    q25, median, q75 = np.percentile(data, [25, 50, 75])
    return {
        "window_count": int(data.size),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def robust_patient_effect(patient_median: float, peer_medians: np.ndarray) -> tuple[float | None, float, float]:
    """Compare one patient with equal-weight peer patient medians.

    The scale is the peer-patient IQR. A zero IQR yields no standardized
    effect rather than an unstable/infinite value.
    """
    peers = np.asarray(peer_medians, dtype=np.float64)
    peers = peers[np.isfinite(peers)]
    if peers.ndim != 1 or peers.size < 3 or not np.isfinite(patient_median):
        return None, float("nan"), float("nan")
    center = float(np.median(peers))
    q25, q75 = np.percentile(peers, [25, 75])
    spread = float(q75 - q25)
    effect = float((patient_median - center) / spread) if spread > 1e-12 else None
    return effect, center, spread


def _load_feature_split(directory: Path):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing robust feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Robust feature cache is incomplete: {directory}")
    X = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
    if (manifest.get("shape") != list(X.shape) or X.ndim != 2
            or X.shape[1] != FEATURE_COUNT or X.dtype != np.float32
            or y.shape != (len(X),) or y.dtype != np.int8
            or not np.isin(y, (0, 1)).all()
            or not np.isfinite(X).all()):
        raise RuntimeError(f"Robust feature cache integrity failure: {directory}")
    return X, y, manifest


def _build_patient_segments(outer_train, audit_path: Path):
    split = build_patient_validation_split(
        outer_train, validation_fraction=0.2, random_state=RANDOM_STATE
    )
    fit_x, fit_y, fit_manifest = _load_feature_split(ROBUST_ROOT / "inner_fit")
    val_x, val_y, val_manifest = _load_feature_split(ROBUST_ROOT / "inner_validation")
    segments: dict[str, list[dict]] = {}
    seen_paths: set[str] = set()

    for split_name, source_x, source_y, records in (
        ("inner_fit", fit_x, fit_y, split.train_recordings),
        ("inner_validation", val_x, val_y, split.validation_recordings),
    ):
        cursor = 0
        for rec in records:
            if rec.patient_id == "patient_01":
                raise RuntimeError("Sealed patient_01 appeared in a feature-cache split")
            if rec.relative_path in seen_paths:
                raise RuntimeError(f"Duplicate recording in cache ordering: {rec.relative_path}")
            source = processed_path_for_recording(ROOT / "processed", rec)
            with np.load(source, allow_pickle=False) as saved:
                labels = np.asarray(saved["y"], dtype=np.int8)
            end = cursor + len(labels)
            if end > len(source_y) or not np.array_equal(source_y[cursor:end], labels):
                raise RuntimeError(f"Feature/label row alignment mismatch: {rec.relative_path}")
            if bool(np.any(labels == 1)) != (rec.seizure_count > 0):
                raise RuntimeError(f"Annotation/label mismatch: {rec.relative_path}")
            segments.setdefault(rec.patient_id, []).append({
                "recording": rec.relative_path,
                "patient_id": rec.patient_id,
                "seizure_count": int(rec.seizure_count),
                "X": source_x,
                "y": source_y,
                "start": cursor,
                "end": end,
                "split": split_name,
            })
            seen_paths.add(rec.relative_path)
            cursor = end
        if cursor != len(source_y):
            raise RuntimeError(f"{split_name} cache rows do not match manifest recordings")

    expected_paths = {rec.relative_path for rec in outer_train}
    if seen_paths != expected_paths:
        missing = sorted(expected_paths - seen_paths)
        extra = sorted(seen_paths - expected_paths)
        raise RuntimeError(f"Feature-cache cohort mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if "patient_01" in segments or len(segments) != EXPECTED_INNER_PATIENTS:
        raise RuntimeError("Expected exactly 22 inner patients and no patient_01")
    return segments, fit_manifest, val_manifest


def _collect_patient_label_values(patient_segments, feature_index: int, label: int):
    chunks = []
    positive_recordings = 0
    recording_count = 0
    for segment in patient_segments:
        labels = np.asarray(segment["y"][segment["start"]:segment["end"]], dtype=np.int8)
        mask = labels == label
        if np.any(mask):
            values = np.asarray(
                segment["X"][segment["start"]:segment["end"], feature_index],
                dtype=np.float32,
            )
            chunks.append(values[mask])
            recording_count += 1
            if segment["seizure_count"] > 0 and label == 1:
                positive_recordings += 1
    if not chunks:
        return np.empty(0, dtype=np.float32), 0, 0
    return np.concatenate(chunks), recording_count, positive_recordings


def run(audit_path: Path, output_dir: Path = DEFAULT_OUTPUT):
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit directory: {output_dir}. "
            "Choose a new --output path to preserve previous results."
        )

    manifest_recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    outer = build_lopo_folds(manifest_recordings)[0]
    if outer.test_patient_id != "patient_01" or len(outer.train_recordings) != 608:
        raise RuntimeError("Unexpected first outer fold; stopping before data access")
    if any(rec.patient_id == "patient_01" for rec in outer.train_recordings):
        raise RuntimeError("Outer-test patient leaked into inner diagnostic cohort")

    segments, fit_manifest, val_manifest = _build_patient_segments(
        outer.train_recordings, audit_path
    )
    oof_report_path = OOF_ROOT / "report.json"
    if not oof_report_path.is_file():
        raise FileNotFoundError(f"Completed OOF report required: {oof_report_path}")
    oof_report = json.loads(oof_report_path.read_text(encoding="utf-8"))
    if (oof_report.get("patient_fold_count") != EXPECTED_INNER_PATIENTS
            or oof_report.get("outer_test_patient_excluded") != "patient_01"
            or oof_report.get("window_count") != 652077):
        raise RuntimeError("OOF report does not match the expected sealed 22-patient cohort")
    oof_metrics = {
        row["patient_id"]: row.get("metrics_at_probability_0_5", {})
        for row in oof_report.get("patient_folds", [])
    }
    if set(oof_metrics) != set(segments):
        raise RuntimeError("OOF report patient IDs do not match the feature-cache cohort")

    patients = sorted(segments)
    patient_summary = []
    for patient in patients:
        recs = segments[patient]
        positive_windows = 0
        total_windows = 0
        positive_recordings = 0
        for rec in recs:
            labels = rec["y"][rec["start"]:rec["end"]]
            total_windows += len(labels)
            positive_windows += int(np.sum(labels == 1))
            positive_recordings += int(rec["seizure_count"] > 0)
        metrics = oof_metrics[patient]
        patient_summary.append({
            "patient_id": patient,
            "recording_count": len(recs),
            "window_count": total_windows,
            "positive_windows": positive_windows,
            "positive_recording_count": positive_recordings,
            "positive_window_fraction": positive_windows / total_windows if total_windows else None,
            "oof_auroc_at_probability_0_5": metrics.get("auroc"),
            "oof_auprc_at_probability_0_5": metrics.get("auprc"),
        })

    # Compute patient-level medians first. Cross-patient reference statistics
    # give every patient equal weight instead of letting long recordings dominate.
    feature_rows = []
    distributions: dict[tuple[str, int], dict[str, dict]] = {}
    for channel_index, channel in enumerate(PREFERRED_CHANNELS):
        for feature_offset, feature_name in enumerate(ROBUST_FEATURE_NAMES):
            feature_index = channel_index * FEATURES_PER_CHANNEL + feature_offset
            for label in (0, 1):
                class_key = (f"{channel}|{feature_name}", label)
                patient_stats = {}
                for patient in patients:
                    values, recording_count, positive_recording_count = (
                        _collect_patient_label_values(segments[patient], feature_index, label)
                    )
                    stats = summarize_vector(values) if len(values) else None
                    patient_stats[patient] = {
                        "stats": stats,
                        "recording_count": recording_count,
                        "positive_recording_count": positive_recording_count,
                    }
                distributions[class_key] = patient_stats

                for patient in patients:
                    stats = patient_stats[patient]["stats"]
                    peer_medians = np.asarray([
                        patient_stats[peer]["stats"]["median"]
                        for peer in patients
                        if peer != patient and patient_stats[peer]["stats"] is not None
                    ], dtype=np.float64)
                    if stats is None:
                        effect, peer_center, peer_iqr = None, None, None
                    else:
                        effect_value, center, spread = robust_patient_effect(
                            stats["median"], peer_medians
                        )
                        effect = effect_value
                        peer_center = center if np.isfinite(center) else None
                        peer_iqr = spread if np.isfinite(spread) else None
                    feature_rows.append({
                        "patient_id": patient,
                        "label": "seizure" if label == 1 else "non_seizure",
                        "channel": channel,
                        "feature": feature_name,
                        "window_count": stats["window_count"] if stats else 0,
                        "recording_count_with_class": patient_stats[patient]["recording_count"],
                        "positive_recording_count": patient_stats[patient]["positive_recording_count"],
                        "q25": stats["q25"] if stats else None,
                        "median": stats["median"] if stats else None,
                        "q75": stats["q75"] if stats else None,
                        "iqr": stats["iqr"] if stats else None,
                        "peer_patient_median_of_medians": peer_center,
                        "peer_patient_iqr_of_medians": peer_iqr,
                        "patient_median_shift_in_peer_iqrs": effect,
                        "peer_patient_count": int(np.count_nonzero(np.isfinite(peer_medians))),
                    })

    output_dir.mkdir(parents=True, exist_ok=False)
    feature_csv = output_dir / "patient_feature_distributions.csv"
    with feature_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    patient_csv = output_dir / "patient_cohort_summary.csv"
    with patient_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_summary[0]))
        writer.writeheader()
        writer.writerows(patient_summary)

    # Rank patient-feature-label cells by absolute shift. This is a screening
    # table only; it is not a feature-selection recommendation.
    ranked = sorted(
        (row for row in feature_rows if row["patient_median_shift_in_peer_iqrs"] is not None),
        key=lambda row: abs(row["patient_median_shift_in_peer_iqrs"]),
        reverse=True,
    )
    report = {
        "status": "completed",
        "diagnostic_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_test_patient_excluded": "patient_01",
        "outer_test_patient_loaded": False,
        "models_trained": False,
        "thresholds_tuned": False,
        "feature_selection_performed": False,
        "cohort": {
            "inner_patients": len(patients),
            "recordings": len(outer.train_recordings),
            "windows": int(sum(row["window_count"] for row in patient_summary)),
            "positive_windows": int(sum(row["positive_windows"] for row in patient_summary)),
            "feature_count": FEATURE_COUNT,
            "channel_count": len(PREFERRED_CHANNELS),
        },
        "cache_signatures": {
            "inner_fit": fit_manifest.get("signature"),
            "inner_validation": val_manifest.get("signature"),
        },
        "method": {
            "feature_source": "existing robust_feature_first_fold_v1 inner_fit and inner_validation caches",
            "row_alignment": "Every recording label vector checked against its processed NPZ before inclusion",
            "patient_weighting": "Cross-patient reference uses one median per patient, not pooled windows",
            "shift_metric": "Patient median minus peer-patient median, divided by peer-patient IQR",
            "zero_peer_iqr": "Effect left null; no division by zero",
            "window_dependence": "10-second windows with 5-second stride overlap and are correlated",
        },
        "oof_ranking_context": {
            "source": "existing inner OOF report; descriptive only",
            "patient_metrics": oof_metrics,
        },
        "largest_descriptive_shifts": ranked[:30],
        "files": {
            "patient_feature_distributions": str(feature_csv.relative_to(ROOT)),
            "patient_cohort_summary": str(patient_csv.relative_to(ROOT)),
        },
        "limitations": [
            "This is a descriptive distribution audit, not an independent statistical test.",
            "The feature cache uses the original first-fold scaler; some inner-fit patients contributed to that scaler. Do not interpret these summaries as unbiased held-out performance.",
            "Window overlap means window counts are not independent sample sizes.",
            "Seizure-positive windows and recordings are sparse and uneven across patients.",
            "A large shift does not establish a biological cause or prove covariate shift is responsible for poor ranking.",
            "No feature removal, retraining, threshold tuning, or outer-test access occurred.",
        ],
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed cross-patient feature audit: {len(patients)} patients, {len(feature_rows):,} patient-feature-class rows", flush=True)
    print(f"Saved: {output_dir}", flush=True)
    print("No model retraining, threshold tuning, feature selection, or patient_01 access.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path,
        default=ROOT.parent.parent / "docs" / "full_annotation_audit.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.audit, args.output)


if __name__ == "__main__":
    main()
