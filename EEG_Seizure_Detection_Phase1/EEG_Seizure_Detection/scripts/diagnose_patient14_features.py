"""Read-only patient-14 robust-feature and affine-invariance diagnostic.

This diagnostic does NOT fit a model, tune thresholds, or modify OOF artifacts.
It recomputes patient_14 features with the exact 21-patient channel scaler used
by that OOF fold, then compares the nine theoretically affine-invariant
features against the existing first-fold feature cache. It also measures
within-recording univariate feature separation for seizure-containing
recordings. Patient_01 remains excluded.

Run from EEG_Seizure_Detection:
    python -m scripts.diagnose_patient14_features --audit ..\\..\\docs\\full_annotation_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, PREFERRED_CHANNELS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import load_recording_windows
from data_management.patient_mapping import load_recording_manifest
from feature_selection.robust_feature_extraction import (
    ROBUST_FEATURE_NAMES, extract_robust_features,
)
from scripts.run_inner_patient_oof import (
    aggregate_patient_statistics, prepare_base_rows, fold_channel_scale,
)

PATIENT = "patient_14"
OOF_ROOT = ROOT / "cache" / "oof_patient_scores_v1"
ROBUST_ROOT = ROOT / "cache" / "robust_feature_first_fold_v1"
STATS_ROOT = ROOT / "cache" / "oof_recording_stats_v1"
OUTPUT_ROOT = ROOT / "results" / "diagnostics"


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _validate_fold_artifact(path, recordings):
    with np.load(path, allow_pickle=False) as saved:
        required = {"y_true", "scores", "recording_offsets"}
        missing = required - set(saved.files)
        if missing:
            raise RuntimeError(f"OOF fold artifact missing fields: {sorted(missing)}")
        labels = np.asarray(saved["y_true"], dtype=np.int8)
        scores = np.asarray(saved["scores"], dtype=np.float32)
        offsets = json.loads(str(saved["recording_offsets"].item()))
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise RuntimeError("Invalid patient_14 OOF labels/scores")
    if len(offsets) != len(recordings):
        raise RuntimeError("patient_14 OOF recording-offset count mismatch")
    cursor = 0
    by_path = {}
    for offset, recording in zip(offsets, recordings):
        start, end = int(offset["start"]), int(offset["end"])
        if (offset.get("relative_path") != recording.relative_path
                or start != cursor or end <= start):
            raise RuntimeError(f"OOF recording identity/offset mismatch: {recording.relative_path}")
        by_path[recording.relative_path] = (start, end)
        cursor = end
    if cursor != len(labels):
        raise RuntimeError("OOF offsets do not cover the score artifact")
    return labels, scores, by_path


def run(audit_path: Path) -> dict:
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    outer = build_lopo_folds(recordings)[0]
    if outer.test_patient_id != "patient_01" or len(outer.train_recordings) != 608:
        raise RuntimeError("Unexpected outer cohort; refusing diagnostic")
    if any(r.patient_id == "patient_01" for r in outer.train_recordings):
        raise RuntimeError("Outer-test patient_01 must remain excluded")

    patient_recordings = sorted(
        (r for r in outer.train_recordings if r.patient_id == PATIENT),
        key=lambda r: r.relative_path,
    )
    if len(patient_recordings) != 26:
        raise RuntimeError(f"Expected 26 patient_14 recordings, found {len(patient_recordings)}")

    stats_manifest = json.loads((STATS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if (not stats_manifest.get("complete")
            or stats_manifest.get("cache_version", stats_manifest.get("version"))
            != "oof_recording_channel_stats_v1"
            or stats_manifest.get("excluded_outer_test_patient") != "patient_01"):
        raise RuntimeError("Invalid/incomplete OOF recording-statistics cache")
    patient_states, _ = aggregate_patient_statistics(
        STATS_ROOT, stats_manifest, outer.train_recordings
    )
    fold_mean, fold_scale = fold_channel_scale(PATIENT, patient_states)

    row_lookup, _, _ = prepare_base_rows(ROBUST_ROOT, outer.train_recordings)
    labels_all, scores_all, offsets = _validate_fold_artifact(
        OOF_ROOT / "folds" / f"{PATIENT}.npz", patient_recordings
    )

    feature_rows = []
    invariance_rows = []
    recording_summary = []
    for rec in patient_recordings:
        windows = load_recording_windows(PROCESSED_DIR, rec)
        start, end = offsets[rec.relative_path]
        if end - start != len(windows.y):
            raise RuntimeError(f"OOF window count mismatch: {rec.relative_path}")
        if not np.array_equal(labels_all[start:end], windows.y.astype(np.int8)):
            raise RuntimeError(f"OOF labels do not align with processed labels: {rec.relative_path}")

        source_x, source_y, base_start, base_end = row_lookup[rec.relative_path]
        if base_end - base_start != len(windows.y):
            raise RuntimeError(f"Base feature row count mismatch: {rec.relative_path}")
        if not np.array_equal(source_y[base_start:base_end], windows.y.astype(np.int8)):
            raise RuntimeError(f"Base feature labels do not align: {rec.relative_path}")

        standardized = (
            (windows.X.astype(np.float64) - fold_mean[None, :, None])
            / fold_scale[None, :, None]
        ).astype(np.float32)
        recomputed = extract_robust_features(standardized, windows.sfreq, chunk_size=128)
        if recomputed.shape != (len(windows.y), len(PREFERRED_CHANNELS) * len(ROBUST_FEATURE_NAMES)):
            raise RuntimeError(f"Unexpected recomputed feature shape: {rec.relative_path}")
        if not np.isfinite(recomputed).all():
            raise RuntimeError(f"Non-finite recomputed features: {rec.relative_path}")

        cached = np.asarray(source_x[base_start:base_end], dtype=np.float32)
        # Compare features 1..9 per channel. log_std is intentionally fold-specific.
        for channel_index, channel in enumerate(PREFERRED_CHANNELS):
            for feature_index, feature_name in enumerate(ROBUST_FEATURE_NAMES[1:], start=1):
                column = channel_index * len(ROBUST_FEATURE_NAMES) + feature_index
                delta = np.abs(recomputed[:, column].astype(np.float64)
                               - cached[:, column].astype(np.float64))
                invariance_rows.append({
                    "recording": rec.relative_path,
                    "channel": channel,
                    "feature": feature_name,
                    "window_count": len(delta),
                    "max_abs_difference": float(np.max(delta)),
                    "p99_abs_difference": float(np.quantile(delta, 0.99)),
                    "mean_abs_difference": float(np.mean(delta)),
                    "over_1e-4_count": int(np.count_nonzero(delta > 1e-4)),
                    "over_1e-3_count": int(np.count_nonzero(delta > 1e-3)),
                })

        positives = int(np.count_nonzero(windows.y == 1))
        negatives = int(np.count_nonzero(windows.y == 0))
        record_aucs = []
        if positives and negatives:
            y = windows.y.astype(np.int8)
            for channel_index, channel in enumerate(PREFERRED_CHANNELS):
                for feature_index, feature_name in enumerate(ROBUST_FEATURE_NAMES):
                    column = channel_index * len(ROBUST_FEATURE_NAMES) + feature_index
                    values = recomputed[:, column]
                    auc = float(roc_auc_score(y, values))
                    row = {
                        "recording": rec.relative_path,
                        "channel": channel,
                        "feature": feature_name,
                        "positive_windows": positives,
                        "negative_windows": negatives,
                        "auc": auc,
                        "direction": "higher_in_seizure" if auc > 0.5 else (
                            "lower_in_seizure" if auc < 0.5 else "no_separation"
                        ),
                        "positive_median": float(np.median(values[y == 1])),
                        "negative_median": float(np.median(values[y == 0])),
                    }
                    feature_rows.append(row)
                    record_aucs.append(auc)
        recording_summary.append({
            "recording": rec.relative_path,
            "windows": len(windows.y),
            "positive_windows": positives,
            "negative_windows": negatives,
            "feature_auc_count": len(record_aucs),
            "median_feature_auc": float(np.median(record_aucs)) if record_aucs else None,
            "mean_feature_auc": float(np.mean(record_aucs)) if record_aucs else None,
        })
        print(f"Audited {rec.relative_path}: {len(windows.y):,} windows, "
              f"{positives} positive windows", flush=True)
        del windows, standardized, recomputed, cached

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    auc_csv = OUTPUT_ROOT / "patient_14_feature_separation.csv"
    inv_csv = OUTPUT_ROOT / "patient_14_affine_invariance.csv"
    for path, rows in ((auc_csv, feature_rows), (inv_csv, invariance_rows)):
        if rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    by_feature = {}
    for row in feature_rows:
        by_feature.setdefault((row["channel"], row["feature"]), []).append(row["auc"])
    feature_summary = [
        {
            "channel": channel,
            "feature": feature,
            "recording_count": len(aucs),
            "median_recording_auc": float(np.median(aucs)),
            "mean_recording_auc": float(np.mean(aucs)),
            "recordings_auc_below_0_5": int(sum(value < 0.5 for value in aucs)),
            "recordings_auc_above_0_5": int(sum(value > 0.5 for value in aucs)),
        }
        for (channel, feature), aucs in sorted(by_feature.items())
    ]
    max_invariance_error = max(
        (row["max_abs_difference"] for row in invariance_rows), default=0.0
    )
    over_1e_3 = sum(row["over_1e-3_count"] for row in invariance_rows)
    summary = {
        "status": "completed",
        "diagnostic_only": True,
        "model_retrained": False,
        "threshold_tuned": False,
        "outer_test_patient_loaded": False,
        "heldout_patient": PATIENT,
        "patient_recording_count": len(patient_recordings),
        "seizure_containing_recordings": sum(r["positive_windows"] > 0 for r in recording_summary),
        "window_count": int(sum(r["windows"] for r in recording_summary)),
        "positive_window_count": int(sum(r["positive_windows"] for r in recording_summary)),
        "negative_window_count": int(sum(r["negative_windows"] for r in recording_summary)),
        "invariance_comparison": {
            "features_compared_per_channel": list(ROBUST_FEATURE_NAMES[1:]),
            "max_absolute_difference": max_invariance_error,
            "total_values_over_1e-4": int(sum(r["over_1e-4_count"] for r in invariance_rows)),
            "total_values_over_1e-3": int(over_1e_3),
            "interpretation": "Numerical audit only; affine invariance is mathematical, while float32 rounding can cause small differences.",
        },
        "recordings": recording_summary,
        "feature_summary": feature_summary,
        "outputs": [str(auc_csv.relative_to(ROOT)), str(inv_csv.relative_to(ROOT))],
        "notes": [
            "Within-recording AUCs are exploratory and can be unstable with few positive windows.",
            "AUC below 0.5 means the feature is lower in seizure windows for that recording; it does not justify reversing a feature or score.",
            "The patient-level OOF model and its predictions are unchanged.",
        ],
    }
    json_path = OUTPUT_ROOT / "patient_14_feature_diagnostic_v1.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_value) + "\n",
                         encoding="utf-8")
    print(f"Saved {json_path.relative_to(ROOT)}", flush=True)
    print(f"Affine-invariance max abs difference: {max_invariance_error:.8g}; "
          f"values >1e-3: {over_1e_3}", flush=True)
    print("No model was trained and patient_01 was not loaded.", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True,
                        help="Path to docs/full_annotation_audit.csv")
    args = parser.parse_args()
    run(args.audit)


if __name__ == "__main__":
    main()
