"""Fold-aware descriptive feature stability audit for the 22 inner OOF patients.

Reconstructs one fold matrix at a time with the exact patient-excluded channel
scaler used by the OOF pipeline. It does not fit models or alter OOF scores.
The sealed outer-test patient_01 is excluded before processed data are accessed.
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

from config import PREFERRED_CHANNELS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import load_recording_manifest
from feature_selection.robust_feature_extraction import ROBUST_FEATURE_NAMES
from scripts.run_inner_patient_oof import (
    aggregate_patient_statistics,
    build_fold_matrix,
    complete_manifest,
    fold_channel_scale,
    prepare_base_rows,
)

CACHE = ROOT / "cache"
OOF_ROOT = CACHE / "oof_patient_scores_v1"
ROBUST_ROOT = CACHE / "robust_feature_first_fold_v1"
STATS_ROOT = CACHE / "oof_recording_stats_v1"
STD_ROOT = CACHE / "oof_window_std_v1"
FEATURES_PER_CHANNEL = len(ROBUST_FEATURE_NAMES)
FEATURE_COUNT = len(PREFERRED_CHANNELS) * FEATURES_PER_CHANNEL
PATIENTS_EXPECTED = 22
DEFAULT_OUTPUT = ROOT / "results" / "diagnostics" / "fold_aware_feature_stability_v1"


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a non-empty finite one-dimensional feature vector")
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"window_count": int(values.size), "q25": float(q25),
            "median": float(median), "q75": float(q75), "iqr": float(q75-q25)}


def _collect(X, y, patient_records, feature_index, label):
    chunks = []
    recording_count = 0
    for rec in patient_records:
        start, end = int(rec["start"]), int(rec["end"])
        labels = y[start:end]
        mask = labels == label
        if np.any(mask):
            chunks.append(np.asarray(X[start:end, feature_index])[mask])
            recording_count += 1
    if not chunks:
        return np.empty(0, dtype=np.float32), 0
    return np.concatenate(chunks), recording_count


def _remove_work_files(folder: Path) -> None:
    # Called only after memmaps are closed by the caller.
    for name in ("X.npy", "y.npy", "manifest.json", "progress.json"):
        (folder / name).unlink(missing_ok=True)


def run(audit: Path, output: Path = DEFAULT_OUTPUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    recordings = filter_primary_cohort(load_recording_manifest(audit))
    outer = build_lopo_folds(recordings)[0]
    if outer.test_patient_id != "patient_01" or len(outer.train_recordings) != 608:
        raise RuntimeError("Unexpected outer cohort; stopping before fold reconstruction")
    if any(r.patient_id == "patient_01" for r in outer.train_recordings):
        raise RuntimeError("Sealed patient_01 leaked into the inner cohort")

    stats_manifest = complete_manifest(STATS_ROOT / "manifest.json", "oof_recording_channel_stats_v1")
    std_manifest = complete_manifest(STD_ROOT / "manifest.json", "oof_window_channel_std_v1")
    if (stats_manifest.get("excluded_outer_test_patient") != "patient_01"
            or std_manifest.get("excluded_outer_test_patient") != "patient_01"
            or stats_manifest.get("recording_count") != 608
            or std_manifest.get("recording_count") != 608):
        raise RuntimeError("Prerequisite OOF caches do not match the sealed 608-recording cohort")

    row_lookup, base_signature, _ = prepare_base_rows(ROBUST_ROOT, outer.train_recordings)
    patient_states, stats_entries = aggregate_patient_statistics(
        STATS_ROOT, stats_manifest, outer.train_recordings
    )
    patients = sorted(patient_states)
    if len(patients) != PATIENTS_EXPECTED or "patient_01" in patients:
        raise RuntimeError("Expected exactly 22 inner patients, excluding patient_01")
    std_paths = {e["relative_path"] for e in std_manifest["entries"]}
    if std_paths != {r.relative_path for r in outer.train_recordings}:
        raise RuntimeError("Raw window-std cache recording identities do not match")

    oof_report = json.loads((OOF_ROOT / "report.json").read_text(encoding="utf-8"))
    if (oof_report.get("patient_fold_count") != PATIENTS_EXPECTED
            or oof_report.get("outer_test_patient_excluded") != "patient_01"
            or oof_report.get("window_count") != 652077):
        raise RuntimeError("Existing OOF report failed cohort integrity checks")
    oof_metrics = {x["patient_id"]: x.get("metrics_at_probability_0_5", {})
                   for x in oof_report.get("patient_folds", [])}
    if set(oof_metrics) != set(patients):
        raise RuntimeError("OOF report patient identities differ from cache cohort")

    # Isolate audit scratch files; never touch the original OOF feature_work cache.
    work_root = CACHE / "fold_aware_feature_stability_work_v1"
    work_root.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_summaries = []

    for fold_number, heldout in enumerate(patients, 1):
        print(f"Fold-aware feature audit {fold_number}/{len(patients)}: {heldout}", flush=True)
        fold_mean, fold_scale = fold_channel_scale(heldout, patient_states)
        X, y, offsets, folder = build_fold_matrix(
            heldout, outer.train_recordings, row_lookup, base_signature,
            stats_entries, STD_ROOT, std_manifest, fold_mean, fold_scale, work_root,
        )
        try:
            if X.shape != (652077, FEATURE_COUNT) or y.shape != (652077,):
                raise RuntimeError(f"Unexpected fold matrix shape for {heldout}")
            if not np.isfinite(X).all() or not np.isin(y, (0, 1)).all():
                raise RuntimeError(f"Non-finite features or invalid labels in {heldout} matrix")
            if any(r["patient_id"] == "patient_01" for r in offsets):
                raise RuntimeError("patient_01 appeared in fold feature offsets")

            patient_offsets = {patient: [r for r in offsets if r["patient_id"] == patient]
                               for patient in patients}
            fold_rows = []
            for channel_index, channel in enumerate(PREFERRED_CHANNELS):
                for feature_offset, feature_name in enumerate(ROBUST_FEATURE_NAMES):
                    feature_index = channel_index * FEATURES_PER_CHANNEL + feature_offset
                    for label in (0, 1):
                        patient_stats = {}
                        for patient in patients:
                            values, rec_count = _collect(
                                X, y, patient_offsets[patient], feature_index, label
                            )
                            patient_stats[patient] = (
                                _quantiles(values) if values.size else None, rec_count
                            )
                        own, own_recs = patient_stats[heldout]
                        peer_medians = np.asarray([
                            stat[0]["median"] for patient, stat in patient_stats.items()
                            if patient != heldout and stat[0] is not None
                        ], dtype=np.float64)
                        if own is None or peer_medians.size < 3:
                            peer_center = peer_iqr = shift = None
                        else:
                            peer_center = float(np.median(peer_medians))
                            q25, q75 = np.percentile(peer_medians, [25, 75])
                            peer_iqr = float(q75 - q25)
                            shift = ((own["median"] - peer_center) / peer_iqr
                                     if peer_iqr > 1e-12 else None)
                        fold_rows.append({
                            "patient_id": heldout,
                            "heldout_scaler_patient": heldout,
                            "label": "seizure" if label else "non_seizure",
                            "channel": channel,
                            "feature": feature_name,
                            "window_count": own["window_count"] if own else 0,
                            "recording_count_with_class": own_recs,
                            "q25": own["q25"] if own else None,
                            "median": own["median"] if own else None,
                            "q75": own["q75"] if own else None,
                            "iqr": own["iqr"] if own else None,
                            "peer_patient_median_of_medians": peer_center,
                            "peer_patient_iqr_of_medians": peer_iqr,
                            "patient_median_shift_in_peer_iqrs": shift,
                            "peer_patient_count": int(peer_medians.size),
                        })
            rows.extend(fold_rows)
            fold_summaries.append({
                "heldout_patient": heldout,
                "recording_count": len(offsets),
                "window_count": len(y),
                "feature_count": int(X.shape[1]),
                "oof_auroc": oof_metrics[heldout].get("auroc"),
                "oof_auprc": oof_metrics[heldout].get("auprc"),
                "fold_scaler_excludes_heldout_patient": True,
            })
        finally:
            del X, y
            _remove_work_files(folder)

    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "fold_aware_patient_feature_distributions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shifts = [abs(r["patient_median_shift_in_peer_iqrs"]) for r in rows
              if r["patient_median_shift_in_peer_iqrs"] is not None]
    report = {
        "status": "completed",
        "diagnostic_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_test_patient_excluded": "patient_01",
        "outer_test_patient_loaded": False,
        "models_trained": False,
        "thresholds_tuned": False,
        "feature_selection_performed": False,
        "oof_scores_modified": False,
        "cohort": {"inner_patients": len(patients), "recordings": len(outer.train_recordings),
                   "windows_per_fold": 652077, "feature_count": FEATURE_COUNT,
                   "channel_count": len(PREFERRED_CHANNELS)},
        "folds": fold_summaries,
        "summary": {"patient_feature_class_rows": len(rows),
                    "finite_absolute_peer_iqr_shifts": len(shifts),
                    "median_absolute_peer_iqr_shift": float(np.median(shifts)) if shifts else None,
                    "p90_absolute_peer_iqr_shift": float(np.percentile(shifts, 90)) if shifts else None},
        "method": {
            "transformation": "For each held-out patient, reconstruct its exact OOF fold matrix using the channel scaler fitted on the other 21 patients.",
            "features": "All 170 robust features; fold-specific log_std reconstructed from raw window standard deviations, other robust features reused from affine-invariant cache.",
            "peer_reference": "Per-patient medians from the same fold matrix; each patient contributes one median, not pooled windows.",
            "shift": "(held-out patient median - peer median of medians) / peer IQR of patient medians.",
            "caveat": "The held-out patient's data are excluded from scaler fitting; peer distributions are transformed under that same fold scaler for a comparable reference.",
            "overlap": "10-second windows with 5-second stride overlap and are not independent observations.",
        },
        "oof_ranking_context": "Existing inner OOF AUROC/AUPRC copied for descriptive context only.",
        "files": {"patient_feature_distributions": str(csv_path.relative_to(ROOT))},
        "limitations": [
            "Descriptive screening only; no inferential or causal claims.",
            "Sparse seizure windows and unequal recording counts affect distribution precision.",
            "OOF metrics are inner development results, not outer-test performance.",
            "No feature selection, model fitting, threshold tuning, or patient_01 access occurred.",
            "The fold-specific channel scaler excludes the held-out patient; the peer reference uses the same fold transform.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Completed: {len(fold_summaries)} folds, {len(rows):,} patient-feature-class rows", flush=True)
    print(f"Saved: {output}", flush=True)
    print("No model retraining, threshold tuning, feature selection, OOF score changes, or patient_01 access.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=ROOT.parent.parent / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.audit, args.output)


if __name__ == "__main__":
    main()
