"""Reconstruct and explain one inner-OOF Logistic Regression patient fold.

This is a read-only model diagnostic. It verifies reconstructed probabilities
against the saved OOF artifact before interpreting coefficients/contributions.
It never loads outer-test patient_01, tunes a threshold, or replaces OOF score artifacts.

Run from EEG_Seizure_Detection:
    python -m scripts.audit_patient14_model_contributions --patient patient_14 --audit ..\\..\\docs\\full_annotation_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RANDOM_STATE
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import load_recording_manifest
from scripts.run_inner_patient_oof import (
    FEATURES_PER_CHANNEL,
    FEATURE_COUNT,
    aggregate_patient_statistics,
    build_fold_matrix,
    fold_channel_scale,
    prepare_base_rows,
    complete_manifest,
)

PATIENT = "patient_14"
OUTPUT_ROOT = ROOT / "results" / "diagnostics"
OOF_ROOT = ROOT / "cache" / "oof_patient_scores_v1"
ROBUST_ROOT = ROOT / "cache" / "robust_feature_first_fold_v1"
STATS_ROOT = ROOT / "cache" / "oof_recording_stats_v1"
STD_ROOT = ROOT / "cache" / "oof_window_std_v1"
FEATURE_NAMES = (
    "log_std", "normalized_line_length", "skewness", "excess_kurtosis",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "spectral_entropy",
)


def _load_saved_fold(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        required = {"signature", "y_true", "scores", "recording_offsets"}
        missing = required - set(saved.files)
        if missing:
            raise RuntimeError(f"Saved OOF artifact missing fields: {sorted(missing)}")
        signature = str(saved["signature"].item())
        y_true = np.asarray(saved["y_true"], dtype=np.int8)
        scores = np.asarray(saved["scores"], dtype=np.float32)
        offsets = json.loads(str(saved["recording_offsets"].item()))
    if y_true.ndim != 1 or scores.shape != y_true.shape:
        raise RuntimeError("Saved OOF labels and scores have incompatible shapes")
    if not np.isfinite(scores).all() or not np.isin(y_true, (0, 1)).all():
        raise RuntimeError("Saved OOF artifact contains invalid values")
    return signature, y_true, scores, offsets


def contribution_summary(transformed: np.ndarray, coefficients: np.ndarray,
                         intercept: float, labels: np.ndarray,
                         recording_offsets: list[dict], channels: list[str]):
    """Return additive logit contributions aggregated by EEG feature family."""
    X = np.asarray(transformed, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int8)
    if X.ndim != 2 or X.shape[1] != len(coef) or len(X) != len(y):
        raise ValueError("transformed features, coefficients, and labels do not align")
    if X.shape[1] != len(channels) * FEATURES_PER_CHANNEL:
        raise ValueError("feature matrix does not match channel/feature schema")
    if not np.isfinite(X).all() or not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError("contribution inputs must be finite")

    raw = X * coef[None, :]
    groups = {}
    for channel_index, channel in enumerate(channels):
        for feature_index, feature in enumerate(FEATURE_NAMES):
            column = channel_index * FEATURES_PER_CHANNEL + feature_index
            groups.setdefault(feature, []).append(column)

    overall = []
    for feature, columns in groups.items():
        values = raw[:, columns].sum(axis=1)
        overall.append({
            "feature": feature,
            "mean_contribution_positive": float(np.mean(values[y == 1])) if np.any(y == 1) else None,
            "mean_contribution_negative": float(np.mean(values[y == 0])) if np.any(y == 0) else None,
            "positive_minus_negative": (
                float(np.mean(values[y == 1]) - np.mean(values[y == 0]))
                if np.any(y == 1) and np.any(y == 0) else None
            ),
            "mean_absolute_contribution": float(np.mean(np.abs(values))),
        })

    channel_features = []
    for channel_index, channel in enumerate(channels):
        for feature_index, feature in enumerate(FEATURE_NAMES):
            column = channel_index * FEATURES_PER_CHANNEL + feature_index
            values = raw[:, column]
            positive_mean = float(np.mean(values[y == 1])) if np.any(y == 1) else None
            negative_mean = float(np.mean(values[y == 0])) if np.any(y == 0) else None
            channel_features.append({
                "channel": channel,
                "feature": feature,
                "mean_contribution_positive": positive_mean,
                "mean_contribution_negative": negative_mean,
                "positive_minus_negative": (
                    positive_mean - negative_mean
                    if positive_mean is not None and negative_mean is not None else None
                ),
                "mean_absolute_contribution": float(np.mean(np.abs(values))),
            })

    per_recording = []
    cursor = 0
    for rec in recording_offsets:
        start, end = int(rec["start"]), int(rec["end"])
        if start != cursor or end <= start or end > len(y):
            raise RuntimeError("Recording offsets are not contiguous or exceed score rows")
        rec_y = y[start:end]
        row = {"recording": rec["relative_path"],
               "window_count": end - start,
               "positive_windows": int(np.sum(rec_y == 1))}
        for feature, columns in groups.items():
            values = raw[start:end, :][:, columns].sum(axis=1)
            row[f"{feature}_positive_mean"] = (
                float(np.mean(values[rec_y == 1])) if np.any(rec_y == 1) else None
            )
            row[f"{feature}_negative_mean"] = (
                float(np.mean(values[rec_y == 0])) if np.any(rec_y == 0) else None
            )
        per_recording.append(row)
        cursor = end
    if cursor != len(y):
        raise RuntimeError("Recording offsets do not cover all held-out rows")

    feature_level = []
    for index, coefficient in enumerate(coef):
        channel = channels[index // FEATURES_PER_CHANNEL]
        feature = FEATURE_NAMES[index % FEATURES_PER_CHANNEL]
        feature_level.append({
            "channel": channel,
            "feature": feature,
            "coefficient": float(coefficient),
            "absolute_coefficient": float(abs(coefficient)),
        })
    feature_level.sort(key=lambda row: row["absolute_coefficient"], reverse=True)

    logits = raw.sum(axis=1) + float(intercept)
    return {
        "feature_group_contributions": overall,
        "channel_feature_contributions": channel_features,
        "per_recording_contributions": per_recording,
        "largest_absolute_coefficients": feature_level[:40],
        "logit_reconstruction_max_abs_error": float(
            np.max(np.abs(logits - (X @ coef + float(intercept))))
        ),
    }


def run(audit_path: Path, patient: str = PATIENT) -> dict:
    from config import PREFERRED_CHANNELS

    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    outer = build_lopo_folds(recordings)[0]
    if outer.test_patient_id != "patient_01" or len(outer.train_recordings) != 608:
        raise RuntimeError("Unexpected outer cohort; refusing diagnostic")
    if any(rec.patient_id == "patient_01" for rec in outer.train_recordings):
        raise RuntimeError("Outer-test patient_01 must remain excluded")

    heldout_records = sorted(
        (rec for rec in outer.train_recordings if rec.patient_id == patient),
        key=lambda rec: rec.relative_path,
    )
    if not heldout_records or patient == "patient_01":
        raise RuntimeError(f"Invalid or empty held-out patient: {patient}")

    stats_manifest = complete_manifest(
        STATS_ROOT / "manifest.json", "oof_recording_channel_stats_v1"
    )
    std_manifest = complete_manifest(
        STD_ROOT / "manifest.json", "oof_window_channel_std_v1"
    )
    if (stats_manifest.get("excluded_outer_test_patient") != "patient_01"
            or std_manifest.get("excluded_outer_test_patient") != "patient_01"):
        raise RuntimeError("A prerequisite cache does not prove patient_01 exclusion")

    row_lookup, base_signature, _ = prepare_base_rows(ROBUST_ROOT, outer.train_recordings)
    patient_states, stats_entries = aggregate_patient_statistics(
        STATS_ROOT, stats_manifest, outer.train_recordings
    )
    fold_mean, fold_scale = fold_channel_scale(patient, patient_states)

    saved_signature, saved_y, saved_scores, saved_offsets = _load_saved_fold(
        OOF_ROOT / "folds" / f"{patient}.npz"
    )
    if len(saved_offsets) != len(heldout_records):
        raise RuntimeError("Saved OOF recording count does not match cohort")
    for offset, rec in zip(saved_offsets, heldout_records):
        if offset.get("relative_path") != rec.relative_path:
            raise RuntimeError("Saved OOF recording order differs from manifest")

    work_root = OOF_ROOT / "feature_work"
    X, y, offsets, work_folder = build_fold_matrix(
        patient, outer.train_recordings, row_lookup, base_signature,
        stats_entries, STD_ROOT, std_manifest, fold_mean, fold_scale, work_root,
    )
    train_records = [rec for rec in offsets if rec["patient_id"] != patient]
    test_records = [rec for rec in offsets if rec["patient_id"] == patient]
    if len(train_records) + len(test_records) != 608 or len(test_records) != len(heldout_records):
        raise RuntimeError("Unexpected held-out patient fold recording split")
    expected_offsets = []
    heldout_cursor = 0
    for rec in test_records:
        count = int(rec["end"]) - int(rec["start"])
        expected_offsets.append({
            "relative_path": rec["relative_path"],
            "start": heldout_cursor,
            "end": heldout_cursor + count,
        })
        heldout_cursor += count
    if expected_offsets != saved_offsets or heldout_cursor != len(saved_y):
        raise RuntimeError("Saved OOF offsets do not match reconstructed fold matrix")

    scaler = StandardScaler()
    for rec in train_records:
        for start in range(rec["start"], rec["end"], 8192):
            scaler.partial_fit(X[start:min(start + 8192, rec["end"])])

    n_train = sum(rec["end"] - rec["start"] for rec in train_records)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="patient14_audit_", dir=OUTPUT_ROOT) as temp:
        train_x_path = Path(temp) / "train_X.npy"
        train_y_path = Path(temp) / "train_y.npy"
        train_x = np.lib.format.open_memmap(
            train_x_path, mode="w+", dtype=np.float32, shape=(n_train, FEATURE_COUNT)
        )
        train_y = np.lib.format.open_memmap(
            train_y_path, mode="w+", dtype=np.int8, shape=(n_train,)
        )
        cursor = 0
        for rec in train_records:
            for start in range(rec["start"], rec["end"], 8192):
                end = min(start + 8192, rec["end"])
                block = scaler.transform(X[start:end]).astype(np.float32, copy=False)
                train_x[cursor:cursor + len(block)] = block
                train_y[cursor:cursor + len(block)] = y[start:end]
                cursor += len(block)
        train_x.flush()
        train_y.flush()
        if cursor != n_train or len(np.unique(train_y)) != 2:
            raise RuntimeError("Reconstructed training matrix failed integrity checks")

        model = LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=1000, random_state=RANDOM_STATE,
        )
        model.fit(train_x, train_y)
        del train_x, train_y

    heldout_y_parts, heldout_score_parts, transformed_parts = [], [], []
    for rec in test_records:
        for start in range(rec["start"], rec["end"], 8192):
            end = min(start + 8192, rec["end"])
            transformed = scaler.transform(X[start:end])
            transformed_parts.append(transformed.astype(np.float64, copy=False))
            heldout_y_parts.append(np.asarray(y[start:end], dtype=np.int8))
            heldout_score_parts.append(model.predict_proba(transformed)[:, 1].astype(np.float32))
    reconstructed_y = np.concatenate(heldout_y_parts)
    reconstructed_scores = np.concatenate(heldout_score_parts)
    transformed_test = np.concatenate(transformed_parts, axis=0)

    if not np.array_equal(reconstructed_y, saved_y):
        raise RuntimeError("Reconstructed Patient 14 labels differ from saved OOF labels")
    if reconstructed_scores.shape != saved_scores.shape:
        raise RuntimeError("Reconstructed Patient 14 score count differs from saved OOF scores")
    score_delta = np.abs(reconstructed_scores.astype(np.float64) - saved_scores.astype(np.float64))
    max_score_error = float(np.max(score_delta))
    if max_score_error > 2e-6:
        raise RuntimeError(
            f"Fold reconstruction did not reproduce saved scores "
            f"(max_abs_error={max_score_error:.8g}); refusing interpretation"
        )

    contribution = contribution_summary(
        transformed_test, model.coef_[0], float(model.intercept_[0]),
        reconstructed_y, saved_offsets, PREFERRED_CHANNELS,
    )
    result = {
        "status": "completed",
        "diagnostic_only": True,
        "model_retrained_for_diagnostic": True,
        "meaning_of_retrained": "Deterministic reconstruction of the already evaluated patient_14 OOF fold; not a new model experiment.",
        "saved_oof_artifact_modified": False,
        "threshold_tuned": False,
        "outer_test_patient_loaded": False,
        "heldout_patient": patient,
        "training_patient_count": 21,
        "training_recording_count": len(train_records),
        "heldout_recording_count": len(test_records),
        "heldout_window_count": int(len(reconstructed_y)),
        "heldout_positive_windows": int(np.sum(reconstructed_y == 1)),
        "saved_fold_signature": saved_signature,
        "reconstruction_check": {
            "max_absolute_probability_difference": max_score_error,
            "mean_absolute_probability_difference": float(np.mean(score_delta)),
            "tolerance": 2e-6,
            "passed": True,
        },
        "interpretation_caveats": [
            "Coefficient contributions are additive contributions to the fitted logit, not causal effects.",
            "Feature groups combine contributions across channels and may hide channel-level cancellation.",
            "Only 48 positive windows occur in patient_14; summaries are descriptive and correlated windows are not independent observations.",
            "A negative contribution means the feature pushes the model logit down relative to zero, not that the feature is biologically protective.",
        ],
        **contribution,
    }
    json_path = OUTPUT_ROOT / f"{patient}_model_contribution_audit_v1.json"
    csv_path = OUTPUT_ROOT / f"{patient}_recording_contributions_v1.csv"
    channel_csv_path = OUTPUT_ROOT / f"{patient}_channel_feature_contributions_v1.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = contribution["per_recording_contributions"]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Reconstruction max probability difference: {max_score_error:.8g}", flush=True)
    if contribution["channel_feature_contributions"]:
        with channel_csv_path.open("w", newline="", encoding="utf-8") as handle:
            channel_rows = contribution["channel_feature_contributions"]
            writer = csv.DictWriter(handle, fieldnames=list(channel_rows[0]))
            writer.writeheader()
            writer.writerows(channel_rows)
    print(f"Saved {json_path.relative_to(ROOT)}", flush=True)
    print(f"Saved {channel_csv_path.relative_to(ROOT)}", flush=True)
    # Keep the completed fold feature matrix on disk. NumPy memmaps can retain
    # Windows file handles until all references are released; retaining this
    # resumable cache is safer than deleting it during process teardown.
    print("OOF score artifact unchanged; threshold not tuned; patient_01 not loaded.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT.parent.parent / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--patient", default=PATIENT, help="Held-out inner-OOF patient, e.g. patient_13")
    args = parser.parse_args()
    if not args.patient.startswith("patient_") or args.patient == "patient_01":
        parser.error("--patient must be an inner-OOF patient other than patient_01")
    run(args.audit, args.patient)


if __name__ == "__main__":
    main()
