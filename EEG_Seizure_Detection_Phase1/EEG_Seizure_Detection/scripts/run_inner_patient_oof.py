"""Run resumable patient-held-out OOF Logistic Regression on 22 inner patients.

Requires the existing robust inner-fit/validation feature caches, the recording
channel-statistics cache, and the compact raw per-window std cache. The outer
test patient is excluded before processed recordings are accessed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, PREFERRED_CHANNELS, RANDOM_STATE
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import load_recording_windows, processed_path_for_recording
from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split
from evaluation.oof_scaler_cache import (
    exact_log_std_fallback_mask, finalize_channel_statistics,
    log_std_from_raw_standard_deviations, merge_channel_statistics,
    patient_grouped_channel_scale, recompute_log_std_features,
)
from modeling.baseline import calculate_metrics

FEATURES_PER_CHANNEL = 10
FEATURE_COUNT = len(PREFERRED_CHANNELS) * FEATURES_PER_CHANNEL
VERSION = "patient_grouped_oof_robust_v3"


def atomic_json(path, value):
    """Write a JSON checkpoint safely, tolerating transient Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_name = None
    try:
        # A unique same-directory temporary file avoids collisions with stale
        # .tmp files and keeps os.replace on the same filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        delays = (0.25, 0.5, 1.0, 2.0, 3.0, 3.0)
        for attempt, delay in enumerate(delays, start=1):
            try:
                os.replace(temp_name, path)
                temp_name = None
                return
            except PermissionError:
                if attempt == len(delays):
                    raise
                time.sleep(delay)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def complete_manifest(path, expected_version):
    if not path.is_file():
        raise FileNotFoundError(f"Required cache manifest missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("complete"):
        raise RuntimeError(f"Cache is incomplete: {path}")
    actual = value.get("cache_version", value.get("version"))
    if actual != expected_version:
        raise RuntimeError(f"Unexpected cache version in {path}: {actual}")
    return value


def validate_feature_split(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Robust feature cache is incomplete: {directory}")
    X = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
    if (manifest.get("shape") != list(X.shape) or X.ndim != 2
            or X.shape[1] != FEATURE_COUNT or X.dtype != np.float32
            or y.shape != (len(X),) or y.dtype != np.int8
            or not np.isin(y, (0, 1)).all()):
        raise RuntimeError(f"Robust feature cache integrity failure: {directory}")
    return X, y, manifest


def prepare_base_rows(robust_root, outer_train):
    split = build_patient_validation_split(
        outer_train, validation_fraction=0.2, random_state=RANDOM_STATE
    )
    fit_x, fit_y, fit_m = validate_feature_split(robust_root / "inner_fit")
    val_x, val_y, val_m = validate_feature_split(robust_root / "inner_validation")
    row_lookup = {}
    for split_name, source_x, source_y, records in (
        ("inner_fit", fit_x, fit_y, split.train_recordings),
        ("inner_validation", val_x, val_y, split.validation_recordings),
    ):
        cursor = 0
        for rec in records:
            path = processed_path_for_recording(PROCESSED_DIR, rec)
            with np.load(path, allow_pickle=False) as saved:
                labels = np.asarray(saved["y"], dtype=np.int8)
            end = cursor + len(labels)
            if end > len(source_y) or not np.array_equal(source_y[cursor:end], labels):
                raise RuntimeError(f"Feature cache row alignment mismatch: {rec.relative_path}")
            if bool(np.any(labels == 1)) != (rec.seizure_count > 0):
                raise RuntimeError(f"Annotation/label mismatch: {rec.relative_path}")
            row_lookup[rec.relative_path] = (source_x, source_y, cursor, end)
            cursor = end
        if cursor != len(source_y):
            raise RuntimeError(f"{split_name} rows do not cover its cache")
    if set(row_lookup) != {r.relative_path for r in outer_train}:
        raise RuntimeError("Feature-cache recording identities do not match outer training")
    base_signature = digest({
        "version": VERSION,
        "fit_cache_signature": fit_m.get("signature"),
        "validation_cache_signature": val_m.get("signature"),
        "ordered_recordings": [r.relative_path for r in outer_train],
    })
    return row_lookup, base_signature, split


def audit_raw_std_approximation(robust_root, std_root, std_manifest, row_lookup, validation_recordings):
    """Compare raw-std reconstruction with cached real validation log_std values."""
    checkpoint = robust_root / "scaler_checkpoint"
    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if not checkpoint_manifest.get("complete"):
        raise RuntimeError("Original inner-fit scaler checkpoint is incomplete")
    state_name = checkpoint_manifest.get("state_file")
    if not isinstance(state_name, str) or Path(state_name).name != state_name:
        raise RuntimeError("Invalid original scaler checkpoint state filename")
    with np.load(checkpoint / state_name, allow_pickle=False) as saved:
        state = {"count": int(saved["count"]), "mean": saved["mean"], "m2": saved["m2"]}
    original_mean, original_scale = finalize_channel_statistics(state)
    fallback_window_count = 0
    entries = {e["relative_path"]: e for e in std_manifest["entries"]}
    columns = np.arange(len(PREFERRED_CHANNELS)) * FEATURES_PER_CHANNEL
    max_error, mismatch_count, floor_mismatches, checked = 0.0, 0, 0, 0
    floor = float(np.log(1e-12))
    for rec in validation_recordings:
        source_x, _, start, end = row_lookup[rec.relative_path]
        entry = entries[rec.relative_path]
        with np.load(std_root / "recordings" / entry["artifact"], allow_pickle=False) as saved:
            raw_std = saved["std"]
            if str(saved["signature"].item()) != entry["signature"]:
                raise RuntimeError(f"Raw-std signature mismatch during numerical audit: {rec.relative_path}")
            reconstructed = log_std_from_raw_standard_deviations(raw_std, original_scale)
            fallback_mask = exact_log_std_fallback_mask(raw_std, original_scale)
            if np.any(fallback_mask):
                original_windows = load_recording_windows(PROCESSED_DIR, rec)
                if len(original_windows.X) != len(raw_std):
                    raise RuntimeError(f"Fallback window alignment mismatch: {rec.relative_path}")
                exact_values = recompute_log_std_features(
                    original_windows.X[fallback_mask], original_mean, original_scale
                )
                reconstructed[fallback_mask] = exact_values
                fallback_window_count += int(np.count_nonzero(fallback_mask))
                del original_windows, exact_values
        actual = np.asarray(source_x[start:end, :][:, columns], dtype=np.float32)
        if reconstructed.shape != actual.shape:
            raise RuntimeError(f"Raw-std numerical audit shape mismatch: {rec.relative_path}")
        delta = np.abs(reconstructed - actual)
        max_error = max(max_error, float(np.max(delta)))
        mismatch_count += int(np.count_nonzero(delta > 5e-4))
        actual_floor = actual <= floor + 1e-6
        reconstructed_floor = reconstructed <= floor + 1e-6
        floor_mismatches += int(np.count_nonzero(actual_floor != reconstructed_floor))
        checked += int(actual.size)
    print(f"Raw-std real-cache audit: {checked:,} values; max_abs_error={max_error:.8g}; "
          f"over_tolerance={mismatch_count}; floor_mismatches={floor_mismatches}", flush=True)
    if mismatch_count or floor_mismatches:
        raise RuntimeError("Raw-std approximation failed real validation-cache audit; refusing OOF training")
    return {"checked_values": checked, "max_absolute_error": max_error,
            "values_over_tolerance": mismatch_count, "floor_mismatches": floor_mismatches,
            "exact_fallback_windows": fallback_window_count,
            "fallback_rule_normalized_std_at_most": 1e-4}


def aggregate_patient_statistics(stats_root, manifest, outer_train):
    entries = {e["relative_path"]: e for e in manifest["entries"]}
    if len(entries) != len(outer_train) or set(entries) != {r.relative_path for r in outer_train}:
        raise RuntimeError("Recording-stat cache does not match the outer-training cohort")
    patient_states = {}
    for rec in outer_train:
        entry = entries[rec.relative_path]
        if entry.get("patient_id") != rec.patient_id:
            raise RuntimeError(f"Patient identity mismatch in stats manifest: {rec.relative_path}")
        artifact = stats_root / "recordings" / entry["artifact"]
        try:
            with np.load(artifact, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != entry["signature"]:
                    raise RuntimeError(f"Stale statistics artifact: {rec.relative_path}")
                state = {"count": int(saved["count"]), "mean": saved["mean"], "m2": saved["m2"]}
        except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Invalid statistics artifact: {rec.relative_path}") from exc
        if (state["count"] <= 0 or state["mean"].shape != (len(PREFERRED_CHANNELS),)
                or state["m2"].shape != (len(PREFERRED_CHANNELS),)):
            raise RuntimeError(f"Invalid channel statistics: {rec.relative_path}")
        if rec.patient_id in patient_states:
            patient_states[rec.patient_id] = merge_channel_statistics(patient_states[rec.patient_id], state)
        else:
            patient_states[rec.patient_id] = state
    if len(patient_states) != 22:
        raise RuntimeError("Expected 22 patient-level statistics groups")
    return patient_states, entries


def fold_channel_scale(heldout_patient, patient_states):
    # This fold's held-out patient's entire recordings are excluded from scaler fit.
    if len(patient_states) != 22:
        raise RuntimeError("OOF channel scaler requires exactly 22 patients")
    mean, scale = patient_grouped_channel_scale(patient_states, heldout_patient)
    return mean, scale


def build_fold_matrix(
    heldout, outer_train, row_lookup, base_signature, stats_entries,
    std_root, std_manifest, fold_mean, fold_scale, work_root,
):
    """Create/resume one fold's features, all using the SAME 21-patient scaler."""
    std_entries = {e["relative_path"]: e for e in std_manifest["entries"]}
    total = sum(int(std_entries[r.relative_path]["window_count"]) for r in outer_train)
    signature = digest({
        "version": VERSION, "base_signature": base_signature,
        "std_manifest": digest(std_manifest), "heldout_patient": heldout,
        "fold_mean": np.asarray(fold_mean, dtype=np.float64).tolist(),
        "fold_scale": np.asarray(fold_scale, dtype=np.float64).tolist(),
    })
    folder = work_root / heldout
    folder.mkdir(parents=True, exist_ok=True)
    x_path, y_path = folder / "X.npy", folder / "y.npy"
    progress_path, manifest_path = folder / "progress.json", folder / "manifest.json"
    completed, cursor, X, y = 0, 0, None, None

    if progress_path.is_file() and x_path.is_file() and y_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("signature") == signature:
                candidate_x = np.load(x_path, mmap_mode="r+", allow_pickle=False)
                candidate_y = np.load(y_path, mmap_mode="r+", allow_pickle=False)
                completed = int(progress["completed_recordings"])
                cursor = int(progress["cursor"])
                expected_cursor = sum(int(std_entries[r.relative_path]["window_count"]) for r in outer_train[:completed])
                if (candidate_x.shape == (total, FEATURE_COUNT) and candidate_x.dtype == np.float32
                        and candidate_y.shape == (total,) and candidate_y.dtype == np.int8
                        and 0 <= completed <= len(outer_train) and cursor == expected_cursor):
                    X, y = candidate_x, candidate_y
                else:
                    completed, cursor = 0, 0
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            completed, cursor = 0, 0
    if X is None:
        if "candidate_x" in locals():
            del candidate_x
        if "candidate_y" in locals():
            del candidate_y
        X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(total, FEATURE_COUNT))
        y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8, shape=(total,))
        atomic_json(progress_path, {"signature": signature, "completed_recordings": 0,
                                    "cursor": 0, "complete": False})

    offsets = []
    cursor_check = 0
    for rec in outer_train:
        count = int(std_entries[rec.relative_path]["window_count"])
        offsets.append({"relative_path": rec.relative_path, "patient_id": rec.patient_id,
                        "start": cursor_check, "end": cursor_check + count})
        cursor_check += count
    if cursor_check != total:
        raise RuntimeError("Fold matrix recording offsets do not sum to total windows")

    for index in range(completed, len(outer_train)):
        rec = outer_train[index]
        source_x, source_y, source_start, source_end = row_lookup[rec.relative_path]
        count = source_end - source_start
        entry = std_entries[rec.relative_path]
        if entry.get("patient_id") != rec.patient_id:
            raise RuntimeError(f"Patient identity mismatch in raw-std manifest: {rec.relative_path}")
        if int(entry["window_count"]) != count:
            raise RuntimeError(f"Raw std/feature row count mismatch: {rec.relative_path}")
        artifact = std_root / "recordings" / entry["artifact"]
        try:
            with np.load(artifact, allow_pickle=False) as saved:
                raw_std = saved["std"]
                if str(saved["signature"].item()) != entry["signature"] or raw_std.shape != (count, len(PREFERRED_CHANNELS)):
                    raise RuntimeError(f"Invalid raw std artifact: {rec.relative_path}")
        except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Could not load raw std for {rec.relative_path}") from exc

        fallback_mask = exact_log_std_fallback_mask(raw_std, fold_scale)
        original_windows = None
        if np.any(fallback_mask):
            original_windows = load_recording_windows(PROCESSED_DIR, rec)
            if len(original_windows.X) != count:
                raise RuntimeError(f"Fallback window alignment mismatch: {rec.relative_path}")

        for chunk_start in range(0, count, 4096):
            chunk_end = min(chunk_start + 4096, count)
            block = np.asarray(source_x[source_start + chunk_start:source_start + chunk_end],
                               dtype=np.float32).copy()
            chunk_raw_std = raw_std[chunk_start:chunk_end]
            chunk_log_std = log_std_from_raw_standard_deviations(chunk_raw_std, fold_scale)
            local_fallback = fallback_mask[chunk_start:chunk_end]
            if np.any(local_fallback):
                exact_values = recompute_log_std_features(
                    original_windows.X[chunk_start:chunk_end][local_fallback],
                    fold_mean, fold_scale,
                )
                chunk_log_std[local_fallback] = exact_values
            block[:, np.arange(len(PREFERRED_CHANNELS)) * FEATURES_PER_CHANNEL] = chunk_log_std
            dest = cursor + chunk_start
            X[dest:dest + len(block)] = block
        if original_windows is not None:
            del original_windows
        y[cursor:cursor + count] = source_y[source_start:source_end]
        cursor += count
        X.flush()
        y.flush()
        atomic_json(progress_path, {"signature": signature, "completed_recordings": index + 1,
                                    "cursor": cursor, "complete": index + 1 == len(outer_train)})
        if (index + 1) % 25 == 0 or index + 1 == len(outer_train):
            print(f"  {heldout} fold features: {index + 1}/{len(outer_train)} recordings", flush=True)
        del raw_std

    if cursor != total or not np.isin(y, (0, 1)).all():
        raise RuntimeError(f"Fold feature matrix integrity failure for {heldout}")
    X.flush()
    atomic_json(manifest_path, {"signature": signature, "complete": True,
                                "window_count": total, "feature_count": FEATURE_COUNT,
                                "recordings": offsets, "heldout_patient": heldout})
    del X, y
    return np.load(x_path, mmap_mode="r", allow_pickle=False), np.load(y_path, mmap_mode="r", allow_pickle=False), offsets, folder


def train_score_fold(heldout, fold_index, X, y, offsets, output, seed):
    test_records = [r for r in offsets if r["patient_id"] == heldout]
    train_records = [r for r in offsets if r["patient_id"] != heldout]
    if not test_records or not train_records:
        raise RuntimeError(f"Empty OOF split for {heldout}")
    scaler = StandardScaler()
    for rec in train_records:
        for start in range(rec["start"], rec["end"], 8192):
            scaler.partial_fit(X[start:min(start + 8192, rec["end"])])
    n_train = sum(r["end"] - r["start"] for r in train_records)
    train_x_path, train_y_path = output / f".{heldout}_train_X.npy", output / f".{heldout}_train_y.npy"
    train_x = np.lib.format.open_memmap(train_x_path, mode="w+", dtype=np.float32, shape=(n_train, FEATURE_COUNT))
    train_y = np.lib.format.open_memmap(train_y_path, mode="w+", dtype=np.int8, shape=(n_train,))
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
        raise RuntimeError(f"Invalid OOF training data/classes for {heldout}")

    model = LogisticRegression(solver="liblinear", class_weight="balanced",
                               max_iter=1000, random_state=seed)
    print(f"Training OOF fold {fold_index}/22: {heldout}; {n_train:,} training windows", flush=True)
    model.fit(train_x, train_y)

    y_parts, score_parts, recording_offsets = [], [], []
    test_cursor = 0
    for rec in test_records:
        rec_start = test_cursor
        for start in range(rec["start"], rec["end"], 8192):
            end = min(start + 8192, rec["end"])
            transformed = scaler.transform(X[start:end])
            score_parts.append(model.predict_proba(transformed)[:, 1].astype(np.float32))
            y_parts.append(np.asarray(y[start:end], dtype=np.int8))
            test_cursor += end - start
        recording_offsets.append({"relative_path": rec["relative_path"],
                                  "start": rec_start, "end": test_cursor})
    fold_y = np.concatenate(y_parts)
    scores = np.concatenate(score_parts)
    del model, scaler, train_x, train_y
    train_x_path.unlink(missing_ok=True)
    train_y_path.unlink(missing_ok=True)
    return fold_y, scores, recording_offsets


def run(audit, seed=RANDOM_STATE, audit_only=False):
    recordings = filter_primary_cohort(load_recording_manifest(audit))
    outer = build_lopo_folds(recordings)[0]
    if outer.test_patient_id != "patient_01" or len(outer.train_recordings) != 608:
        raise RuntimeError("Unexpected outer cohort; stopping before OOF")
    if "patient_01" in {r.patient_id for r in outer.train_recordings}:
        raise RuntimeError("Outer-test patient leaked into OOF")

    cache_root = ROOT / "cache"
    robust_root = cache_root / "robust_feature_first_fold_v1"
    stats_root = cache_root / "oof_recording_stats_v1"
    std_root = cache_root / "oof_window_std_v1"
    stats_manifest = complete_manifest(stats_root / "manifest.json", "oof_recording_channel_stats_v1")
    std_manifest = complete_manifest(std_root / "manifest.json", "oof_window_channel_std_v1")
    if stats_manifest.get("excluded_outer_test_patient") != "patient_01" or std_manifest.get("excluded_outer_test_patient") != "patient_01":
        raise RuntimeError("A prerequisite cache did not exclude patient_01")

    row_lookup, base_signature, inner_split = prepare_base_rows(robust_root, outer.train_recordings)
    patient_states, stats_entries = aggregate_patient_statistics(stats_root, stats_manifest, outer.train_recordings)
    std_paths = {e["relative_path"] for e in std_manifest["entries"]}
    if std_paths != {r.relative_path for r in outer.train_recordings}:
        raise RuntimeError("Raw window-std cache does not match outer training")
    if std_manifest.get("recording_count") != 608 or stats_manifest.get("recording_count") != 608:
        raise RuntimeError("Expected 608 recordings in both OOF caches")
    numerical_audit = audit_raw_std_approximation(
        robust_root, std_root, std_manifest, row_lookup, inner_split.validation_recordings
    )
    if audit_only:
        print("Numerical audit passed. Audit-only mode: no OOF models were trained.", flush=True)
        return {"status": "audit_passed", "numerical_audit": numerical_audit}

    output = cache_root / "oof_patient_scores_v1"
    fold_dir = output / "folds"
    work_root = output / "feature_work"
    fold_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    patient_ids = sorted(patient_states)
    results, all_y, all_scores = [], [], []

    for fold_index, heldout in enumerate(patient_ids, 1):
        fold_mean, fold_scale = fold_channel_scale(heldout, patient_states)
        signature = digest({"version": VERSION, "base_signature": base_signature,
                            "stats_manifest": digest(stats_manifest), "std_manifest": digest(std_manifest),
                            "heldout_patient": heldout, "seed": seed,
                            "model": "LogisticRegression(liblinear,balanced,max_iter=1000)",
                            "fold_mean": fold_mean.tolist(),
                            "fold_scale": fold_scale.tolist()})
        artifact = fold_dir / f"{heldout}.npz"
        cached = False
        if artifact.is_file():
            try:
                with np.load(artifact, allow_pickle=False) as saved:
                    cached = (str(saved["signature"].item()) == signature
                              and np.isfinite(saved["scores"]).all()
                              and len(saved["scores"]) == sum(
                                  int(e["window_count"]) for e in std_manifest["entries"]
                                  if e["patient_id"] == heldout))
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
                cached = False
        if cached:
            with np.load(artifact, allow_pickle=False) as saved:
                fold_y, scores = saved["y_true"], saved["scores"]
                recording_offsets = json.loads(str(saved["recording_offsets"].item()))
            print(f"Reusing completed OOF fold {heldout}.", flush=True)
        else:
            X, y, offsets, work_folder = build_fold_matrix(
                heldout, outer.train_recordings, row_lookup, base_signature,
                stats_entries, std_root, std_manifest, fold_mean, fold_scale, work_root,
            )
            fold_y, scores, recording_offsets = train_score_fold(
                heldout, fold_index, X, y, offsets, output, seed
            )
            tmp = artifact.with_suffix(".npz.tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, signature=np.asarray(signature),
                                    y_true=fold_y, scores=scores,
                                    recording_offsets=np.asarray(json.dumps(recording_offsets)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, artifact)
            del X, y
            # Keep the feature work only until the score artifact is committed.
            for filename in ("X.npy", "y.npy", "manifest.json", "progress.json"):
                (work_folder / filename).unlink(missing_ok=True)

        metrics = calculate_metrics(fold_y, scores, threshold=0.5)
        results.append({"patient_id": heldout, "recording_count": len(recording_offsets),
                        "window_count": len(fold_y), "positive_windows": int(np.sum(fold_y == 1)),
                        "metrics_at_probability_0_5": metrics,
                        "artifact": str(artifact.relative_to(ROOT))})
        all_y.append(np.asarray(fold_y, dtype=np.int8))
        all_scores.append(np.asarray(scores, dtype=np.float32))
        atomic_json(output / "progress.json", {"completed_folds": fold_index,
                    "total_folds": 22, "last_completed_patient": heldout, "complete": False})
        print(f"Completed {heldout}: AUROC={metrics['auroc']} AUPRC={metrics['auprc']}", flush=True)

    pooled_y = np.concatenate(all_y)
    pooled_scores = np.concatenate(all_scores)
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(pooled_y, pooled_scores)
    p, r = precision[:-1], recall[:-1]
    f2 = np.divide(5.0 * p * r, 4.0 * p + r, out=np.zeros_like(p), where=(4.0 * p + r) > 0)
    threshold = float(thresholds[int(np.argmax(f2))]) if len(thresholds) else 0.5
    pooled_metrics = calculate_metrics(pooled_y, pooled_scores, threshold)
    np.savez_compressed(output / "pooled_oof_scores.npz", y_true=pooled_y, scores=pooled_scores)
    report = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "outer_test_patient_excluded": "patient_01", "patient_fold_count": 22,
              "window_count": len(pooled_y), "positive_windows": int(np.sum(pooled_y == 1)),
              "model": "LogisticRegression(solver=liblinear,class_weight=balanced,max_iter=1000)",
              "raw_std_real_cache_audit": numerical_audit,
              "threshold": {"value": threshold, "objective": "pooled OOF window F2",
                            "status": "provisional; event-level metrics and fold calibration review required"},
              "pooled_metrics_at_oof_f2_threshold": pooled_metrics, "patient_folds": results,
              "limitations": ["Each OOF model trains on 21 patients; the final outer model may train on all 22.",
                              "Pooled probabilities from distinct fold models may differ in calibration.",
                              "This is inner OOF only; no outer-test result is included.",
                              "Research prototype, not a clinical device."]}
    atomic_json(output / "report.json", report)
    atomic_json(output / "progress.json", {"completed_folds": 22, "total_folds": 22,
                "last_completed_patient": patient_ids[-1], "complete": True})
    print(f"Saved OOF report: {output / 'report.json'}", flush=True)
    print(f"Provisional pooled OOF F2 threshold: {threshold:.8f}", flush=True)
    print("Outer-test patient_01 was not opened.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT.parent.parent / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Run cache and numerical integrity checks without training OOF models.",
    )
    args = parser.parse_args()
    run(args.audit, args.seed, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
