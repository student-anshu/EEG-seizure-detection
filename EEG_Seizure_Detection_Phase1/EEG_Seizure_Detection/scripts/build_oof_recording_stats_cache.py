"""Build reusable per-recording raw channel statistics for inner OOF folds.

Run from the EEG project root:
    python -m scripts.build_oof_recording_stats_cache --audit ..\\..\\docs\\full_annotation_audit.csv

Only the 22 outer-training patients are read. The outer held-out patient is
excluded before any processed recording is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR, PREFERRED_CHANNELS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import load_recording_windows, processed_path_for_recording
from data_management.patient_mapping import load_recording_manifest
from evaluation.oof_scaler_cache import channel_sufficient_statistics

CACHE_VERSION = "oof_recording_channel_stats_v1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_name(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest() + ".npz"


def build(audit_path: Path, cache_root: Path) -> dict:
    recordings = filter_primary_cohort(load_recording_manifest(audit_path))
    fold = build_lopo_folds(recordings)[0]
    if fold.fold_id != 1 or fold.test_patient_id != "patient_01":
        raise RuntimeError("Unexpected first outer fold; refusing to build OOF cache")
    train_patients = {r.patient_id for r in fold.train_recordings}
    if fold.test_patient_id in train_patients:
        raise RuntimeError("Outer test patient leaked into OOF statistics request")

    records_dir = cache_root / "recordings"
    records_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, recording in enumerate(fold.train_recordings, start=1):
        source = processed_path_for_recording(PROCESSED_DIR, recording)
        stat = source.stat()
        signature_payload = {
            "cache_version": CACHE_VERSION,
            "relative_path": recording.relative_path,
            "patient_id": recording.patient_id,
            "subject": recording.subject,
            "edf_file": recording.edf_file,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }
        signature = hashlib.sha256(json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        artifact = records_dir / _artifact_name(recording.relative_path)
        valid = False
        if artifact.is_file():
            try:
                with np.load(artifact, allow_pickle=False) as saved:
                    valid = (
                        str(saved["signature"].item()) == signature
                        and int(saved["count"]) > 0
                        and saved["mean"].shape == (len(PREFERRED_CHANNELS),)
                        and saved["m2"].shape == (len(PREFERRED_CHANNELS),)
                        and np.isfinite(saved["mean"]).all()
                        and np.isfinite(saved["m2"]).all()
                        and np.all(saved["m2"] >= 0)
                    )
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
                valid = False
        if not valid:
            item = load_recording_windows(PROCESSED_DIR, recording)
            state = channel_sufficient_statistics(item.X)
            if len(item.y) == 0:
                raise RuntimeError(f"Empty recording windows: {recording.relative_path}")
            temporary = artifact.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    signature=np.asarray(signature),
                    count=np.asarray(state["count"], dtype=np.int64),
                    mean=state["mean"], m2=state["m2"],
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, artifact)
        with np.load(artifact, allow_pickle=False) as saved:
            count = int(saved["count"])
        entries.append({
            **signature_payload,
            "signature": signature,
            "artifact": artifact.name,
            "count": count,
        })
        # Durable per-recording checkpoint; valid artifacts are reused on restart.
        _atomic_json(cache_root / "progress.json", {
            "cache_version": CACHE_VERSION,
            "outer_fold_id": fold.fold_id,
            "excluded_outer_test_patient": fold.test_patient_id,
            "completed_recordings": index,
            "total_recordings": len(fold.train_recordings),
            "last_recording": recording.relative_path,
            "entries": entries,
            "complete": False,
        })
        if index % 10 == 0 or index == len(fold.train_recordings):
            print(f"OOF statistics cache: {index}/{len(fold.train_recordings)} recordings", flush=True)

    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold_id": fold.fold_id,
        "excluded_outer_test_patient": fold.test_patient_id,
        "training_patient_ids": sorted(train_patients),
        "recording_count": len(entries),
        "channel_count": len(PREFERRED_CHANNELS),
        "statistics": "raw_processed_window_population_count_mean_m2",
        "entries": entries,
        "complete": len(entries) == len(fold.train_recordings),
    }
    _atomic_json(cache_root / "manifest.json", manifest)
    _atomic_json(cache_root / "progress.json", {
        "cache_version": CACHE_VERSION,
        "outer_fold_id": fold.fold_id,
        "excluded_outer_test_patient": fold.test_patient_id,
        "completed_recordings": len(entries),
        "total_recordings": len(fold.train_recordings),
        "last_recording": entries[-1]["relative_path"] if entries else None,
        "complete": True,
    })
    if not manifest["complete"] or len(train_patients) != 22:
        raise RuntimeError("OOF statistics cache failed cohort integrity checks")
    print(f"Saved OOF recording statistics manifest: {cache_root / 'manifest.json'}", flush=True)
    print("Outer-test recordings were not opened.", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT.parent.parent / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "cache" / "oof_recording_stats_v1")
    args = parser.parse_args()
    build(args.audit, args.cache)


if __name__ == "__main__":
    main()
