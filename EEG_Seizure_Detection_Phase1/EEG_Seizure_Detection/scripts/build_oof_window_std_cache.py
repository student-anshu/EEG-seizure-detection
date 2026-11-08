"""Build resumable per-recording raw window/channel standard deviations for OOF."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, PREFERRED_CHANNELS
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import load_recording_windows, processed_path_for_recording
from data_management.patient_mapping import load_recording_manifest
from evaluation.oof_scaler_cache import window_channel_standard_deviations

VERSION = "oof_window_channel_std_v1"

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)

def build(audit, cache):
    recordings = filter_primary_cohort(load_recording_manifest(audit))
    fold = build_lopo_folds(recordings)[0]
    if fold.fold_id != 1 or fold.test_patient_id != "patient_01":
        raise RuntimeError("Unexpected outer fold; refusing OOF cache build")
    if len({r.patient_id for r in fold.train_recordings}) != 22:
        raise RuntimeError("Expected 22 outer-training patients")
    folder = cache / "recordings"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, rec in enumerate(fold.train_recordings, 1):
        source = processed_path_for_recording(PROCESSED_DIR, rec)
        st = source.stat()
        identity = {"version": VERSION, "relative_path": rec.relative_path,
                    "patient_id": rec.patient_id, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        name = hashlib.sha256(rec.relative_path.encode()).hexdigest() + ".npz"
        artifact = folder / name
        valid = False
        if artifact.is_file():
            try:
                with np.load(artifact, allow_pickle=False) as z:
                    a = z["std"]
                    valid = (str(z["signature"].item()) == signature and a.ndim == 2
                             and int(z["window_count"]) == a.shape[0]
                             and a.shape[1] == len(PREFERRED_CHANNELS) and a.shape[0] > 0
                             and a.dtype == np.float32 and np.isfinite(a).all() and np.all(a >= 0))
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
                valid = False
        if not valid:
            item = load_recording_windows(PROCESSED_DIR, rec)
            std = window_channel_standard_deviations(item.X, chunk_size=256)
            if len(std) != len(item.y):
                raise RuntimeError("Window/std alignment mismatch: " + rec.relative_path)
            tmp = artifact.with_suffix(".npz.tmp")
            with tmp.open("wb") as h:
                np.savez_compressed(h, signature=np.asarray(signature), std=std,
                                    window_count=np.asarray(len(std), dtype=np.int64))
                h.flush()
                os.fsync(h.fileno())
            os.replace(tmp, artifact)
            del item, std
        with np.load(artifact, allow_pickle=False) as z:
            count = int(z["window_count"])
            if z["std"].shape != (count, len(PREFERRED_CHANNELS)):
                raise RuntimeError("Invalid std artifact: " + rec.relative_path)
        entries.append({**identity, "signature": signature, "artifact": name, "window_count": count})
        atomic_json(cache / "progress.json", {"completed": i, "total": len(fold.train_recordings),
                                               "last_recording": rec.relative_path, "complete": False})
        if i % 10 == 0 or i == len(fold.train_recordings):
            print(f"Window std cache: {i}/{len(fold.train_recordings)}", flush=True)
    manifest = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "outer_fold_id": 1, "excluded_outer_test_patient": "patient_01",
                "training_patient_ids": sorted({r.patient_id for r in fold.train_recordings}),
                "recording_count": len(entries), "channel_count": len(PREFERRED_CHANNELS),
                "entries": entries, "complete": len(entries) == 608}
    atomic_json(cache / "manifest.json", manifest)
    atomic_json(cache / "progress.json", {"completed": len(entries), "total": len(entries), "complete": True})
    print("Saved window std manifest:", cache / "manifest.json", flush=True)
    print("Outer-test recordings were not opened.", flush=True)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT.parent.parent / "docs" / "full_annotation_audit.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache" / "oof_window_std_v1")
    args = parser.parse_args()
    build(args.audit, args.cache)

if __name__ == "__main__":
    main()
