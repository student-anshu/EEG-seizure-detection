"""Inspect the first nested LOPO fold before allocating feature matrices.

Run from the EEG_Seizure_Detection project directory:
    python .\\scripts\\inspect_first_baseline_fold.py

This script reads only processed NPZ metadata/labels. It does not extract
features, fit scalers, train models, or inspect outer-test performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

# Running a file inside scripts/ puts that directory, not the project root,
# on sys.path. Add the project root so top-level project modules are importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import processed_path_for_recording
from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split

FEATURE_COLUMNS = 221
BYTES_PER_FEATURE = np.dtype(np.float32).itemsize


def inspect_group(name, recordings, processed_dir):
    """Count labels without loading/decompressing the large EEG signal array."""
    counts = Counter()
    total_windows = 0
    total = len(recordings)
    print(f"  Inspecting {name}: {total} recordings (labels only)...", flush=True)

    for index, recording in enumerate(recordings, start=1):
        path = processed_path_for_recording(processed_dir, recording)
        if not path.is_file():
            raise FileNotFoundError(f"Processed windows missing: {path}")

        # NPZ members are loaded lazily. Accessing y avoids loading the much
        # larger X array, which is unnecessary for a label-count preflight.
        with np.load(path, allow_pickle=False) as saved:
            if "y" not in saved.files:
                raise ValueError(f"{path.name} is missing the y array.")
            y = np.asarray(saved["y"])

        if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
            raise ValueError(f"{path.name} has empty or non-binary labels.")
        if bool(np.any(y == 1)) != (recording.seizure_count > 0):
            raise ValueError(f"{path.name} labels disagree with audited seizure annotations.")

        counts.update(int(value) for value in y)
        total_windows += len(y)
        if index % 25 == 0 or index == total:
            print(f"    {index}/{total} recordings inspected", flush=True)

    normal = counts.get(0, 0)
    seizure = counts.get(1, 0)
    estimated_bytes = total_windows * FEATURE_COLUMNS * BYTES_PER_FEATURE
    print(f"  {name}:")
    print(f"    patients: {len({r.patient_id for r in recordings})}")
    print(f"    recordings: {len(recordings)}")
    print(f"    windows: {total_windows:,}")
    print(f"    normal windows: {normal:,}")
    print(f"    seizure windows: {seizure:,}")
    print(f"    estimated feature matrix: {estimated_bytes / (1024 ** 2):,.1f} MiB")
    if total_windows == 0:
        raise RuntimeError(f"{name} has no windows")
    return {"normal": normal, "seizure": seizure, "windows": total_windows,
            "feature_bytes": estimated_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "docs" / "full_annotation_audit.csv",
        help="Path to docs/full_annotation_audit.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    recordings = filter_primary_cohort(load_recording_manifest(args.audit))
    folds = build_lopo_folds(recordings)
    fold = folds[0]
    inner = build_patient_validation_split(
        fold.train_recordings,
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
    )

    print("FIRST OUTER LOPO FOLD — PREFLIGHT ONLY")
    print(f"  fold id: {fold.fold_id}")
    print(f"  held-out patient: {fold.test_patient_id}")
    print(f"  total primary recordings: {len(recordings)}")
    print(f"  total independent patients: {len({r.patient_id for r in recordings})}")
    print(f"  validation seed: {args.seed}")
    print()
    print("SPLIT SUMMARY")
    fit = inspect_group("inner fit", inner.train_recordings, PROCESSED_DIR)
    validation = inspect_group("inner validation", inner.validation_recordings, PROCESSED_DIR)
    test = inspect_group("outer test (held out)", fold.test_recordings, PROCESSED_DIR)

    for name, stats in (("inner fit", fit), ("inner validation", validation)):
        if stats["seizure"] == 0 or stats["normal"] == 0:
            raise RuntimeError(f"{name} must contain both classes before training")

    combined = fit["feature_bytes"] + validation["feature_bytes"] + test["feature_bytes"]
    largest_split = max(fit["feature_bytes"], validation["feature_bytes"], test["feature_bytes"])
    print()
    print("MEMORY PREFLIGHT")
    print(f"  estimated all three feature matrices: {combined / (1024 ** 2):,.1f} MiB")
    print(f"  largest single split matrix: {largest_split / (1024 ** 2):,.1f} MiB")
    print("  estimates cover float32 feature arrays only, not temporary arrays/model overhead.")
    print()
    print("No features were extracted and no model was trained.")
    print("If the estimates are acceptable, the next step is a one-fold fit/validation/test runner.")


if __name__ == "__main__":
    main()
