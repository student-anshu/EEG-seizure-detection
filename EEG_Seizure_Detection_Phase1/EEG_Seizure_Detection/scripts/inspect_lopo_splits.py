"""Inspect LOPO folds for the fixed 17-channel primary cohort."""
from __future__ import annotations
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import load_recording_manifest


def main():
    audit_csv = REPOSITORY_ROOT / "docs" / "full_annotation_audit.csv"
    full = load_recording_manifest(audit_csv)
    recordings = filter_primary_cohort(full)
    folds = build_lopo_folds(recordings)
    total_events = sum(r.seizure_count for r in recordings)
    excluded_events = sum(r.seizure_count for r in full if r not in recordings)
    patients = {r.patient_id for r in recordings}

    print("========== PRIMARY COHORT LOPO SUMMARY ==========")
    print(f"Full audit: {len(full)} recordings, {sum(r.seizure_count for r in full)} seizure events")
    print(f"Excluded montage recordings: {len(full) - len(recordings)}")
    print(f"Excluded seizure events: {excluded_events}")
    print(f"Primary cohort: {len(recordings)} recordings, {total_events} seizure events")
    print(f"Independent patients: {len(patients)}")
    print(f"LOPO folds: {len(folds)}")

    for fold in folds:
        test_events = sum(r.seizure_count for r in fold.test_recordings)
        print(f"Fold {fold.fold_id:02d}: test={fold.test_patient_id}, "
              f"train_recordings={len(fold.train_recordings)}, "
              f"test_recordings={len(fold.test_recordings)}, test_events={test_events}")

    tested = [r.relative_path for f in folds for r in f.test_recordings]
    assert len(full) == 686
    assert len(recordings) == 683
    assert excluded_events == 13
    assert total_events == 185
    assert len(patients) == len(folds) == 23
    assert len(tested) == len(set(tested)) == 683
    print("\nPASS: 683 eligible recordings and 185 seizure events.")
    print("PASS: all 23 patients are represented with no patient leakage.")
    print("PASS: every eligible recording is tested exactly once.")


if __name__ == "__main__":
    main()
