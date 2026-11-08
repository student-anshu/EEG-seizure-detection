"""Inspect patient-grouped validation within primary-cohort LOPO folds."""
from __future__ import annotations
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split


def main():
    audit_csv = REPOSITORY_ROOT / "docs" / "full_annotation_audit.csv"
    recordings = filter_primary_cohort(load_recording_manifest(audit_csv))
    folds = build_lopo_folds(recordings)
    print("========== NESTED PRIMARY-COHORT VALIDATION ==========")
    print(f"Primary cohort: {len(recordings)} recordings, {sum(r.seizure_count for r in recordings)} seizure events")
    print(f"Outer LOPO folds: {len(folds)}")

    for fold in folds:
        split = build_patient_validation_split(fold.train_recordings, validation_fraction=0.2, random_state=42)
        test_patients = {r.patient_id for r in fold.test_recordings}
        train_patients = set(split.train_patient_ids)
        validation_patients = set(split.validation_patient_ids)
        assert test_patients.isdisjoint(train_patients)
        assert test_patients.isdisjoint(validation_patients)
        assert train_patients.isdisjoint(validation_patients)
        assert len(test_patients | train_patients | validation_patients) == 23
        train_events = sum(r.seizure_count for r in split.train_recordings)
        validation_events = sum(r.seizure_count for r in split.validation_recordings)
        assert train_events > 0 and validation_events > 0
        print(f"Fold {fold.fold_id:02d}: test={fold.test_patient_id}, "
              f"fit_patients={len(train_patients)}, validation_patients={len(validation_patients)}, "
              f"fit_events={train_events}, validation_events={validation_events}")

    assert len(folds) == 23 and len(recordings) == 683
    assert sum(r.seizure_count for r in recordings) == 185
    print("\nPASS: nested splits are patient-disjoint and contain seizure events.")
    print("PASS: all 23 primary-cohort LOPO folds are valid.")


if __name__ == "__main__":
    main()
