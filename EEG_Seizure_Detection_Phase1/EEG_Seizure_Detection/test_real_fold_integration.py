"""Real-data integration check for patient-wise LOPO fold preparation.

This script reads existing processed NPZ files. It does not train models or
modify the source dataset.
"""

from pathlib import Path

import numpy as np

from data_management.patient_mapping import load_recording_manifest
from data_management.cohort import filter_primary_cohort
from data_management.dataset_split import build_lopo_folds
from data_management.fold_data import prepare_lopo_fold


AUDIT = Path(r"..\..\docs\full_annotation_audit.csv")
PROCESSED = Path("processed")

SAMPLE_PATIENTS = {"patient_01", "patient_02", "patient_03"}
EXPECTED_RECORDINGS = 11


def select_deterministic_sample(recordings, available_files):
    """Select exactly N recordings while guaranteeing every target patient."""
    eligible = [
        recording
        for recording in recordings
        if recording.patient_id in SAMPLE_PATIENTS
        and f"{recording.subject}_{Path(recording.edf_file).stem}.npz"
        in available_files
    ]

    by_patient = {
        patient_id: [
            recording
            for recording in eligible
            if recording.patient_id == patient_id
        ]
        for patient_id in sorted(SAMPLE_PATIENTS)
    }

    missing_patients = [
        patient_id
        for patient_id, patient_recordings in by_patient.items()
        if not patient_recordings
    ]
    if missing_patients:
        raise RuntimeError(
            "No processed recordings available for: "
            + ", ".join(missing_patients)
        )

    if len(eligible) < EXPECTED_RECORDINGS:
        raise RuntimeError(
            f"Need {EXPECTED_RECORDINGS} eligible recordings, "
            f"but found {len(eligible)}."
        )

    # Reserve one recording per patient first, then fill remaining slots
    # in the original deterministic manifest order.
    selected = [by_patient[patient_id][0] for patient_id in sorted(SAMPLE_PATIENTS)]
    selected_paths = {recording.edf_file for recording in selected}

    for recording in eligible:
        if len(selected) == EXPECTED_RECORDINGS:
            break
        if recording.edf_file not in selected_paths:
            selected.append(recording)
            selected_paths.add(recording.edf_file)

    if len(selected) != EXPECTED_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDINGS} unique recordings, "
            f"selected {len(selected)}."
        )

    selected_patients = {recording.patient_id for recording in selected}
    if selected_patients != SAMPLE_PATIENTS:
        raise AssertionError(
            f"Expected patients {sorted(SAMPLE_PATIENTS)}, "
            f"selected {sorted(selected_patients)}."
        )

    return selected


def main():
    print("1. Loading audited manifest...")
    recordings = filter_primary_cohort(load_recording_manifest(AUDIT))

    print("Primary cohort recordings:", len(recordings))
    print("Independent patients:", len({r.patient_id for r in recordings}))

    print("\n2. Selecting deterministic processed sample...")
    if not PROCESSED.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {PROCESSED.resolve()}"
        )

    available = {path.name for path in PROCESSED.glob("*.npz")}
    sample = select_deterministic_sample(recordings, available)

    print("Selected recordings:", len(sample))
    print(
        "Selected patient counts:",
        {
            patient_id: sum(r.patient_id == patient_id for r in sample)
            for patient_id in sorted(SAMPLE_PATIENTS)
        },
    )

    print("\n3. Building patient-wise LOPO folds...")
    folds = build_lopo_folds(sample)
    assert len(folds) == len(SAMPLE_PATIENTS), (
        f"Expected {len(SAMPLE_PATIENTS)} folds, found {len(folds)}"
    )

    for fold in folds:
        print(f"\nFold {fold.fold_id}: test={fold.test_patient_id}")
        prepared = prepare_lopo_fold(fold, PROCESSED)

        fit_patients = {r.patient_id for r in prepared.fit_recordings}
        val_patients = {r.patient_id for r in prepared.validation_recordings}
        test_patients = {r.patient_id for r in prepared.test_recordings}

        print("Fit patients:", sorted(fit_patients))
        print("Validation patients:", sorted(val_patients))
        print("Test patients:", sorted(test_patients))

        assert len(fit_patients) == 1
        assert len(val_patients) == 1
        assert test_patients == {fold.test_patient_id}
        assert not (fit_patients & val_patients)
        assert not (fit_patients & test_patients)
        assert not (val_patients & test_patients)

        for split in ("fit", "validation", "test"):
            items = list(prepared.iter_split(split))
            assert items, f"{split} split is empty"

            for item in items:
                assert item.X.ndim == 3
                assert item.X.shape[1] == 17
                assert item.X.shape[2] == 2560
                assert len(item.X) == len(item.y) == len(item.starts)
                assert np.isfinite(item.X).all()

            windows = sum(len(item.y) for item in items)
            positives = sum(int(item.y.sum()) for item in items)

            print(
                f"{split}: {len(items)} recordings, "
                f"{windows} windows, {positives} seizure windows"
            )

        assert prepared.standardizer.mean_.shape == (17,)
        assert prepared.standardizer.scale_.shape == (17,)
        print("Scaler mean shape:", prepared.standardizer.mean_.shape)
        print("Scaler scale shape:", prepared.standardizer.scale_.shape)

    print("\nSUCCESS: All three real-data fold integration checks passed.")


if __name__ == "__main__":
    main()
