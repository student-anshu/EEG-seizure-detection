"""Patient-wise outer evaluation folds for CHB-MIT recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from data_management.patient_mapping import Recording


@dataclass(frozen=True, slots=True)
class LopoFold:
    """One Leave-One-Patient-Out outer evaluation fold."""

    fold_id: int
    test_patient_id: str
    train_recordings: tuple[Recording, ...]
    test_recordings: tuple[Recording, ...]


def build_lopo_folds(recordings: Sequence[Recording]) -> list[LopoFold]:
    """Build deterministic LOPO folds, holding out each patient exactly once.

    Every recording belonging to the held-out patient is test data; all
    recordings from every other patient are training data. This function
    creates outer folds only. Any model-selection validation split must be
    made from the training patients inside each fold, never from its test
    patient.
    """
    if not recordings:
        raise ValueError("Cannot create LOPO folds from an empty recording list")

    by_patient: dict[str, list[Recording]] = {}
    seen_paths: set[str] = set()
    for recording in recordings:
        if recording.relative_path in seen_paths:
            raise ValueError(f"Duplicate recording path: {recording.relative_path}")
        seen_paths.add(recording.relative_path)
        by_patient.setdefault(recording.patient_id, []).append(recording)

    patient_ids = sorted(by_patient)
    if len(patient_ids) < 2:
        raise ValueError("LOPO requires at least two independent patients")

    folds: list[LopoFold] = []
    for fold_id, test_patient_id in enumerate(patient_ids, start=1):
        test_recordings = tuple(
            sorted(by_patient[test_patient_id], key=lambda item: item.relative_path)
        )
        train_recordings = tuple(
            sorted(
                (
                    recording
                    for patient_id, patient_recordings in by_patient.items()
                    if patient_id != test_patient_id
                    for recording in patient_recordings
                ),
                key=lambda item: item.relative_path,
            )
        )

        train_patients = {item.patient_id for item in train_recordings}
        test_patients = {item.patient_id for item in test_recordings}
        if train_patients & test_patients:
            raise RuntimeError("Patient leakage detected while building LOPO fold")
        if test_patients != {test_patient_id}:
            raise RuntimeError("LOPO fold contains an unexpected test patient")

        folds.append(
            LopoFold(
                fold_id=fold_id,
                test_patient_id=test_patient_id,
                train_recordings=train_recordings,
                test_recordings=test_recordings,
            )
        )

    # Every input recording must appear in exactly one outer test fold.
    test_paths = [
        recording.relative_path
        for fold in folds
        for recording in fold.test_recordings
    ]
    if len(test_paths) != len(set(test_paths)) or set(test_paths) != seen_paths:
        raise RuntimeError("LOPO folds do not test every recording exactly once")

    return folds
