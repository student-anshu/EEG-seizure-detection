"""Patient-grouped inner validation splits for outer LOPO training data."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from data_management.patient_mapping import Recording


@dataclass(frozen=True, slots=True)
class PatientValidationSplit:
    """A leakage-safe train/validation split made only from outer training data."""

    train_recordings: tuple[Recording, ...]
    validation_recordings: tuple[Recording, ...]
    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]


def build_patient_validation_split(
    recordings: Sequence[Recording],
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> PatientValidationSplit:
    """Hold out complete patients for validation using a reproducible shuffle.

    Call this function with an outer LOPO fold's train_recordings only.
    The outer test patient must never be passed to this function.

    The validation size is ceil(validation_fraction * patient_count), with at
    least one patient on each side. All recordings from a selected patient
    remain together. This is a deterministic group holdout, not a random
    window split and not a substitute for the untouched outer test fold.
    """
    if not recordings:
        raise ValueError("Cannot split an empty recording list")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    by_patient: dict[str, list[Recording]] = {}
    seen_paths: set[str] = set()
    for recording in recordings:
        if recording.relative_path in seen_paths:
            raise ValueError(f"Duplicate recording path: {recording.relative_path}")
        seen_paths.add(recording.relative_path)
        by_patient.setdefault(recording.patient_id, []).append(recording)

    patient_ids = sorted(by_patient)
    if len(patient_ids) < 2:
        raise ValueError("Patient validation split requires at least two patients")

    validation_count = max(1, math.ceil(len(patient_ids) * validation_fraction))
    validation_count = min(validation_count, len(patient_ids) - 1)

    shuffled_patient_ids = patient_ids.copy()
    random.Random(random_state).shuffle(shuffled_patient_ids)
    validation_patient_ids = tuple(sorted(shuffled_patient_ids[:validation_count]))
    validation_patient_set = set(validation_patient_ids)
    train_patient_ids = tuple(
        patient_id for patient_id in patient_ids
        if patient_id not in validation_patient_set
    )

    train_recordings = tuple(sorted(
        (
            recording
            for patient_id in train_patient_ids
            for recording in by_patient[patient_id]
        ),
        key=lambda item: item.relative_path,
    ))
    validation_recordings = tuple(sorted(
        (
            recording
            for patient_id in validation_patient_ids
            for recording in by_patient[patient_id]
        ),
        key=lambda item: item.relative_path,
    ))

    train_paths = {item.relative_path for item in train_recordings}
    validation_paths = {item.relative_path for item in validation_recordings}
    if train_paths & validation_paths:
        raise RuntimeError("Recording leakage detected between train and validation")
    if train_paths | validation_paths != seen_paths:
        raise RuntimeError("Patient validation split lost or added recordings")

    if {item.patient_id for item in train_recordings} != set(train_patient_ids):
        raise RuntimeError("Train patient assignment is inconsistent")
    if {item.patient_id for item in validation_recordings} != validation_patient_set:
        raise RuntimeError("Validation patient assignment is inconsistent")

    return PatientValidationSplit(
        train_recordings=train_recordings,
        validation_recordings=validation_recordings,
        train_patient_ids=train_patient_ids,
        validation_patient_ids=validation_patient_ids,
    )
