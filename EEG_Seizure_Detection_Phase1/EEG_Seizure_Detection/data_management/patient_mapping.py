"""Patient identity mapping and validated CHB-MIT audit manifest loading."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_SUBJECT_PATTERN = re.compile(r"^chb(?P<number>\d{2})$", re.IGNORECASE)
_REQUIRED_COLUMNS = {
    "subject",
    "edf_file",
    "relative_path",
    "seizure_count",
    "seizure_intervals_seconds",
    "annotation_source",
    "status",
}


@dataclass(frozen=True, slots=True)
class Recording:
    """One audited EDF recording and its patient-level identity."""

    subject: str
    patient_id: str
    edf_file: str
    relative_path: str
    seizure_count: int
    seizure_intervals_seconds: tuple[str, ...]
    annotation_source: str
    status: str


def patient_id_for_subject(subject: str) -> str:
    """Return the canonical patient ID for a CHB-MIT subject folder.

    CHB01 and CHB21 are recordings from the same individual and therefore
    intentionally share patient_01. All other CHB folders retain their
    numeric subject identity (for example, CHB12 -> patient_12).
    """
    normalized = subject.strip().lower()
    match = _SUBJECT_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError(f"Invalid CHB-MIT subject identifier: {subject!r}")

    subject_number = int(match.group("number"))
    if subject_number == 21:
        subject_number = 1
    return f"patient_{subject_number:02d}"


def load_recording_manifest(csv_path: str | Path) -> list[Recording]:
    """Load and validate the full annotation audit CSV.

    The loader intentionally fails closed: incomplete columns, failed audit
    rows, invalid counts, or duplicate recording paths must be corrected
    before the manifest can be used for cross-patient evaluation.
    """
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Annotation audit CSV not found: {path}")

    recordings: list[Recording] = []
    seen_paths: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Annotation audit CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            subject = (row.get("subject") or "").strip().lower()
            edf_file = (row.get("edf_file") or "").strip()
            relative_path = (row.get("relative_path") or "").strip().replace("\\", "/")
            status = (row.get("status") or "").strip().upper()

            if status != "SUCCESS":
                raise ValueError(
                    f"Audit row {line_number} is not successful: {status!r}"
                )
            if not subject or not edf_file or not relative_path:
                raise ValueError(f"Audit row {line_number} has an empty recording identity")
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate recording path in audit CSV: {relative_path}")

            try:
                seizure_count = int(row["seizure_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid seizure_count on audit row {line_number}"
                ) from exc
            if seizure_count < 0:
                raise ValueError(f"Negative seizure_count on audit row {line_number}")

            raw_intervals = (row.get("seizure_intervals_seconds") or "").strip()
            intervals = tuple(
                item.strip() for item in raw_intervals.split(";") if item.strip()
            )
            if len(intervals) != seizure_count:
                raise ValueError(
                    f"Audit row {line_number} declares {seizure_count} seizure(s) "
                    f"but contains {len(intervals)} interval(s)"
                )

            recordings.append(
                Recording(
                    subject=subject,
                    patient_id=patient_id_for_subject(subject),
                    edf_file=edf_file,
                    relative_path=relative_path,
                    seizure_count=seizure_count,
                    seizure_intervals_seconds=intervals,
                    annotation_source=(row.get("annotation_source") or "").strip(),
                    status=status,
                )
            )
            seen_paths.add(relative_path)

    if not recordings:
        raise ValueError("Annotation audit CSV contains no recording rows")

    return recordings
