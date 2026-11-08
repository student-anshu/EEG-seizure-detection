"""Identity-validated recording duration lookup from the committed EDF inventory."""
from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path
import re
from typing import Sequence

from data_management.patient_mapping import Recording

_REQUIRED_COLUMNS = {
    "subject", "edf_file", "relative_path", "sampling_frequency_hz",
    "n_samples", "duration_seconds", "status",
}


def normalize_recording_path(value: str) -> str:
    """Normalize a repository-relative CHB recording path for identity joins."""
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = re.sub(r"/+", "/", path).lower()
    if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"Invalid relative recording path: {value!r}")
    return path


def load_recording_durations(
    inventory_path: str | Path,
    recordings: Sequence[Recording],
    *,
    duration_tolerance_seconds: float = 0.01,
) -> tuple[dict[str, float], dict]:
    """Load exact durations and join by path + subject + EDF filename."""
    path = Path(inventory_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Recording inventory not found: {path}")
    tolerance = float(duration_tolerance_seconds)
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Duration tolerance must be finite and non-negative")

    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Recording inventory missing columns: " + ", ".join(sorted(missing)))
        for line_number, row in enumerate(reader, start=2):
            key = normalize_recording_path(row.get("relative_path", ""))
            if key in rows:
                raise ValueError(f"Duplicate inventory recording path: {key}")
            subject = (row.get("subject") or "").strip().lower()
            edf_file = (row.get("edf_file") or "").strip().lower()
            parts = key.split("/")
            if len(parts) != 2 or subject != parts[0] or edf_file != parts[1]:
                raise ValueError(f"Inventory identity columns disagree on row {line_number}: {key}")
            if (row.get("status") or "").strip().upper() != "SUCCESS":
                raise ValueError(f"Inventory row {line_number} is not SUCCESS: {key}")
            try:
                samples = int(row["n_samples"])
                sampling_frequency = float(row["sampling_frequency_hz"])
                duration = float(row["duration_seconds"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid duration fields on inventory row {line_number}") from exc
            if (
                samples <= 0
                or not isfinite(sampling_frequency) or sampling_frequency <= 0
                or not isfinite(duration) or duration <= 0
            ):
                raise ValueError(f"Invalid duration fields on inventory row {line_number}: {key}")
            calculated = samples / sampling_frequency
            if abs(duration - calculated) > tolerance:
                raise ValueError(
                    f"Duration arithmetic mismatch for {key}: duration={duration}, samples/fs={calculated}"
                )
            rows[key] = {
                "subject": subject, "edf_file": edf_file,
                "duration_seconds": duration,
            }

    duration_by_path: dict[str, float] = {}
    seen: set[str] = set()
    for recording in recordings:
        key = normalize_recording_path(recording.relative_path)
        if key in seen:
            raise ValueError(f"Duplicate primary-cohort recording path: {key}")
        seen.add(key)
        item = rows.get(key)
        if item is None:
            raise ValueError(f"Recording missing from duration inventory: {key}")
        if item["subject"] != recording.subject.lower() or item["edf_file"] != recording.edf_file.lower():
            raise ValueError(f"Recording identity mismatch against inventory: {key}")
        duration_by_path[recording.relative_path] = item["duration_seconds"]

    audit = {
        "inventory_path": str(path),
        "inventory_recording_count": len(rows),
        "matched_primary_recording_count": len(duration_by_path),
        "unmatched_inventory_recording_count": len(rows) - len(duration_by_path),
        "duration_tolerance_seconds": tolerance,
        "join_key": "normalized relative_path, cross-checked against subject and edf_file",
        "duration_validation": "duration_seconds ~= n_samples / sampling_frequency_hz",
    }
    return duration_by_path, audit
