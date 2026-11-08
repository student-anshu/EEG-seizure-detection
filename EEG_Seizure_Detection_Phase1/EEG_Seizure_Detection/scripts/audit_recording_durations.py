"""Read-only audit of exact recording durations in retained processed artifacts.

This script never reads EDF files and never modifies project data.
Run from the EEG_Seizure_Detection directory:
    python -m scripts.audit_recording_durations --audit ..\\..\\docs\\full_annotation_audit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from data_management.cohort import filter_primary_cohort
from data_management.patient_mapping import Recording, load_recording_manifest


def _read_duration_sources(
    recording: Recording,
    processed_dir: Path,
    manifest_entries: dict,
) -> dict[str, float]:
    stem = f"{recording.subject}_{Path(recording.edf_file).stem}"
    npz_path = processed_dir / f"{stem}.npz"
    metadata_path = processed_dir / f"{stem}_metadata.csv"
    sources: dict[str, float] = {}

    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as saved:
            if "recording_duration_seconds" in saved.files:
                value = np.asarray(saved["recording_duration_seconds"])
                if value.size != 1:
                    raise ValueError(f"{npz_path.name}: duration field must be scalar")
                sources["npz"] = float(value.reshape(-1)[0])

    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        if "recording_duration_seconds" in metadata.columns and len(metadata):
            values = metadata["recording_duration_seconds"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.allclose(
                values, values[0], rtol=0.0, atol=1e-9
            ):
                raise ValueError(f"{metadata_path.name}: duration values are invalid or inconsistent")
            sources["metadata_csv"] = float(values[0])

    entry = manifest_entries.get(recording.relative_path)
    if isinstance(entry, dict) and "recording_duration_seconds" in entry:
        sources["duration_manifest"] = float(entry["recording_duration_seconds"])

    for source, value in sources.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{recording.relative_path}: invalid {source} duration {value!r}")
    return sources


def audit_recording_durations(
    recordings: Iterable[Recording],
    *,
    processed_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Count exact duration availability and disagreements without writing files."""
    root = Path(processed_dir)
    manifest_file = (
        Path(manifest_path) if manifest_path is not None
        else root / "recording_durations.json"
    )
    manifest_entries: dict = {}
    if manifest_file.is_file():
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        raw_entries = payload.get("recordings", {})
        if not isinstance(raw_entries, dict):
            raise ValueError("Duration manifest 'recordings' must be an object")
        manifest_entries = raw_entries

    counts = {
        "recordings": 0,
        "npz_duration": 0,
        "metadata_duration": 0,
        "manifest_duration": 0,
        "exact_duration_available": 0,
        "missing_duration": 0,
        "conflicts": 0,
        "artifact_errors": 0,
    }
    missing: list[str] = []
    conflicts: list[dict] = []
    errors: list[dict] = []

    for recording in recordings:
        counts["recordings"] += 1
        try:
            sources = _read_duration_sources(recording, root, manifest_entries)
            counts["npz_duration"] += "npz" in sources
            counts["metadata_duration"] += "metadata_csv" in sources
            counts["manifest_duration"] += "duration_manifest" in sources
            if not sources:
                counts["missing_duration"] += 1
                missing.append(recording.relative_path)
                continue
            values = list(sources.values())
            if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
                counts["conflicts"] += 1
                conflicts.append({"relative_path": recording.relative_path, "sources": sources})
            else:
                counts["exact_duration_available"] += 1
        except Exception as exc:
            counts["artifact_errors"] += 1
            errors.append({
                "relative_path": recording.relative_path,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "counts": counts,
        "missing_examples": missing[:10],
        "conflict_examples": conflicts[:10],
        "error_examples": errors[:10],
        "manifest_path": str(manifest_file),
        "read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of recording durations in retained processed artifacts"
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    recordings = filter_primary_cohort(load_recording_manifest(args.audit))
    result = audit_recording_durations(
        recordings,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2))
    counts = result["counts"]
    if (
        counts["missing_duration"] or counts["conflicts"]
        or counts["artifact_errors"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
