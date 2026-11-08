"""Safely backfill exact EDF durations without loading EEG signals into memory.

The utility updates per-recording metadata CSVs and a central JSON manifest.
It deliberately does not rewrite large processed NPZ archives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import mne
import numpy as np
import pandas as pd

from config import DATASET_DIR, PROCESSED_DIR
from data_management.cohort import filter_primary_cohort
from data_management.fold_data import processed_path_for_recording
from data_management.patient_mapping import Recording, load_recording_manifest
from evaluation.event_metrics import parse_seizure_intervals


def _read_edf_header(path: Path) -> tuple[float, float, int]:
    """Return duration seconds, sampling frequency and sample count, no preload."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration = n_times / sfreq
    finally:
        raw.close()
    if not np.isfinite(sfreq) or sfreq <= 0 or n_times <= 0:
        raise ValueError(f"Invalid EDF header timing values: {path}")
    return float(duration), sfreq, n_times


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "recordings": {}}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("recordings"), dict
    ):
        raise ValueError(f"Unsupported or corrupt duration manifest: {path}")
    return payload


def backfill_recording_durations(
    recordings: Sequence[Recording],
    *,
    dataset_dir: str | Path,
    processed_dir: str | Path,
    manifest_path: str | Path | None = None,
    dry_run: bool = False,
    header_reader: Callable[[Path], tuple[float, float, int]] = _read_edf_header,
) -> dict[str, int]:
    """Backfill one recording at a time; completed recordings are restartable."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    processed_root = Path(processed_dir).expanduser().resolve()
    destination = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else processed_root / "recording_durations.json"
    )
    manifest = _load_manifest(destination)
    entries = manifest["recordings"]
    counts = {"updated": 0, "skipped_valid": 0, "failed": 0}
    errors: list[dict[str, str]] = []

    for recording in recordings:
        relative = Path(recording.relative_path)
        edf_path = (dataset_root / relative).resolve()
        try:
            edf_path.relative_to(dataset_root)
            if not edf_path.is_file():
                raise FileNotFoundError(f"EDF not found: {edf_path}")

            npz_path = processed_path_for_recording(processed_root, recording)
            if not npz_path.is_file():
                raise FileNotFoundError(f"Processed NPZ not found: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as saved:
                required = {"y", "starts", "subject", "source_file"}
                missing = required - set(saved.files)
                if missing:
                    raise ValueError(
                        f"{npz_path.name} missing arrays: {', '.join(sorted(missing))}"
                    )
                labels = np.asarray(saved["y"])
                starts = np.asarray(saved["starts"], dtype=np.float64)
                if str(saved["subject"]) != recording.subject:
                    raise ValueError(f"NPZ subject mismatch: {npz_path.name}")
                if str(saved["source_file"]) != recording.edf_file:
                    raise ValueError(f"NPZ source filename mismatch: {npz_path.name}")
            if labels.ndim != 1 or starts.ndim != 1 or len(labels) == 0 or len(labels) != len(starts):
                raise ValueError(f"Invalid window metadata arrays: {npz_path.name}")

            metadata_path = processed_root / (
                f"{recording.subject}_{Path(recording.edf_file).stem}_metadata.csv"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
            metadata = pd.read_csv(metadata_path)
            required_columns = {"window_start_seconds", "label"}
            if not required_columns.issubset(metadata.columns):
                raise ValueError(f"Metadata CSV lacks required columns: {metadata_path.name}")
            if len(metadata) != len(labels):
                raise ValueError(f"Metadata row count mismatch: {metadata_path.name}")
            if not np.allclose(
                metadata["window_start_seconds"].to_numpy(dtype=float),
                starts, rtol=0.0, atol=1e-9,
            ) or not np.array_equal(metadata["label"].to_numpy(), labels):
                raise ValueError(f"Metadata does not align with NPZ windows: {metadata_path.name}")

            stat = edf_path.stat()
            prior = entries.get(recording.relative_path, {})
            signature_matches = (
                prior.get("edf_size_bytes") == stat.st_size
                and prior.get("edf_mtime_ns") == stat.st_mtime_ns
            )
            if signature_matches:
                duration = float(prior["recording_duration_seconds"])
                sfreq = float(prior["sampling_frequency_hz"])
                n_times = int(prior["sample_count"])
            else:
                duration, sfreq, n_times = header_reader(edf_path)

            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration for {recording.relative_path}")
            if len(starts) and np.any(starts < 0):
                raise ValueError(f"Negative window start in {npz_path.name}")
            if len(starts) and np.any(starts + (metadata["window_end_seconds"].iloc[0] - metadata["window_start_seconds"].iloc[0] if "window_end_seconds" in metadata.columns else 10.0) > duration + 1e-6):
                raise ValueError(f"Saved windows exceed EDF duration: {npz_path.name}")
            for interval in parse_seizure_intervals(recording.seizure_intervals_seconds):
                if interval.end > duration + 1e-6:
                    raise ValueError(
                        f"Annotated seizure interval ends beyond EDF duration: {recording.relative_path}"
                    )

            if "recording_duration_seconds" in metadata.columns:
                existing = metadata["recording_duration_seconds"].to_numpy(dtype=float)
                metadata_is_current = (
                    np.isfinite(existing).all()
                    and np.allclose(existing, duration, rtol=0.0, atol=1e-9)
                )
            else:
                metadata_is_current = False
            if (
                signature_matches and metadata_is_current
                and prior.get("sample_count") == n_times
                and prior.get("sampling_frequency_hz") == sfreq
            ):
                counts["skipped_valid"] += 1
                continue

            entry = {
                "subject": recording.subject,
                "patient_id": recording.patient_id,
                "relative_path": recording.relative_path,
                "recording_duration_seconds": float(duration),
                "sampling_frequency_hz": float(sfreq),
                "sample_count": int(n_times),
                "edf_size_bytes": int(stat.st_size),
                "edf_mtime_ns": int(stat.st_mtime_ns),
            }
            if not dry_run:
                metadata["recording_duration_seconds"] = float(duration)
                _atomic_write_csv(metadata_path, metadata)
                entries[recording.relative_path] = entry
                manifest["recordings"] = entries
                _atomic_write_json(destination, manifest)
            counts["updated"] += 1
        except Exception as exc:
            counts["failed"] += 1
            error = {
                "relative_path": recording.relative_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(
                f"FAILED {recording.relative_path}: {error['error']}",
                file=sys.stderr,
            )

    if errors and not dry_run:
        _atomic_write_json(
            destination.with_name("recording_duration_backfill_errors.json"),
            {"errors": errors},
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill exact EDF durations using header-only reads."
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recordings = filter_primary_cohort(load_recording_manifest(args.audit))
    result = backfill_recording_durations(
        recordings,
        dataset_dir=args.dataset_dir,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )
    print(
        "Duration backfill: "
        f"updated={result['updated']}, "
        f"skipped_valid={result['skipped_valid']}, "
        f"failed={result['failed']}"
    )
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
