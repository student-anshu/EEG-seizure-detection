import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATASET_DIR, PROCESSED_DIR, LOG_DIR, LOW_CUT_HZ, HIGH_CUT_HZ,
    NOTCH_HZ, WINDOW_SECONDS, STRIDE_SECONDS, PREFERRED_CHANNELS,
)
from data_management.cohort import filter_primary_cohort
from data_management.fold_data import load_recording_windows
from data_management.patient_mapping import load_recording_manifest
from preprocessing.load_data import load_edf
from preprocessing.annotations import read_seizure_intervals
from preprocessing.signal_processing import bandpass_and_notch
from preprocessing.segmentation import make_windows

NORMALIZATION_POLICY = "none_fold_train_only_scaling_required"
_LOG_FIELDS = ("relative_path", "status", "window_count", "normal_windows",
               "seizure_windows", "message")


def _normalize_channel_name(name: str) -> str:
    return "".join(str(name).upper().split())


def choose_channels(raw):
    """Select all canonical channels in fixed order or reject the recording."""
    lookup = {_normalize_channel_name(name): name for name in raw.ch_names}
    missing = [name for name in PREFERRED_CHANNELS if _normalize_channel_name(name) not in lookup]
    if missing:
        raise ValueError(
            "Recording is incompatible with the fixed 17-channel primary montage. "
            "Missing channels: " + ", ".join(missing)
        )
    return [lookup[_normalize_channel_name(name)] for name in PREFERRED_CHANNELS]


def process_edf(edf_path: str | Path):
    edf_path = Path(edf_path)
    print(f"\nLoading: {edf_path}")
    raw = load_edf(edf_path)
    channels = choose_channels(raw)
    raw.pick(channels)
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()
    recording_duration_seconds = float(data.shape[-1] / sfreq)
    intervals = read_seizure_intervals(edf_path)
    print(f"Channels selected: {len(channels)}")
    print(f"Sampling frequency: {sfreq:g} Hz")
    print(f"Seizure intervals: {intervals if intervals else 'None'}")
    filtered = bandpass_and_notch(data, sfreq, LOW_CUT_HZ, HIGH_CUT_HZ, NOTCH_HZ)
    # Keep filtered windows unnormalized here. A scaler must be fitted only
    # on the outer-fold training patients and then reused for validation/test.
    X, y, starts = make_windows(filtered, sfreq, intervals, WINDOW_SECONDS, STRIDE_SECONDS)
    if len(y) == 0:
        print("Recording is shorter than one configured window; skipped.")
        return None
    subject = edf_path.parent.name
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{subject}_{edf_path.stem}"
    npz_path = PROCESSED_DIR / f"{output_name}.npz"
    np.savez_compressed(
        npz_path, X=X, y=y, starts=starts, channels=np.asarray(channels),
        sfreq=sfreq, subject=subject, source_file=edf_path.name,
        recording_duration_seconds=recording_duration_seconds,
        normalization=NORMALIZATION_POLICY,
    )
    metadata = pd.DataFrame({
        "subject": subject, "source_file": edf_path.name,
        "window_start_seconds": starts, "window_end_seconds": starts + WINDOW_SECONDS,
        "sampling_frequency_hz": sfreq,
        "recording_duration_seconds": recording_duration_seconds,
        "channels": "|".join(channels),
        "normalization": NORMALIZATION_POLICY, "label": y,
    })
    metadata.to_csv(PROCESSED_DIR / f"{output_name}_metadata.csv", index=False)
    print(f"Saved {len(y)} windows: {np.bincount(y, minlength=2).tolist()} [normal, seizure]")
    print(f"Saved: {npz_path}")
    return npz_path


def _metadata_is_consistent(recording, metadata_path: Path, windows) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = pd.read_csv(metadata_path)
        required = {"window_start_seconds", "label", "normalization"}
        if not required.issubset(metadata.columns):
            return False
        if len(metadata) != len(windows.y):
            return False
        if not np.array_equal(metadata["window_start_seconds"].to_numpy(), windows.starts):
            return False
        if not np.array_equal(metadata["label"].to_numpy(), windows.y):
            return False
        if "recording_duration_seconds" in metadata.columns:
            durations = metadata["recording_duration_seconds"].to_numpy(dtype=float)
            if not np.isfinite(durations).all() or not np.allclose(
                durations, durations[0], rtol=0.0, atol=1e-9
            ):
                return False
        return bool((metadata["normalization"] == NORMALIZATION_POLICY).all())
    except Exception:
        return False


def _processed_recording_is_valid(recording, processed_dir: Path) -> bool:
    try:
        windows = load_recording_windows(processed_dir, recording)
        metadata_path = processed_dir / (
            f"{recording.subject}_{Path(recording.edf_file).stem}_metadata.csv"
        )
        return _metadata_is_consistent(recording, metadata_path, windows)
    except Exception:
        # Any unreadable, corrupt, or incompatible artifact must be regenerated.
        return False


def _append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_primary_cohort(
    audit_csv: str | Path,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Process the audited primary cohort, resuming valid outputs by default.

    Each recording is validated and logged immediately, so an interrupted run
    can be restarted safely without reprocessing completed recordings.
    """
    dataset_root = DATASET_DIR.resolve()
    processed_dir = PROCESSED_DIR
    log_path = LOG_DIR / "full_cohort_preprocessing.csv"
    recordings = filter_primary_cohort(load_recording_manifest(audit_csv))
    counts = {"processed": 0, "skipped_valid": 0, "failed": 0}
    print(f"Primary cohort recordings: {len(recordings)}")
    print(f"Dataset directory: {dataset_root}")
    print(f"Processed directory: {processed_dir}")

    for index, recording in enumerate(recordings, start=1):
        relative_path = Path(recording.relative_path)
        edf_path = (dataset_root / relative_path).resolve()
        try:
            # Reject paths that escape the configured dataset root.
            edf_path.relative_to(dataset_root)
            if not edf_path.is_file():
                raise FileNotFoundError(f"EDF file not found: {edf_path}")

            if not force and _processed_recording_is_valid(recording, processed_dir):
                row = {
                    "relative_path": recording.relative_path,
                    "status": "SKIPPED_VALID",
                    "window_count": "",
                    "normal_windows": "",
                    "seizure_windows": "",
                    "message": "Existing NPZ and metadata passed validation",
                }
                counts["skipped_valid"] += 1
            else:
                saved_path = process_edf(edf_path)
                if saved_path is None:
                    raise ValueError("Recording produced no complete windows")
                windows = load_recording_windows(processed_dir, recording)
                metadata_path = processed_dir / (
                    f"{recording.subject}_{Path(recording.edf_file).stem}_metadata.csv"
                )
                if not _metadata_is_consistent(recording, metadata_path, windows):
                    raise ValueError("Saved metadata does not match NPZ windows")
                row = {
                    "relative_path": recording.relative_path,
                    "status": "PROCESSED",
                    "window_count": len(windows.y),
                    "normal_windows": int(np.sum(windows.y == 0)),
                    "seizure_windows": int(np.sum(windows.y == 1)),
                    "message": "",
                }
                counts["processed"] += 1
        except Exception as exc:
            row = {
                "relative_path": recording.relative_path,
                "status": "FAILED",
                "window_count": "",
                "normal_windows": "",
                "seizure_windows": "",
                "message": f"{type(exc).__name__}: {exc}",
            }
            counts["failed"] += 1

        _append_log(log_path, row)
        print(
            f"[{index}/{len(recordings)}] {row['status']}: "
            f"{recording.relative_path}"
            + (f" — {row['message']}" if row["message"] else "")
        )

    print(
        "\nCohort preprocessing summary: "
        f"processed={counts['processed']}, "
        f"skipped_valid={counts['skipped_valid']}, failed={counts['failed']}"
    )
    print(f"Run log: {log_path}")
    return counts


def inspect_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}\nDownload and extract CHB-MIT into dataset/chbmit.")
    edfs = sorted(DATASET_DIR.rglob("*.edf"))
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"EDF files found: {len(edfs)}")
    for path in edfs[:20]:
        print(f"  {path.relative_to(DATASET_DIR)}")
    if len(edfs) > 20:
        print(f"  ... and {len(edfs) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="EEG seizure data preparation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--edf", type=str, help="Process one EDF file")
    group.add_argument("--all", action="store_true", help="Legacy: process every EDF without the audited cohort policy")
    group.add_argument("--prepare-cohort", action="store_true", help="Process the audited 17-channel primary cohort with resume support")
    parser.add_argument("--audit", type=Path, help="Path to full_annotation_audit.csv (required with --prepare-cohort)")
    parser.add_argument("--force", action="store_true", help="Reprocess recordings even when existing outputs pass validation")
    args = parser.parse_args()

    if args.prepare_cohort:
        if args.audit is None:
            parser.error("--audit is required with --prepare-cohort")
        counts = prepare_primary_cohort(args.audit, force=args.force)
        if counts["failed"]:
            raise SystemExit(1)
    elif args.audit is not None or args.force:
        parser.error("--audit and --force can only be used with --prepare-cohort")
    elif args.inspect:
        inspect_dataset()
    elif args.edf:
        process_edf(args.edf)
    elif args.all:
        print("WARNING: --all bypasses the audited primary-cohort filter; use --prepare-cohort instead.")
        edfs = sorted(DATASET_DIR.rglob("*.edf"))
        if not edfs:
            raise FileNotFoundError(f"No EDF files found under {DATASET_DIR}")
        processed = skipped = 0
        for path in edfs:
            try:
                result = process_edf(path)
                processed += result is not None
                skipped += result is None
            except Exception as exc:
                skipped += 1
                print(f"SKIPPED {path}: {exc}")
        print(f"\nDataset processing finished: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
