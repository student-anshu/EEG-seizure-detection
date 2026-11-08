import argparse
import csv
from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_DIR, LOG_DIR, PREFERRED_CHANNELS, WINDOW_SECONDS
)
from data_management.cohort import filter_primary_cohort
from data_management.fold_data import load_recording_windows
from data_management.patient_mapping import load_recording_manifest
from main import NORMALIZATION_POLICY, _metadata_is_consistent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=r"..\..\docs\full_annotation_audit.csv"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write metadata only after validation"
    )
    args = parser.parse_args()

    processed_dir = Path(PROCESSED_DIR)
    log_path = Path(LOG_DIR) / "full_cohort_preprocessing.csv"

    if not log_path.is_file():
        raise FileNotFoundError(f"Preprocessing log missing: {log_path}")

    latest = {}
    with log_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latest[row["relative_path"]] = row

    failed = [
        path for path, row in latest.items()
        if row["status"] == "FAILED"
    ]

    if not failed:
        print("No currently failed recordings found.")
        return

    unexpected = [
        path for path in failed
        if "No space left on device" not in latest[path]["message"]
    ]
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} failures have unexpected causes. "
            "Recovery stopped."
        )

    recordings = filter_primary_cohort(
        load_recording_manifest(args.audit)
    )
    recording_map = {r.relative_path: r for r in recordings}

    missing_from_manifest = [
        path for path in failed if path not in recording_map
    ]
    if missing_from_manifest:
        raise RuntimeError(
            f"Failed paths missing from primary cohort: "
            f"{missing_from_manifest[:5]}"
        )

    validated = 0
    written = 0
    already_present = 0
    errors = []

    print(f"Failed recordings to inspect: {len(failed)}")
    print(f"Apply mode: {args.apply}")

    for index, relative_path in enumerate(failed, start=1):
        recording = recording_map[relative_path]
        output_name = (
            f"{recording.subject}_{Path(recording.edf_file).stem}"
        )
        metadata_path = processed_dir / f"{output_name}_metadata.csv"
        temp_path = processed_dir / f"{output_name}_metadata.recovery.tmp"

        try:
            if metadata_path.exists():
                already_present += 1
                print(f"[{index}/{len(failed)}] EXISTS: {relative_path}")
                continue

            windows = load_recording_windows(processed_dir, recording)
            validated += 1

            metadata = pd.DataFrame({
                "subject": recording.subject,
                "source_file": recording.edf_file,
                "window_start_seconds": windows.starts,
                "window_end_seconds": windows.starts + WINDOW_SECONDS,
                "sampling_frequency_hz": windows.sfreq,
                "channels": "|".join(PREFERRED_CHANNELS),
                "normalization": NORMALIZATION_POLICY,
                "label": windows.y,
            })

            if args.apply:
                metadata.to_csv(temp_path, index=False)

                if not _metadata_is_consistent(
                    recording, temp_path, windows
                ):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError(
                        "Reconstructed metadata failed validation"
                    )

                temp_path.replace(metadata_path)
                written += 1

            print(
                f"[{index}/{len(failed)}] "
                f"{'RECOVERED' if args.apply else 'VALIDATED'}: "
                f"{relative_path} ({len(windows.y)} windows)"
            )

        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            errors.append((relative_path, str(exc)))
            print(f"[{index}/{len(failed)}] ERROR: {relative_path}: {exc}")

    print("\nRecovery audit summary")
    print(f"Failed recordings inspected: {len(failed)}")
    print(f"NPZ files validated: {validated}")
    print(f"Metadata files written: {written}")
    print(f"Metadata already present: {already_present}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for path, message in errors:
            print(f"{path}: {message}")

    if not args.apply:
        print("\nDRY RUN ONLY: No metadata files were written.")


if __name__ == "__main__":
    main()
