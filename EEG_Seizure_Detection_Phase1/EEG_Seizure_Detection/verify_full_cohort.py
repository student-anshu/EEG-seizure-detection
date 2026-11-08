import csv
from pathlib import Path

from config import PROCESSED_DIR, LOG_DIR
from data_management.cohort import filter_primary_cohort
from data_management.fold_data import load_recording_windows
from data_management.patient_mapping import load_recording_manifest
from main import _metadata_is_consistent

AUDIT = Path(r"..\..\docs\full_annotation_audit.csv")


def main():
    recordings = filter_primary_cohort(load_recording_manifest(AUDIT))
    errors = []
    total_windows = 0
    total_seizure_windows = 0
    total_normal_windows = 0

    print(f"Expected primary cohort: {len(recordings)} recordings")

    for index, recording in enumerate(recordings, 1):
        metadata_path = Path(PROCESSED_DIR) / (
            f"{recording.subject}_{Path(recording.edf_file).stem}_metadata.csv"
        )

        try:
            windows = load_recording_windows(PROCESSED_DIR, recording)

            if not _metadata_is_consistent(
                recording, metadata_path, windows
            ):
                raise ValueError("Metadata missing or inconsistent")

            total_windows += len(windows.y)
            total_seizure_windows += int((windows.y == 1).sum())
            total_normal_windows += int((windows.y == 0).sum())

        except Exception as exc:
            errors.append((recording.relative_path, str(exc)))

        if index % 50 == 0 or index == len(recordings):
            print(f"Checked {index}/{len(recordings)}")

    print("\nFULL COHORT INTEGRITY REPORT")
    print(f"Expected recordings: {len(recordings)}")
    print(f"Valid recordings: {len(recordings) - len(errors)}")
    print(f"Invalid recordings: {len(errors)}")
    print(f"Total windows: {total_windows}")
    print(f"Normal windows: {total_normal_windows}")
    print(f"Seizure windows: {total_seizure_windows}")

    if errors:
        print("\nERROR DETAILS:")
        for path, message in errors:
            print(f"{path}: {message}")
    else:
        print("\nSUCCESS: All recordings and metadata passed validation.")


if __name__ == "__main__":
    main()
