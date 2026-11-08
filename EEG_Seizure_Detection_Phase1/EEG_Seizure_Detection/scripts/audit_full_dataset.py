import csv
import sys
from collections import Counter
from pathlib import Path

# Make the project code directory importable when this file is run directly
# with: python .\\scripts\\audit_full_dataset.py
PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from config import DATASET_DIR
from preprocessing.annotations import (
    is_officially_unannotated_recording,
    read_seizure_intervals,
)


def main() -> int:
    if not DATASET_DIR.is_dir():
        print(f"ERROR: Dataset directory does not exist: {DATASET_DIR}")
        return 2

    edf_files = sorted(
        path for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".edf"
    )
    if not edf_files:
        print(f"ERROR: No EDF recordings found in: {DATASET_DIR}")
        return 2

    print(f"Dataset: {DATASET_DIR}")
    print(f"EDF recordings discovered: {len(edf_files)}")
    print("Starting annotation audit...")

    rows = []
    subject_counts = Counter()
    successful = 0
    failed = 0
    total_seizure_events = 0
    zero_seizure_recordings = 0
    manifest_confirmed_negative = 0
    failed_rows = []

    for index, edf_path in enumerate(edf_files, start=1):
        subject = edf_path.parent.name
        subject_counts[subject] += 1

        try:
            manifest_negative = is_officially_unannotated_recording(edf_path)
            intervals = read_seizure_intervals(edf_path)
            successful += 1
            total_seizure_events += len(intervals)

            if not intervals:
                zero_seizure_recordings += 1
            if manifest_negative:
                manifest_confirmed_negative += 1

            rows.append({
                "subject": subject,
                "edf_file": edf_path.name,
                "relative_path": edf_path.relative_to(DATASET_DIR).as_posix(),
                "seizure_count": len(intervals),
                "seizure_intervals_seconds": "; ".join(
                    f"{start:g}-{end:g}" for start, end in intervals
                ),
                "annotation_source": (
                    "official_manifests_no_seizure"
                    if manifest_negative else "subject_summary"
                ),
                "status": "SUCCESS",
                "error": "",
            })
        except Exception as exc:
            failed += 1
            row = {
                "subject": subject,
                "edf_file": edf_path.name,
                "relative_path": edf_path.relative_to(DATASET_DIR).as_posix(),
                "seizure_count": "",
                "seizure_intervals_seconds": "",
                "annotation_source": "",
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            failed_rows.append(row)

        if index % 25 == 0 or index == len(edf_files):
            print(f"Audited {index}/{len(edf_files)} recordings")

    # scripts/audit_full_dataset.py -> repository root is parents[3].
    repository_root = Path(__file__).resolve().parents[3]
    report_dir = repository_root / "docs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "full_annotation_audit.csv"

    fieldnames = [
        "subject",
        "edf_file",
        "relative_path",
        "seizure_count",
        "seizure_intervals_seconds",
        "annotation_source",
        "status",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== FULL ANNOTATION AUDIT ==========")
    print(f"EDF recordings discovered: {len(edf_files)}")
    print(f"Successfully parsed: {successful}")
    print(f"Failed: {failed}")
    print(f"Zero-seizure recordings: {zero_seizure_recordings}")
    print(f"Manifest-confirmed negative recordings: {manifest_confirmed_negative}")
    print(f"Annotated seizure events: {total_seizure_events}")
    print(f"Subjects represented: {len(subject_counts)}")
    print(f"CSV report: {report_path}")

    if failed_rows:
        print("\nFailed recordings (up to 30):")
        for row in failed_rows[:30]:
            print(f"- {row['relative_path']}: {row['error']}")
        return 1

    print("\nAll discovered EDF annotations were parsed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
