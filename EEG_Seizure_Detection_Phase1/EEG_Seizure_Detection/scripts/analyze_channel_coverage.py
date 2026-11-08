"""Analyze channel coverage from the header-only EDF inventory.

Run from EEG_Seizure_Detection directory:
    python -m scripts.analyze_channel_coverage

Reads docs/signal_inventory.csv and writes docs/channel_coverage.csv.
No EDF signal samples are loaded.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from config import PREFERRED_CHANNELS, PROJECT_ROOT


INVENTORY_PATH = PROJECT_ROOT.parent.parent / "docs" / "signal_inventory.csv"
OUTPUT_PATH = PROJECT_ROOT.parent.parent / "docs" / "channel_coverage.csv"


def normalize_channel_name(name: str) -> str:
    """Match the project's existing case/whitespace-insensitive channel policy."""
    return "".join(str(name).upper().split())


def main() -> None:
    if not INVENTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Signal inventory not found: {INVENTORY_PATH}. "
            "Run: python -m scripts.audit_recording_signals"
        )

    rows: list[dict[str, str]] = []
    exact_counts: Counter[str] = Counter()
    normalized_counts: Counter[str] = Counter()
    preferred_lookup = {
        normalize_channel_name(name): name for name in PREFERRED_CHANNELS
    }
    preferred_counts: Counter[str] = Counter()
    failed_rows: list[str] = []

    with INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "channel_names", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Signal inventory is missing columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            if row["status"].strip().upper() != "SUCCESS":
                failed_rows.append(row.get("relative_path", "unknown"))
                continue

            labels = [
                label.strip()
                for label in (row.get("channel_names") or "").split("|")
                if label.strip()
            ]
            normalized = {normalize_channel_name(label) for label in labels}
            for label in labels:
                exact_counts[label] += 1
            normalized_counts.update(normalized)

            for key in normalized & preferred_lookup.keys():
                preferred_counts[preferred_lookup[key]] += 1

            rows.append({
                "relative_path": row["relative_path"],
                "channel_count": str(len(labels)),
                "preferred_channels_present": str(
                    sum(1 for key in normalized if key in preferred_lookup)
                ),
                "preferred_channels_missing": "|".join(
                    name for name in PREFERRED_CHANNELS
                    if normalize_channel_name(name) not in normalized
                ),
                "normalized_channels": "|".join(sorted(normalized)),
            })

    total = len(rows)
    if total == 0:
        raise ValueError("No successful EDF rows were found in the signal inventory.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "relative_path",
            "channel_count",
            "preferred_channels_present",
            "preferred_channels_missing",
            "normalized_channels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== CHANNEL COVERAGE ANALYSIS ==========")
    print(f"Successful EDF recordings: {total}")
    print(f"Failed inventory rows: {len(failed_rows)}")
    print(f"Distinct exact channel labels: {len(exact_counts)}")
    print(f"Distinct normalized channel labels: {len(normalized_counts)}")

    print("\nPreferred channel coverage:")
    for name in PREFERRED_CHANNELS:
        count = preferred_counts[name]
        percentage = 100.0 * count / total
        print(f"  {name:8s} {count:3d}/{total} recordings ({percentage:6.2f}%)")

    common = sorted(
        (name, count)
        for name, count in normalized_counts.items()
        if count == total
    )
    print(f"\nChannels present in every successful recording: {len(common)}")
    for name, count in common:
        print(f"  {name}: {count}/{total}")

    print("\nMost frequent normalized channel labels:")
    for name, count in normalized_counts.most_common(30):
        print(f"  {name:12s} {count:3d}/{total} ({100.0 * count / total:6.2f}%)")

    coverage_95 = sorted(
        (name, count)
        for name, count in normalized_counts.items()
        if count / total >= 0.95
    )
    print(f"\nChannels present in at least 95% of recordings: {len(coverage_95)}")
    for name, count in coverage_95:
        print(f"  {name}: {count}/{total} ({100.0 * count / total:.2f}%)")

    print(f"\nPer-recording coverage CSV: {OUTPUT_PATH}")
    if failed_rows:
        print("WARNING: failed inventory rows were excluded from coverage counts.")
    else:
        print("PASS: coverage calculated across all successful inventory rows.")


if __name__ == "__main__":
    main()
