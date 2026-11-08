"""Inventory CHB-MIT EDF headers without loading signal samples into memory.

Run from the EEG_Seizure_Detection directory:
    python scripts/audit_recording_signals.py

The resulting CSV contains recording metadata and channel labels only; it does
not contain EEG samples.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import mne

from config import DATASET_DIR, PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT.parent.parent / "docs" / "signal_inventory.csv"


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"CHB-MIT dataset directory not found: {DATASET_DIR}. "
            "Set EEG_DATASET_DIR to the extracted dataset root."
        )

    edf_paths = sorted(DATASET_DIR.rglob("*.edf"))
    if not edf_paths:
        raise FileNotFoundError(f"No EDF files found under {DATASET_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "edf_file",
        "relative_path",
        "sampling_frequency_hz",
        "n_samples",
        "duration_seconds",
        "n_channels",
        "channel_names",
        "status",
        "error",
    ]

    sampling_rates: Counter[float] = Counter()
    channel_counts: Counter[int] = Counter()
    all_channels: Counter[str] = Counter()
    successful = 0
    failed = 0

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, edf_path in enumerate(edf_paths, start=1):
            raw = None
            try:
                raw = mne.io.read_raw_edf(
                    edf_path, preload=False, verbose="ERROR"
                )
                sfreq = float(raw.info["sfreq"])
                n_samples = int(raw.n_times)
                channels = list(raw.ch_names)
                duration = n_samples / sfreq

                writer.writerow({
                    "subject": edf_path.parent.name,
                    "edf_file": edf_path.name,
                    "relative_path": edf_path.relative_to(DATASET_DIR).as_posix(),
                    "sampling_frequency_hz": sfreq,
                    "n_samples": n_samples,
                    "duration_seconds": round(duration, 6),
                    "n_channels": len(channels),
                    "channel_names": "|".join(channels),
                    "status": "SUCCESS",
                    "error": "",
                })

                sampling_rates[sfreq] += 1
                channel_counts[len(channels)] += 1
                all_channels.update(channels)
                successful += 1
            except Exception as exc:
                writer.writerow({
                    "subject": edf_path.parent.name,
                    "edf_file": edf_path.name,
                    "relative_path": edf_path.relative_to(DATASET_DIR).as_posix(),
                    "sampling_frequency_hz": "",
                    "n_samples": "",
                    "duration_seconds": "",
                    "n_channels": "",
                    "channel_names": "",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                failed += 1
            finally:
                if raw is not None:
                    raw.close()

            if index % 25 == 0 or index == len(edf_paths):
                print(f"Inspected {index}/{len(edf_paths)} EDF headers")

    print("\n========== EDF SIGNAL INVENTORY ==========")
    print(f"Dataset: {DATASET_DIR}")
    print(f"EDF files discovered: {len(edf_paths)}")
    print(f"Successfully inspected: {successful}")
    print(f"Failed: {failed}")
    print(f"Sampling frequencies (Hz): {dict(sorted(sampling_rates.items()))}")
    print(f"Channel counts: {dict(sorted(channel_counts.items()))}")
    print(f"Distinct channel labels: {len(all_channels)}")
    print(f"Inventory CSV: {OUTPUT_PATH}")

    if failed:
        print("\nSome EDF headers could not be inspected. Review the FAILED rows.")
    else:
        print("\nPASS: all discovered EDF headers were inspected.")


if __name__ == "__main__":
    main()
