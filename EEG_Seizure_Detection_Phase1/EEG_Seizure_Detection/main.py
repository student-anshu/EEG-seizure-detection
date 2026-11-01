import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import (DATASET_DIR, PROCESSED_DIR, LOW_CUT_HZ, HIGH_CUT_HZ,
                    NOTCH_HZ, WINDOW_SECONDS, STRIDE_SECONDS, PREFERRED_CHANNELS)
from preprocessing.load_data import load_edf
from preprocessing.annotations import read_seizure_intervals
from preprocessing.signal_processing import bandpass_and_notch, robust_normalize
from preprocessing.segmentation import make_windows

def choose_channels(raw):
    """Choose available preferred bipolar channels, otherwise EEG channels."""
    names = list(raw.ch_names)
    lookup = {name.upper().replace(" ", ""): name for name in names}
    selected = []
    for wanted in PREFERRED_CHANNELS:
        key = wanted.upper().replace(" ", "")
        if key in lookup:
            selected.append(lookup[key])
    if len(selected) < 2:
        selected = [name for name in names
                    if "EEG" in name.upper() or "-" in name]
    if not selected:
        raise ValueError(f"No suitable EEG channels found. EDF channels: {names}")
    return selected

def process_edf(edf_path: str | Path):
    edf_path = Path(edf_path)
    print(f"\nLoading: {edf_path}")
    raw = load_edf(edf_path)
    channels = choose_channels(raw)
    raw.pick(channels)
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()  # MNE returns volts, shape (channels, samples)

    intervals = read_seizure_intervals(edf_path)
    print(f"Channels selected: {len(channels)}")
    print(f"Sampling frequency: {sfreq:g} Hz")
    print(f"Seizure intervals: {intervals if intervals else 'None'}")

    filtered = bandpass_and_notch(
        data, sfreq, LOW_CUT_HZ, HIGH_CUT_HZ, NOTCH_HZ
    )
    normalized = robust_normalize(filtered)
    X, y, starts = make_windows(
        normalized, sfreq, intervals, WINDOW_SECONDS, STRIDE_SECONDS
    )

    if len(y) == 0:
        print("Recording is shorter than one configured window; skipped.")
        return None

    subject = edf_path.parent.name
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{subject}_{edf_path.stem}"
    npz_path = PROCESSED_DIR / f"{output_name}.npz"
    np.savez_compressed(npz_path, X=X, y=y, starts=starts,
                        channels=np.asarray(channels), sfreq=sfreq,
                        subject=subject, source_file=edf_path.name)

    metadata = pd.DataFrame({
        "subject": subject,
        "source_file": edf_path.name,
        "window_start_seconds": starts,
        "label": y
    })
    csv_path = PROCESSED_DIR / f"{output_name}_metadata.csv"
    metadata.to_csv(csv_path, index=False)

    print(f"Saved {len(y)} windows: {np.bincount(y, minlength=2).tolist()} "
          f"[normal, seizure]")
    print(f"Saved: {npz_path}")
    return npz_path

def inspect_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Download and extract CHB-MIT into dataset/chbmit."
        )
    edfs = sorted(DATASET_DIR.rglob("*.edf"))
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"EDF files found: {len(edfs)}")
    for path in edfs[:20]:
        print(f"  {path.relative_to(DATASET_DIR)}")
    if len(edfs) > 20:
        print(f"  ... and {len(edfs)-20} more")

def main():
    parser = argparse.ArgumentParser(description="EEG seizure data preparation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--edf", type=str, help="Process one EDF file")
    group.add_argument("--all", action="store_true", help="Process all EDF files")
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset()
    elif args.edf:
        process_edf(args.edf)
    elif args.all:
        edfs = sorted(DATASET_DIR.rglob("*.edf"))
        if not edfs:
            raise FileNotFoundError(f"No EDF files found under {DATASET_DIR}")
        for path in edfs:
            try:
                process_edf(path)
            except Exception as exc:
                print(f"SKIPPED {path}: {exc}")

if __name__ == "__main__":
    main()


