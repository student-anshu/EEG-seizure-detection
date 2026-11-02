from pathlib import Path
import numpy as np
from config import PROCESSED_DIR

files = sorted(PROCESSED_DIR.glob("*.npz"))

print("\nPROCESSED EEG DATASET VERIFICATION")
print("=" * 55)

total_windows = 0
total_seizure = 0
total_normal = 0

for file in files:
    data = np.load(file)
    X = data["X"]
    y = data["y"]
    channels = data["channels"]

    normal = int(np.sum(y == 0))
    seizure = int(np.sum(y == 1))

    total_windows += len(y)
    total_normal += normal
    total_seizure += seizure

    print(f"\nFile: {file.name}")
    print(f"EEG shape: {X.shape}")
    print(f"Channels: {len(channels)}")
    print(f"Normal windows: {normal}")
    print(f"Seizure windows: {seizure}")

print("\n" + "=" * 55)
print("FINAL DATASET SUMMARY")
print("=" * 55)
print("Total recordings:", len(files))
print("Total windows:", total_windows)
print("Normal windows:", total_normal)
print("Seizure windows:", total_seizure)