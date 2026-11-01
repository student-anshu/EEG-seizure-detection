from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "chbmit"
PROCESSED_DIR = PROJECT_ROOT / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Signal-processing settings
LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 40.0
NOTCH_HZ = 60.0
WINDOW_SECONDS = 10
STRIDE_SECONDS = 5

# A practical initial common montage list. A channel is used only when it
# exists in the EDF file (case/spacing-insensitive matching).
PREFERRED_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

RANDOM_STATE = 42
