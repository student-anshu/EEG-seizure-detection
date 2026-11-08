import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_default_dataset_dir = PROJECT_ROOT.parent / "dataset" / "chbmit"
DATASET_DIR = Path(os.environ.get("EEG_DATASET_DIR", str(_default_dataset_dir))).expanduser().resolve()
PROCESSED_DIR = PROJECT_ROOT / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 40.0
NOTCH_HZ = 60.0
WINDOW_SECONDS = 10
STRIDE_SECONDS = 5

# Fixed 17-channel longitudinal bipolar montage for the primary cohort.
# T8-P8 is omitted because it is not consistently available under the
# channel-label audit. Every recording must contain all channels below.
PREFERRED_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
RANDOM_STATE = 42
