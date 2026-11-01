from pathlib import Path
import mne

def load_edf(edf_path: str | Path, preload: bool = True):
    """Load an EDF recording and return an MNE Raw object."""
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file does not exist: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
    return raw
