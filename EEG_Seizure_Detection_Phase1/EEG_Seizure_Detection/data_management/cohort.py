"""Primary cohort policy for the CHB-MIT 17-channel experiment."""
from __future__ import annotations
from typing import Sequence
from data_management.patient_mapping import Recording

EXCLUDED_MONTAGE_RECORDINGS = frozenset({
    "chb12/chb12_27.edf",
    "chb12/chb12_28.edf",
    "chb12/chb12_29.edf",
})

def filter_primary_cohort(recordings: Sequence[Recording]) -> list[Recording]:
    """Exclude only the three documented non-longitudinal-bipolar recordings."""
    paths = {recording.relative_path for recording in recordings}
    missing = EXCLUDED_MONTAGE_RECORDINGS - paths
    if missing:
        raise ValueError("Expected montage-exclusion recordings are absent from the audit: " + ", ".join(sorted(missing)))
    return [r for r in recordings if r.relative_path not in EXCLUDED_MONTAGE_RECORDINGS]
