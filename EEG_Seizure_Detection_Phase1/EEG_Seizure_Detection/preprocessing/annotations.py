from pathlib import Path
import re

def find_summary_file(edf_path: str | Path) -> Path:
    """Find the matching subject summary file in the EDF's directory."""
    edf_path = Path(edf_path)
    subject = edf_path.parent.name
    summary = edf_path.parent / f"{subject}-summary.txt"
    if not summary.exists():
        raise FileNotFoundError(
            f"Could not find {summary}. Keep the matching subject summary "
            "file in the same folder as the EDF files."
        )
    return summary

def read_seizure_intervals(edf_path: str | Path) -> list[tuple[float, float]]:
    """
    Return seizure intervals (start_seconds, end_seconds) for this EDF file.
    Parses CHB-MIT summary text lines such as:
      File Name: chb01_03.edf
      Number of Seizures in File: 1
      Seizure 1 Start Time: 2996 seconds
      Seizure 1 End Time: 3036 seconds
    """
    edf_path = Path(edf_path)
    summary_path = find_summary_file(edf_path)
    text = summary_path.read_text(encoding="utf-8", errors="ignore")

    blocks = re.split(r"(?=File Name\s*:)", text, flags=re.IGNORECASE)
    for block in blocks:
        name_match = re.search(r"File Name\s*:\s*(.+)", block, flags=re.IGNORECASE)
        if not name_match:
            continue
        filename = name_match.group(1).strip()
        if filename.lower() != edf_path.name.lower():
            continue

        intervals = []
        starts = re.findall(
            r"Seizure\s+\d+\s+Start Time\s*:\s*(\d+)\s*seconds?",
            block, flags=re.IGNORECASE
        )
        ends = re.findall(
            r"Seizure\s+\d+\s+End Time\s*:\s*(\d+)\s*seconds?",
            block, flags=re.IGNORECASE
        )
        if len(starts) != len(ends):
            raise ValueError(
                f"Unmatched seizure start/end annotations for {edf_path.name}"
            )
        intervals = [(float(a), float(b)) for a, b in zip(starts, ends)]
        return intervals

    raise ValueError(
        f"{edf_path.name} was not listed in {summary_path.name}. "
        "Check that the EDF and summary file belong to the same CHB-MIT subject."
    )
