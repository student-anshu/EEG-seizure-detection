from pathlib import Path
import re


def find_summary_file(edf_path: str | Path) -> Path:
    """Find the matching subject summary file beside the EDF."""
    edf_path = Path(edf_path)
    summary = edf_path.parent / f"{edf_path.parent.name}-summary.txt"
    if not summary.is_file():
        raise FileNotFoundError(f"Could not find matching summary file: {summary}")
    return summary


def read_seizure_intervals(edf_path: str | Path) -> list[tuple[float, float]]:
    """Read seizure intervals from numbered or unnumbered CHB-MIT summary fields."""
    edf_path = Path(edf_path)
    summary_path = find_summary_file(edf_path)
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"(?=^File Name\s*:)", text, flags=re.IGNORECASE | re.MULTILINE)

    for block in blocks:
        name = re.search(r"^File Name\s*:\s*(.*?)\s*$", block,
                         flags=re.IGNORECASE | re.MULTILINE)
        if not name or name.group(1).lower() != edf_path.name.lower():
            continue

        count = re.search(r"^Number of Seizures in File\s*:\s*(\d+)\s*$",
                          block, flags=re.IGNORECASE | re.MULTILINE)
        if not count:
            raise ValueError(f"Missing seizure count for {edf_path.name}")
        expected = int(count.group(1))

        pattern = r"^Seizure(?:\s+\d+)?\s+{0}\s+Time\s*:\s*(\d+(?:\.\d+)?)\s*seconds?\s*$"
        starts = re.findall(pattern.format("Start"), block,
                            flags=re.IGNORECASE | re.MULTILINE)
        ends = re.findall(pattern.format("End"), block,
                          flags=re.IGNORECASE | re.MULTILINE)

        if expected == 0:
            if starts or ends:
                raise ValueError(f"{edf_path.name} declares zero seizures but has time annotations")
            return []
        if len(starts) != expected or len(ends) != expected:
            raise ValueError(
                f"{edf_path.name} declares {expected} seizure(s), but found "
                f"{len(starts)} start and {len(ends)} end annotation(s)"
            )

        intervals = []
        for index, (start_text, end_text) in enumerate(zip(starts, ends), start=1):
            start, end = float(start_text), float(end_text)
            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid seizure interval #{index} for {edf_path.name}: "
                    f"start={start}, end={end}; expected 0 <= start < end"
                )
            intervals.append((start, end))
        return intervals

    raise ValueError(
        f"{edf_path.name} was not listed in {summary_path.name}. "
        "Check that the EDF and summary file belong to the same subject."
    )
