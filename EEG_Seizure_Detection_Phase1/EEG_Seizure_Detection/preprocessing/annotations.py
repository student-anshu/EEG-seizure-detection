from pathlib import Path
import re


def find_summary_file(edf_path: str | Path) -> Path:
    """Find the matching subject summary file beside the EDF."""
    edf_path = Path(edf_path)
    summary = edf_path.parent / f"{edf_path.parent.name}-summary.txt"
    if not summary.is_file():
        raise FileNotFoundError(f"Could not find matching summary file: {summary}")
    return summary


def is_officially_unannotated_recording(edf_path: str | Path) -> bool:
    """Confirm an unlisted EDF is negative using both official CHB-MIT manifests."""
    edf_path = Path(edf_path)
    summary_path = find_summary_file(edf_path)
    summary_text = summary_path.read_text(encoding="utf-8", errors="ignore")
    if re.search(
        rf"^File Name\s*:\s*{re.escape(edf_path.name)}\s*$",
        summary_text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return False

    dataset_root = edf_path.parent.parent
    records_path = dataset_root / "RECORDS"
    seizure_records_path = dataset_root / "RECORDS-WITH-SEIZURES"
    if not records_path.is_file() or not seizure_records_path.is_file():
        return False

    relative_path = edf_path.relative_to(dataset_root).as_posix()
    records = {
        line.strip()
        for line in records_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }
    seizure_records = {
        line.strip()
        for line in seizure_records_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }
    return relative_path in records and relative_path not in seizure_records


def read_seizure_intervals(edf_path: str | Path) -> list[tuple[float, float]]:
    """Read intervals from a subject summary; use official manifests for omitted negatives."""
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
        starts = re.findall(pattern.format("Start"), block, flags=re.IGNORECASE | re.MULTILINE)
        ends = re.findall(pattern.format("End"), block, flags=re.IGNORECASE | re.MULTILINE)

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

    if is_officially_unannotated_recording(edf_path):
        return []

    raise ValueError(
        f"{edf_path.name} was not listed in {summary_path.name}, and the official "
        "RECORDS manifests do not confirm it as a non-seizure recording."
    )
