from pathlib import Path
import re

root = Path(".")
output = Path("inference_source_audit.txt")

patterns = [
    "bandpass_and_notch",
    "robust_normalize",
    "extract_robust_features",
    "load_recording_windows",
    "FoldChannelStandardizer",
    "PREFERRED_CHANNELS",
    "WINDOW_SECONDS",
    "STRIDE_SECONDS",
    "sfreq",
]

excluded = {
    ".venv", ".venv311", "__pycache__",
    ".git", "cache", "node_modules"
}

with output.open("w", encoding="utf-8") as report:
    for path in sorted(root.rglob("*.py")):
        if any(part in excluded for part in path.parts):
            continue

        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue

        matches = []
        for i, line in enumerate(lines):
            if any(term in line for term in patterns):
                matches.append(i)

        if not matches:
            continue

        report.write(f"\n{'=' * 80}\n")
        report.write(f"FILE: {path}\n")
        report.write(f"{'=' * 80}\n")

        selected = set()
        for index in matches:
            start = max(0, index - 8)
            end = min(len(lines), index + 15)
            selected.update(range(start, end))

        previous = -2
        for index in sorted(selected):
            if index > previous + 1:
                report.write("\n...\n")
            report.write(f"{index + 1}: {lines[index]}\n")
            previous = index

print(f"Audit report created: {output}")
