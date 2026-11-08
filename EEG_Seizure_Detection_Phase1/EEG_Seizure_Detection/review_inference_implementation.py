from pathlib import Path

files = {
    "ROBUST FEATURE EXTRACTOR": (
        Path("feature_selection/robust_feature_extraction.py"), 1, 180
    ),
    "CHANNEL STANDARDIZER": (
        Path("preprocessing/normalization.py"), 55, 180
    ),
    "EDF PREPROCESSING": (
        Path("main.py"), 25, 85
    ),
}

output = Path("inference_implementation_review.txt")

with output.open("w", encoding="utf-8") as f:
    for title, (path, start, end) in files.items():
        f.write(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n")
        f.write(f"File: {path}\n")

        lines = path.read_text(encoding="utf-8").splitlines()

        for index in range(start - 1, min(end, len(lines))):
            f.write(f"{index + 1}: {lines[index]}\n")

print(f"Created: {output}")
