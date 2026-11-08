from pathlib import Path

targets = {
    "MAIN IMPORTS AND CONFIGURATION": ("main.py", 1, 25),
    "EDF LOADING AND FILTERING FUNCTIONS": ("main.py", 85, 190),
    "WINDOW CREATION FUNCTIONS": ("main.py", 190, 310),
    "CHANNEL STANDARDIZER TRANSFORM": ("preprocessing/normalization.py", 180, 300),
}

output = Path("inference_preprocessing_review.txt")

with output.open("w", encoding="utf-8") as f:
    for title, (filename, start, end) in targets.items():
        path = Path(filename)
        f.write(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n")
        f.write(f"File: {path}\n")

        if not path.exists():
            f.write("FILE NOT FOUND\n")
            continue

        lines = path.read_text(encoding="utf-8").splitlines()

        for i in range(start - 1, min(end, len(lines))):
            f.write(f"{i + 1}: {lines[i]}\n")

print(f"Created: {output.resolve()}")
