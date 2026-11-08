from pathlib import Path

files = [
    "preprocessing/load_data.py",
    "preprocessing/signal_processing.py",
    "preprocessing/segmentation.py",
    "preprocessing/normalization.py",
    "config.py",
]

output = Path("inference_core_source_review.txt")

with output.open("w", encoding="utf-8") as f:
    for filename in files:
        path = Path(filename)

        f.write(f"\n{'=' * 75}\n")
        f.write(f"SOURCE FILE: {filename}\n")
        f.write(f"{'=' * 75}\n")

        if not path.exists():
            f.write("FILE NOT FOUND\n")
            continue

        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            f.write(f"{number}: {line}\n")

print(f"Created: {output.resolve()}")
