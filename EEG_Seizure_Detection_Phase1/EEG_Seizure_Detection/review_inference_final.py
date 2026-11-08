from pathlib import Path

output = Path("inference_final_source_review.txt")
patterns = (
    "FoldChannelStandardizer",
    "load_state_dict",
    "scaler_checkpoint",
    "collect_robust_batch",
    "extract_robust_features",
    "load_recording_windows",
    "inner_validation",
)

with output.open("w", encoding="utf-8") as f:
    fold_file = Path("data_management/fold_data.py")

    f.write("=" * 75 + "\nPROCESSED RECORDING LOADER\n" + "=" * 75 + "\n")

    if fold_file.exists():
        for i, line in enumerate(
            fold_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            f.write(f"{i}: {line}\n")
    else:
        f.write("fold_data.py NOT FOUND\n")

    f.write("\n" + "=" * 75 + "\nROBUST EXPERIMENT REFERENCES\n" + "=" * 75 + "\n")

    for path in sorted(Path("scripts").glob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = [
            i for i, line in enumerate(lines)
            if any(pattern in line for pattern in patterns)
        ]

        if not matches:
            continue

        f.write(f"\nFILE: {path}\n")

        included = set()
        for index in matches:
            for j in range(max(0, index - 8), min(len(lines), index + 15)):
                included.add(j)

        previous = -2
        for j in sorted(included):
            if j > previous + 1:
                f.write("...\n")
            f.write(f"{j + 1}: {lines[j]}\n")
            previous = j

print(f"Created: {output.resolve()}")
