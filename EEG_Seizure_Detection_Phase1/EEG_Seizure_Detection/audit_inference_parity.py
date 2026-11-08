from pathlib import Path
import numpy as np

root = Path("results/baselines/robust_feature_first_fold_v1")

output = Path("inference_parity_audit.txt")

with output.open("w", encoding="utf-8") as f:
    for filename in ["validation_scores.npz", "outer_test_scores.npz"]:
        path = root / "logistic_regression" / filename

        f.write(f"\n=== {filename} ===\n")

        if not path.exists():
            f.write("FILE NOT FOUND\n")
            continue

        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = data[key]
                f.write(
                    f"{key}: shape={value.shape}, dtype={value.dtype}\n"
                )

    f.write("\n=== MODEL ARTIFACT ===\n")
    model_path = root / "logistic_regression" / "model.joblib"
    f.write(f"Exists: {model_path.exists()}\n")
    if model_path.exists():
        f.write(f"Size: {model_path.stat().st_size} bytes\n")

print(f"Created: {output}")
