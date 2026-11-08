import json
from pathlib import Path
import numpy as np

root = Path("cache/robust_feature_first_fold_v1/inner_validation")
report = Path("validation_cache_verification.txt")

with report.open("w", encoding="utf-8") as f:
    f.write(f"Cache directory exists: {root.is_dir()}\n")

    for name in ["manifest.json", "X.npy", "y.npy"]:
        path = root / name
        f.write(f"{name} exists: {path.is_file()}\n")

    manifest_path = root / "manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        f.write("\n=== MANIFEST ===\n")
        f.write(json.dumps(manifest, indent=2) + "\n")

    x_path, y_path = root / "X.npy", root / "y.npy"

    if x_path.is_file() and y_path.is_file():
        X = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)

        f.write("\n=== ARRAY INFORMATION ===\n")
        f.write(f"X shape: {X.shape}, dtype: {X.dtype}\n")
        f.write(f"y shape: {y.shape}, dtype: {y.dtype}\n")

        f.write("\n=== VALIDATION CHECKS ===\n")
        f.write(f"Labels binary: {bool(np.isin(y, [0, 1]).all())}\n")
        f.write(f"Positive labels: {int(np.count_nonzero(y == 1))}\n")
        f.write(f"Negative labels: {int(np.count_nonzero(y == 0))}\n")

        finite = True
        for start in range(0, X.shape[0], 10000):
            if not np.isfinite(X[start:start + 10000]).all():
                finite = False
                break

        f.write(f"All feature values finite: {finite}\n")

        scores_path = Path(
            "results/baselines/robust_feature_first_fold_v1/"
            "logistic_regression/validation_scores.npz"
        )

        if scores_path.is_file():
            with np.load(scores_path, allow_pickle=False) as saved:
                original_y = saved["y_true"]
                f.write(
                    f"Labels match saved validation scores: "
                    f"{np.array_equal(y, original_y)}\n"
                )

print(f"Report created: {report}")
