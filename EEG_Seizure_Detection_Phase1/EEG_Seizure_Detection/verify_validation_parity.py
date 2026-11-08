from pathlib import Path
import joblib
import numpy as np
from modeling.baseline import positive_scores

cache_dir = Path("cache/robust_feature_first_fold_v1/inner_validation")
model_path = Path(
    "results/baselines/robust_feature_first_fold_v1/"
    "logistic_regression/model.joblib"
)
scores_path = Path(
    "results/baselines/robust_feature_first_fold_v1/"
    "logistic_regression/validation_scores.npz"
)
report_path = Path("validation_prediction_parity.txt")

with report_path.open("w", encoding="utf-8") as f:

    f.write("=== VALIDATION PREDICTION PARITY AUDIT ===\n")

    # Verify required files
    required = [cache_dir / "X.npy", cache_dir / "y.npy",
                model_path, scores_path]

    for path in required:
        f.write(f"{path}: exists={path.is_file()}\n")

    if not all(path.is_file() for path in required):
        raise FileNotFoundError("One or more required artifacts are missing.")

    # Load model and read-only arrays
    model = joblib.load(model_path)
    X = np.load(cache_dir / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(cache_dir / "y.npy", mmap_mode="r", allow_pickle=False)

    with np.load(scores_path, allow_pickle=False) as saved:
        original_y = saved["y_true"]
        original_scores = saved["scores"]

        f.write(f"\nModel type: {type(model).__name__}\n")
        f.write(f"Feature matrix shape: {X.shape}\n")
        f.write(f"Model expected features: {model.n_features_in_}\n")
        f.write(f"Archived scores shape: {original_scores.shape}\n")

        if X.shape[1] != model.n_features_in_:
            raise ValueError("Feature count does not match model.")

        if X.shape[0] != len(original_scores):
            raise ValueError("Prediction count does not match archived scores.")

        if not np.array_equal(y, original_y):
            raise ValueError("Cached labels do not match archived labels.")

        # Reproduce original positive_scores() in batches
        recalculated = np.empty(len(y), dtype=np.float64)
        batch_size = 8192

        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))
            recalculated[start:end] = positive_scores(
                model, X[start:end]
            )

        difference = np.abs(recalculated - original_scores)

        tolerance = 1e-8
        within_tolerance = np.isclose(
            recalculated,
            original_scores,
            rtol=1e-7,
            atol=tolerance
        )

        f.write("\n=== PARITY RESULTS ===\n")
        f.write(f"Predictions recalculated: {len(recalculated)}\n")
        f.write(f"Recalculated scores finite: {np.isfinite(recalculated).all()}\n")
        f.write(f"Archived scores finite: {np.isfinite(original_scores).all()}\n")
        f.write(f"Maximum absolute difference: {difference.max():.15g}\n")
        f.write(f"Mean absolute difference: {difference.mean():.15g}\n")
        f.write(f"Scores outside tolerance: {int((~within_tolerance).sum())}\n")
        f.write(f"All scores within tolerance: {bool(within_tolerance.all())}\n")
        f.write(f"Labels exactly match: {np.array_equal(y, original_y)}\n")

        passed = (
            np.isfinite(recalculated).all()
            and np.isfinite(original_scores).all()
            and within_tolerance.all()
            and np.array_equal(y, original_y)
        )

        f.write(f"\nFINAL RESULT: {'PASS' if passed else 'FAIL'}\n")

print(f"Audit completed. Report: {report_path}")
