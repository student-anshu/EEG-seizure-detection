import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.baseline import (
    FeatureBatch, calculate_metrics, choose_f1_threshold, choose_fbeta_threshold,
    fit_validate_model, positive_scores, build_baseline_estimators,
    collect_features,
)


class BaselineUtilityTests(unittest.TestCase):
    def test_builds_two_documented_baselines(self):
        models = build_baseline_estimators()
        self.assertEqual(set(models), {"logistic_regression", "random_forest"})
        self.assertIn("feature_scaler", models["logistic_regression"].named_steps)

    def test_threshold_is_selected_from_validation_scores(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([-2.0, -1.0, 1.0, 2.0])
        threshold = choose_f1_threshold(y, scores)
        self.assertEqual(calculate_metrics(y, scores, threshold)["f1"], 1.0)

    def test_f2_threshold_matches_brute_force_and_prioritizes_recall(self):
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.4, 0.3, 0.5, 0.9])
        threshold = choose_fbeta_threshold(y, scores, beta=2.0)
        from sklearn.metrics import fbeta_score
        candidates = np.unique(scores)
        expected = max(
            candidates,
            key=lambda value: fbeta_score(y, scores >= value, beta=2.0, zero_division=0),
        )
        self.assertEqual(threshold, float(expected))
        self.assertEqual(choose_f1_threshold(y, scores), choose_fbeta_threshold(y, scores, beta=1.0))

    def test_fbeta_rejects_invalid_beta(self):
        y = np.array([0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.9])
        for beta in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                choose_fbeta_threshold(y, scores, beta=beta)

    def test_threshold_matches_brute_force_for_repeated_scores(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.4, 0.4, 0.8, 0.2, 0.8, 0.1, 0.4, 0.6])
        candidates = np.unique(scores)
        expected = None
        best_f1 = -1.0
        from sklearn.metrics import f1_score
        for candidate in candidates:
            current = f1_score(y, scores >= candidate, zero_division=0)
            if current > best_f1:
                expected, best_f1 = float(candidate), float(current)
        self.assertEqual(choose_f1_threshold(y, scores), expected)

    def test_threshold_scales_to_large_validation_arrays(self):
        rng = np.random.default_rng(123)
        y = rng.integers(0, 2, size=200_000, dtype=np.int8)
        y[0] = 1
        scores = rng.random(200_000)
        threshold = choose_f1_threshold(y, scores)
        self.assertTrue(np.isfinite(threshold))
        self.assertGreaterEqual(threshold, float(scores.min()))
        self.assertLessEqual(threshold, float(scores.max()))

    def test_metrics_report_confusion_counts_and_imbalance_metrics(self):
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.7, 0.9])
        result = calculate_metrics(y, scores, 0.5)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 1, 0, 2))
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertIn("f2", result)
        self.assertEqual(result["specificity"], 2 / 3)
        self.assertIsNotNone(result["auprc"])
        self.assertIsNotNone(result["auroc"])

    def test_auroc_is_none_for_single_class_test_patient(self):
        result = calculate_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.2)
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["auprc"])

    def test_fit_validation_keeps_validation_out_of_estimator_fit(self):
        X_fit = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y_fit = np.array([0, 0, 1, 1])
        X_val = np.array([[-3.0], [3.0]])
        y_val = np.array([0, 1])
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=1)),
        ])
        fitted, threshold, metrics = fit_validate_model(
            model, FeatureBatch(X_fit, y_fit, ()),
            FeatureBatch(X_val, y_val, ()),
        )
        self.assertEqual(fitted.named_steps["scale"].mean_[0], 0.0)
        self.assertTrue(np.isfinite(positive_scores(fitted, X_val)).all())
        self.assertGreaterEqual(threshold, float(np.min(positive_scores(fitted, X_val))))
        self.assertIn("f1", metrics)

    def test_collect_features_preallocates_aligned_float32_matrix(self):
        from data_management.patient_mapping import Recording

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recordings = [
                Recording("chb01", "patient_01", "chb01_01.edf", "chb01/chb01_01.edf",
                          0, (), "test", "SUCCESS"),
                Recording("chb02", "patient_02", "chb02_01.edf", "chb02/chb02_01.edf",
                          1, ("10-20",), "test", "SUCCESS"),
            ]
            labels = [np.array([0, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8)]
            for recording, target in zip(recordings, labels):
                filename = f"{recording.subject}_{Path(recording.edf_file).stem}.npz"
                np.savez_compressed(directory / filename, y=target)

            loaded = [
                SimpleNamespace(X=np.zeros((2, 17, 4), dtype=np.float32), y=target, sfreq=256.0)
                for target in labels
            ]
            scaler = SimpleNamespace(transform=lambda values: values)
            feature_matrices = [
                np.full((2, 221), 1.0, dtype=np.float32),
                np.full((2, 221), 2.0, dtype=np.float32),
            ]
            with (
                patch("modeling.baseline.load_recording_windows", side_effect=loaded),
                patch("modeling.baseline.extract_features", side_effect=feature_matrices),
            ):
                batch = collect_features(recordings, directory, scaler)

            self.assertEqual(batch.X.shape, (4, 221))
            self.assertEqual(batch.X.dtype, np.float32)
            self.assertTrue(np.array_equal(batch.y, np.array([0, 0, 0, 1], dtype=np.int8)))
            self.assertTrue(np.all(batch.X[:2] == 1.0))
            self.assertTrue(np.all(batch.X[2:] == 2.0))
            self.assertEqual(batch.recording_paths, ())

    def test_collect_features_resumes_from_last_completed_recording(self):
        from data_management.patient_mapping import Recording

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            cache = root / "cache"
            processed.mkdir()
            recordings = [
                Recording("chb01", "patient_01", "chb01_01.edf", "chb01/chb01_01.edf",
                          0, (), "test", "SUCCESS"),
                Recording("chb02", "patient_02", "chb02_01.edf", "chb02/chb02_01.edf",
                          1, ("10-20",), "test", "SUCCESS"),
            ]
            labels = [np.array([0, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8)]
            for recording, target in zip(recordings, labels):
                filename = f"{recording.subject}_{Path(recording.edf_file).stem}.npz"
                np.savez_compressed(processed / filename, y=target)

            scaler = SimpleNamespace(
                mean_=np.zeros(17), scale_=np.ones(17), epsilon=1e-6,
                transform=lambda values: values,
            )
            loaded = [
                SimpleNamespace(X=np.zeros((2, 17, 4), dtype=np.float32), y=target, sfreq=256.0)
                for target in labels
            ]
            calls = {"count": 0}

            def interrupt_after_first_recording(values, sfreq):
                calls["count"] += 1
                if calls["count"] == 1:
                    return np.full((2, 221), 1.0, dtype=np.float32)
                raise RuntimeError("simulated interruption")

            with (
                patch("modeling.baseline.load_recording_windows", side_effect=loaded),
                patch("modeling.baseline.extract_features", side_effect=interrupt_after_first_recording),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    collect_features(recordings, processed, scaler, cache_dir=cache, cache_key="resume_test")

            self.assertEqual(calls["count"], 2)
            resumed_loaded = [
                SimpleNamespace(X=np.zeros((2, 17, 4), dtype=np.float32), y=labels[1], sfreq=256.0)
            ]
            with (
                patch("modeling.baseline.load_recording_windows", side_effect=resumed_loaded),
                patch("modeling.baseline.extract_features", return_value=np.full((2, 221), 2.0, dtype=np.float32)) as extractor,
            ):
                batch = collect_features(recordings, processed, scaler, cache_dir=cache, cache_key="resume_test")

            self.assertEqual(extractor.call_count, 1)
            self.assertTrue(np.array_equal(batch.X[:2], np.full((2, 221), 1.0, dtype=np.float32)))
            self.assertTrue(np.array_equal(batch.X[2:], np.full((2, 221), 2.0, dtype=np.float32)))
            self.assertTrue(np.array_equal(batch.y, np.array([0, 0, 0, 1], dtype=np.int8)))

            with (
                patch("modeling.baseline.load_recording_windows", side_effect=AssertionError("cache should be reused")),
                patch("modeling.baseline.extract_features", side_effect=AssertionError("cache should be reused")),
            ):
                cached = collect_features(recordings, processed, scaler, cache_dir=cache, cache_key="resume_test")
            self.assertTrue(np.array_equal(cached.X, batch.X))
            # Close .npy memory maps before TemporaryDirectory cleanup (required on Windows).
            for array in (batch.X, batch.y, cached.X, cached.y):
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
    def test_collect_features_rejects_manifest_label_disagreement(self):
        from data_management.patient_mapping import Recording

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recording = Recording(
                "chb01", "patient_01", "chb01_01.edf", "chb01/chb01_01.edf",
                1, ("10-20",), "test", "SUCCESS",
            )
            filename = f"{recording.subject}_{Path(recording.edf_file).stem}.npz"
            np.savez_compressed(directory / filename, y=np.array([0, 0], dtype=np.int8))
            with self.assertRaisesRegex(ValueError, "disagree"):
                collect_features([recording], directory, SimpleNamespace(transform=lambda x: x))


if __name__ == "__main__":
    unittest.main()
