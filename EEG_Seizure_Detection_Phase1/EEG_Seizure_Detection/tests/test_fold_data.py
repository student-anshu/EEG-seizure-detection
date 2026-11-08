import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from config import PREFERRED_CHANNELS
from data_management.dataset_split import LopoFold
from data_management.fold_data import (
    load_recording_windows, prepare_lopo_fold, processed_path_for_recording,
)
from data_management.patient_mapping import Recording


def make_recording(number, seizures=0):
    subject = f"chb{number:02d}"
    filename = f"{subject}_01.edf"
    patient = "patient_01" if number in (1, 21) else f"patient_{number:02d}"
    return Recording(
        subject, patient, filename, f"{subject}/{filename}", seizures,
        tuple("10-20" for _ in range(seizures)), "subject_summary", "SUCCESS",
    )


def save_recording(directory, recording, value):
    X = np.full((2, len(PREFERRED_CHANNELS), 2560), value, dtype=np.float32)
    y = np.array([0, 1] if recording.seizure_count else [0, 0], dtype=np.int64)
    path = processed_path_for_recording(directory, recording)
    np.savez_compressed(
        path, X=X, y=y, starts=np.array([0.0, 5.0]),
        channels=np.asarray(PREFERRED_CHANNELS), sfreq=256.0,
        subject=recording.subject, source_file=recording.edf_file,
        normalization="none_fold_train_only_scaling_required",
    )


class FoldDataTests(unittest.TestCase):
    def test_loader_reads_and_validates_one_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            item = make_recording(2, seizures=1)
            save_recording(directory, item, 2.0)
            loaded = load_recording_windows(directory, item)
            self.assertEqual(loaded.X.shape, (2, len(PREFERRED_CHANNELS), 2560))
            np.testing.assert_array_equal(loaded.y, [0, 1])

    def test_loader_rejects_missing_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_recording_windows(directory, make_recording(2))

    def test_fold_scaler_excludes_validation_and_outer_test_patients(self):
        with tempfile.TemporaryDirectory() as directory:
            train = [make_recording(n, seizures=(1 if n == 1 else 0))
                     for n in range(1, 7)]
            for item in train:
                save_recording(directory, item, float(int(item.subject[-2:]) * 10))
            outer_test = make_recording(7, seizures=1)
            save_recording(directory, outer_test, 10000.0)

            fold = LopoFold(7, outer_test.patient_id, tuple(train), (outer_test,))
            prepared = prepare_lopo_fold(
                fold, directory, validation_fraction=0.33, random_state=3
            )

            fit_ids = {r.patient_id for r in prepared.fit_recordings}
            validation_ids = {r.patient_id for r in prepared.validation_recordings}
            self.assertTrue(fit_ids.isdisjoint(validation_ids))
            self.assertNotIn(outer_test.patient_id, fit_ids | validation_ids)
            self.assertEqual(prepared.test_patient_id, outer_test.patient_id)
            self.assertLess(float(np.max(prepared.standardizer.mean_)), 10000.0)

            test_items = list(prepared.iter_split("test"))
            self.assertEqual(len(test_items), 1)
            self.assertEqual(test_items[0].X.dtype, np.float32)
            self.assertGreater(float(np.mean(test_items[0].X)), 100.0)

            feature_items = list(prepared.iter_features("test"))
            self.assertEqual(len(feature_items), 1)
            result = feature_items[0]
            self.assertEqual(result.features.shape, (2, 221))
            self.assertEqual(result.features.dtype, np.float32)
            self.assertTrue(np.isfinite(result.features).all())
            np.testing.assert_array_equal(result.y, test_items[0].y)
            np.testing.assert_array_equal(result.starts, test_items[0].starts)
            self.assertEqual(result.recording, outer_test)

    def test_fold_scaler_checkpoint_is_reused_without_reloading_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            train = [make_recording(n, seizures=(1 if n == 1 else 0))
                     for n in range(1, 7)]
            for item in train:
                save_recording(directory, item, float(int(item.subject[-2:]) * 10))
            outer_test = make_recording(7, seizures=1)
            save_recording(directory, outer_test, 10000.0)
            fold = LopoFold(7, outer_test.patient_id, tuple(train), (outer_test,))
            checkpoint = __import__("pathlib").Path(directory) / "scaler_checkpoint"

            first = prepare_lopo_fold(
                fold, directory, validation_fraction=0.33, random_state=3,
                scaler_checkpoint_dir=checkpoint,
            )
            manifest_before = json.loads((checkpoint / "manifest.json").read_text())
            self.assertTrue(manifest_before["complete"])
            self.assertEqual(manifest_before["completed_recordings"], len(first.fit_recordings))

            with patch(
                "data_management.fold_data.load_recording_windows",
                side_effect=AssertionError("completed scaler checkpoint should be reused"),
            ):
                second = prepare_lopo_fold(
                    fold, directory, validation_fraction=0.33, random_state=3,
                    scaler_checkpoint_dir=checkpoint,
                )

            np.testing.assert_array_equal(
                second.standardizer.mean_, first.standardizer.mean_
            )
            np.testing.assert_array_equal(
                second.standardizer.scale_, first.standardizer.scale_
            )

    def test_feature_iteration_supports_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            train = [make_recording(n, seizures=(1 if n == 1 else 0))
                     for n in range(1, 7)]
            outer_test = make_recording(7, seizures=1)
            for index, item in enumerate([*train, outer_test], start=1):
                save_recording(directory, item, float(index))
            fold = LopoFold(7, outer_test.patient_id, tuple(train), (outer_test,))
            prepared = prepare_lopo_fold(
                fold, directory, validation_fraction=0.33, random_state=3
            )

            for split in ("fit", "validation", "test"):
                items = list(prepared.iter_features(split))
                expected = {
                    "fit": prepared.fit_recordings,
                    "validation": prepared.validation_recordings,
                    "test": prepared.test_recordings,
                }[split]
                self.assertEqual(
                    [item.recording for item in items], list(expected)
                )
                for item in items:
                    self.assertEqual(item.features.shape, (2, 221))
                    self.assertEqual(item.y.shape, (2,))
                    self.assertEqual(item.starts.shape, (2,))

    def test_invalid_split_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            item = make_recording(7)
            save_recording(directory, item, 7.0)
            from data_management.fold_data import PreparedFold
            from preprocessing.normalization import FoldChannelStandardizer
            prepared = PreparedFold(
                1, item.patient_id, (), (), (item,),
                FoldChannelStandardizer(), __import__("pathlib").Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "split must be"):
                list(prepared.iter_split("unknown"))


if __name__ == "__main__":
    unittest.main()
