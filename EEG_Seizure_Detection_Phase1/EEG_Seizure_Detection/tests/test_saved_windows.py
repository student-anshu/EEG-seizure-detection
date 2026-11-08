import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
from config import PREFERRED_CHANNELS


class FakeRaw:
    def __init__(self):
        self.ch_names = list(PREFERRED_CHANNELS)
        self.info = {"sfreq": 1.0}
        self._data = np.zeros((len(PREFERRED_CHANNELS), 30), dtype=np.float32)

    def pick(self, channels):
        self.ch_names = list(channels)

    def get_data(self):
        return self._data


class SavedWindowsIntegrationTests(unittest.TestCase):
    def test_saved_npz_and_metadata_keep_timestamps_and_labels_aligned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            edf_path = Path(temp_dir) / "chb01" / "chb01_03.edf"

            with (
                patch.object(main, "load_edf", return_value=FakeRaw()),
                patch.object(main, "read_seizure_intervals", return_value=[(12, 14)]),
                patch.object(main, "bandpass_and_notch", side_effect=lambda data, *args: data),
                patch.object(main, "PROCESSED_DIR", output_dir),
            ):
                saved_path = main.process_edf(edf_path)

            self.assertIsNotNone(saved_path)
            self.assertTrue(saved_path.exists())

            with np.load(saved_path, allow_pickle=False) as saved:
                X = saved["X"]
                labels = saved["y"]
                starts = saved["starts"]

                self.assertEqual(X.shape, (5, len(PREFERRED_CHANNELS), 10))
                self.assertEqual(labels.shape, (5,))
                self.assertEqual(starts.shape, (5,))
                np.testing.assert_array_equal(starts, [0, 5, 10, 15, 20])
                np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0])
                self.assertEqual(len(saved["channels"]), len(PREFERRED_CHANNELS))
                self.assertEqual(float(saved["sfreq"]), 1.0)
                self.assertEqual(float(saved["recording_duration_seconds"]), 30.0)
                self.assertEqual(str(saved["subject"]), "chb01")
                self.assertEqual(str(saved["source_file"]), "chb01_03.edf")
                self.assertEqual(str(saved["normalization"]), "none_fold_train_only_scaling_required")

            metadata_path = output_dir / "chb01_chb01_03_metadata.csv"
            self.assertTrue(metadata_path.exists())
            metadata = pd.read_csv(metadata_path)
            self.assertEqual(len(metadata), len(starts))
            self.assertTrue((metadata["recording_duration_seconds"] == 30.0).all())
            np.testing.assert_array_equal(metadata["window_start_seconds"], starts)
            np.testing.assert_array_equal(
                metadata["window_end_seconds"], starts + main.WINDOW_SECONDS
            )
            np.testing.assert_array_equal(metadata["label"], labels)
            self.assertTrue((metadata["normalization"] == "none_fold_train_only_scaling_required").all())


if __name__ == "__main__":
    unittest.main()
