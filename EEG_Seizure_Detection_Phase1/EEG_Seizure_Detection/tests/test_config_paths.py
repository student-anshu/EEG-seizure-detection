import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class ConfigPathTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config)

    def test_default_dataset_path_is_preserved(self):
        expected = config.PROJECT_ROOT.parent / "dataset" / "chbmit"

        with patch.dict(os.environ):
            os.environ.pop("EEG_DATASET_DIR", None)
            importlib.reload(config)
            self.assertEqual(config.DATASET_DIR, expected.resolve())

    def test_external_dataset_path_from_environment(self):
        external_path = Path(
            r"D:\EEG-Datasets\CHB-MIT\chb-mit-scalp-eeg-database-1.0.0"
        )

        with patch.dict(
            os.environ,
            {"EEG_DATASET_DIR": str(external_path)}
        ):
            importlib.reload(config)
            self.assertEqual(config.DATASET_DIR, external_path.resolve())

    def test_dataset_dir_is_not_inside_code_directory(self):
        self.assertNotEqual(config.DATASET_DIR, config.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
