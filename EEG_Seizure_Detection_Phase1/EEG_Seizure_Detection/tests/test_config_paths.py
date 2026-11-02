import unittest
from pathlib import Path

from config import DATASET_DIR, PROJECT_ROOT


class ConfigPathTests(unittest.TestCase):
    def test_dataset_dir_points_to_sibling_dataset_folder(self):
        expected = PROJECT_ROOT.parent / "dataset" / "chbmit"
        self.assertEqual(DATASET_DIR, expected)

    def test_dataset_dir_is_not_inside_code_directory(self):
        self.assertNotEqual(DATASET_DIR.parent, PROJECT_ROOT)
        self.assertEqual(DATASET_DIR.name, "chbmit")


if __name__ == "__main__":
    unittest.main()
