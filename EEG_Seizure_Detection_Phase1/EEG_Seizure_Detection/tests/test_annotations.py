import tempfile
import unittest
from pathlib import Path

from preprocessing.annotations import read_seizure_intervals


class ReadSeizureIntervalsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)
        self.subject_dir = self.dataset_dir / "chb_test"
        self.subject_dir.mkdir()
        self.edf_path = self.subject_dir / "chb_test_01.edf"
        self.edf_path.touch()
        self.summary_path = self.subject_dir / "chb_test-summary.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_summary(self, block):
        self.summary_path.write_text(block, encoding="utf-8")

    def write_manifests(self, records, seizure_records):
        (self.dataset_dir / "RECORDS").write_text(records, encoding="utf-8")
        (self.dataset_dir / "RECORDS-WITH-SEIZURES").write_text(
            seizure_records, encoding="utf-8"
        )

    def test_actual_unnumbered_chbmit_format(self):
        self.write_summary(
            "File Name: chb_test_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 362 seconds\n"
            "Seizure End Time: 414 seconds\n"
        )
        self.assertEqual(read_seizure_intervals(self.edf_path), [(362.0, 414.0)])

    def test_numbered_annotation_format(self):
        self.write_summary(
            "File Name: chb_test_01.edf\n"
            "Number of Seizures in File: 2\n"
            "Seizure 1 Start Time: 100 seconds\n"
            "Seizure 1 End Time: 120 seconds\n"
            "Seizure 2 Start Time: 300 seconds\n"
            "Seizure 2 End Time: 330 seconds\n"
        )
        self.assertEqual(
            read_seizure_intervals(self.edf_path),
            [(100.0, 120.0), (300.0, 330.0)],
        )

    def test_zero_seizure_recording(self):
        self.write_summary(
            "File Name: chb_test_01.edf\n"
            "Number of Seizures in File: 0\n"
        )
        self.assertEqual(read_seizure_intervals(self.edf_path), [])

    def test_mismatched_annotation_count_raises(self):
        self.write_summary(
            "File Name: chb_test_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 100 seconds\n"
        )
        with self.assertRaisesRegex(ValueError, "declares 1 seizure"):
            read_seizure_intervals(self.edf_path)

    def test_invalid_interval_raises(self):
        self.write_summary(
            "File Name: chb_test_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 200 seconds\n"
            "Seizure End Time: 100 seconds\n"
        )
        with self.assertRaisesRegex(ValueError, "Invalid seizure interval"):
            read_seizure_intervals(self.edf_path)

    def test_unlisted_edf_raises_without_official_manifest_evidence(self):
        self.write_summary(
            "File Name: another_recording.edf\n"
            "Number of Seizures in File: 0\n"
        )
        with self.assertRaisesRegex(ValueError, "official RECORDS manifests"):
            read_seizure_intervals(self.edf_path)

    def test_official_manifests_confirm_unlisted_recording_is_negative(self):
        self.write_summary(
            "File Name: another_recording.edf\n"
            "Number of Seizures in File: 0\n"
        )
        self.write_manifests("chb_test/chb_test_01.edf\n", "")
        self.assertEqual(read_seizure_intervals(self.edf_path), [])

    def test_manifest_seizure_entry_does_not_allow_missing_summary(self):
        self.write_summary(
            "File Name: another_recording.edf\n"
            "Number of Seizures in File: 0\n"
        )
        self.write_manifests(
            "chb_test/chb_test_01.edf\n",
            "chb_test/chb_test_01.edf\n",
        )
        with self.assertRaisesRegex(ValueError, "official RECORDS manifests"):
            read_seizure_intervals(self.edf_path)


if __name__ == "__main__":
    unittest.main()
