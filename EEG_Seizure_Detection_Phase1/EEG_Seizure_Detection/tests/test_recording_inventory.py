"""Tests for identity-safe recording duration inventory joins."""
import tempfile
import unittest
from pathlib import Path

from data_management.patient_mapping import Recording
from evaluation.recording_inventory import load_recording_durations


class RecordingInventoryTests(unittest.TestCase):
    def setUp(self):
        self.recording = Recording(
            subject="chb01", patient_id="patient_01", edf_file="chb01_03.edf",
            relative_path="chb01/chb01_03.edf", seizure_count=1,
            seizure_intervals_seconds=("10-15",),
            annotation_source="subject_summary", status="SUCCESS",
        )

    def write_inventory(self, root: Path, rows: str) -> Path:
        path = root / "signal_inventory.csv"
        path.write_text(
            "subject,edf_file,relative_path,sampling_frequency_hz,n_samples,"
            "duration_seconds,n_channels,channel_names,status,error\n" + rows,
            encoding="utf-8",
        )
        return path

    def test_joins_by_normalized_path_and_validates_duration_arithmetic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(
                Path(temp),
                "chb01,chb01_03.edf,CHB01\\chb01_03.edf,256,25600,100,23,x,SUCCESS,\n",
            )
            durations, audit = load_recording_durations(path, [self.recording])
            self.assertEqual(durations[self.recording.relative_path], 100.0)
            self.assertEqual(audit["matched_primary_recording_count"], 1)

    def test_missing_primary_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(Path(temp), "")
            with self.assertRaisesRegex(ValueError, "missing from duration inventory"):
                load_recording_durations(path, [self.recording])

    def test_duplicate_inventory_path_fails_closed(self):
        row = "chb01,chb01_03.edf,chb01/chb01_03.edf,256,25600,100,23,x,SUCCESS,\n"
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(Path(temp), row + row)
            with self.assertRaisesRegex(ValueError, "Duplicate inventory"):
                load_recording_durations(path, [self.recording])

    def test_inconsistent_duration_arithmetic_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(
                Path(temp),
                "chb01,chb01_03.edf,chb01/chb01_03.edf,256,25600,101,23,x,SUCCESS,\n",
            )
            with self.assertRaisesRegex(ValueError, "arithmetic mismatch"):
                load_recording_durations(path, [self.recording])

    def test_non_finite_duration_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(
                Path(temp),
                "chb01,chb01_03.edf,chb01/chb01_03.edf,256,25600,nan,23,x,SUCCESS,\n",
            )
            with self.assertRaisesRegex(ValueError, "Invalid duration fields"):
                load_recording_durations(path, [self.recording])

    def test_subject_or_filename_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_inventory(
                Path(temp),
                "chb02,chb01_03.edf,chb01/chb01_03.edf,256,25600,100,23,x,SUCCESS,\n",
            )
            with self.assertRaisesRegex(ValueError, "identity columns disagree"):
                load_recording_durations(path, [self.recording])


if __name__ == "__main__":
    unittest.main()
