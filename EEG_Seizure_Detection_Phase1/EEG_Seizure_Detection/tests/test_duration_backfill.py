import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_management.patient_mapping import Recording
from scripts.backfill_recording_durations import backfill_recording_durations


class DurationBackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.processed = self.root / "processed"
        self.edf = self.dataset / "chb01" / "chb01_03.edf"
        self.edf.parent.mkdir(parents=True)
        self.edf.write_bytes(b"fake-edf-header-source")
        self.processed.mkdir()
        self.recording = Recording(
            subject="chb01",
            patient_id="patient_01",
            edf_file="chb01_03.edf",
            relative_path="chb01/chb01_03.edf",
            seizure_count=1,
            seizure_intervals_seconds=("12-14",),
            annotation_source="test",
            status="SUCCESS",
        )
        np.savez_compressed(
            self.processed / "chb01_chb01_03.npz",
            X=np.zeros((5, 17, 10), dtype=np.float32),
            y=np.asarray([0, 1, 1, 0, 0]),
            starts=np.asarray([0, 5, 10, 15, 20], dtype=np.float64),
            subject="chb01",
            source_file="chb01_03.edf",
        )
        pd.DataFrame({
            "window_start_seconds": [0, 5, 10, 15, 20],
            "window_end_seconds": [10, 15, 20, 25, 30],
            "label": [0, 1, 1, 0, 0],
        }).to_csv(self.processed / "chb01_chb01_03_metadata.csv", index=False)
        self.reader = lambda path: (30.0, 1.0, 30)

    def tearDown(self):
        self.temp.cleanup()

    def test_backfills_metadata_and_manifest_without_rewriting_npz(self):
        npz = self.processed / "chb01_chb01_03.npz"
        before = npz.read_bytes()
        result = backfill_recording_durations(
            [self.recording], dataset_dir=self.dataset,
            processed_dir=self.processed, header_reader=self.reader,
        )
        self.assertEqual(result, {"updated": 1, "skipped_valid": 0, "failed": 0})
        metadata = pd.read_csv(self.processed / "chb01_chb01_03_metadata.csv")
        self.assertTrue((metadata["recording_duration_seconds"] == 30.0).all())
        manifest = json.loads((self.processed / "recording_durations.json").read_text())
        self.assertEqual(
            manifest["recordings"]["chb01/chb01_03.edf"]["sample_count"], 30
        )
        self.assertEqual(npz.read_bytes(), before)

    def test_resume_skips_unchanged_recording(self):
        first = backfill_recording_durations(
            [self.recording], dataset_dir=self.dataset,
            processed_dir=self.processed, header_reader=self.reader,
        )
        def unexpected_header_read(path):
            raise AssertionError("header should not be reread for an unchanged EDF")

        second = backfill_recording_durations(
            [self.recording], dataset_dir=self.dataset,
            processed_dir=self.processed,
            header_reader=unexpected_header_read,
        )
        self.assertEqual(first["updated"], 1)
        self.assertEqual(second["skipped_valid"], 1)

    def test_dry_run_does_not_modify_metadata_or_manifest(self):
        metadata_path = self.processed / "chb01_chb01_03_metadata.csv"
        before = metadata_path.read_bytes()
        result = backfill_recording_durations(
            [self.recording], dataset_dir=self.dataset,
            processed_dir=self.processed, dry_run=True,
            header_reader=self.reader,
        )
        self.assertEqual(result["updated"], 1)
        self.assertEqual(metadata_path.read_bytes(), before)
        self.assertFalse((self.processed / "recording_durations.json").exists())

    def test_missing_edf_is_reported_without_aborting_other_recordings(self):
        self.edf.unlink()
        result = backfill_recording_durations(
            [self.recording], dataset_dir=self.dataset,
            processed_dir=self.processed, header_reader=self.reader,
        )
        self.assertEqual(result["failed"], 1)
        self.assertTrue(
            (self.processed / "recording_duration_backfill_errors.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
