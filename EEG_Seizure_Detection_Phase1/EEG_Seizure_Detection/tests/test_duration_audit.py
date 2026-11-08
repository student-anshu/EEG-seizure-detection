import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_management.patient_mapping import Recording
from scripts.audit_recording_durations import audit_recording_durations


class DurationAuditTests(unittest.TestCase):
    def setUp(self):
        self.recording = Recording(
            subject="chb01",
            patient_id="patient_01",
            edf_file="chb01_03.edf",
            relative_path="chb01/chb01_03.edf",
            seizure_count=1,
            seizure_intervals_seconds=("2996-3036",),
            annotation_source="subject_summary",
            status="SUCCESS",
        )

    def test_reads_duration_from_existing_npz_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "chb01_chb01_03.npz"
            np.savez_compressed(
                path, recording_duration_seconds=np.asarray(3600.0),
                marker=np.asarray([1, 2, 3]),
            )
            before = path.read_bytes()
            result = audit_recording_durations([self.recording], processed_dir=root)
            self.assertEqual(result["counts"]["exact_duration_available"], 1)
            self.assertEqual(result["counts"]["npz_duration"], 1)
            self.assertEqual(path.read_bytes(), before)

    def test_detects_conflicting_metadata_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(
                root / "chb01_chb01_03.npz",
                recording_duration_seconds=np.asarray(3600.0),
            )
            pd.DataFrame({
                "recording_duration_seconds": [3590.0, 3590.0],
            }).to_csv(root / "chb01_chb01_03_metadata.csv", index=False)
            result = audit_recording_durations([self.recording], processed_dir=root)
            self.assertEqual(result["counts"]["conflicts"], 1)

    def test_reports_missing_duration_without_guessing(self):
        with tempfile.TemporaryDirectory() as temp:
            result = audit_recording_durations([self.recording], processed_dir=temp)
            self.assertEqual(result["counts"]["missing_duration"], 1)
            self.assertEqual(result["counts"]["exact_duration_available"], 0)
            self.assertTrue(result["read_only"])

    def test_reads_central_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "recording_durations.json"
            manifest.write_text(json.dumps({
                "recordings": {
                    self.recording.relative_path: {
                        "recording_duration_seconds": 3600.0
                    }
                }
            }), encoding="utf-8")
            result = audit_recording_durations(
                [self.recording], processed_dir=root, manifest_path=manifest
            )
            self.assertEqual(result["counts"]["manifest_duration"], 1)
            self.assertEqual(result["counts"]["exact_duration_available"], 1)


if __name__ == "__main__":
    unittest.main()
