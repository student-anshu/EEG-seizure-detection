import csv
import tempfile
import unittest
from pathlib import Path

from data_management.dataset_split import build_lopo_folds
from data_management.patient_mapping import (
    load_recording_manifest,
    patient_id_for_subject,
)


class PatientMappingTests(unittest.TestCase):
    def test_chb01_and_chb21_map_to_same_patient(self):
        self.assertEqual(patient_id_for_subject("chb01"), "patient_01")
        self.assertEqual(patient_id_for_subject("CHB21"), "patient_01")

    def test_other_subject_keeps_numeric_patient_identity(self):
        self.assertEqual(patient_id_for_subject("chb12"), "patient_12")
        self.assertEqual(patient_id_for_subject("chb24"), "patient_24")

    def test_invalid_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            patient_id_for_subject("patient_01")

    def _write_manifest(self, directory: str, rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "audit.csv"
        fieldnames = [
            "subject",
            "edf_file",
            "relative_path",
            "seizure_count",
            "seizure_intervals_seconds",
            "annotation_source",
            "status",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _row(self, subject: str, filename: str, count: int = 0) -> dict[str, str]:
        interval = "100-120" if count else ""
        return {
            "subject": subject,
            "edf_file": filename,
            "relative_path": f"{subject}/{filename}",
            "seizure_count": str(count),
            "seizure_intervals_seconds": interval,
            "annotation_source": "subject_summary",
            "status": "SUCCESS",
            "error": "",
        }

    def test_load_manifest_maps_chb01_and_chb21_together(self):
        rows = [
            self._row("chb01", "chb01_01.edf"),
            self._row("chb21", "chb21_01.edf", 1),
            self._row("chb02", "chb02_01.edf"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(directory, rows)
            recordings = load_recording_manifest(path)

        self.assertEqual(len(recordings), 3)
        self.assertEqual(recordings[0].patient_id, recordings[1].patient_id)
        self.assertEqual(recordings[2].patient_id, "patient_02")
        self.assertEqual(recordings[1].seizure_intervals_seconds, ("100-120",))

    def test_manifest_rejects_failed_audit_row(self):
        row = self._row("chb01", "chb01_01.edf")
        row["status"] = "FAILED"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(directory, [row])
            with self.assertRaisesRegex(ValueError, "not successful"):
                load_recording_manifest(path)

    def test_manifest_rejects_interval_count_mismatch(self):
        row = self._row("chb01", "chb01_01.edf", 2)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(directory, [row])
            with self.assertRaisesRegex(ValueError, "contains 1 interval"):
                load_recording_manifest(path)

    def test_manifest_rejects_duplicate_recording_path(self):
        row = self._row("chb01", "chb01_01.edf")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(directory, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate recording path"):
                load_recording_manifest(path)


class LopoSplitTests(unittest.TestCase):
    def _recordings(self):
        helper = PatientMappingTests()
        rows = [
            helper._row("chb01", "chb01_01.edf"),
            helper._row("chb21", "chb21_01.edf", 1),
            helper._row("chb02", "chb02_01.edf"),
            helper._row("chb02", "chb02_02.edf", 1),
            helper._row("chb03", "chb03_01.edf"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = helper._write_manifest(directory, rows)
            return load_recording_manifest(path)

    def test_builds_one_fold_per_independent_patient(self):
        recordings = self._recordings()
        folds = build_lopo_folds(recordings)
        self.assertEqual(len(folds), 3)
        self.assertEqual(
            {fold.test_patient_id for fold in folds},
            {"patient_01", "patient_02", "patient_03"},
        )

    def test_no_patient_leakage_in_any_fold(self):
        folds = build_lopo_folds(self._recordings())
        for fold in folds:
            train_patients = {item.patient_id for item in fold.train_recordings}
            test_patients = {item.patient_id for item in fold.test_recordings}
            self.assertEqual(test_patients, {fold.test_patient_id})
            self.assertTrue(train_patients.isdisjoint(test_patients))

    def test_every_recording_is_tested_exactly_once(self):
        recordings = self._recordings()
        folds = build_lopo_folds(recordings)
        tested_paths = [
            item.relative_path for fold in folds for item in fold.test_recordings
        ]
        self.assertEqual(len(tested_paths), len(recordings))
        self.assertEqual(set(tested_paths), {item.relative_path for item in recordings})

    def test_requires_at_least_two_patients(self):
        recordings = self._recordings()
        only_one_patient = [item for item in recordings if item.patient_id == "patient_01"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_lopo_folds(only_one_patient)


if __name__ == "__main__":
    unittest.main()
