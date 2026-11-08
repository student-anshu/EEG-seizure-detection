import tempfile
import unittest

from data_management.patient_mapping import load_recording_manifest
from data_management.validation_split import build_patient_validation_split
from tests.test_dataset_split import PatientMappingTests


class PatientValidationSplitTests(unittest.TestCase):
    def _recordings(self):
        helper = PatientMappingTests()
        rows = [
            helper._row("chb01", "chb01_01.edf", 1),
            helper._row("chb21", "chb21_01.edf", 1),
            helper._row("chb02", "chb02_01.edf", 1),
            helper._row("chb02", "chb02_02.edf"),
            {
                **helper._row("chb03", "chb03_01.edf", 2),
                "seizure_intervals_seconds": "100-120; 200-220",
            },
            helper._row("chb04", "chb04_01.edf", 1),
            helper._row("chb05", "chb05_01.edf", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = helper._write_manifest(directory, rows)
            return load_recording_manifest(path)

    def test_keeps_all_recordings_from_a_patient_together(self):
        split = build_patient_validation_split(self._recordings())
        train = set(split.train_patient_ids)
        validation = set(split.validation_patient_ids)
        self.assertTrue(train.isdisjoint(validation))
        self.assertEqual({item.patient_id for item in split.train_recordings}, train)
        self.assertEqual(
            {item.patient_id for item in split.validation_recordings}, validation
        )

    def test_train_and_validation_cover_all_recordings_without_overlap(self):
        recordings = self._recordings()
        split = build_patient_validation_split(recordings)
        train_paths = {item.relative_path for item in split.train_recordings}
        validation_paths = {item.relative_path for item in split.validation_recordings}
        all_paths = {item.relative_path for item in recordings}
        self.assertTrue(train_paths.isdisjoint(validation_paths))
        self.assertEqual(train_paths | validation_paths, all_paths)

    def test_split_is_reproducible_for_same_seed(self):
        recordings = self._recordings()
        first = build_patient_validation_split(recordings, random_state=17)
        second = build_patient_validation_split(recordings, random_state=17)
        self.assertEqual(first.validation_patient_ids, second.validation_patient_ids)

    def test_different_seed_can_change_validation_patients(self):
        recordings = self._recordings()
        first = build_patient_validation_split(recordings, random_state=1)
        second = build_patient_validation_split(recordings, random_state=17)
        self.assertNotEqual(first.validation_patient_ids, second.validation_patient_ids)

    def test_validation_fraction_uses_ceiling(self):
        split = build_patient_validation_split(
            self._recordings(), validation_fraction=0.4
        )
        self.assertEqual(len(split.validation_patient_ids), 2)
        self.assertEqual(len(split.train_patient_ids), 3)

    def test_rejects_invalid_validation_fraction(self):
        for fraction in (0, 1, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    build_patient_validation_split(
                        self._recordings(), validation_fraction=fraction
                    )

    def test_requires_at_least_two_patients(self):
        recordings = [
            item for item in self._recordings() if item.patient_id == "patient_01"
        ]
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_patient_validation_split(recordings)

    def test_rejects_duplicate_recording_paths(self):
        recordings = self._recordings()
        with self.assertRaisesRegex(ValueError, "Duplicate recording path"):
            build_patient_validation_split(recordings + [recordings[0]])


if __name__ == "__main__":
    unittest.main()
