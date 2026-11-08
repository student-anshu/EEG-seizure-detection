import unittest
from data_management.cohort import EXCLUDED_MONTAGE_RECORDINGS, filter_primary_cohort
from data_management.patient_mapping import Recording

def make_recording(path, count=0):
    subject, filename = path.split("/")
    intervals = tuple(f"{i * 100}-{i * 100 + 20}" for i in range(count))
    return Recording(subject, "patient_12", filename, path, count, intervals, "subject_summary", "SUCCESS")

class PrimaryCohortTests(unittest.TestCase):
    def test_excludes_only_three_documented_recordings(self):
        paths = ["chb12/chb12_26.edf", *sorted(EXCLUDED_MONTAGE_RECORDINGS), "chb12/chb12_32.edf"]
        selected = filter_primary_cohort([make_recording(path) for path in paths])
        self.assertEqual({r.relative_path for r in selected}, {"chb12/chb12_26.edf", "chb12/chb12_32.edf"})

    def test_requires_exclusions_to_exist_in_audit(self):
        with self.assertRaisesRegex(ValueError, "absent from the audit"):
            filter_primary_cohort([make_recording("chb12/chb12_32.edf")])

    def test_preserves_remaining_recording_annotations(self):
        paths = [*sorted(EXCLUDED_MONTAGE_RECORDINGS), "chb12/chb12_33.edf"]
        selected = filter_primary_cohort([make_recording(p, 2 if p.endswith("_33.edf") else 0) for p in paths])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].seizure_count, 2)

if __name__ == "__main__":
    unittest.main()
