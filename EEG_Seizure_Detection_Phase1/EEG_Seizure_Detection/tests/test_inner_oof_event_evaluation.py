import unittest

from data_management.patient_mapping import Recording
from scripts.evaluate_inner_oof_events import (
    _patient_summary,
    validate_recording_offsets,
)


def recording(name: str) -> Recording:
    return Recording(
        subject="chb02",
        patient_id="patient_02",
        edf_file=name,
        relative_path=f"chb02/{name}",
        seizure_count=0,
        seizure_intervals_seconds=(),
        annotation_source="subject_summary",
        status="SUCCESS",
    )


class InnerOofEventEvaluationTests(unittest.TestCase):
    def test_offsets_require_exact_identity_order_and_contiguous_slices(self):
        records = [recording("chb02_01.edf"), recording("chb02_02.edf")]
        offsets = [
            {"relative_path": records[0].relative_path, "start": 0, "end": 3},
            {"relative_path": records[1].relative_path, "start": 3, "end": 8},
        ]
        validate_recording_offsets(offsets, records, 8)

    def test_offsets_reject_wrong_recording_identity(self):
        records = [recording("chb02_01.edf")]
        offsets = [{"relative_path": "chb02/chb02_99.edf", "start": 0, "end": 3}]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_recording_offsets(offsets, records, 3)

    def test_offsets_reject_gaps_and_wrong_total(self):
        records = [recording("chb02_01.edf"), recording("chb02_02.edf")]
        gap = [
            {"relative_path": records[0].relative_path, "start": 0, "end": 3},
            {"relative_path": records[1].relative_path, "start": 4, "end": 8},
        ]
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            validate_recording_offsets(gap, records, 8)
        with self.assertRaisesRegex(ValueError, "cover 3 scores"):
            validate_recording_offsets(gap[:1], records[:1], 9)

    def test_patient_summary_pools_exposure_before_calculating_rate(self):
        rows = [
            {
                "patient_id": "patient_02",
                "event_count": 2,
                "detected_events": 1,
                "false_alarm_count": 2,
                "non_seizure_duration_seconds": 1800.0,
                "events": [
                    {"detected": True, "detection_delay_seconds": 5.0},
                    {"detected": False, "detection_delay_seconds": None},
                ],
            },
            {
                "patient_id": "patient_02",
                "event_count": 0,
                "detected_events": 0,
                "false_alarm_count": 1,
                "non_seizure_duration_seconds": 1800.0,
                "events": [],
            },
        ]
        summary = _patient_summary(rows)[0]
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["detected_events"], 1)
        self.assertEqual(summary["missed_events"], 1)
        self.assertAlmostEqual(summary["event_sensitivity"], 0.5)
        self.assertEqual(summary["false_alarm_count"], 3)
        self.assertEqual(summary["non_seizure_duration_seconds"], 3600.0)
        self.assertAlmostEqual(summary["false_alarms_per_hour"], 3.0)
        self.assertAlmostEqual(summary["mean_detection_delay_seconds"], 5.0)

    def test_patient_summary_keeps_rate_null_if_any_exposure_is_missing(self):
        row = {
            "patient_id": "patient_02",
            "event_count": 0,
            "detected_events": 0,
            "false_alarm_count": 1,
            "non_seizure_duration_seconds": None,
            "events": [],
        }
        summary = _patient_summary([row])[0]
        self.assertIsNone(summary["non_seizure_duration_seconds"])
        self.assertIsNone(summary["false_alarms_per_hour"])


if __name__ == "__main__":
    unittest.main()
