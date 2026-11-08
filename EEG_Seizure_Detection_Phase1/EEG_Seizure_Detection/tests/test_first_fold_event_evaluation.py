import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_management.patient_mapping import Recording
from scripts.evaluate_first_fold_events import (
    _score_slices, _patient_summaries, _add_event_score_diagnostics,
)


class FirstFoldEventEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.recording = Recording(
            subject="chb01", patient_id="patient_01",
            edf_file="chb01_03.edf", relative_path="chb01/chb01_03.edf",
            seizure_count=1, seizure_intervals_seconds=("10-15",),
            annotation_source="subject_summary", status="SUCCESS",
        )

    def test_score_slicing_aligns_saved_window_labels_and_counts_event_without_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(
                root / "chb01_chb01_03.npz",
                y=np.asarray([0, 1, 0], dtype=np.int8),
                starts=np.asarray([0.0, 5.0, 10.0]),
            )
            results = _score_slices(
                [self.recording],
                np.asarray([0, 1, 0], dtype=np.int8),
                np.asarray([0.1, 0.9, 0.2]),
                root,
                0.5,
                "test",
            )
            self.assertEqual(results[0]["event_count"], 1)
            self.assertEqual(results[0]["detected_events"], 1)
            self.assertEqual(results[0]["false_alarm_count"], 0)
            self.assertIsNone(results[0]["false_alarms_per_hour"])

    def test_score_label_misalignment_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(
                root / "chb01_chb01_03.npz",
                y=np.asarray([1, 0, 0], dtype=np.int8),
                starts=np.asarray([0.0, 5.0, 10.0]),
            )
            with self.assertRaisesRegex(ValueError, "do not align"):
                _score_slices(
                    [self.recording],
                    np.asarray([0, 1, 0], dtype=np.int8),
                    np.asarray([0.1, 0.9, 0.2]),
                    root,
                    0.5,
                    "test",
                )

    def test_patient_summary_keeps_unknown_exposure_null(self):
        rows = [{
            "patient_id": "patient_01",
            "event_count": 2,
            "detected_events": 1,
            "missed_events": 1,
            "mean_detection_delay_seconds": 3.0,
            "false_alarm_count": 4,
        }]
        summary = _patient_summaries(rows)[0]
        self.assertEqual(summary["event_sensitivity"], 0.5)
        self.assertEqual(summary["false_alarm_count"], 4)
        self.assertIsNone(summary["false_alarms_per_hour"])

    
    def test_duration_inventory_is_used_for_false_alarms_per_hour(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(
                root / "chb01_chb01_03.npz",
                y=np.asarray([0, 1, 0], dtype=np.int8),
                starts=np.asarray([0.0, 5.0, 10.0]),
            )
            results = _score_slices(
                [self.recording],
                np.asarray([0, 1, 0], dtype=np.int8),
                np.asarray([0.9, 0.1, 0.1]),
                root,
                0.5,
                "test",
                {self.recording.relative_path: 20.0},
            )
            row = results[0]
            self.assertEqual(row["false_alarm_count"], 1)
            self.assertEqual(row["non_seizure_duration_seconds"], 15.0)
            self.assertAlmostEqual(row["false_alarms_per_hour"], 240.0)

    def test_patient_rate_uses_summed_false_alarms_and_exposure(self):
        rows = [
            {
                "patient_id": "patient_01", "event_count": 1,
                "detected_events": 1, "missed_events": 0,
                "mean_detection_delay_seconds": 2.0,
                "false_alarm_count": 1, "non_seizure_duration_seconds": 900.0,
            },
            {
                "patient_id": "patient_01", "event_count": 1,
                "detected_events": 0, "missed_events": 1,
                "mean_detection_delay_seconds": None,
                "false_alarm_count": 1, "non_seizure_duration_seconds": 900.0,
            },
        ]
        summary = _patient_summaries(rows)[0]
        self.assertEqual(summary["false_alarm_count"], 2)
        self.assertEqual(summary["non_seizure_duration_seconds"], 1800.0)
        self.assertAlmostEqual(summary["false_alarms_per_hour"], 4.0)

    def test_event_score_diagnostics_explain_threshold_miss(self):
        events = [{
            "event_index": 1,
            "seizure_start_seconds": 10.0,
            "seizure_end_seconds": 20.0,
            "detected": False,
        }]
        starts = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 40.0])
        scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.2, 0.1])
        _add_event_score_diagnostics(
            events, starts, scores, threshold=0.5, window_seconds=10.0
        )
        event = events[0]
        self.assertEqual(event["overlapping_window_count"], 3)
        self.assertAlmostEqual(event["max_overlapping_window_score"], 0.4)
        self.assertAlmostEqual(event["max_score_minus_threshold"], -0.1)
        self.assertEqual(event["top_scoring_window_start_seconds"], 15.0)
        self.assertGreaterEqual(event["context_window_count"], 2)

    def test_event_score_diagnostics_handles_no_overlapping_windows(self):
        events = [{
            "event_index": 1,
            "seizure_start_seconds": 100.0,
            "seizure_end_seconds": 105.0,
            "detected": False,
        }]
        _add_event_score_diagnostics(
            events, np.asarray([0.0, 5.0]), np.asarray([0.1, 0.2]),
            threshold=0.5, window_seconds=10.0,
        )
        self.assertEqual(events[0]["overlapping_window_count"], 0)
        self.assertIsNone(events[0]["max_overlapping_window_score"])


if __name__ == "__main__":
    unittest.main()
