import unittest

from scripts.simulate_validation_alarm_policies import (
    _evaluate_recording, merge_intervals, MIN_DURATIONS, MERGE_GAPS,
)


class AlarmPolicySimulationTests(unittest.TestCase):
    def test_grid_has_24_combinations(self):
        self.assertEqual(len(MIN_DURATIONS) * len(MERGE_GAPS), 24)

    def test_gap_merge_combines_intervals_at_boundary(self):
        self.assertEqual(merge_intervals([(0, 10), (15, 20)], 5), [(0, 20)])

    def test_gap_larger_than_tolerance_does_not_merge(self):
        self.assertEqual(merge_intervals([(0, 10), (15.01, 20)], 5), [(0, 10), (15.01, 20)])

    def test_minimum_duration_applies_to_each_raw_alarm_before_grouping(self):
        record = {
            "relative_path": "chb01/test.edf", "patient_id": "patient_01",
            "alarms": [{"start": 0, "end": 10}, {"start": 15, "end": 20}],
            "events": [], "non_seizure_duration_seconds": 3600,
        }
        result = _evaluate_recording(record, 15, 5)
        self.assertEqual(result["alarm_count"], 0)
        self.assertEqual(result["false_alarm_count"], 0)

    def test_gap_merge_cannot_create_seizure_detection(self):
        record = {
            "relative_path": "chb01/test.edf", "patient_id": "patient_01",
            "alarms": [{"start": 0, "end": 5}, {"start": 15, "end": 20}],
            "events": [{"seizure_start_seconds": 8, "seizure_end_seconds": 12}],
            "non_seizure_duration_seconds": 3596,
        }
        result = _evaluate_recording(record, 0, 10)
        self.assertEqual(result["detected_events"], 0)
        self.assertEqual(result["missed_events"], 1)
        # The grouped alarm remains a false alarm because no actual alarm overlaps seizure.
        self.assertEqual(result["alarm_count"], 1)
        self.assertEqual(result["false_alarm_count"], 1)

    def test_filter_can_remove_short_true_detection(self):
        record = {
            "relative_path": "chb01/test.edf", "patient_id": "patient_01",
            "alarms": [{"start": 10, "end": 20}],
            "events": [{"seizure_start_seconds": 15, "seizure_end_seconds": 25}],
            "non_seizure_duration_seconds": 3590,
        }
        self.assertEqual(_evaluate_recording(record, 0, 0)["detected_events"], 1)
        self.assertEqual(_evaluate_recording(record, 15, 0)["detected_events"], 0)

    def test_group_containing_actual_event_alarm_is_not_false(self):
        record = {
            "relative_path": "chb01/test.edf", "patient_id": "patient_01",
            "alarms": [{"start": 0, "end": 5}, {"start": 15, "end": 20}],
            "events": [{"seizure_start_seconds": 16, "seizure_end_seconds": 18}],
            "non_seizure_duration_seconds": 3596,
        }
        result = _evaluate_recording(record, 0, 10)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["false_alarm_count"], 0)


if __name__ == "__main__":
    unittest.main()
