import unittest

from scripts.sweep_event_thresholds import (
    event_f2_proxy,
    _candidate_thresholds,
    _select_candidate,
)


class EventThresholdSweepTests(unittest.TestCase):
    def test_event_f2_proxy_rewards_recall_and_penalizes_false_alarm_episodes(self):
        clean = event_f2_proxy(8, 10, 2)
        noisy = event_f2_proxy(8, 10, 20)
        low_recall = event_f2_proxy(2, 10, 0)
        self.assertGreater(clean, noisy)
        self.assertGreater(clean, low_recall)

    def test_event_f2_proxy_handles_no_events(self):
        self.assertEqual(event_f2_proxy(0, 0, 0), 0.0)

    def test_candidate_thresholds_include_existing_thresholds_and_are_unique(self):
        import numpy as np

        values = _candidate_thresholds(
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            [0.25, 0.25, 0.9],
        )
        self.assertIn(0.25, values)
        self.assertIn(0.9, values)
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(values, sorted(values))

    def test_selection_uses_macro_f2_then_sensitivity_then_false_alarms(self):
        rows = [
            {
                "threshold": 0.8,
                "macro_patient_event_f2_proxy": 0.5,
                "macro_patient_event_sensitivity": 0.7,
                "aggregate": {"false_alarm_count": 20},
            },
            {
                "threshold": 0.9,
                "macro_patient_event_f2_proxy": 0.5,
                "macro_patient_event_sensitivity": 0.8,
                "aggregate": {"false_alarm_count": 25},
            },
            {
                "threshold": 0.95,
                "macro_patient_event_f2_proxy": 0.5,
                "macro_patient_event_sensitivity": 0.8,
                "aggregate": {"false_alarm_count": 10},
            },
        ]
        self.assertEqual(_select_candidate(rows)["threshold"], 0.95)


if __name__ == "__main__":
    unittest.main()
