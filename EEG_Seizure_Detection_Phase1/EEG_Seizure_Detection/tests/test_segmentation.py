import unittest

import numpy as np

from preprocessing.segmentation import make_windows


class MakeWindowsTests(unittest.TestCase):
    def test_window_count_shape_and_starts(self):
        data = np.arange(2 * 30, dtype=np.float32).reshape(2, 30)
        windows, labels, starts = make_windows(
            data, sfreq=1, seizure_intervals=[], window_seconds=10, stride_seconds=5
        )
        self.assertEqual(windows.shape, (5, 2, 10))
        np.testing.assert_array_equal(labels, np.zeros(5, dtype=np.int64))
        np.testing.assert_allclose(starts, [0, 5, 10, 15, 20])

    def test_seizure_overlap_labels_positive(self):
        data = np.zeros((1, 30), dtype=np.float32)
        _, labels, _ = make_windows(
            data, 1, [(12, 14)], window_seconds=10, stride_seconds=5
        )
        np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0])

    def test_boundary_touch_without_overlap_is_negative(self):
        data = np.zeros((1, 20), dtype=np.float32)
        _, labels, _ = make_windows(
            data, 1, [(10, 12)], window_seconds=10, stride_seconds=10
        )
        np.testing.assert_array_equal(labels, [0, 1])

    def test_short_recording_returns_empty_arrays(self):
        data = np.zeros((2, 9), dtype=np.float32)
        windows, labels, starts = make_windows(
            data, 1, [], window_seconds=10, stride_seconds=5
        )
        self.assertEqual(windows.shape, (0, 2, 10))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(starts.shape, (0,))

    def test_invalid_sampling_frequency_raises(self):
        with self.assertRaisesRegex(ValueError, "Sampling frequency"):
            make_windows(np.ones((1, 10)), 0, [])

    def test_invalid_seizure_interval_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid seizure interval"):
            make_windows(np.ones((1, 20)), 1, [(5, 3)])

    def test_non_finite_data_raises(self):
        data = np.ones((1, 20))
        data[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            make_windows(data, 1, [])


if __name__ == "__main__":
    unittest.main()
