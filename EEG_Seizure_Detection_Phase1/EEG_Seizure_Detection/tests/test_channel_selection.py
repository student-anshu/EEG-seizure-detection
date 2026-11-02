import unittest

import numpy as np

from feature_selection.channel_selection import channel_features, rank_channels


class ChannelFeatureTests(unittest.TestCase):
    def test_feature_shape(self):
        windows = np.random.default_rng(42).normal(size=(8, 3, 100))
        features = channel_features(windows)
        self.assertEqual(features.shape, (8, 9))
        self.assertEqual(features.dtype, np.float32)

    def test_empty_or_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty 3D"):
            channel_features(np.ones((2, 3)))

    def test_non_finite_windows_raise(self):
        windows = np.ones((2, 2, 10))
        windows[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            channel_features(windows)

    def test_rank_channels_returns_channel_indices_and_scores(self):
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(20, 3, 50))
        labels = np.array([0] * 10 + [1] * 10)
        windows[labels == 1, 0, :] += 2.0

        indices, scores = rank_channels(windows, labels)

        self.assertEqual(indices.shape, (3,))
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(set(indices.tolist()), {0, 1, 2})

    def test_rank_channels_rejects_single_class(self):
        windows = np.ones((5, 2, 20))
        labels = np.zeros(5, dtype=int)
        with self.assertRaisesRegex(ValueError, "at least two label classes"):
            rank_channels(windows, labels)

    def test_rank_channels_rejects_mismatched_labels(self):
        windows = np.ones((5, 2, 20))
        with self.assertRaisesRegex(ValueError, "one value per window"):
            rank_channels(windows, np.array([0, 1]))

    def test_rank_channels_rejects_non_binary_labels(self):
        windows = np.ones((4, 2, 20))
        labels = np.array([0, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "binary"):
            rank_channels(windows, labels)


if __name__ == "__main__":
    unittest.main()
