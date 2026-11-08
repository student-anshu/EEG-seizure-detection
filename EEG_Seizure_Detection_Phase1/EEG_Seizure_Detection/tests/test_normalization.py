import unittest

import numpy as np

from preprocessing.normalization import FoldChannelStandardizer


class FoldChannelStandardizerTests(unittest.TestCase):
    def test_fit_uses_training_windows_channel_statistics(self):
        training = np.array(
            [
                [[1.0, 3.0], [10.0, 14.0]],
                [[5.0, 7.0], [18.0, 22.0]],
            ],
            dtype=np.float32,
        )
        scaler = FoldChannelStandardizer().fit(training)

        np.testing.assert_allclose(scaler.mean_, [4.0, 16.0])
        np.testing.assert_allclose(scaler.scale_, [np.sqrt(5.0), np.sqrt(20.0)])

    def test_transform_reuses_training_statistics_for_validation(self):
        training = np.array([[[0.0, 2.0], [10.0, 14.0]]])
        validation = np.array([[[100.0, 102.0], [1000.0, 1004.0]]])
        scaler = FoldChannelStandardizer().fit(training)
        fitted_mean = scaler.mean_.copy()
        fitted_scale = scaler.scale_.copy()

        transformed = scaler.transform(validation)

        np.testing.assert_array_equal(scaler.mean_, fitted_mean)
        np.testing.assert_array_equal(scaler.scale_, fitted_scale)
        np.testing.assert_allclose(transformed[0, 0], [99.0, 101.0])
        np.testing.assert_allclose(transformed[0, 1], [494.0, 496.0])
        self.assertEqual(transformed.dtype, np.float32)

    def test_fit_transform_standardizes_training_windows(self):
        training = np.array(
            [
                [[1.0, 3.0, 5.0]],
                [[7.0, 9.0, 11.0]],
            ],
            dtype=np.float32,
        )
        transformed = FoldChannelStandardizer().fit_transform(training)

        self.assertEqual(transformed.shape, training.shape)
        np.testing.assert_allclose(np.mean(transformed, axis=(0, 2)), [0.0])
        np.testing.assert_allclose(np.std(transformed, axis=(0, 2)), [1.0])

    def test_constant_channel_remains_finite(self):
        training = np.ones((2, 2, 5), dtype=np.float32)
        scaler = FoldChannelStandardizer().fit(training)
        transformed = scaler.transform(training)

        self.assertTrue(np.all(np.isfinite(transformed)))
        np.testing.assert_array_equal(transformed, np.zeros_like(transformed))

    def test_transform_before_fit_raises(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            FoldChannelStandardizer().transform(np.ones((1, 2, 5)))

    def test_rejects_wrong_channel_count(self):
        scaler = FoldChannelStandardizer().fit(np.ones((1, 2, 5)))
        with self.assertRaisesRegex(ValueError, "Expected 2 channels"):
            scaler.transform(np.ones((1, 3, 5)))

    def test_rejects_invalid_shape_and_non_finite_values(self):
        scaler = FoldChannelStandardizer()
        with self.assertRaisesRegex(ValueError, "shape"):
            scaler.fit(np.ones((2, 5)))
        invalid = np.ones((1, 2, 5))
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            scaler.fit(invalid)


    def test_streaming_batches_match_single_array_statistics(self):
        first = np.array([[[1.0, 3.0], [10.0, 14.0]]])
        second = np.array([[[5.0, 7.0], [18.0, 22.0]]])
        combined = np.concatenate([first, second], axis=0)
        one_shot = FoldChannelStandardizer().fit(combined)
        streamed = FoldChannelStandardizer().fit_batches([first, second])
        np.testing.assert_allclose(streamed.mean_, one_shot.mean_)
        np.testing.assert_allclose(streamed.scale_, one_shot.scale_)
        self.assertEqual(streamed.n_values_per_channel_, 4)

    def test_streaming_fit_rejects_empty_batches(self):
        with self.assertRaisesRegex(ValueError, "no training windows"):
            FoldChannelStandardizer().fit_batches([])


if __name__ == "__main__":
    unittest.main()
