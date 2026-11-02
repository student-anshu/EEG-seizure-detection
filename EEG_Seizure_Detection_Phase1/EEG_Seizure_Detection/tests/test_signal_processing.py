import unittest

import numpy as np

from preprocessing.signal_processing import bandpass_and_notch, robust_normalize


class BandpassAndNotchTests(unittest.TestCase):
    def test_filter_preserves_shape_and_returns_float32(self):
        sfreq = 256.0
        time = np.arange(int(sfreq * 5)) / sfreq
        signal = np.sin(2 * np.pi * 10 * time)
        data = np.vstack([signal, signal])

        result = bandpass_and_notch(data, sfreq)

        self.assertEqual(result.shape, data.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_60_hz_notch_reduces_60_hz_component(self):
        sfreq = 256.0
        time = np.arange(int(sfreq * 10)) / sfreq
        data = np.sin(2 * np.pi * 60 * time)[None, :]

        filtered = bandpass_and_notch(data, sfreq, low_hz=1.0, high_hz=100.0)

        # Ignore filter edge transients when measuring the remaining amplitude.
        edge = int(sfreq)
        rms_before = np.sqrt(np.mean(data[:, edge:-edge] ** 2))
        rms_after = np.sqrt(np.mean(filtered[:, edge:-edge] ** 2))
        self.assertLess(rms_after, rms_before * 0.2)

    def test_notch_is_skipped_when_at_or_above_nyquist(self):
        sfreq = 80.0
        time = np.arange(int(sfreq * 5)) / sfreq
        data = np.sin(2 * np.pi * 10 * time)[None, :]

        result = bandpass_and_notch(data, sfreq, notch_hz=60.0)

        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_invalid_sampling_frequency_raises(self):
        data = np.ones((2, 1000))
        with self.assertRaisesRegex(ValueError, "Sampling frequency"):
            bandpass_and_notch(data, 0)

    def test_invalid_data_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty 2D"):
            bandpass_and_notch(np.ones(1000), 256)

    def test_non_finite_input_raises(self):
        data = np.ones((2, 1000))
        data[0, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            bandpass_and_notch(data, 256)


class RobustNormalizeTests(unittest.TestCase):
    def test_normalizes_each_channel_by_median_and_iqr(self):
        data = np.array(
            [[-2.0, -1.0, 0.0, 1.0, 2.0],
             [10.0, 12.0, 14.0, 16.0, 18.0]]
        )

        result = robust_normalize(data)

        self.assertEqual(result.shape, data.shape)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.median(result, axis=1), 0.0)
        np.testing.assert_allclose(
            np.percentile(result, 75, axis=1)
            - np.percentile(result, 25, axis=1),
            1.0,
            rtol=1e-6,
        )

    def test_constant_channel_remains_finite(self):
        data = np.ones((1, 100)) * 5.0

        result = robust_normalize(data)

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty 2D"):
            robust_normalize(np.ones(100))

    def test_non_finite_input_raises(self):
        data = np.ones((1, 100))
        data[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            robust_normalize(data)


if __name__ == "__main__":
    unittest.main()
