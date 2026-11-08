import unittest

import numpy as np

from feature_selection.feature_extraction import (
    FEATURE_NAMES,
    extract_features,
)


class FeatureExtractionTests(unittest.TestCase):

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        windows = rng.normal(size=(8, 3, 2560))

        features = extract_features(windows, sfreq=256)

        self.assertEqual(
            features.shape,
            (8, 3 * len(FEATURE_NAMES)),
        )

    def test_output_dtype_is_float32(self):
        windows = np.random.default_rng(42).normal(
            size=(2, 3, 2560)
        )

        features = extract_features(windows, 256)

        self.assertEqual(features.dtype, np.float32)

    def test_output_is_finite(self):
        windows = np.random.default_rng(42).normal(
            size=(2, 3, 2560)
        )

        features = extract_features(windows, 256)

        self.assertTrue(np.all(np.isfinite(features)))

    def test_constant_signal_has_zero_variance(self):
        windows = np.ones((1, 1, 2560)) * 5.0

        features = extract_features(windows, 256)

        self.assertAlmostEqual(float(features[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(features[0, 2]), 0.0, places=6)

    def test_rms_of_constant_signal(self):
        windows = np.ones((1, 1, 2560)) * 5.0

        features = extract_features(windows, 256)

        self.assertAlmostEqual(float(features[0, 3]), 5.0, places=5)

    def test_two_hz_sine_power_is_approximately_half_in_delta(self):
        sfreq = 256
        time = np.arange(2560) / sfreq
        signal = np.sin(2 * np.pi * 2.0 * time)
        windows = signal[np.newaxis, np.newaxis, :]

        features = extract_features(windows, sfreq)

        delta_power = float(features[0, FEATURE_NAMES.index("delta_power")])
        self.assertAlmostEqual(delta_power, 0.5, delta=0.015)

    def test_ten_hz_half_amplitude_sine_power_is_alpha(self):
        sfreq = 256
        time = np.arange(2560) / sfreq
        signal = 0.5 * np.sin(2 * np.pi * 10.0 * time)
        windows = signal[np.newaxis, np.newaxis, :]

        features = extract_features(windows, sfreq)

        alpha_power = float(features[0, FEATURE_NAMES.index("alpha_power")])
        self.assertAlmostEqual(alpha_power, 0.125, delta=0.01)

    def test_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty EEG windows"):
            extract_features(np.ones((2, 100)), 256)

    def test_empty_input_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty EEG windows"):
            extract_features(np.empty((0, 2, 100)), 256)

    def test_non_finite_input_raises(self):
        windows = np.ones((1, 2, 100))
        windows[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            extract_features(windows, 256)

    def test_invalid_sampling_frequency_raises(self):
        windows = np.ones((1, 2, 2560))

        with self.assertRaisesRegex(ValueError, "at least 80 Hz"):
            extract_features(windows, 64)

    def test_invalid_zero_sampling_frequency_raises(self):
        windows = np.ones((1, 2, 2560))

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            extract_features(windows, 0)


if __name__ == "__main__":
    unittest.main()
