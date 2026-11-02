import unittest

from main import choose_channels


class FakeRaw:
    def __init__(self, ch_names):
        self.ch_names = ch_names


class ChooseChannelsTests(unittest.TestCase):
    def test_preferred_channels_are_selected_case_insensitively(self):
        raw = FakeRaw(["fp1-f7", "F7-T7", "ECG"])

        selected = choose_channels(raw)

        self.assertEqual(selected, ["fp1-f7", "F7-T7"])

    def test_fallback_selects_eeg_like_channels_in_edf_order(self):
        raw = FakeRaw(["ECG", "EEG F3", "C3-P3", "EOG"])

        selected = choose_channels(raw)

        self.assertEqual(selected, ["EEG F3", "C3-P3"])

    def test_insufficient_suitable_channels_raises(self):
        raw = FakeRaw(["ECG", "EOG"])

        with self.assertRaisesRegex(ValueError, "Fewer than two suitable EEG channels"):
            choose_channels(raw)


if __name__ == "__main__":
    unittest.main()
