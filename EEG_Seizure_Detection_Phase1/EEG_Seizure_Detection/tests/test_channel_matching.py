import unittest
from config import PREFERRED_CHANNELS
from main import choose_channels

class FakeRaw:
    def __init__(self, ch_names):
        self.ch_names = ch_names

class ChooseChannelsTests(unittest.TestCase):
    def test_complete_montage_is_selected_in_configured_order(self):
        raw = FakeRaw([name.lower() for name in reversed(PREFERRED_CHANNELS)])
        self.assertEqual(choose_channels(raw), [name.lower() for name in PREFERRED_CHANNELS])

    def test_matching_is_case_and_whitespace_insensitive(self):
        raw = FakeRaw([name.lower().replace("-", " - ") for name in PREFERRED_CHANNELS])
        selected = choose_channels(raw)
        self.assertEqual(len(selected), 17)
        self.assertEqual([name.replace(" ", "").upper() for name in selected], PREFERRED_CHANNELS)

    def test_missing_canonical_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed 17-channel primary montage"):
            choose_channels(FakeRaw(PREFERRED_CHANNELS[:-1]))

    def test_auxiliary_channels_cannot_replace_missing_canonical_channel(self):
        channels = PREFERRED_CHANNELS[:-1] + ["ECG", "EOG", "EKG1-CHIN"]
        with self.assertRaisesRegex(ValueError, "CZ-PZ"):
            choose_channels(FakeRaw(channels))

if __name__ == "__main__":
    unittest.main()
