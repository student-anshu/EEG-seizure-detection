import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing.normalization import (
    FoldChannelStandardizer,
    scaler_checkpoint_signature,
)


class CheckpointedScalerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.batches = [
            rng.normal(loc=index * 2.0, scale=1.0 + index * 0.1, size=(3, 2, 5))
            .astype(np.float32)
            for index in range(5)
        ]

    def _factory(self, batches, starts):
        def make(start_index):
            starts.append(start_index)
            return iter(batches[start_index:])
        return make

    def test_interrupted_fit_resumes_from_last_complete_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            calls = []

            def interrupted_factory(start_index):
                calls.append(start_index)
                for index in range(start_index, 2):
                    yield self.batches[index]
                raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                FoldChannelStandardizer().fit_batches_checkpointed(
                    interrupted_factory, len(self.batches), checkpoint, "fold-a"
                )

            manifest = json.loads((checkpoint / "manifest.json").read_text())
            self.assertEqual(manifest["completed_recordings"], 2)
            self.assertFalse(manifest["complete"])

            resumed_starts = []
            resumed = FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, resumed_starts),
                len(self.batches), checkpoint, "fold-a",
            )
            uninterrupted = FoldChannelStandardizer().fit_batches(self.batches)

            self.assertEqual(resumed_starts, [2])
            np.testing.assert_allclose(resumed.mean_, uninterrupted.mean_, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(resumed.scale_, uninterrupted.scale_, rtol=1e-12, atol=1e-12)
            self.assertEqual(resumed.n_values_per_channel_, uninterrupted.n_values_per_channel_)

    def test_completed_checkpoint_is_reused_without_reading_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            expected = FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, []), len(self.batches), checkpoint, "fold-b"
            )

            def should_not_run(start_index):
                raise AssertionError("completed scaler checkpoint should be reused")

            actual = FoldChannelStandardizer().fit_batches_checkpointed(
                should_not_run, len(self.batches), checkpoint, "fold-b"
            )
            np.testing.assert_array_equal(actual.mean_, expected.mean_)
            np.testing.assert_array_equal(actual.scale_, expected.scale_)

    def test_incompatible_signature_restarts_from_first_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, []), len(self.batches), checkpoint, "old-fold"
            )
            starts = []
            FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, starts),
                len(self.batches), checkpoint, "new-fold",
            )
            self.assertEqual(starts, [0])

    def test_corrupt_state_checkpoint_is_rejected_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, []), len(self.batches), checkpoint, "fold-c"
            )
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            (checkpoint / manifest["state_file"]).write_bytes(b"not a numpy archive")
            starts = []
            rebuilt = FoldChannelStandardizer().fit_batches_checkpointed(
                self._factory(self.batches, starts),
                len(self.batches), checkpoint, "fold-c",
            )
            expected = FoldChannelStandardizer().fit_batches(self.batches)
            self.assertEqual(starts, [0])
            np.testing.assert_allclose(rebuilt.mean_, expected.mean_)
            np.testing.assert_allclose(rebuilt.scale_, expected.scale_)

    def test_checkpoint_signature_changes_with_fold_context(self):
        from data_management.patient_mapping import Recording

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez_compressed(
                directory / "chb01_chb01_01.npz",
                y=np.array([0, 1], dtype=np.int8),
            )
            recording = Recording(
                "chb01", "patient_01", "chb01_01.edf", "chb01/chb01_01.edf",
                1, ("10-20",), "test", "SUCCESS",
            )
            first = scaler_checkpoint_signature(
                [recording], directory, {"fold": 1, "role": "inner_fit"}
            )
            second = scaler_checkpoint_signature(
                [recording], directory, {"fold": 2, "role": "inner_fit"}
            )
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
