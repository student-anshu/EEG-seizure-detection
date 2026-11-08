"""Fold-specific EEG channel normalization with resumable statistics."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np


_SCALER_CHECKPOINT_VERSION = 1


def scaler_checkpoint_signature(recordings, processed_dir, context, epsilon=1e-6):
    """Fingerprint the exact recording set, processed inputs and scaler context."""
    directory = Path(processed_dir)
    sources = []
    identities = []
    for recording in recordings:
        path = directory / (
            f"{recording.subject}_{Path(recording.edf_file).stem}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed windows missing for {recording.relative_path}: {path}"
            )
        stat = path.stat()
        sources.append({
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
        identities.append({
            "relative_path": recording.relative_path,
            "patient_id": recording.patient_id,
            "subject": recording.subject,
            "edf_file": recording.edf_file,
            "seizure_count": recording.seizure_count,
        })

    payload = {
        "checkpoint_version": _SCALER_CHECKPOINT_VERSION,
        "algorithm": "parallel_welford_population_variance_v1",
        "epsilon": float(epsilon),
        "context": context,
        "recordings": identities,
        "processed_sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FoldChannelStandardizer:
    """Standardize channels using statistics fitted on training-fold windows.

    Input shape: (windows, channels, samples). The caller must enforce that
    only the permitted training recordings are passed to fit methods.
    """

    def __init__(self, epsilon=1e-6):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number.")
        self.epsilon = float(epsilon)
        self.mean_ = None
        self.scale_ = None
        self.n_channels_ = None
        self.n_values_per_channel_ = None
        self._count = 0
        self._mean = None
        self._m2 = None
        self._n_channels = None

    @staticmethod
    def _validate_windows(windows):
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("EEG windows must have shape (windows, channels, samples).")
        if any(size == 0 for size in values.shape):
            raise ValueError("EEG windows must be non-empty on every axis.")
        if not np.all(np.isfinite(values)):
            raise ValueError("EEG windows contain NaN or infinite values.")
        return values

    def partial_fit(self, training_windows):
        """Merge one recording/batch into the running population statistics."""
        if self.mean_ is not None:
            raise RuntimeError("Cannot partial_fit after the scaler has been finalized.")
        values = self._validate_windows(training_windows)
        if self._n_channels is None:
            self._n_channels = values.shape[1]
        elif values.shape[1] != self._n_channels:
            raise ValueError("Training batches have inconsistent channel counts.")

        flattened = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
        batch_count = flattened.shape[1]
        batch_mean = np.mean(flattened, axis=1)
        centered = flattened - batch_mean[:, None]
        batch_m2 = np.sum(centered * centered, axis=1)

        if self._count == 0:
            self._count = int(batch_count)
            self._mean = batch_mean
            self._m2 = batch_m2
        else:
            total = self._count + batch_count
            delta = batch_mean - self._mean
            self._mean = self._mean + delta * (batch_count / total)
            self._m2 = (
                self._m2 + batch_m2
                + delta * delta * (self._count * batch_count / total)
            )
            self._count = int(total)
        return self

    def state_dict(self):
        """Return a detached, serializable snapshot of the running accumulator."""
        return {
            "count": int(self._count),
            "n_channels": int(self._n_channels) if self._n_channels is not None else 0,
            "mean": None if self._mean is None else self._mean.copy(),
            "m2": None if self._m2 is None else self._m2.copy(),
        }

    def load_state_dict(self, state):
        """Restore and validate an unfinished accumulator checkpoint."""
        if self.mean_ is not None:
            raise RuntimeError("Cannot restore state after the scaler has been finalized.")
        try:
            count = int(state["count"])
            n_channels = int(state["n_channels"])
            mean = np.asarray(state["mean"], dtype=np.float64)
            m2 = np.asarray(state["m2"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid scaler accumulator checkpoint.") from exc
        if count <= 0 or n_channels <= 0:
            raise ValueError("Scaler checkpoint must contain completed samples and channels.")
        if mean.shape != (n_channels,) or m2.shape != (n_channels,):
            raise ValueError("Scaler checkpoint arrays have inconsistent dimensions.")
        if not np.isfinite(mean).all() or not np.isfinite(m2).all() or np.any(m2 < 0):
            raise ValueError("Scaler checkpoint contains invalid statistics.")
        self._count = count
        self._n_channels = n_channels
        self._mean = mean.copy()
        self._m2 = m2.copy()
        return self

    def finalize(self):
        """Convert running accumulators into immutable transform statistics."""
        if self._count == 0 or self._mean is None or self._m2 is None:
            raise ValueError("Cannot fit standardizer on no training windows.")
        variance = np.maximum(self._m2 / self._count, 0.0)
        self.n_channels_ = int(self._n_channels)
        self.n_values_per_channel_ = int(self._count)
        self.mean_ = self._mean.copy()
        self.scale_ = np.maximum(np.sqrt(variance), self.epsilon)
        return self

    def fit(self, training_windows):
        return self.fit_batches([training_windows])

    def fit_batches(self, training_batches):
        """Incrementally fit exact population statistics over recording batches."""
        for batch in training_batches:
            self.partial_fit(batch)
        return self.finalize()

    def fit_batches_checkpointed(
        self, batch_factory, total_batches, checkpoint_dir, signature,
        progress_label="channel-scaler fit", progress_every=25,
    ):
        """Fit from a per-recording atomic checkpoint, resuming after the last commit.

        batch_factory(start_index) must return an iterator over batches beginning
        at the zero-based recording index supplied. A checkpoint advances only
        after a complete batch has been merged and its state file is durable.
        """
        if not callable(batch_factory):
            raise TypeError("batch_factory must be callable.")
        if int(total_batches) <= 0:
            raise ValueError("total_batches must be positive.")
        if not signature:
            raise ValueError("A non-empty scaler checkpoint signature is required.")
        if int(progress_every) <= 0:
            raise ValueError("progress_every must be positive.")
        if self._count or self.mean_ is not None:
            raise RuntimeError("Checkpointed fitting requires a fresh, unfitted scaler.")

        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        completed = 0

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema_version") != _SCALER_CHECKPOINT_VERSION:
                    raise ValueError("unsupported checkpoint schema")
                if manifest.get("signature") != signature:
                    raise ValueError("checkpoint signature mismatch")
                completed = int(manifest["completed_recordings"])
                state_name = manifest["state_file"]
                if not isinstance(state_name, str) or Path(state_name).name != state_name:
                    raise ValueError("invalid checkpoint state filename")
                if not 0 < completed <= int(total_batches):
                    raise ValueError("invalid completed-recording cursor")
                if bool(manifest.get("complete")) != (completed == int(total_batches)):
                    raise ValueError("checkpoint completion flag mismatch")
                state_path = root / state_name
                with np.load(state_path, allow_pickle=False) as saved:
                    state = {
                        "count": int(saved["count"]),
                        "n_channels": int(saved["n_channels"]),
                        "mean": saved["mean"],
                        "m2": saved["m2"],
                    }
                self.load_state_dict(state)
                if int(manifest["sample_count"]) != self._count:
                    raise ValueError("checkpoint sample count mismatch")
                print(
                    f"    resuming channel-scaler checkpoint: "
                    f"{completed}/{total_batches} recordings",
                    flush=True,
                )
            except (OSError, EOFError, ValueError, TypeError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                print(
                    f"    rejecting invalid scaler checkpoint ({exc}); restarting scaler fit",
                    flush=True,
                )
                self._reset_accumulator()
                completed = 0

        if completed == int(total_batches):
            return self.finalize()

        processed_count = completed
        for index, batch in enumerate(batch_factory(completed), start=completed + 1):
            if index > int(total_batches):
                raise RuntimeError("Scaler batch factory yielded too many recordings.")
            self.partial_fit(batch)
            state_name = f"state_{index:06d}.npz"
            state_path = root / state_name
            temporary_state = root / (state_name + ".tmp")
            with temporary_state.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    count=np.asarray(self._count, dtype=np.int64),
                    n_channels=np.asarray(self._n_channels, dtype=np.int64),
                    mean=self._mean,
                    m2=self._m2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_state, state_path)

            manifest = {
                "schema_version": _SCALER_CHECKPOINT_VERSION,
                "signature": signature,
                "completed_recordings": index,
                "sample_count": self._count,
                "state_file": state_name,
                "complete": index == int(total_batches),
            }
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            with temporary_manifest.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest, manifest_path)
            processed_count = index

            if index % int(progress_every) == 0 or index == int(total_batches):
                print(
                    f"    {progress_label}: {index}/{total_batches} recordings",
                    flush=True,
                )

        if processed_count != int(total_batches):
            raise RuntimeError(
                f"Scaler batch factory ended at {processed_count}/{total_batches} recordings."
            )
        return self.finalize()

    def _reset_accumulator(self):
        self.mean_ = None
        self.scale_ = None
        self.n_channels_ = None
        self.n_values_per_channel_ = None
        self._count = 0
        self._mean = None
        self._m2 = None
        self._n_channels = None

    def transform(self, windows):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted before transform().")
        values = self._validate_windows(windows)
        if values.shape[1] != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels, received {values.shape[1]}."
            )
        return ((values - self.mean_[None, :, None]) /
                self.scale_[None, :, None]).astype(np.float32)

    def fit_transform(self, training_windows):
        return self.fit(training_windows).transform(training_windows)
