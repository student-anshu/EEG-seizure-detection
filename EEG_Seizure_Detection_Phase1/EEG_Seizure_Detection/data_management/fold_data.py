"""Leakage-safe, recording-at-a-time data preparation for one LOPO fold."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from config import PREFERRED_CHANNELS
from data_management.dataset_split import LopoFold
from data_management.patient_mapping import Recording
from data_management.validation_split import build_patient_validation_split
from feature_selection.feature_extraction import FEATURE_NAMES, extract_features
from preprocessing.normalization import (
    FoldChannelStandardizer, scaler_checkpoint_signature,
)

_NORMALIZATION_POLICY = "none_fold_train_only_scaling_required"
_REQUIRED_ARRAYS = {
    "X", "y", "starts", "channels", "sfreq", "subject",
    "source_file", "normalization",
}


@dataclass(frozen=True, slots=True)
class RecordingWindows:
    recording: Recording
    X: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    sfreq: float


@dataclass(frozen=True, slots=True)
class RecordingFeatures:
    """Features and aligned labels/metadata for one recording."""

    recording: Recording
    features: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    sfreq: float


@dataclass(frozen=True, slots=True)
class PreparedFold:
    fold_id: int
    test_patient_id: str
    fit_recordings: tuple[Recording, ...]
    validation_recordings: tuple[Recording, ...]
    test_recordings: tuple[Recording, ...]
    standardizer: FoldChannelStandardizer
    processed_dir: Path

    def iter_split(self, split: str) -> Iterator[RecordingWindows]:
        """Stream fold-standardized EEG windows, one recording at a time."""
        groups = {
            "fit": self.fit_recordings,
            "validation": self.validation_recordings,
            "test": self.test_recordings,
        }
        if split not in groups:
            raise ValueError("split must be one of: fit, validation, test")
        for recording in groups[split]:
            item = load_recording_windows(self.processed_dir, recording)
            yield RecordingWindows(
                recording, self.standardizer.transform(item.X),
                item.y, item.starts, item.sfreq,
            )

    def iter_features(self, split: str) -> Iterator[RecordingFeatures]:
        """Stream deterministic feature matrices, one recording at a time.

        Channel standardization is applied by iter_split before extraction.
        The extractor itself does not fit on labels or data, so validation
        and outer-test recordings cannot influence fitted preprocessing.
        """
        expected_features = len(PREFERRED_CHANNELS) * len(FEATURE_NAMES)
        for item in self.iter_split(split):
            matrix = extract_features(item.X, item.sfreq)
            if matrix.shape != (len(item.y), expected_features):
                raise RuntimeError(
                    "Feature extractor returned an unexpected shape: "
                    f"{matrix.shape}; expected ({len(item.y)}, {expected_features})."
                )
            if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                raise RuntimeError("Feature matrix must be finite float32.")
            yield RecordingFeatures(
                item.recording, matrix, item.y, item.starts, item.sfreq
            )


def processed_path_for_recording(processed_dir: str | Path, recording: Recording) -> Path:
    return Path(processed_dir) / f"{recording.subject}_{Path(recording.edf_file).stem}.npz"


def load_recording_windows(processed_dir: str | Path, recording: Recording) -> RecordingWindows:
    path = processed_path_for_recording(processed_dir, recording)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed windows missing for {recording.relative_path}: {path}"
        )
    with np.load(path, allow_pickle=False) as saved:
        missing = _REQUIRED_ARRAYS - set(saved.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {', '.join(sorted(missing))}")
        X = np.asarray(saved["X"])
        y = np.asarray(saved["y"])
        starts = np.asarray(saved["starts"], dtype=np.float64)
        channels = [str(value) for value in saved["channels"].tolist()]
        sfreq = float(saved["sfreq"])
        subject = str(saved["subject"])
        source_file = str(saved["source_file"])
        normalization = str(saved["normalization"])

    if subject != recording.subject or source_file != recording.edf_file:
        raise ValueError(f"Saved recording identity does not match manifest: {path.name}")
    if normalization != _NORMALIZATION_POLICY:
        raise ValueError(
            f"{path.name} uses incompatible normalization policy; regenerate windows."
        )
    if channels != list(PREFERRED_CHANNELS):
        raise ValueError(f"{path.name} does not use the configured fixed channel order.")
    if X.ndim != 3 or X.shape[1] != len(PREFERRED_CHANNELS):
        raise ValueError(f"{path.name} has invalid X shape {X.shape}.")
    if y.ndim != 1 or starts.ndim != 1 or len(X) != len(y) or len(y) != len(starts):
        raise ValueError(f"{path.name} has inconsistent X/y/starts lengths.")
    if len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path.name} has empty or non-binary labels.")
    if not np.isfinite(X).all() or not np.isfinite(starts).all() or not np.isfinite(sfreq):
        raise ValueError(f"{path.name} contains non-finite values.")
    if sfreq <= 0:
        raise ValueError(f"{path.name} has invalid sampling frequency.")
    if bool(np.any(y == 1)) != (recording.seizure_count > 0):
        raise ValueError(
            f"{path.name} labels disagree with audited seizure annotation presence."
        )
    return RecordingWindows(recording, X, y.astype(np.int64), starts, sfreq)


def prepare_lopo_fold(
    fold: LopoFold,
    processed_dir: str | Path,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    scaler_checkpoint_dir: str | Path | None = None,
) -> PreparedFold:
    """Split outer-training patients, then fit scaler on inner-fit recordings only."""
    train_patients = {item.patient_id for item in fold.train_recordings}
    test_patients = {item.patient_id for item in fold.test_recordings}
    if test_patients != {fold.test_patient_id}:
        raise ValueError("Outer fold test recordings do not match its test patient.")
    if train_patients & test_patients:
        raise ValueError("Patient leakage detected in supplied outer fold.")

    inner = build_patient_validation_split(
        fold.train_recordings, validation_fraction, random_state
    )
    fit_patients = set(inner.train_patient_ids)
    validation_patients = set(inner.validation_patient_ids)
    if fit_patients & validation_patients or fit_patients & test_patients:
        raise RuntimeError("Patient leakage detected before normalization fitting.")
    if validation_patients & test_patients:
        raise RuntimeError("Outer test patient entered inner validation.")

    directory = Path(processed_dir)
    scaler = FoldChannelStandardizer()

    def training_batches(start_index=0):
        for recording in inner.train_recordings[start_index:]:
            yield load_recording_windows(directory, recording).X

    if scaler_checkpoint_dir is None:
        scaler.fit_batches(training_batches())
    else:
        checkpoint_context = {
            "stage": "inner_fit_channel_scaler",
            "fold_id": fold.fold_id,
            "outer_test_patient": fold.test_patient_id,
            "inner_fit_patients": sorted(fit_patients),
            "inner_validation_patients": sorted(validation_patients),
            "random_state": int(random_state),
            "validation_fraction": float(validation_fraction),
        }
        signature = scaler_checkpoint_signature(
            inner.train_recordings, directory, checkpoint_context, scaler.epsilon
        )
        scaler.fit_batches_checkpointed(
            training_batches,
            total_batches=len(inner.train_recordings),
            checkpoint_dir=scaler_checkpoint_dir,
            signature=signature,
            progress_label="inner-fit channel-scaler fit",
        )
    return PreparedFold(
        fold.fold_id, fold.test_patient_id,
        inner.train_recordings, inner.validation_recordings,
        fold.test_recordings, scaler, directory,
    )
