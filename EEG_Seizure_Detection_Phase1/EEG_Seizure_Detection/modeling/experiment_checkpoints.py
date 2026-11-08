"""Atomic, signature-checked checkpoints for completed model stages.

These checkpoints persist completed estimator fits and score arrays. They do
not serialize a live scikit-learn solver, so an interrupted estimator.fit()
must be rerun from the beginning.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np


class ModelStageCheckpointStore:
    """Persist completed model stages under one immutable experiment signature."""

    VERSION = 1

    def __init__(self, root: Path, context: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.context = context

    @staticmethod
    def _safe_stage(stage: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage)).strip("._")
        if not value:
            raise ValueError("Checkpoint stage must have a safe non-empty name")
        return value

    def _paths(self, stage: str, suffix: str):
        name = self._safe_stage(stage)
        return self.root / f"{name}.{suffix}", self.root / f"{name}.manifest.json"

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write(chr(10))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _manifest(self, stage: str, artifact: Path, kind: str) -> dict:
        return {
            "version": self.VERSION,
            "signature": self.signature,
            "stage": self._safe_stage(stage),
            "kind": kind,
            "artifact": artifact.name,
            "sha256": self._sha256(artifact),
            "complete": True,
        }

    def _valid_manifest(self, stage: str, artifact: Path, kind: str) -> bool:
        _, manifest_path = self._paths(stage, artifact.suffix.lstrip("."))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return (
                manifest.get("version") == self.VERSION
                and manifest.get("signature") == self.signature
                and manifest.get("stage") == self._safe_stage(stage)
                and manifest.get("kind") == kind
                and manifest.get("artifact") == artifact.name
                and manifest.get("complete") is True
                and artifact.is_file()
                and manifest.get("sha256") == self._sha256(artifact)
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def save_json(self, stage: str, payload: dict) -> None:
        artifact, manifest = self._paths(stage, "json")
        self._atomic_json(artifact, payload)
        self._atomic_json(manifest, self._manifest(stage, artifact, "json"))

    def load_json(self, stage: str) -> dict | None:
        artifact, _ = self._paths(stage, "json")
        if not self._valid_manifest(stage, artifact, "json"):
            return None
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save_estimator(self, stage: str, estimator) -> None:
        artifact, manifest = self._paths(stage, "joblib")
        fd, temporary_name = tempfile.mkstemp(
            prefix=artifact.name + ".", suffix=".tmp", dir=self.root
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            joblib.dump(estimator, temporary)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, artifact)
            self._atomic_json(manifest, self._manifest(stage, artifact, "estimator"))
        finally:
            temporary.unlink(missing_ok=True)

    def load_estimator(self, stage: str):
        artifact, _ = self._paths(stage, "joblib")
        if not self._valid_manifest(stage, artifact, "estimator"):
            return None
        try:
            return joblib.load(artifact)
        except (OSError, ValueError, EOFError, ImportError, AttributeError):
            return None

    def save_scores(self, stage: str, y_true: np.ndarray, scores: np.ndarray) -> None:
        artifact, manifest = self._paths(stage, "npz")
        y = np.asarray(y_true)
        values = np.asarray(scores, dtype=np.float64)
        if y.ndim != 1 or values.shape != y.shape or len(y) == 0:
            raise ValueError("Score checkpoint requires non-empty aligned 1-D arrays")
        if not np.isin(y, (0, 1)).all() or not np.isfinite(values).all():
            raise ValueError("Score checkpoint requires binary labels and finite scores")
        fd, temporary_name = tempfile.mkstemp(
            prefix=artifact.name + ".", suffix=".tmp", dir=self.root
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, y_true=y.astype(np.int8, copy=False), scores=values
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, artifact)
            self._atomic_json(manifest, self._manifest(stage, artifact, "scores"))
        finally:
            temporary.unlink(missing_ok=True)

    def load_scores(self, stage: str) -> tuple[np.ndarray, np.ndarray] | None:
        artifact, _ = self._paths(stage, "npz")
        if not self._valid_manifest(stage, artifact, "scores"):
            return None
        try:
            with np.load(artifact, allow_pickle=False) as saved:
                y = np.asarray(saved["y_true"], dtype=np.int8)
                scores = np.asarray(saved["scores"], dtype=np.float64)
            if y.ndim != 1 or scores.shape != y.shape or len(y) == 0:
                return None
            if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
                return None
            return y, scores
        except (OSError, ValueError, KeyError, EOFError):
            return None
