"""Persistence: the database, the artifact tree, and the dataset memory.

Import surface is deliberately small — :class:`RunRepository` for structured
records, the :mod:`artifacts` helpers for files on disk, and
:class:`DatasetMemory` for cross-run recall.
"""

from __future__ import annotations

from .artifacts import (
    ArtifactAccessError,
    ArtifactInfo,
    artifact_path,
    list_artifacts,
    load_model,
    model_path,
    resolve_artifact,
    run_dir,
    save_model,
)
from .memory import DatasetMemory, build_fingerprint, score_similarity
from .repository import RunRepository

__all__ = [
    "ArtifactAccessError",
    "ArtifactInfo",
    "DatasetMemory",
    "RunRepository",
    "artifact_path",
    "build_fingerprint",
    "list_artifacts",
    "load_model",
    "model_path",
    "resolve_artifact",
    "run_dir",
    "save_model",
    "score_similarity",
]
