"""Artifact layout on disk, plus save/load of fitted models.

Every run owns one directory (``<workspace>/runs/<run_id>``) with a fixed
sub-layout, so the API can serve artifacts by relative path and the report writer
can reference them without coordinating naming with anyone.

The security-relevant function here is :func:`resolve_artifact`. Anything that
turns a client-supplied string into a filesystem path must go through it: it
resolves symlinks and rejects any result that is not inside the run directory,
which is the difference between serving a chart and serving ``/etc/passwd``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..core.errors import AutoMLArchitectError

logger = logging.getLogger(__name__)

MODELS_SUBDIR = "models"
CHARTS_SUBDIR = "charts"
REPORTS_SUBDIR = "reports"
DATA_SUBDIR = "data"
UPLOADS_DIRNAME = "uploads"

MODEL_SUFFIX = ".joblib"

#: Extension -> coarse artifact kind, used by the artifact listing endpoint.
_KIND_BY_SUFFIX = {
    ".joblib": "model",
    ".pkl": "model",
    ".html": "chart",
    ".png": "image",
    ".svg": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".json": "data",
    ".csv": "data",
    ".parquet": "data",
    ".md": "report",
    ".pdf": "report",
    ".pptx": "report",
    ".txt": "text",
    ".log": "text",
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactAccessError(AutoMLArchitectError):
    """A requested artifact path escaped its run directory, or does not exist."""


@dataclass(slots=True)
class ArtifactInfo:
    """One file inside a run directory."""

    relative_path: str
    kind: str
    size_bytes: int
    modified_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, for the API listing."""
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def run_dir(run_id: str, *, settings: Settings | None = None) -> Path:
    """The directory owning every artifact for ``run_id``, created if absent."""
    resolved = settings or get_settings()
    return resolved.run_dir(run_id)


def artifact_path(run_id: str, *parts: str, settings: Settings | None = None) -> Path:
    """A path inside the run directory, with parent directories created.

    Args:
        run_id: Owning run.
        *parts: Path segments below the run directory.
        settings: Settings override.

    Returns:
        The absolute path. The file itself is not created.
    """
    path = run_dir(run_id, settings=settings).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model_path(run_id: str, name: str = "best_model", *, settings: Settings | None = None) -> Path:
    """Canonical location for a fitted estimator."""
    stem = safe_filename(name)
    if not stem.endswith(MODEL_SUFFIX):
        stem = f"{stem}{MODEL_SUFFIX}"
    return artifact_path(run_id, MODELS_SUBDIR, stem, settings=settings)


def chart_path(run_id: str, filename: str, *, settings: Settings | None = None) -> Path:
    """Canonical location for a rendered chart."""
    return artifact_path(run_id, CHARTS_SUBDIR, safe_filename(filename), settings=settings)


def report_path(run_id: str, filename: str, *, settings: Settings | None = None) -> Path:
    """Canonical location for a rendered report."""
    return artifact_path(run_id, REPORTS_SUBDIR, safe_filename(filename), settings=settings)


def uploads_dir(*, settings: Settings | None = None) -> Path:
    """Where API uploads land. Outside the run directories: one upload, many runs."""
    resolved = settings or get_settings()
    path = resolved.workspace / UPLOADS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Reduce arbitrary text to a single safe path segment.

    Strips directory components and anything outside ``[A-Za-z0-9._-]``, which
    neutralises ``../`` and absolute paths before they reach the filesystem.
    """
    base = os.path.basename(str(name).replace("\\", "/")).strip()
    cleaned = _SAFE_NAME.sub("_", base).lstrip(".")
    return cleaned or "artifact"


def store_upload(
    data: bytes,
    filename: str,
    *,
    settings: Settings | None = None,
) -> Path:
    """Write uploaded bytes into the uploads directory under a unique name.

    Args:
        data: Raw file content.
        filename: Client-supplied name; only its sanitised basename is used.
        settings: Settings override.

    Returns:
        The path written.
    """
    target = uploads_dir(settings=settings) / f"{uuid.uuid4().hex[:12]}_{safe_filename(filename)}"
    target.write_bytes(data)
    # Absolute, because the returned path becomes a DataSource URI that may be
    # resolved by a different process with a different working directory.
    return target.resolve()


# ---------------------------------------------------------------------------
# Traversal-safe resolution and listing
# ---------------------------------------------------------------------------


def resolve_artifact(
    run_id: str,
    relative_path: str,
    *,
    settings: Settings | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a client-supplied relative path inside a run directory.

    Args:
        run_id: Owning run.
        relative_path: Untrusted path, relative to the run directory.
        settings: Settings override.
        must_exist: Require an existing regular file.

    Returns:
        The resolved absolute path.

    Raises:
        ArtifactAccessError: If the path is absolute, escapes the run directory
            after symlink resolution, or (when ``must_exist``) is not a file.
    """
    root = run_dir(run_id, settings=settings).resolve()
    candidate = Path(str(relative_path).replace("\\", "/"))
    if candidate.is_absolute() or candidate.drive:
        raise ArtifactAccessError("artifact paths must be relative to the run directory")

    resolved = (root / candidate).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ArtifactAccessError(f"path {relative_path!r} escapes the run directory")
    if must_exist and not resolved.is_file():
        raise ArtifactAccessError(f"artifact {relative_path!r} not found")
    return resolved


def list_artifacts(
    run_id: str,
    *,
    settings: Settings | None = None,
    max_entries: int = 2000,
) -> list[ArtifactInfo]:
    """Every file under the run directory, as relative paths.

    Symlinks are skipped rather than followed: a listing that advertises a path
    outside the run directory would be a path-traversal vector by another route.
    """
    root = run_dir(run_id, settings=settings).resolve()
    if not root.is_dir():
        return []
    out: list[ArtifactInfo] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_entries:
            logger.warning("artifact listing for %s truncated at %d entries", run_id, max_entries)
            break
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        out.append(
            ArtifactInfo(
                relative_path=relative.as_posix(),
                kind=_KIND_BY_SUFFIX.get(path.suffix.lower(), "file"),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return out


def relative_to_run(run_id: str, path: str | Path, *, settings: Settings | None = None) -> str | None:
    """Express an absolute artifact path relative to its run directory.

    Returns ``None`` when the path is outside the run directory, which callers
    should treat as "not servable over HTTP".
    """
    root = run_dir(run_id, settings=settings).resolve()
    try:
        candidate = Path(path).resolve()
    except OSError:
        return None
    if candidate == root or not candidate.is_relative_to(root):
        return None
    return candidate.relative_to(root).as_posix()


def human_bytes(count: int | float) -> str:
    """Render a byte count for humans, e.g. ``2.4 MB``."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    run_id: str,
    name: str = "best_model",
    *,
    metadata: dict[str, Any] | None = None,
    settings: Settings | None = None,
    compress: int = 3,
) -> Path:
    """Persist a fitted estimator (or pipeline) with joblib.

    A sidecar ``<name>.meta.json`` records the library versions the object was
    pickled under. A joblib file that silently fails to load two sklearn
    releases later is a support nightmare; the sidecar makes the cause obvious.

    Args:
        model: Any picklable fitted object.
        run_id: Owning run.
        name: Base filename, without extension.
        metadata: Extra facts to record beside the model (family, feature names).
        settings: Settings override.
        compress: joblib compression level, 0-9.

    Returns:
        The path the model was written to.
    """
    import joblib  # heavy-ish; import at call time to keep CLI startup fast

    target = model_path(run_id, name, settings=settings)
    joblib.dump(model, target, compress=compress)

    sidecar = {
        "artifact": target.name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_repr": _safe_repr(model),
        "versions": _library_versions(),
    }
    if metadata:
        sidecar["metadata"] = metadata
    try:
        target.with_suffix(".meta.json").write_text(
            json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
        )
    except OSError:  # the model itself is what matters
        logger.warning("could not write model sidecar for %s", target.name)
    return target


def load_model(path: str | Path) -> Any:
    """Load a joblib-persisted model.

    Args:
        path: Path produced by :func:`save_model`.

    Returns:
        The unpickled object.

    Raises:
        ArtifactAccessError: If the file is missing or cannot be unpickled.
    """
    import joblib

    target = Path(path)
    if not target.is_file():
        raise ArtifactAccessError(f"model artifact not found: {target}")
    try:
        return joblib.load(target)
    except Exception as exc:  # version skew, truncated file, missing class
        sidecar = target.with_suffix(".meta.json")
        hint = ""
        if sidecar.is_file():
            try:
                versions = json.loads(sidecar.read_text(encoding="utf-8")).get("versions", {})
                hint = f" (saved under {versions})"
            except (OSError, ValueError):
                hint = ""
        raise ArtifactAccessError(f"could not load model {target.name}{hint}: {exc}") from exc


def load_run_model(
    run_id: str, name: str = "best_model", *, settings: Settings | None = None
) -> Any:
    """Load a model by run id and name, using the canonical layout."""
    return load_model(model_path(run_id, name, settings=settings))


def save_json(payload: Any, path: str | Path) -> Path:
    """Write JSON with parents created and non-serialisable values stringified."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def load_json(path: str | Path) -> Any:
    """Read a JSON file, returning ``None`` if it is missing or malformed."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not read JSON artifact %s", target)
        return None


def _safe_repr(obj: Any, limit: int = 500) -> str:
    try:
        text = repr(obj)
    except Exception:
        return type(obj).__name__
    return text[:limit]


def _library_versions() -> dict[str, str]:
    """Versions of the libraries that determine pickle compatibility."""
    out: dict[str, str] = {}
    import sys

    out["python"] = sys.version.split()[0]
    for module_name, key in (
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("joblib", "joblib"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version:
            out[key] = str(version)
    return out


__all__ = [
    "ArtifactAccessError",
    "ArtifactInfo",
    "CHARTS_SUBDIR",
    "DATA_SUBDIR",
    "MODELS_SUBDIR",
    "REPORTS_SUBDIR",
    "artifact_path",
    "chart_path",
    "human_bytes",
    "list_artifacts",
    "load_json",
    "load_model",
    "load_run_model",
    "model_path",
    "relative_to_run",
    "report_path",
    "resolve_artifact",
    "run_dir",
    "safe_filename",
    "save_json",
    "save_model",
    "store_upload",
    "uploads_dir",
]
