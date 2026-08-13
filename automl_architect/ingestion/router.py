"""The single entry point into ingestion.

:func:`load_source` dispatches on :class:`SourceKind` through the connector
registry — no ``if`` chain — then runs the shared post-load pipeline so that all
thirteen source kinds return an identically-shaped
:class:`~automl_architect.core.schemas.IngestionResult`.

The division of labour is deliberate: a connector knows how to get bytes and
nothing else, while header normalisation, validation, schema inference, and
credential redaction happen exactly once, here and in :mod:`.base`. Adding a
fourteenth source kind means writing one class with one method.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..core.errors import (
    ConfigurationError,
    IngestionError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from ..core.schemas import DataSource, IngestionResult, Param, SourceKind
from .base import (
    build_result,
    ensure_frame,
    get_connector,
    normalise_columns,
    redact_uri,
    registered_kinds,
    scrub_secrets,
)

logger = logging.getLogger(__name__)

_SCHEME_KINDS: dict[str, SourceKind] = {
    "s3": SourceKind.S3,
    "s3a": SourceKind.S3,
    "s3n": SourceKind.S3,
    "gs": SourceKind.GCS,
    "gcs": SourceKind.GCS,
    "az": SourceKind.AZURE_BLOB,
    "abfs": SourceKind.AZURE_BLOB,
    "abfss": SourceKind.AZURE_BLOB,
    "wasb": SourceKind.AZURE_BLOB,
    "wasbs": SourceKind.AZURE_BLOB,
    "adl": SourceKind.AZURE_BLOB,
    "postgresql": SourceKind.POSTGRES,
    "postgres": SourceKind.POSTGRES,
    "mysql": SourceKind.MYSQL,
    "mariadb": SourceKind.MYSQL,
    "snowflake": SourceKind.SNOWFLAKE,
    "databricks": SourceKind.DATABRICKS,
    "duckdb": SourceKind.DUCKDB,
    "kaggle": SourceKind.KAGGLE,
}

_EXTENSION_KINDS: dict[str, SourceKind] = {
    ".csv": SourceKind.CSV,
    ".tsv": SourceKind.CSV,
    ".txt": SourceKind.CSV,
    ".psv": SourceKind.CSV,
    ".dat": SourceKind.CSV,
    ".json": SourceKind.JSON,
    ".jsonl": SourceKind.JSON,
    ".ndjson": SourceKind.JSON,
    ".geojson": SourceKind.JSON,
    ".parquet": SourceKind.PARQUET,
    ".pq": SourceKind.PARQUET,
    ".parq": SourceKind.PARQUET,
    ".xlsx": SourceKind.EXCEL,
    ".xlsm": SourceKind.EXCEL,
    ".xls": SourceKind.EXCEL,
    ".xlsb": SourceKind.EXCEL,
    ".ods": SourceKind.EXCEL,
    ".duckdb": SourceKind.DUCKDB,
    ".ddb": SourceKind.DUCKDB,
}

_COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zip", ".zst", ".zstd", ".lzma"}


def load_source(
    source: DataSource, *, max_rows: int | None = None
) -> tuple[Any, IngestionResult]:
    """Load a data source into a dataframe and describe what was loaded.

    Args:
        source: Where the data comes from. ``source.kind`` selects the connector.
        max_rows: Optional row cap. When it applies,
            ``IngestionResult.truncated`` is set to ``True``.

    Returns:
        ``(frame, result)``. The frame has whitespace-stripped, de-duplicated
        column names; the result carries row/column counts, per-column schema
        fields, validation findings, memory footprint, and load time. The copy of
        the source embedded in the result is redacted.

    Raises:
        UnsupportedSourceError: No connector is registered for ``source.kind``.
        MissingDependencyError: The connector needs an optional package.
        ConfigurationError: The source is under-specified or a credential is
            missing.
        IngestionError: The source exists but could not be read. Structural
            problems with data that *did* load are reported in
            ``result.validation_errors`` instead, so the run can degrade rather
            than crash.
    """
    if not isinstance(source, DataSource):  # pragma: no cover - guard for callers
        raise ConfigurationError(
            f"load_source expects a DataSource, got {type(source).__name__}."
        )

    connector_cls = get_connector(source.kind)
    connector = connector_cls(source)
    started = time.perf_counter()

    try:
        outcome = connector.load(max_rows=max_rows)
    except (
        UnsupportedSourceError,
        MissingDependencyError,
        ConfigurationError,
        IngestionError,
    ):
        raise
    except Exception as exc:
        message = scrub_secrets(
            f"{source.kind.value} ingestion failed for "
            f"{redact_uri(source.uri) or '<no uri>'}: {type(exc).__name__}: {exc}",
            getattr(connector, "resolved_secrets", []),
        )
        raise IngestionError(message) from exc

    elapsed = time.perf_counter() - started
    frame = ensure_frame(outcome.frame)
    frame, rename_notes = normalise_columns(frame)

    result = build_result(
        source,
        frame,
        load_seconds=elapsed,
        truncated=outcome.truncated,
        notes=list(outcome.notes) + rename_notes,
        delimited=outcome.text_delimited,
    )

    logger.info(
        "ingested %s (%s): %d rows x %d cols in %.2fs%s",
        source.kind.value,
        outcome.detail or "n/a",
        result.n_rows,
        result.n_columns,
        result.load_seconds,
        " [truncated]" if result.truncated else "",
    )
    for message in result.validation_errors:
        logger.error("ingestion validation: %s", message)
    for message in result.validation_warnings:
        logger.warning("ingestion validation: %s", message)

    return frame, result


def available_kinds() -> list[SourceKind]:
    """Source kinds that have a registered connector in this installation."""
    return registered_kinds()


def infer_source(target: Any, **options: Any) -> DataSource:
    """Build a :class:`DataSource` from a loose reference.

    Convenience for callers that accept "a path, a URL, or a dataframe" — the
    library entry point and the CLI both do. Kind detection is by URI scheme
    first, then file extension.

    Args:
        target: A :class:`DataSource` (returned unchanged), a dataframe, a local
            path, or a URI.
        **options: Extra connector options, appended as ``Param`` entries.

    Returns:
        A source ready for :func:`load_source`.

    Raises:
        ConfigurationError: ``target`` is not a recognisable reference.
    """
    if isinstance(target, DataSource):
        if options:
            extra = [Param(key=str(k), value=_render(v)) for k, v in options.items()]
            return target.model_copy(update={"options": list(target.options) + extra})
        return target

    if hasattr(target, "columns") and hasattr(target, "shape"):
        from .dataframe import make_dataframe_source

        source = make_dataframe_source(target, name=str(options.pop("name", "")))
        return infer_source(source, **options) if options else source

    if isinstance(target, Path):
        target = str(target)
    if not isinstance(target, str) or not target.strip():
        raise ConfigurationError(
            f"Cannot infer a data source from {type(target).__name__}; pass a path, "
            "a URI, a DataFrame, or a DataSource."
        )

    text = target.strip()
    kind = _infer_kind(text)
    params = [Param(key=str(k), value=_render(v)) for k, v in options.items()]
    return DataSource(kind=kind, uri=text, options=params)


def _infer_kind(text: str) -> SourceKind:
    """Map a path or URI to the most likely :class:`SourceKind`."""
    scheme = text.split("://", 1)[0].lower() if "://" in text else ""
    if scheme:
        base = scheme.split("+", 1)[0]
        if base in _SCHEME_KINDS:
            return _SCHEME_KINDS[base]
        if base in {"http", "https"}:
            by_extension = _kind_from_extension(text.split("?", 1)[0])
            return by_extension or SourceKind.REST_API
        if base == "file":
            return _kind_from_extension(text[7:]) or SourceKind.CSV
        return SourceKind.SQL

    by_extension = _kind_from_extension(text)
    if by_extension:
        return by_extension
    if text.lower().startswith("kaggle:"):
        return SourceKind.KAGGLE
    raise ConfigurationError(
        f"Could not infer a source kind from {text!r}. Set DataSource.kind "
        f"explicitly; registered kinds: "
        f"{', '.join(k.value for k in registered_kinds())}."
    )


def _kind_from_extension(text: str) -> SourceKind | None:
    """Look up a source kind by file extension, ignoring compression suffixes."""
    suffixes = [s.lower() for s in Path(text).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes.pop()
    if suffixes and suffixes[-1] in _EXTENSION_KINDS:
        return _EXTENSION_KINDS[suffixes[-1]]
    return None


def _render(value: Any) -> str:
    """Render an option value for a :class:`Param`."""
    import json

    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["available_kinds", "infer_source", "load_source"]
