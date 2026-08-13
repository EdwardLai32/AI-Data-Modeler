"""HTTP request and response models.

These are the wire contract, distinct from the domain contracts in
:mod:`automl_architect.core.schemas`. Domain models are returned unchanged where
one exists (``RunSummary``, ``RunEvent``, ``QuestionAnswer``) — wrapping them in
near-identical API types would be duplication that drifts. What lives here is
only what HTTP itself needs: request bodies, cursors, list envelopes, and health.

This module deliberately imports no FastAPI: the CLI reads
:func:`data_source_from_uri` from here, and paying for a web-framework import on
``amla --help`` would be silly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from ..core.schemas import (
    ApprovalRequest,
    DataSource,
    Param,
    RunConfig,
    RunEvent,
    RunStatus,
    SourceKind,
    TaskType,
)

#: Extension -> source kind. Everything the ingestion router can read from a path.
_SUFFIX_KINDS: dict[str, SourceKind] = {
    ".csv": SourceKind.CSV,
    ".tsv": SourceKind.CSV,
    ".txt": SourceKind.CSV,
    ".csv.gz": SourceKind.CSV,
    ".xlsx": SourceKind.EXCEL,
    ".xlsm": SourceKind.EXCEL,
    ".xls": SourceKind.EXCEL,
    ".json": SourceKind.JSON,
    ".jsonl": SourceKind.JSON,
    ".ndjson": SourceKind.JSON,
    ".parquet": SourceKind.PARQUET,
    ".pq": SourceKind.PARQUET,
    ".duckdb": SourceKind.DUCKDB,
    ".ddb": SourceKind.DUCKDB,
}

#: URL scheme -> source kind, checked before the extension.
_SCHEME_KINDS: dict[str, SourceKind] = {
    "s3": SourceKind.S3,
    "s3a": SourceKind.S3,
    "gs": SourceKind.GCS,
    "gcs": SourceKind.GCS,
    "az": SourceKind.AZURE_BLOB,
    "abfs": SourceKind.AZURE_BLOB,
    "abfss": SourceKind.AZURE_BLOB,
    "wasb": SourceKind.AZURE_BLOB,
    "wasbs": SourceKind.AZURE_BLOB,
    "postgres": SourceKind.POSTGRES,
    "postgresql": SourceKind.POSTGRES,
    "mysql": SourceKind.MYSQL,
    "mariadb": SourceKind.MYSQL,
    "snowflake": SourceKind.SNOWFLAKE,
    "databricks": SourceKind.DATABRICKS,
    "duckdb": SourceKind.DUCKDB,
    "sqlite": SourceKind.SQL,
    "kaggle": SourceKind.KAGGLE,
}


def infer_source_kind(uri: str) -> SourceKind:
    """Classify a URI or path into a :class:`SourceKind`.

    Scheme wins over extension, because ``s3://bucket/data.csv`` is an S3 source
    that happens to hold a CSV — the connector, not the parser, is what the
    router dispatches on.

    Args:
        uri: Path, URL, connection string, or ``kaggle:owner/slug``.

    Returns:
        The best-guess kind, defaulting to :attr:`SourceKind.CSV`.
    """
    text = str(uri).strip()
    if not text:
        return SourceKind.CSV

    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme in _SCHEME_KINDS:
        return _SCHEME_KINDS[scheme]

    lowered = text.lower()
    for suffix, kind in _SUFFIX_KINDS.items():
        if lowered.endswith(suffix):
            return kind

    if scheme in ("http", "https"):
        return SourceKind.REST_API
    if "://" in text:
        return SourceKind.SQL
    return SourceKind.CSV


def data_source_from_uri(
    uri: str,
    *,
    query: str | None = None,
    options: dict[str, Any] | None = None,
    kind: SourceKind | None = None,
) -> DataSource:
    """Build a :class:`DataSource` from a single user-supplied string.

    Args:
        uri: Path, URL, connection string, or dataset slug.
        query: SQL query or table name, for database sources.
        options: Connector-specific options; converted to ``list[Param]``.
        kind: Force a kind instead of inferring one.

    Returns:
        A populated :class:`DataSource`.
    """
    resolved_kind = kind or infer_source_kind(uri)
    params = [Param(key=str(k), value=str(v)) for k, v in (options or {}).items()]
    text = str(uri).strip()
    # A local path is normalised to absolute so a run started over HTTP and a run
    # started from a different working directory resolve to the same file.
    if resolved_kind in (
        SourceKind.CSV,
        SourceKind.EXCEL,
        SourceKind.JSON,
        SourceKind.PARQUET,
    ) and "://" not in text:
        candidate = Path(text).expanduser()
        if candidate.exists():
            text = str(candidate.resolve())
    return DataSource(kind=resolved_kind, uri=text, query=query, options=params)


class ApiModel(BaseModel):
    """Base for wire models: unknown fields are an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Starting runs
# ---------------------------------------------------------------------------


class StartRunRequest(ApiModel):
    """Body for ``POST /api/runs``.

    Supply exactly one of ``source``, ``uri``, or ``upload_path`` (the value
    returned by ``POST /api/upload``). The multipart form of the same endpoint
    accepts a file directly and fills ``upload_path`` for you.
    """

    source: DataSource | None = None
    uri: str | None = Field(
        default=None, description="Path/URL, when you do not want to build a DataSource."
    )
    upload_path: str | None = Field(
        default=None, description="Path returned by POST /api/upload."
    )
    query: str | None = Field(default=None, description="SQL query for database sources.")

    project: str = "default"
    target_column: str | None = None
    task_type: TaskType | None = None
    primary_metric: str | None = None
    time_budget_seconds: int = Field(default=900, ge=10, le=86_400)
    max_experiments: int = Field(default=8, ge=1, le=64)
    max_rows: int | None = Field(default=None, ge=10)
    test_size: float = Field(default=0.2, gt=0.0, lt=0.9)
    validation_size: float = Field(default=0.15, ge=0.0, lt=0.9)
    cv_folds: int = Field(default=5, ge=2, le=20)
    random_state: int = 42
    require_approval: bool = False
    enable_tuning: bool = True
    enable_explainability: bool = True
    enable_self_improvement: bool = True
    max_replans: int = Field(default=2, ge=0, le=10)
    min_acceptable_score: float | None = None
    report_formats: list[str] = Field(
        default_factory=lambda: ["markdown", "html", "json"]
    )
    fairness_attributes: list[str] = Field(default_factory=list)
    notes: str = ""

    def resolve_source(self) -> DataSource:
        """Turn whichever source field was supplied into a :class:`DataSource`.

        Raises:
            ValueError: If no source was supplied at all.
        """
        if self.source is not None:
            return self.source
        target = self.upload_path or self.uri
        if not target:
            raise ValueError(
                "provide one of 'source', 'uri', or 'upload_path' (or upload a file)"
            )
        return data_source_from_uri(target, query=self.query)

    def to_run_config(self) -> RunConfig:
        """Build the :class:`RunConfig` this request describes."""
        return RunConfig(
            project=self.project,
            source=self.resolve_source(),
            target_column=self.target_column,
            task_type_override=self.task_type,
            primary_metric_override=self.primary_metric,
            time_budget_seconds=self.time_budget_seconds,
            max_experiments=self.max_experiments,
            max_rows=self.max_rows,
            test_size=self.test_size,
            validation_size=self.validation_size,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            require_approval=self.require_approval,
            enable_tuning=self.enable_tuning,
            enable_explainability=self.enable_explainability,
            enable_self_improvement=self.enable_self_improvement,
            max_replans=self.max_replans,
            min_acceptable_score=self.min_acceptable_score,
            report_formats=self.report_formats,
            fairness_attributes=self.fairness_attributes,
            notes=self.notes,
        )


class StartRunResponse(ApiModel):
    """Acknowledgement that a run is queued; it executes in the background."""

    run_id: str
    project: str
    status: RunStatus
    stream_url: str
    events_url: str
    run_url: str
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Reading runs
# ---------------------------------------------------------------------------


class RunListItem(ApiModel):
    """One row of the run list. Cheap enough to render a dashboard from."""

    run_id: str
    project: str
    status: RunStatus
    task_type: TaskType | None = None
    target: str | None = None
    primary_metric: str | None = None
    best_score: float | None = None
    best_family: str | None = None
    grade: str | None = None
    n_experiments: int = 0
    n_rows: int | None = None
    n_columns: int | None = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    is_active: bool = False


class RunListResponse(ApiModel):
    """A page of run list items, newest first."""

    project: str | None = None
    count: int
    runs: list[RunListItem]


class EventPage(ApiModel):
    """A cursor-paged slice of the event log."""

    run_id: str
    after: int
    last_sequence: int
    count: int
    status: RunStatus | None = None
    finished: bool = False
    events: list[RunEvent]


class CancelResponse(ApiModel):
    """Outcome of a cancellation request.

    ``cancellation_requested`` is False when there was nothing to cancel — the
    run had already finished, or it is executing in another process.
    """

    run_id: str
    status: RunStatus | None = None
    cancellation_requested: bool
    detail: str


class ApprovalListResponse(ApiModel):
    """Approval requests for a run.

    ``approvals`` is the whole set, oldest first; ``pending`` and ``resolved``
    are the two partitions a UI actually renders. Both are returned so a client
    does not have to filter, and so a generic client can read one list.
    """

    run_id: str
    approvals: list[ApprovalRequest]
    pending: list[ApprovalRequest]
    resolved: list[ApprovalRequest]


class ApprovalDecisionRequest(ApiModel):
    """Body for ``POST /api/runs/{run_id}/approvals/{request_id}``."""

    decision: Literal["approved", "rejected"]
    note: str | None = None
    decided_by: str | None = None


class ApprovalDecisionResponse(ApiModel):
    """Result of approving or rejecting one request."""

    run_id: str
    request_id: str
    approval: ApprovalRequest | None = None
    resumed: bool = Field(
        description="True when the paused run was released by this decision."
    )
    detail: str = ""


class AskRequest(ApiModel):
    """Body for ``POST /api/runs/{run_id}/ask``."""

    question: str = Field(min_length=3, max_length=2000)


class UploadResponse(ApiModel):
    """Where an uploaded dataset landed, and the source that reads it."""

    filename: str
    path: str
    size_bytes: int
    source: DataSource


class ArtifactEntry(ApiModel):
    """One file a run wrote, with the URL that serves it."""

    relative_path: str
    kind: str
    size_bytes: int
    modified_at: datetime
    url: str


class ArtifactListResponse(ApiModel):
    """Every artifact under a run directory."""

    run_id: str
    count: int
    total_bytes: int
    artifacts: list[ArtifactEntry]


class FeatureStatus(ApiModel):
    """Whether one optional capability is usable in this install."""

    name: str
    available: bool
    packages: list[str] = Field(default_factory=list)
    install_hint: str = ""
    detail: str = ""


class HealthResponse(ApiModel):
    """Answer to "is this deployment actually able to run anything?"."""

    status: Literal["ok", "degraded"]
    version: str
    model: str
    python_version: str
    credentials_configured: bool
    workspace: str
    workspace_writable: bool
    database_url: str
    database_reachable: bool
    active_runs: int
    stored_runs: int
    features: list[FeatureStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(ApiModel):
    """Uniform error envelope; every non-2xx response uses this shape."""

    error: str
    detail: str = ""
    run_id: str | None = None


__all__ = [
    "ApiModel",
    "ApprovalDecisionRequest",
    "ApprovalDecisionResponse",
    "ApprovalListResponse",
    "ArtifactEntry",
    "ArtifactListResponse",
    "AskRequest",
    "CancelResponse",
    "ErrorResponse",
    "EventPage",
    "FeatureStatus",
    "HealthResponse",
    "RunListItem",
    "RunListResponse",
    "StartRunRequest",
    "StartRunResponse",
    "UploadResponse",
    "data_source_from_uri",
    "infer_source_kind",
]
