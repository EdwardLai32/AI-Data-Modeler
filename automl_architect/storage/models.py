"""SQLAlchemy 2.0 ORM tables backing run persistence.

Two ideas shape this schema.

*   **The JSON blob is authoritative; the columns are the index.** Every table
    that mirrors a Pydantic model stores the model's full ``model_dump(mode="json")``
    alongside a handful of scalar columns. Reads reconstruct the typed object
    from the blob, so a new field on :class:`~automl_architect.core.schemas.RunSummary`
    never needs a migration; the scalar columns exist only so the API can filter
    and sort without deserialising every run.
*   **Nothing here may depend on SQLite.** The default URL points at a file
    under the workspace, but the same DDL has to run on Postgres. JSON columns
    are therefore declared as portable :class:`sqlalchemy.JSON` with a
    ``JSONB`` variant, every ``String`` carries an explicit length (MySQL
    requires one on indexed columns), and timestamps default in Python rather
    than via dialect-specific SQL functions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Portable JSON column type. ``JSON`` works on SQLite/MySQL/Postgres; the
#: variant upgrades Postgres to ``JSONB`` so operators can index into payloads.
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")

ID_LEN = 64
NAME_LEN = 128


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every table in this package."""


class RunRow(Base):
    """One analysis run, with the full :class:`RunSummary` JSON attached."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    project: Mapped[str] = mapped_column(String(NAME_LEN), default="default", index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    task_type: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    target: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    primary_metric: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    best_score: Mapped[float | None] = mapped_column(Float, default=None)
    best_family: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    n_rows: Mapped[int | None] = mapped_column(Integer, default=None)
    n_columns: Mapped[int | None] = mapped_column(Integer, default=None)
    n_experiments: Mapped[int] = mapped_column(Integer, default=0)
    grade: Mapped[str | None] = mapped_column(String(8), default=None)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    artifact_dir: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)


class EventRow(Base):
    """One :class:`RunEvent`. The ``(run_id, sequence)`` pair is the cursor."""

    __tablename__ = "run_events"
    __table_args__ = (
        # Unique so re-persisting a batch after a retry cannot duplicate rows,
        # and ordered so the API's `after=` cursor scan is an index range read.
        Index("ix_run_events_run_sequence", "run_id", "sequence", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_id: Mapped[str] = mapped_column(String(ID_LEN), default="")
    kind: Mapped[str] = mapped_column(String(32), default="log", index=True)
    agent: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    step_id: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    message: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    cost_usd: Mapped[float | None] = mapped_column(Float, default=None)


class ExperimentRow(Base):
    """A leaderboard row, denormalised out of the run summary for querying."""

    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_run", "run_id"),
        Index("ix_experiments_score", "primary_metric", "primary_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    project: Mapped[str] = mapped_column(String(NAME_LEN), default="default")
    experiment_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    family: Mapped[str] = mapped_column(String(NAME_LEN), default="")
    label: Mapped[str] = mapped_column(String(255), default="")
    task_type: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    primary_metric: Mapped[str] = mapped_column(String(NAME_LEN), default="")
    primary_score: Mapped[float | None] = mapped_column(Float, default=None)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    tuned: Mapped[bool] = mapped_column(Boolean, default=False)
    failed: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    train_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    predict_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    n_features_in: Mapped[int] = mapped_column(Integer, default=0)
    model_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)


class ApprovalRow(Base):
    """A human-in-the-loop gate on a destructive step."""

    __tablename__ = "approvals"
    __table_args__ = (
        Index("ix_approvals_run_request", "run_id", "request_id", unique=True),
        Index("ix_approvals_run_decision", "run_id", "decision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    request_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    step_id: Mapped[str] = mapped_column(String(NAME_LEN), default="")
    agent: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    action_summary: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(32), default="medium")
    decision: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    decided_by: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)


class FingerprintRow(Base):
    """Structural signature of a dataset, for the dataset-memory lookup.

    The scalar columns are enough to pre-filter candidates cheaply; scoring
    needs the column-name list, so the full fingerprint JSON rides along.
    """

    __tablename__ = "fingerprints"
    __table_args__ = (
        Index("ix_fingerprints_project_task", "project", "task_type"),
        Index("ix_fingerprints_run", "run_id"),
    )

    fingerprint_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    project: Mapped[str] = mapped_column(String(NAME_LEN), default="default")
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_columns: Mapped[int] = mapped_column(Integer, default=0)
    n_numeric: Mapped[int] = mapped_column(Integer, default=0)
    n_categorical: Mapped[int] = mapped_column(Integer, default=0)
    n_datetime: Mapped[int] = mapped_column(Integer, default=0)
    n_text: Mapped[int] = mapped_column(Integer, default=0)
    missing_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    duplicate_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    task_type: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    target_kind: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    imbalance_ratio: Mapped[float | None] = mapped_column(Float, default=None)
    primary_metric: Mapped[str] = mapped_column(String(NAME_LEN), default="")
    best_score: Mapped[float | None] = mapped_column(Float, default=None)
    best_family: Mapped[str | None] = mapped_column(String(NAME_LEN), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    fingerprint_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)


ALL_TABLES = (RunRow, EventRow, ExperimentRow, ApprovalRow, FingerprintRow)

__all__ = [
    "ALL_TABLES",
    "ApprovalRow",
    "Base",
    "EventRow",
    "ExperimentRow",
    "FingerprintRow",
    "JSON_TYPE",
    "RunRow",
]
