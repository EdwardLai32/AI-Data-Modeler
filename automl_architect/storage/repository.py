"""Durable storage for runs, events, experiments, approvals, and fingerprints.

The repository is the only module that touches the database, and it deals
exclusively in the Pydantic contracts from :mod:`automl_architect.core.schemas`
— callers never see an ORM row. That boundary is what lets the API serve a
``RunSummary`` for a run that finished in a different process, and lets the SSE
stream replay events it never saw on its own bus.

Concurrency note: a run executes on a background thread while HTTP handlers read
from the event loop's threadpool, so engines are cached per URL and SQLite gets
``check_same_thread=False`` plus WAL. Those are conditional tunings, not
assumptions — the same code runs unchanged against Postgres.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, create_engine, delete, event, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import Settings, get_settings
from ..core.schemas import (
    AgentName,
    ApprovalRequest,
    DatasetFingerprint,
    EventKind,
    ExperimentResult,
    ModelFamily,
    RunEvent,
    RunStatus,
    RunSummary,
    SimilarRun,
    TaskType,
    dict_to_params,
    params_to_dict,
)
from .models import (
    ApprovalRow,
    Base,
    EventRow,
    ExperimentRow,
    FingerprintRow,
    RunRow,
)

logger = logging.getLogger(__name__)

_ENGINES: dict[str, Engine] = {}
_ENGINE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum_value(value: Any) -> Any:
    """Unwrap an ``Enum`` to its value, passing anything else through."""
    return getattr(value, "value", value)


def _install_sqlite_pragmas(engine: Engine) -> None:
    """Make SQLite tolerate concurrent readers while a run writes events."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        except Exception:  # a pragma failure must not break the connection
            logger.debug("could not apply SQLite pragmas", exc_info=True)
        finally:
            cursor.close()


def get_engine(database_url: str) -> Engine:
    """Return the process-wide engine for ``database_url``, creating it once.

    Args:
        database_url: A SQLAlchemy URL.

    Returns:
        A cached :class:`sqlalchemy.Engine` with the schema already created.
    """
    with _ENGINE_LOCK:
        existing = _ENGINES.get(database_url)
        if existing is not None:
            return existing

        kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in database_url or "mode=memory" in database_url:
                # An in-memory database is per-connection; one shared connection
                # is the only way threads see the same data.
                kwargs["poolclass"] = StaticPool

        engine = create_engine(database_url, **kwargs)
        if engine.dialect.name == "sqlite":
            _install_sqlite_pragmas(engine)
        Base.metadata.create_all(engine)
        _ENGINES[database_url] = engine
        return engine


def dispose_engines() -> None:
    """Close every cached engine. Used by tests and by process shutdown."""
    with _ENGINE_LOCK:
        for engine in _ENGINES.values():
            engine.dispose()
        _ENGINES.clear()


class RunRepository:
    """Read/write access to persisted runs.

    Args:
        database_url: Override the URL from settings. Defaults to
            ``settings.resolved_database_url`` (SQLite under the workspace).
        settings: Settings instance; the cached singleton by default.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.resolved_database_url
        self.engine = get_engine(self.database_url)
        self._sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False)

    # -- session plumbing --------------------------------------------------

    def session(self) -> Session:
        """A new session. Callers are responsible for closing it."""
        return self._sessionmaker()

    @property
    def dialect(self) -> str:
        """Name of the active SQLAlchemy dialect, e.g. ``sqlite``/``postgresql``."""
        return self.engine.dialect.name

    def ping(self) -> bool:
        """Whether the database accepts a connection right now."""
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return True
        except SQLAlchemyError as exc:
            logger.warning("database ping failed: %s", exc)
            return False

    def close(self) -> None:
        """Drop pooled connections held by this repository's engine."""
        self.engine.dispose()

    # -- runs --------------------------------------------------------------

    def save_run(self, summary: RunSummary) -> None:
        """Upsert a run and refresh its derived experiment/approval rows.

        Safe to call repeatedly during a run (the API saves once at start and
        again at every status transition); the JSON blob is replaced wholesale.

        Args:
            summary: The run to persist.
        """
        payload = summary.model_dump(mode="json")
        best = summary.experiments.best() if summary.experiments else None
        best_score = best.primary_score if best else None
        if best_score is None and summary.tuning is not None:
            best_score = summary.tuning.best_score

        with self._sessionmaker.begin() as session:
            row = session.get(RunRow, summary.run_id)
            if row is None:
                row = RunRow(run_id=summary.run_id)
                session.add(row)

            row.project = summary.project
            row.status = _enum_value(summary.status)
            row.task_type = (
                _enum_value(summary.problem.task_type) if summary.problem else None
            )
            row.target = (
                summary.problem.target_column
                if summary.problem
                else summary.config.target_column
            )
            row.primary_metric = (
                summary.experiments.primary_metric
                if summary.experiments and summary.experiments.primary_metric
                else (summary.problem.primary_metric if summary.problem else None)
            )
            row.best_score = best_score
            row.best_family = _enum_value(best.family) if best else None
            row.n_rows = summary.profile.n_rows if summary.profile else None
            row.n_columns = summary.profile.n_columns if summary.profile else None
            row.n_experiments = len(summary.experiments.results) if summary.experiments else 0
            row.grade = summary.evaluation.overall_grade if summary.evaluation else None
            row.cost_usd = summary.usage.cost_usd
            row.duration_seconds = summary.duration_seconds
            row.error = summary.error
            row.artifact_dir = summary.artifact_dir
            row.started_at = summary.started_at
            row.finished_at = summary.finished_at
            row.updated_at = _utcnow()
            row.summary_json = payload

            self._replace_experiments(session, summary)
            self._replace_approvals(session, summary)

    def get_run(self, run_id: str) -> RunSummary | None:
        """Load one run's full summary, or ``None`` if it is unknown."""
        with self.session() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                return None
            return self._decode_summary(row)

    def list_runs(self, project: str | None = None, limit: int = 50) -> list[RunSummary]:
        """Most recent runs first, optionally filtered to one project."""
        statement = select(RunRow).order_by(RunRow.created_at.desc()).limit(max(1, limit))
        if project:
            statement = statement.where(RunRow.project == project)
        with self.session() as session:
            rows = list(session.scalars(statement))
        decoded = [self._decode_summary(row) for row in rows]
        return [item for item in decoded if item is not None]

    def list_run_index(
        self, project: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Scalar columns only, for list views.

        Decoding every run's full summary JSON just to render a table of twenty
        rows is the obvious way to make a dashboard slow, so the indexed columns
        are read directly.
        """
        columns = (
            RunRow.run_id,
            RunRow.project,
            RunRow.status,
            RunRow.task_type,
            RunRow.target,
            RunRow.primary_metric,
            RunRow.best_score,
            RunRow.best_family,
            RunRow.grade,
            RunRow.n_experiments,
            RunRow.n_rows,
            RunRow.n_columns,
            RunRow.duration_seconds,
            RunRow.cost_usd,
            RunRow.started_at,
            RunRow.finished_at,
            RunRow.error,
        )
        statement = (
            select(*columns).order_by(RunRow.created_at.desc()).limit(max(1, limit))
        )
        if project:
            statement = statement.where(RunRow.project == project)
        with self.session() as session:
            return [dict(row._mapping) for row in session.execute(statement)]

    def list_projects(self) -> list[str]:
        """Distinct project names, alphabetically."""
        with self.session() as session:
            names = session.scalars(select(RunRow.project).distinct()).all()
        return sorted({name for name in names if name})

    def delete_run(self, run_id: str) -> bool:
        """Remove a run and everything derived from it. Artifacts are untouched."""
        with self._sessionmaker.begin() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                return False
            session.delete(row)
            session.execute(delete(EventRow).where(EventRow.run_id == run_id))
            session.execute(delete(ExperimentRow).where(ExperimentRow.run_id == run_id))
            session.execute(delete(ApprovalRow).where(ApprovalRow.run_id == run_id))
            session.execute(delete(FingerprintRow).where(FingerprintRow.run_id == run_id))
        return True

    def _decode_summary(self, row: RunRow) -> RunSummary | None:
        try:
            return RunSummary.model_validate(row.summary_json)
        except Exception:  # stored blob predates a contract change
            logger.exception(
                "run %s could not be decoded into RunSummary; skipping", row.run_id
            )
            return None

    # -- events ------------------------------------------------------------

    def append_events(self, run_id: str, events: list[RunEvent]) -> None:
        """Persist events for a run, skipping any sequence already stored.

        Idempotent by ``(run_id, sequence)``: the event writer may re-submit a
        batch after a transient failure, and a duplicate must not raise.

        The already-stored set is read as a range from the batch's lowest
        sequence, not as a single high-water mark. Sequences are assigned under
        the bus lock but delivered to subscribers outside it, so a later event
        can reach the writer first; comparing against the maximum would then
        discard the earlier one permanently and break the replay guarantee the
        SSE cursor depends on.

        Args:
            run_id: Owning run.
            events: Events to append, in any order.
        """
        if not events:
            return
        lowest = min(event.sequence for event in events)
        stored = self._stored_sequences(run_id, lowest)
        fresh = [event for event in events if event.sequence not in stored]
        if not fresh:
            return
        rows = [self._event_row(run_id, event) for event in fresh]
        try:
            with self._sessionmaker.begin() as session:
                session.add_all(rows)
        except IntegrityError:
            # Two writers raced on the same sequence. Fall back to row-at-a-time
            # so one collision does not discard the rest of the batch.
            logger.debug("event batch collided; retrying individually", exc_info=True)
            for event_model in fresh:
                try:
                    with self._sessionmaker.begin() as session:
                        session.add(self._event_row(run_id, event_model))
                except IntegrityError:
                    continue

    def get_events(
        self, run_id: str, after: int = 0, *, limit: int | None = None
    ) -> list[RunEvent]:
        """Events with ``sequence > after``, in order. The SSE replay path.

        Args:
            run_id: Owning run.
            after: Exclusive lower bound on the sequence cursor.
            limit: Cap the window. Omit it to replay everything after the cursor.
        """
        statement = (
            select(EventRow)
            .where(EventRow.run_id == run_id, EventRow.sequence > after)
            .order_by(EventRow.sequence.asc())
        )
        if limit is not None:
            statement = statement.limit(max(1, limit))
        with self.session() as session:
            rows = list(session.scalars(statement))
        return [self._decode_event(row) for row in rows]

    def _stored_sequences(self, run_id: str, at_or_above: int) -> set[int]:
        """Sequences already persisted for a run from ``at_or_above`` upwards.

        An index range read on ``(run_id, sequence)``. For the normal case — a
        batch of new events — the range starts past the end and returns nothing.
        """
        statement = select(EventRow.sequence).where(
            EventRow.run_id == run_id, EventRow.sequence >= at_or_above
        )
        with self.session() as session:
            return {int(value) for value in session.scalars(statement)}

    def latest_sequence(self, run_id: str) -> int:
        """Highest persisted sequence for a run, or 0 if none."""
        with self.session() as session:
            value = session.scalar(
                select(EventRow.sequence)
                .where(EventRow.run_id == run_id)
                .order_by(EventRow.sequence.desc())
                .limit(1)
            )
        return int(value or 0)

    @staticmethod
    def _event_row(run_id: str, model: RunEvent) -> EventRow:
        return EventRow(
            run_id=run_id,
            sequence=model.sequence,
            event_id=model.event_id,
            kind=_enum_value(model.kind),
            agent=_enum_value(model.agent) if model.agent else None,
            step_id=model.step_id,
            message=model.message,
            payload_json=params_to_dict(model.payload),
            occurred_at=model.at,
            duration_seconds=model.duration_seconds,
            tokens_in=model.tokens_in,
            tokens_out=model.tokens_out,
            cache_read_tokens=model.cache_read_tokens,
            cost_usd=model.cost_usd,
        )

    @staticmethod
    def _decode_event(row: EventRow) -> RunEvent:
        try:
            kind = EventKind(row.kind)
        except ValueError:
            kind = EventKind.LOG
        agent: AgentName | None = None
        if row.agent:
            try:
                agent = AgentName(row.agent)
            except ValueError:
                agent = None
        return RunEvent(
            event_id=row.event_id or f"ev_{row.id}",
            run_id=row.run_id,
            sequence=row.sequence,
            kind=kind,
            agent=agent,
            step_id=row.step_id,
            message=row.message or "",
            payload=dict_to_params(row.payload_json or {}),
            at=row.occurred_at,
            duration_seconds=row.duration_seconds,
            tokens_in=row.tokens_in,
            tokens_out=row.tokens_out,
            cache_read_tokens=row.cache_read_tokens,
            cost_usd=row.cost_usd,
        )

    # -- experiments -------------------------------------------------------

    def _replace_experiments(self, session: Session, summary: RunSummary) -> None:
        session.execute(delete(ExperimentRow).where(ExperimentRow.run_id == summary.run_id))
        log = summary.experiments
        if log is None:
            return
        task_type = _enum_value(summary.problem.task_type) if summary.problem else None
        for result in log.results:
            session.add(
                ExperimentRow(
                    run_id=summary.run_id,
                    project=summary.project,
                    experiment_id=result.experiment_id,
                    family=_enum_value(result.family),
                    label=result.label[:255],
                    task_type=task_type,
                    primary_metric=result.primary_metric or log.primary_metric,
                    primary_score=result.primary_score,
                    is_best=result.experiment_id == log.best_experiment_id,
                    is_baseline=result.is_baseline,
                    tuned=result.tuned,
                    failed=result.failed,
                    error=result.error,
                    train_seconds=result.train_seconds,
                    predict_seconds=result.predict_seconds,
                    n_features_in=result.n_features_in,
                    model_size_bytes=result.model_size_bytes,
                    created_at=result.created_at,
                    result_json=result.model_dump(mode="json"),
                )
            )

    def get_experiments(self, run_id: str) -> list[ExperimentResult]:
        """Leaderboard rows for one run, best-scoring first where known."""
        statement = (
            select(ExperimentRow)
            .where(ExperimentRow.run_id == run_id)
            .order_by(ExperimentRow.is_best.desc(), ExperimentRow.primary_score.desc())
        )
        with self.session() as session:
            rows = list(session.scalars(statement))
        out: list[ExperimentResult] = []
        for row in rows:
            try:
                out.append(ExperimentResult.model_validate(row.result_json))
            except Exception:
                logger.debug("skipping undecodable experiment row %s", row.id)
        return out

    def top_experiments(
        self,
        *,
        project: str | None = None,
        family: ModelFamily | None = None,
        limit: int = 20,
    ) -> list[tuple[str, ExperimentResult]]:
        """Cross-run leaderboard as ``(run_id, result)`` pairs."""
        statement = (
            select(ExperimentRow)
            .where(ExperimentRow.failed.is_(False))
            .order_by(ExperimentRow.primary_score.desc())
            .limit(max(1, limit))
        )
        if project:
            statement = statement.where(ExperimentRow.project == project)
        if family:
            statement = statement.where(ExperimentRow.family == _enum_value(family))
        with self.session() as session:
            rows = list(session.scalars(statement))
        out: list[tuple[str, ExperimentResult]] = []
        for row in rows:
            try:
                out.append((row.run_id, ExperimentResult.model_validate(row.result_json)))
            except Exception:
                continue
        return out

    # -- approvals ---------------------------------------------------------

    def _replace_approvals(self, session: Session, summary: RunSummary) -> None:
        session.execute(delete(ApprovalRow).where(ApprovalRow.run_id == summary.run_id))
        for approval in summary.approvals:
            session.add(self._approval_row(summary.run_id, approval))

    @staticmethod
    def _approval_row(run_id: str, approval: ApprovalRequest) -> ApprovalRow:
        return ApprovalRow(
            run_id=run_id,
            request_id=approval.request_id,
            step_id=approval.step_id,
            agent=_enum_value(approval.agent),
            action_summary=approval.action_summary,
            severity=_enum_value(approval.severity),
            decision=approval.decision,
            created_at=approval.created_at,
            decided_at=approval.decided_at,
            decided_by=approval.decided_by,
            note=approval.note,
            request_json=approval.model_dump(mode="json"),
        )

    def save_approval(self, run_id: str, approval: ApprovalRequest) -> None:
        """Upsert one approval request, keyed by ``(run_id, request_id)``."""
        with self._sessionmaker.begin() as session:
            row = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.run_id == run_id,
                    ApprovalRow.request_id == approval.request_id,
                )
            )
            fresh = self._approval_row(run_id, approval)
            if row is None:
                session.add(fresh)
                return
            row.step_id = fresh.step_id
            row.agent = fresh.agent
            row.action_summary = fresh.action_summary
            row.severity = fresh.severity
            row.decision = fresh.decision
            row.decided_at = fresh.decided_at
            row.decided_by = fresh.decided_by
            row.note = fresh.note
            row.request_json = fresh.request_json

    def get_approvals(
        self, run_id: str, *, pending_only: bool = False
    ) -> list[ApprovalRequest]:
        """Approval requests for a run, oldest first."""
        statement = (
            select(ApprovalRow)
            .where(ApprovalRow.run_id == run_id)
            .order_by(ApprovalRow.created_at.asc())
        )
        if pending_only:
            statement = statement.where(ApprovalRow.decision == "pending")
        with self.session() as session:
            rows = list(session.scalars(statement))
        out: list[ApprovalRequest] = []
        for row in rows:
            try:
                out.append(ApprovalRequest.model_validate(row.request_json))
            except Exception:
                logger.debug("skipping undecodable approval row %s", row.id)
        return out

    # -- fingerprints ------------------------------------------------------

    def save_fingerprint(self, fp: DatasetFingerprint) -> None:
        """Upsert a dataset fingerprint for the memory lookup."""
        with self._sessionmaker.begin() as session:
            row = session.get(FingerprintRow, fp.fingerprint_id)
            if row is None:
                row = FingerprintRow(fingerprint_id=fp.fingerprint_id)
                session.add(row)
            row.run_id = fp.run_id
            row.project = fp.project
            row.n_rows = fp.n_rows
            row.n_columns = fp.n_columns
            row.n_numeric = fp.n_numeric
            row.n_categorical = fp.n_categorical
            row.n_datetime = fp.n_datetime
            row.n_text = fp.n_text
            row.missing_fraction = fp.missing_fraction
            row.duplicate_fraction = fp.duplicate_fraction
            row.task_type = _enum_value(fp.task_type) if fp.task_type else None
            row.target_kind = _enum_value(fp.target_kind) if fp.target_kind else None
            row.imbalance_ratio = fp.imbalance_ratio
            row.primary_metric = fp.primary_metric
            row.best_score = fp.best_score
            row.best_family = _enum_value(fp.best_family) if fp.best_family else None
            row.created_at = fp.created_at
            row.fingerprint_json = fp.model_dump(mode="json")

    def list_fingerprints(
        self,
        *,
        project: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 500,
    ) -> list[DatasetFingerprint]:
        """Stored fingerprints, newest first.

        Args:
            project: Restrict to one project. ``None`` searches every project,
                which is usually what you want — precedent from another project
                is still precedent.
            exclude_run_id: Skip this run (a run must not match itself).
            limit: Candidate cap.
        """
        statement = (
            select(FingerprintRow)
            .order_by(FingerprintRow.created_at.desc())
            .limit(max(1, limit))
        )
        if project:
            statement = statement.where(FingerprintRow.project == project)
        if exclude_run_id:
            statement = statement.where(FingerprintRow.run_id != exclude_run_id)
        with self.session() as session:
            rows = list(session.scalars(statement))
        out: list[DatasetFingerprint] = []
        for row in rows:
            try:
                out.append(DatasetFingerprint.model_validate(row.fingerprint_json))
            except Exception:
                logger.debug("skipping undecodable fingerprint %s", row.fingerprint_id)
        return out

    def get_fingerprint_for_run(self, run_id: str) -> DatasetFingerprint | None:
        """The most recent fingerprint recorded for a run."""
        statement = (
            select(FingerprintRow)
            .where(FingerprintRow.run_id == run_id)
            .order_by(FingerprintRow.created_at.desc())
            .limit(1)
        )
        with self.session() as session:
            row = session.scalar(statement)
        if row is None:
            return None
        try:
            return DatasetFingerprint.model_validate(row.fingerprint_json)
        except Exception:
            return None

    def find_similar(self, fp: DatasetFingerprint, limit: int = 5) -> list[SimilarRun]:
        """Rank stored fingerprints against ``fp`` by structural similarity.

        Scoring lives in :mod:`automl_architect.storage.memory` so it can be
        unit-tested without a database; this method only supplies candidates.

        Args:
            fp: The fingerprint to match.
            limit: Maximum matches to return.

        Returns:
            Matches above the similarity floor, most similar first. Empty when
            there is no usable history.
        """
        from .memory import rank_fingerprints  # circular at module scope

        candidates = self.list_fingerprints(exclude_run_id=fp.run_id)
        return [similar for similar, _ in rank_fingerprints(fp, candidates, limit=limit)]

    # -- convenience -------------------------------------------------------

    def run_status(self, run_id: str) -> RunStatus | None:
        """Cheap status read that avoids decoding the whole summary."""
        with self.session() as session:
            value = session.scalar(select(RunRow.status).where(RunRow.run_id == run_id))
        if value is None:
            return None
        try:
            return RunStatus(value)
        except ValueError:
            return None

    def count_runs(self, project: str | None = None) -> int:
        """Number of stored runs, optionally scoped to a project."""
        statement = select(RunRow.run_id)
        if project:
            statement = statement.where(RunRow.project == project)
        with self.session() as session:
            return len(session.scalars(statement).all())

    def task_types_seen(self) -> list[TaskType]:
        """Distinct task types across stored runs, for the doctor/health views."""
        with self.session() as session:
            values = session.scalars(select(RunRow.task_type).distinct()).all()
        out: list[TaskType] = []
        for value in values:
            if not value:
                continue
            try:
                out.append(TaskType(value))
            except ValueError:
                continue
        return out


__all__ = ["RunRepository", "dispose_engines", "get_engine"]
