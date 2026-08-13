"""The run blackboard.

``RunState`` is the single mutable object threaded through the orchestrator.
Agents read from it and write their typed output back; executors mutate the
dataframes. Keeping it in one place (rather than passing tuples between steps)
is what makes replanning possible — a revised plan re-runs steps against the
same state, and each step overwrites only its own slot.

One deliberate constraint: :meth:`freeze_context` builds the run-context digest
**once**, right after profiling, and every subsequent agent reuses that exact
string. It is the cached prompt prefix, so it must not vary between agents
within a run — see the caching notes in ``core/llm.py``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from .events import EventBus, NullEventBus
from .schemas import (
    AgentName,
    ApprovalRequest,
    CleaningPlan,
    DatasetProfile,
    DatasetUnderstanding,
    EvaluationVerdict,
    ExecutionPlan,
    ExperimentLog,
    ExplainabilityReport,
    FeaturePlan,
    FinalReport,
    IngestionResult,
    InsightReport,
    MemorySuggestion,
    ModelSelection,
    ProblemDefinition,
    ReportBundle,
    RunConfig,
    RunStatus,
    RunSummary,
    StepRecord,
    StepStatus,
    TaskType,
    TuningDecision,
    TuningResult,
    UsageTotals,
    VisualizationBundle,
    VisualizationPlan,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DataSplits:
    """Materialised train/validation/test partitions.

    ``strategy`` records *how* they were made (stratified, grouped, temporal),
    which the Evaluation Agent needs in order to judge whether a score is
    trustworthy.
    """

    X_train: Any = None
    X_valid: Any = None
    X_test: Any = None
    y_train: Any = None
    y_valid: Any = None
    y_test: Any = None
    strategy: str = ""
    rationale: str = ""

    @property
    def has_validation(self) -> bool:
        return self.X_valid is not None and len(self.X_valid) > 0

    @property
    def has_test(self) -> bool:
        return self.X_test is not None and len(self.X_test) > 0

    def sizes(self) -> dict[str, int]:
        def n(frame: Any) -> int:
            return 0 if frame is None else int(len(frame))

        return {
            "train": n(self.X_train),
            "validation": n(self.X_valid),
            "test": n(self.X_test),
        }


@dataclass
class RunState:
    """Everything known about one analysis run."""

    config: RunConfig
    settings: Settings = field(default_factory=get_settings)
    bus: EventBus = field(default_factory=NullEventBus)

    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    _monotonic_start: float | None = field(default=None, repr=False)

    # --- data ------------------------------------------------------------
    raw_df: Any = field(default=None, repr=False)
    working_df: Any = field(default=None, repr=False)
    feature_frame: Any = field(default=None, repr=False)
    splits: DataSplits = field(default_factory=DataSplits, repr=False)
    feature_names: list[str] = field(default_factory=list)

    # --- agent + executor outputs ----------------------------------------
    ingestion: IngestionResult | None = None
    profile: DatasetProfile | None = None
    understanding: DatasetUnderstanding | None = None
    problem: ProblemDefinition | None = None
    plan: ExecutionPlan | None = None
    plan_history: list[ExecutionPlan] = field(default_factory=list)
    memory: MemorySuggestion | None = None
    cleaning: CleaningPlan | None = None
    features: FeaturePlan | None = None
    model_selection: ModelSelection | None = None
    experiments: ExperimentLog | None = None
    tuning_decision: TuningDecision | None = None
    tuning: TuningResult | None = None
    explainability: ExplainabilityReport | None = None
    evaluation: EvaluationVerdict | None = None
    insights: InsightReport | None = None
    visualization_plan: VisualizationPlan | None = None
    visualizations: VisualizationBundle | None = None
    report: FinalReport | None = None
    report_bundle: ReportBundle | None = None

    # --- fitted objects (never serialised into the summary) ---------------
    preprocessor: Any = field(default=None, repr=False)
    best_model: Any = field(default=None, repr=False)
    best_pipeline: Any = field(default=None, repr=False)
    label_encoder: Any = field(default=None, repr=False)

    # --- bookkeeping ------------------------------------------------------
    steps: list[StepRecord] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)
    usage: UsageTotals = field(default_factory=UsageTotals)
    replans: int = 0
    warnings: list[str] = field(default_factory=list)
    applied_cleaning: list[str] = field(default_factory=list)
    applied_features: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict, repr=False)

    _frozen_context: str | None = field(default=None, repr=False)

    # -- identity ---------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self.config.run_id

    @property
    def artifact_dir(self) -> Path:
        return self.settings.run_dir(self.run_id)

    def artifact_path(self, *parts: str) -> Path:
        path = self.artifact_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def elapsed_seconds(self) -> float:
        if self._monotonic_start is None:
            return 0.0
        return time.monotonic() - self._monotonic_start

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.config.time_budget_seconds - self.elapsed_seconds)

    def mark_started(self) -> None:
        self.started_at = _utcnow()
        self._monotonic_start = time.monotonic()
        self.status = RunStatus.RUNNING

    def mark_finished(self, status: RunStatus, error: str | None = None) -> None:
        self.finished_at = _utcnow()
        self.status = status
        if error:
            self.error = error

    # -- convenience accessors --------------------------------------------

    @property
    def task_type(self) -> TaskType | None:
        if self.config.task_type_override:
            return self.config.task_type_override
        return self.problem.task_type if self.problem else None

    @property
    def target(self) -> str | None:
        if self.problem and self.problem.target_column:
            return self.problem.target_column
        return self.config.target_column

    @property
    def primary_metric(self) -> str:
        if self.config.primary_metric_override:
            return self.config.primary_metric_override
        if self.problem:
            return self.problem.primary_metric
        return "accuracy"

    @property
    def df(self) -> Any:
        """The most-processed frame available."""
        for candidate in (self.feature_frame, self.working_df, self.raw_df):
            if candidate is not None:
                return candidate
        return None

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        self.bus.warn(message)

    # -- step tracking ----------------------------------------------------

    def step(self, step_id: str) -> StepRecord | None:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def upsert_step(
        self,
        step_id: str,
        *,
        title: str = "",
        agent: AgentName | None = None,
        status: StepStatus | None = None,
        error: str | None = None,
        summary: str | None = None,
    ) -> StepRecord:
        record = self.step(step_id)
        if record is None:
            record = StepRecord(step_id=step_id, title=title, agent=agent)
            self.steps.append(record)
        if title:
            record.title = title
        if agent:
            record.agent = agent
        if status is not None:
            record.status = status
            if status is StepStatus.RUNNING:
                record.started_at = _utcnow()
                record.attempts += 1
            elif status in (
                StepStatus.COMPLETED,
                StepStatus.FAILED,
                StepStatus.SKIPPED,
            ):
                record.finished_at = _utcnow()
                if record.started_at:
                    record.duration_seconds = (
                        record.finished_at - record.started_at
                    ).total_seconds()
        if error is not None:
            record.error = error
        if summary is not None:
            record.summary = summary
        return record

    def pending_approval(self) -> ApprovalRequest | None:
        return next((a for a in self.approvals if a.decision == "pending"), None)

    # -- prompt context ---------------------------------------------------

    def freeze_context(self, digest: str) -> str:
        """Pin the run-context prompt block. Idempotent: first call wins."""
        if self._frozen_context is None:
            self._frozen_context = digest
        return self._frozen_context

    @property
    def run_context(self) -> str | None:
        return self._frozen_context

    # -- serialisation ----------------------------------------------------

    def to_summary(self) -> RunSummary:
        duration = 0.0
        if self.started_at and self.finished_at:
            duration = (self.finished_at - self.started_at).total_seconds()
        elif self._monotonic_start is not None:
            duration = self.elapsed_seconds

        return RunSummary(
            run_id=self.run_id,
            project=self.config.project,
            status=self.status,
            config=self.config,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_seconds=duration,
            error=self.error,
            ingestion=self.ingestion,
            profile=self.profile,
            understanding=self.understanding,
            problem=self.problem,
            plan=self.plan,
            plan_history=self.plan_history,
            cleaning=self.cleaning,
            features=self.features,
            model_selection=self.model_selection,
            experiments=self.experiments,
            tuning_decision=self.tuning_decision,
            tuning=self.tuning,
            explainability=self.explainability,
            evaluation=self.evaluation,
            insights=self.insights,
            visualization_plan=self.visualization_plan,
            visualizations=self.visualizations,
            report=self.report,
            report_bundle=self.report_bundle,
            steps=self.steps,
            approvals=self.approvals,
            usage=self.usage,
            replans=self.replans,
            warnings=self.warnings,
            artifact_dir=str(self.artifact_dir),
        )

    def save_summary(self) -> Path:
        path = self.artifact_path("run_summary.json")
        payload = self.to_summary().model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
