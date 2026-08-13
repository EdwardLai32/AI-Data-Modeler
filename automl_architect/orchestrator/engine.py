"""The orchestration engine.

:class:`Orchestrator` is the deterministic spine of the platform. It owns the run
state, dispatches every step, and is the only component allowed to decide that
something failed, should be retried, needs a human, or is worth doing again.
Agents decide *what* to do; this decides *whether and when* their decisions run.

Four design decisions here are load-bearing:

*   **The prompt context is frozen exactly once**, immediately after profiling
    and before the first agent call. Every agent in the run then shares a
    byte-identical cached prefix. Rebuilding it mid-run would silently cost a
    cache read on every subsequent call, so :meth:`RunState.freeze_context` is
    called from one place only.
*   **Failure degrades by default.** Only ingest, profile, problem
    identification, planning, splitting, and training are critical. Tuning,
    explainability, insights, and visualisation are allowed to fail into a
    warning, because a report that says "SHAP was unavailable" is worth far more
    than a stack trace.
*   **Agents and executors are resolved lazily.** A module that has not been
    written, or an optional dependency that is not installed, becomes a warning
    at the step that needed it rather than an ImportError at process start.
*   **The self-improvement loop keeps the better model.** A replan that produces
    a worse model restores the previous one and says so in a warning. Shipping a
    silent regression would defeat the entire purpose of retrying.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
import pkgutil
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..config import Settings, get_settings
from ..core.context import build_run_context
from ..core.errors import (
    ApprovalRejected,
    ApprovalRequired,
    ConfigurationError,
    NoViableModelError,
    RunCancelled,
    StepFailedError,
)
from ..core.events import EventBus
from ..core.schemas import (
    AgentName,
    CleaningAction,
    CleaningPlan,
    ColumnKind,
    DatasetFingerprint,
    EventKind,
    FeatureOp,
    FeaturePlan,
    MemorySuggestion,
    MissingStrategy,
    ModelCandidate,
    ModelSelection,
    PlanStep,
    RunConfig,
    RunStatus,
    RunSummary,
    Severity,
    StepStatus,
    TuningResult,
)
from ..core.state import DataSplits, RunState
from .graph import (
    EVALUATION_STEP_ID,
    PlanReconciliation,
    ResolvedStep,
    StepDefinition,
    canonical_steps,
    reconcile_plan,
    require_step,
    step_index,
)
from .policies import (
    ApprovalPolicy,
    BudgetPolicy,
    ReplanDecision,
    ReplanPolicy,
    RetryPolicy,
)

logger = logging.getLogger(__name__)

StepHandler = Callable[[PlanStep | None], str]


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------

# (module suffixes, class names) tried in order for each agent. The fallback is a
# scan of the agents package for a BaseAgent subclass whose `name` matches, so a
# renamed module still resolves.
_AGENT_CANDIDATES: dict[AgentName, tuple[tuple[str, ...], tuple[str, ...]]] = {
    AgentName.DATASET: (
        ("dataset", "understanding"),
        ("DatasetAgent", "DatasetUnderstandingAgent"),
    ),
    AgentName.PROBLEM: (
        ("problem", "problem_identification"),
        ("ProblemAgent", "ProblemIdentificationAgent"),
    ),
    AgentName.PLANNER: (("planner", "planning"), ("PlannerAgent", "PlanningAgent")),
    AgentName.CLEANING: (("cleaning", "clean"), ("CleaningAgent", "DataCleaningAgent")),
    AgentName.FEATURES: (
        ("features", "feature", "feature_engineering"),
        ("FeatureAgent", "FeaturesAgent", "FeatureEngineeringAgent"),
    ),
    AgentName.MODEL_SELECTION: (
        ("model_selection", "models", "selection"),
        ("ModelSelectionAgent", "ModelSelectorAgent"),
    ),
    AgentName.EXPERIMENT: (
        ("experiment", "experiments", "training"),
        ("ExperimentAgent", "ExperimentationAgent", "TrainingAgent"),
    ),
    AgentName.TUNING: (
        ("tuning", "tuner", "hpo"),
        ("TuningAgent", "HyperparameterAgent", "HyperparameterOptimizationAgent"),
    ),
    AgentName.EXPLAIN: (
        ("explain", "explainability", "explainer"),
        ("ExplainAgent", "ExplainabilityAgent"),
    ),
    AgentName.EVALUATION: (
        ("evaluation", "evaluate", "evaluator"),
        ("EvaluationAgent", "EvaluatorAgent"),
    ),
    AgentName.INSIGHT: (
        ("insight", "insights", "business_insight"),
        ("InsightAgent", "InsightsAgent", "BusinessInsightAgent"),
    ),
    AgentName.VISUALIZATION: (
        ("visualization", "visualisation", "charts", "viz"),
        ("VisualizationAgent", "VisualisationAgent", "ChartAgent"),
    ),
    AgentName.REPORT: (
        ("report", "reporting", "writer"),
        ("ReportAgent", "ReportingAgent"),
    ),
}

_AGENT_CLASS_CACHE: dict[AgentName, type | None] = {}


def _optional_module(dotted: str) -> Any | None:
    """Import a module that may not exist yet, returning ``None`` if it does not.

    Every sibling layer is optional from the orchestrator's point of view: a
    missing or broken module has to surface as a warning on the step that needed
    it, not as an ImportError at process start.
    """
    try:
        return importlib.import_module(dotted)
    except Exception:  # noqa: BLE001 - absent or broken sibling module
        logger.debug("optional module %s unavailable", dotted)
        return None


def _scan_module_for_agent(module: Any, name: AgentName) -> type | None:
    from ..core.agent import BaseAgent  # deferred: pulls in the Anthropic SDK

    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseAgent)
            and not inspect.isabstract(attr)
            and getattr(attr, "name", None) is name
        ):
            return attr
    return None


def _scan_package_for_agent(name: AgentName) -> type | None:
    try:
        package = importlib.import_module("automl_architect.agents")
    except ImportError:
        return None
    for info in pkgutil.iter_modules(getattr(package, "__path__", [])):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"automl_architect.agents.{info.name}")
        except Exception:  # a half-written sibling module must not break dispatch
            logger.debug("could not import agents.%s while scanning", info.name)
            continue
        found = _scan_module_for_agent(module, name)
        if found is not None:
            return found
    return None


def resolve_agent_class(name: AgentName) -> type | None:
    """Find the class implementing an agent, or ``None`` when it is absent.

    The agents package's own registry is authoritative; the candidate scan below
    it exists so the orchestrator still runs against a partially built or
    reorganised agents layer instead of failing at import time.

    Args:
        name: The agent to locate.

    Returns:
        A concrete ``BaseAgent`` subclass, or ``None`` if no module provides one.
    """
    if name in _AGENT_CLASS_CACHE:
        return _AGENT_CLASS_CACHE[name]

    registry = _optional_module("automl_architect.agents")
    get_agent = getattr(registry, "get_agent", None) if registry else None
    if callable(get_agent):
        try:
            resolved = get_agent(name)
        except Exception:  # noqa: BLE001 - fall through to the scan
            resolved = None
        if isinstance(resolved, type):
            _AGENT_CLASS_CACHE[name] = resolved
            return resolved

    modules, class_names = _AGENT_CANDIDATES.get(name, ((), ()))
    found: type | None = None
    for suffix in modules:
        try:
            module = importlib.import_module(f"automl_architect.agents.{suffix}")
        except ImportError:
            continue
        for class_name in class_names:
            candidate = getattr(module, class_name, None)
            if isinstance(candidate, type):
                found = candidate
                break
        if found is None:
            found = _scan_module_for_agent(module, name)
        if found is not None:
            break

    if found is None:
        found = _scan_package_for_agent(name)

    _AGENT_CLASS_CACHE[name] = found
    return found


# ---------------------------------------------------------------------------
# Replan bookkeeping
# ---------------------------------------------------------------------------

# State slots cleared when a step is re-run, so a replan recomputes rather than
# reusing stale derived data. Frames are nulled deliberately: `RunState.df` falls
# back to the raw frame, so cleaning restarts from the original data.
_RESET_SLOTS: dict[str, tuple[str, ...]] = {
    "clean": (
        "cleaning",
        "working_df",
        "feature_frame",
        "feature_names",
        "applied_cleaning",
        "dropped_columns",
        "preprocessor",
    ),
    "engineer_features": (
        "features",
        "feature_frame",
        "feature_names",
        "applied_features",
        "preprocessor",
    ),
    "split": ("splits",),
    "select_models": ("model_selection",),
    "run_experiments": ("experiments", "best_model", "best_pipeline", "label_encoder"),
    "tune": ("tuning", "tuning_decision"),
    "explain": ("explainability",),
    EVALUATION_STEP_ID: ("evaluation",),
    "insights": ("insights",),
    "visualise": ("visualization_plan", "visualizations"),
    "report": ("report", "report_bundle"),
}

_SLOT_DEFAULTS: dict[str, Callable[[], Any]] = {
    "splits": DataSplits,
    "feature_names": list,
    "applied_cleaning": list,
    "applied_features": list,
    "dropped_columns": list,
}

# ``state.extras`` keys each step's executor owns, cleared alongside that step's
# slots. Not optional bookkeeping: consumers cache what they find there — the
# Evaluation Agent reuses ``extras['diagnostics']`` rather than recomputing, and
# the splitter writes ``class_names``/``positive_label`` with ``setdefault`` — so
# a key left behind would make the replanned attempt reason over the discarded
# one's numbers.
_RESET_EXTRA_KEYS: dict[str, tuple[str, ...]] = {
    "clean": (
        "cleaning_fill_values",
        "cleaning_clip_bounds",
        "cleaning_dtype_casts",
        "cleaning_datetime_columns",
        "cleaning_imputers",
    ),
    "engineer_features": (
        "feature_encodings",
        "feature_excluded_columns",
        "feature_preprocessor_steps",
        "feature_column_specs",
    ),
    "split": (
        "split_strategy",
        "split_sizes",
        "split_feature_columns",
        "class_names",
        "positive_label",
    ),
    EVALUATION_STEP_ID: ("diagnostics",),
}

# Slots compared before and after a replan to decide which model ships.
_OUTCOME_SLOTS: tuple[str, ...] = (
    "cleaning",
    "features",
    "model_selection",
    "experiments",
    "tuning_decision",
    "tuning",
    "explainability",
    "evaluation",
    "best_model",
    "best_pipeline",
    "preprocessor",
    "label_encoder",
    "working_df",
    "feature_frame",
    "feature_names",
    "splits",
    # The audit trail of what was actually applied has to travel with the model
    # it describes: the report renders these next to the decisions that produced
    # them, so restoring one without the other would document transformations the
    # shipped model never saw.
    "applied_cleaning",
    "applied_features",
    "dropped_columns",
    "extras",
)

# Outcome slots holding a mutable container the next attempt writes into, so the
# snapshot must hold a copy rather than a reference.
_COPIED_SLOTS: frozenset[str] = frozenset(
    {"applied_cleaning", "applied_features", "dropped_columns", "extras"}
)

_DESTRUCTIVE_CLEANING: frozenset[CleaningAction] = frozenset(
    {
        CleaningAction.DROP_COLUMN,
        CleaningAction.DROP_DUPLICATE_ROWS,
        CleaningAction.DROP_ROWS_MISSING_TARGET,
        CleaningAction.REMOVE_OUTLIER_ROWS,
        CleaningAction.DROP_CONSTANT_COLUMN,
        CleaningAction.DROP_LEAKAGE_COLUMN,
    }
)

_DESTRUCTIVE_FEATURE_OPS: frozenset[FeatureOp] = frozenset(
    {
        FeatureOp.DROP_CORRELATED,
        FeatureOp.VARIANCE_THRESHOLD,
        FeatureOp.SELECT_K_BEST,
        FeatureOp.PCA,
        FeatureOp.SVD,
    }
)


class _StepSkipped(Exception):
    """Internal signal: a handler declined to act, with a reason to record."""


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------


def state_from_summary(
    summary: RunSummary,
    *,
    settings: Settings | None = None,
    bus: EventBus | None = None,
) -> RunState:
    """Rebuild a read-only :class:`RunState` from a persisted summary.

    Fitted models and dataframes are not serialised, so the result supports
    narration and question answering, not further training.

    Args:
        summary: A previously persisted run.
        settings: Settings override.
        bus: Event bus to attach; a fresh one is created when omitted.

    Returns:
        A state whose agent-output slots mirror the summary, with the run context
        re-frozen so agents see the same cached prefix the run used.
    """
    resolved = settings or get_settings()
    state = RunState(
        config=summary.config,
        settings=resolved,
        bus=bus or EventBus(summary.run_id),
    )
    state.status = summary.status
    state.started_at = summary.started_at
    state.finished_at = summary.finished_at
    state.error = summary.error
    for slot in (
        "ingestion",
        "profile",
        "understanding",
        "problem",
        "plan",
        "cleaning",
        "features",
        "model_selection",
        "experiments",
        "tuning_decision",
        "tuning",
        "explainability",
        "evaluation",
        "insights",
        "visualization_plan",
        "visualizations",
        "report",
        "report_bundle",
    ):
        setattr(state, slot, getattr(summary, slot))
    state.plan_history = list(summary.plan_history)
    state.steps = list(summary.steps)
    state.approvals = list(summary.approvals)
    state.usage = summary.usage
    state.replans = summary.replans
    state.warnings = list(summary.warnings)
    if summary.profile is not None:
        state.freeze_context(build_run_context(summary.profile, summary.config))
    return state


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runs one analysis end to end.

    Args:
        config: The run configuration, including its data source.
        settings: Process settings; the cached singleton by default.
        repository: Optional persistence. When supplied, the run summary, event
            log, and dataset fingerprint are stored, and dataset memory is
            consulted before planning.
        bus: Optional event bus, for a caller that already has subscribers.

    Example:
        >>> orch = Orchestrator(config)              # doctest: +SKIP
        >>> summary = orch.run()                     # doctest: +SKIP
        >>> if summary.status is RunStatus.AWAITING_APPROVAL:  # doctest: +SKIP
        ...     request = summary.approvals[-1]
        ...     summary = orch.resume(request.request_id, approved=True)
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        settings: Settings | None = None,
        repository: Any | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository
        self._bus = bus or EventBus(config.run_id)
        self._state = RunState(config=config, settings=self._settings, bus=self._bus)

        self.cancel_event = threading.Event()

        self._retry_policy = RetryPolicy()
        self._budget_policy = BudgetPolicy()
        self._approval_policy = ApprovalPolicy(require_approval=config.require_approval)
        self._replan_policy = ReplanPolicy(
            max_replans=config.max_replans,
            enabled=config.enable_self_improvement,
        )

        self._agents: dict[AgentName, Any] = {}
        self._memory: Any | None = None
        self._memory_cached = False
        self._reported_notes: set[str] = set()
        self._events_persisted = 0
        self._suspended_at: float | None = None
        self._pending_comparison: tuple[dict[str, Any], float | None] | None = None
        self._loop_locked = False

        self._dispatch: dict[str, StepHandler] = {
            "ingest": self._h_ingest,
            "profile": self._h_profile,
            "understand": self._h_understand,
            "identify_problem": self._h_identify_problem,
            "plan": self._h_plan,
            "clean": self._h_clean,
            "engineer_features": self._h_engineer_features,
            "split": self._h_split,
            "select_models": self._h_select_models,
            "run_experiments": self._h_run_experiments,
            "tune": self._h_tune,
            "explain": self._h_explain,
            EVALUATION_STEP_ID: self._h_evaluate,
            "insights": self._h_insights,
            "visualise": self._h_visualise,
            "report": self._h_report,
        }

    # -- public surface ----------------------------------------------------

    @property
    def state(self) -> RunState:
        """The live run blackboard."""
        return self._state

    @property
    def bus(self) -> EventBus:
        """The run's event bus, for streaming subscribers."""
        return self._bus

    def cancel(self) -> None:
        """Ask the run to stop at the next step boundary."""
        self.cancel_event.set()
        self._bus.log("cancellation requested")

    def run(self) -> RunSummary:
        """Execute the pipeline.

        Returns:
            The run summary. ``status`` is ``COMPLETED``, ``FAILED``,
            ``CANCELLED``, or ``AWAITING_APPROVAL`` — the last meaning a
            destructive step needs a human and :meth:`resume` should be called.
        """
        if self._state.status is RunStatus.AWAITING_APPROVAL:
            raise ConfigurationError(
                f"run {self._state.run_id} is awaiting approval; call resume() instead"
            )
        if self._state.started_at is None:
            self._state.mark_started()
            self._bus.emit(
                EventKind.RUN_STARTED,
                f"run {self._state.run_id} started ({self._state.config.project})",
                payload={
                    "project": self._state.config.project,
                    "source_kind": self._state.config.source.kind.value,
                    "time_budget_seconds": self._state.config.time_budget_seconds,
                },
            )
        return self._drive()

    def resume(
        self, request_id: str, approved: bool, note: str | None = None
    ) -> RunSummary:
        """Record an approval decision and continue the suspended run.

        Args:
            request_id: Id of the request the run suspended on.
            approved: Whether the destructive step may proceed.
            note: Optional justification, kept in the audit trail.

        Returns:
            The summary of the continued run.
        """
        self._approval_policy.resolve(self._state, request_id, approved, note)
        if not approved:
            self._state.add_warning(
                f"approval {request_id} was rejected"
                f"{f' ({note})' if note else ''}; the step will be skipped"
            )
        self._discount_suspension()
        self._state.status = RunStatus.RUNNING
        self._state.finished_at = None
        if self._state.started_at is None:  # resumed without ever having run
            self._state.mark_started()
        return self._drive()

    # -- driver ------------------------------------------------------------

    def _drive(self) -> RunSummary:
        try:
            self._prologue()
            self._plan_loop()
        except ApprovalRequired as exc:
            self._suspended_at = time.monotonic()
            self._state.upsert_step(
                exc.step_id, status=StepStatus.AWAITING_APPROVAL, summary=exc.summary
            )
            self._state.status = RunStatus.AWAITING_APPROVAL
            self._bus.log(f"run suspended awaiting approval: {exc.summary}")
            return self._finalise()
        except RunCancelled as exc:
            self._state.mark_finished(RunStatus.CANCELLED, str(exc))
            self._bus.emit(EventKind.RUN_CANCELLED, str(exc))
            return self._finalise()
        except Exception as exc:  # noqa: BLE001 - a run must always end in a summary
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("run %s failed", self._state.run_id)
            self._state.mark_finished(RunStatus.FAILED, message)
            self._bus.emit(EventKind.RUN_FAILED, message)
            return self._finalise()

        self._state.mark_finished(RunStatus.COMPLETED)
        self._bus.emit(
            EventKind.RUN_COMPLETED,
            f"run {self._state.run_id} completed in "
            f"{self._state.elapsed_seconds:.1f}s",
            payload={
                "replans": self._state.replans,
                "warnings": len(self._state.warnings),
                "cost_usd": round(self._state.usage.cost_usd, 6),
            },
        )
        return self._finalise()

    def _prologue(self) -> None:
        """Fixed steps that must precede any plan."""
        self._run_step(require_step("ingest"))
        self._run_step(require_step("profile"))
        # Exactly one freeze, before the first agent, so every agent in the run
        # shares one cached prompt prefix.
        self._freeze_context()
        self._run_step(require_step("understand"))
        self._run_step(require_step("identify_problem"))
        self._consult_memory()
        self._run_step(require_step("plan"))

    def _plan_loop(self) -> None:
        """Execute the plan, replanning while the evaluation says to."""
        reconciliation = self._reconcile()
        while True:
            self._execute(reconciliation.head())

            if self._pending_comparison is not None:
                snapshot, previous = self._pending_comparison
                self._pending_comparison = None
                if not self._keep_better(snapshot, previous):
                    # The restored model already failed its own evaluation once;
                    # looping again would just repeat the same replan.
                    self._loop_locked = True

            decision = self._replan_decision()
            if decision.reason:
                self._bus.log(f"replan policy: {decision.reason}")
            if not decision.should_replan or self._loop_locked:
                self._note_shipping_verdict(decision)
                break

            self._replan(decision)
            reconciliation = self._reconcile()

        self._execute(reconciliation.tail())

    def _finalise(self) -> RunSummary:
        """Persist everything and return the summary."""
        self._persist(self._state.to_summary())
        # Written after the repository so the on-disk copy also carries any
        # warning that persistence itself produced.
        try:
            self._state.save_summary()
        except Exception as exc:  # noqa: BLE001 - persistence must not mask results
            self._state.add_warning(f"could not write run_summary.json: {exc}")
        return self._state.to_summary()

    def _persist(self, summary: RunSummary) -> None:
        repository = self._repository
        if repository is None:
            return
        try:
            repository.save_run(summary)
        except Exception as exc:  # noqa: BLE001
            self._state.add_warning(f"could not save the run to storage: {exc}")
        try:
            fresh = self._bus.since(self._events_persisted)
            if fresh:
                repository.append_events(self._state.run_id, fresh)
                self._events_persisted = fresh[-1].sequence
        except Exception as exc:  # noqa: BLE001
            self._state.add_warning(f"could not save the event log: {exc}")

        if self._state.status is RunStatus.COMPLETED:
            self._register_fingerprint(summary)

    # -- step dispatch -----------------------------------------------------

    def _execute(self, steps: Sequence[ResolvedStep]) -> None:
        for resolved in steps:
            self._run_step(resolved.definition, resolved)

    def _run_step(
        self, definition: StepDefinition, resolved: ResolvedStep | None = None
    ) -> StepStatus:
        """Run one step through the budget, approval, and retry policies."""
        self._check_cancelled()
        state = self._state
        existing = state.step(definition.step_id)
        if existing is not None and existing.status in (
            StepStatus.COMPLETED,
            StepStatus.SKIPPED,
        ):
            return existing.status

        plan_step = resolved.plan_step if resolved else None
        title = resolved.title if resolved else definition.title
        state.upsert_step(definition.step_id, title=title, agent=definition.agent)

        gate = self._feature_gate(definition)
        if gate is not None:
            return self._skip(definition, gate)

        verdict = self._budget_policy.decide(state, definition)
        if not verdict.allowed:
            # A budget skip is lost capability, not an operator choice, so it
            # belongs in the run's warnings where a report reader will see it.
            state.add_warning(verdict.reason)
            return self._skip(definition, verdict.reason)
        if verdict.over_budget and verdict.reason:
            state.add_warning(verdict.reason)

        handler = self._dispatch.get(definition.step_id)
        if handler is None:  # pragma: no cover - the dispatch table is exhaustive
            return self._skip(
                definition, "no executor is bound to this step in the engine"
            )

        attempt = 0
        while True:
            attempt += 1
            self._check_cancelled()
            state.upsert_step(definition.step_id, status=StepStatus.RUNNING)
            self._bus.emit(
                EventKind.STEP_STARTED,
                f"{title} (attempt {attempt})",
                agent=definition.agent,
                step_id=definition.step_id,
                payload={"budget": self._budget_policy.summary(state)},
            )
            started = time.perf_counter()
            try:
                summary = handler(plan_step) or ""
            except _StepSkipped as exc:
                return self._skip(definition, str(exc))
            except (ApprovalRequired, RunCancelled):
                raise
            except ApprovalRejected as exc:
                if definition.skippable:
                    return self._skip(definition, f"human rejected the step: {exc}")
                raise
            except Exception as exc:  # noqa: BLE001 - policy decides what happens
                if self._retry_policy.should_retry(exc, attempt):
                    delay = self._retry_policy.delay_for(attempt)
                    state.upsert_step(definition.step_id, error=str(exc))
                    self._bus.emit(
                        EventKind.STEP_RETRIED,
                        f"{title} failed ({type(exc).__name__}: {exc}); "
                        f"retrying in {delay:.1f}s",
                        agent=definition.agent,
                        step_id=definition.step_id,
                    )
                    time.sleep(delay)
                    continue
                return self._fail(definition, exc)

            state.upsert_step(
                definition.step_id,
                status=StepStatus.COMPLETED,
                summary=summary,
                error=None,
            )
            self._bus.emit(
                EventKind.STEP_COMPLETED,
                f"{title}: {summary}" if summary else title,
                agent=definition.agent,
                step_id=definition.step_id,
                duration_seconds=time.perf_counter() - started,
            )
            return StepStatus.COMPLETED

    def _skip(self, definition: StepDefinition, reason: str) -> StepStatus:
        self._state.upsert_step(
            definition.step_id, status=StepStatus.SKIPPED, summary=reason
        )
        self._bus.emit(
            EventKind.STEP_SKIPPED,
            f"{definition.title} skipped: {reason}",
            agent=definition.agent,
            step_id=definition.step_id,
        )
        logger.info("step %s skipped: %s", definition.step_id, reason)
        return StepStatus.SKIPPED

    def _fail(self, definition: StepDefinition, exc: BaseException) -> StepStatus:
        message = f"{type(exc).__name__}: {exc}"
        self._state.upsert_step(
            definition.step_id, status=StepStatus.FAILED, error=message
        )
        self._bus.emit(
            EventKind.STEP_FAILED,
            f"{definition.title} failed: {message}",
            agent=definition.agent,
            step_id=definition.step_id,
        )
        logger.error("step %s failed: %s", definition.step_id, message, exc_info=exc)
        if definition.critical:
            raise StepFailedError(definition.step_id, message) from exc
        self._state.add_warning(
            f"step '{definition.step_id}' failed ({message}); the run continued with "
            "reduced capability"
        )
        return StepStatus.FAILED

    def _feature_gate(self, definition: StepDefinition) -> str | None:
        """Config-level opt-outs, checked before any work is done."""
        config = self._state.config
        if definition.step_id == "tune" and not config.enable_tuning:
            return "hyperparameter tuning is disabled in the run config"
        if definition.step_id == "explain" and not config.enable_explainability:
            return "explainability is disabled in the run config"
        if definition.step_id == "report" and not config.report_formats:
            return "no report formats were requested in the run config"
        return None

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled(f"run {self._state.run_id} was cancelled by the caller")

    def _discount_suspension(self) -> None:
        """Exclude human deliberation time from the wall-clock budget.

        A run suspended for an hour must not resume with a spent budget: the time
        was not compute, and penalising it would make approvals unusable.
        """
        if self._suspended_at is None:
            return
        waited = time.monotonic() - self._suspended_at
        self._suspended_at = None
        start = getattr(self._state, "_monotonic_start", None)
        if start is None:
            return
        self._state._monotonic_start = start + waited
        self._bus.log(
            f"resumed after {waited:.0f}s awaiting approval; that time was not "
            "charged to the run's time budget"
        )

    # -- context, memory, fingerprint --------------------------------------

    def _freeze_context(self) -> None:
        profile = self._state.profile
        if profile is None:
            self._state.add_warning(
                "no dataset profile was produced, so agents will reason without "
                "measured facts"
            )
            return
        digest = build_run_context(profile, self._state.config)
        self._state.freeze_context(digest)
        self._bus.log(f"run context frozen ({len(digest):,} characters)")

    def _memory_service(self) -> Any | None:
        """A :class:`DatasetMemory` bound to this run's repository, if available."""
        if self._repository is None:
            return None
        if self._memory_cached:
            return self._memory
        self._memory_cached = True
        module = _optional_module("automl_architect.storage.memory")
        factory = getattr(module, "DatasetMemory", None) if module else None
        if factory is None:
            self._state.add_warning(
                "dataset memory is unavailable in this installation; the run cannot "
                "learn from previous runs"
            )
            return None
        try:
            self._memory = factory(self._repository, settings=self._settings)
        except Exception as exc:  # noqa: BLE001 - memory is an optimisation
            self._state.add_warning(f"dataset memory could not be opened: {exc}")
            self._memory = None
        return self._memory

    def _consult_memory(self) -> None:
        """Attach a :class:`MemorySuggestion` from prior runs, if any exist."""
        if self._state.profile is None:
            return
        memory = self._memory_service()
        if memory is None:
            return
        try:
            suggestion = memory.lookup_for_state(self._state, limit=5)
        except Exception as exc:  # noqa: BLE001 - precedent is never required
            self._state.add_warning(f"dataset memory lookup failed: {exc}")
            return
        if not isinstance(suggestion, MemorySuggestion):
            return
        self._state.memory = suggestion
        self._bus.log(
            "dataset memory: "
            + (
                f"{len(suggestion.similar_runs)} similar prior run(s) found"
                if suggestion.has_precedent
                else "no comparable prior run"
            )
        )

    def _register_fingerprint(self, summary: RunSummary) -> None:
        """Record this dataset's signature so future runs can find it."""
        if self._repository is None:
            return
        memory = self._memory_service()
        if memory is not None:
            try:
                if memory.remember(summary) is not None:
                    return
            except Exception as exc:  # noqa: BLE001
                self._state.add_warning(f"could not remember this dataset: {exc}")

        fingerprint = self._local_fingerprint()
        if fingerprint is None:
            return
        try:
            self._repository.save_fingerprint(fingerprint)
        except Exception as exc:  # noqa: BLE001
            self._state.add_warning(
                f"could not register the dataset fingerprint: {exc}"
            )

    def _local_fingerprint(self) -> DatasetFingerprint | None:
        """Fingerprint built without the memory module, as a last resort."""
        profile = self._state.profile
        if profile is None:
            return None

        counts: dict[ColumnKind, int] = {}
        for column in profile.columns:
            counts[column.kind] = counts.get(column.kind, 0) + 1

        def total(*kinds: ColumnKind) -> int:
            return sum(counts.get(kind, 0) for kind in kinds)

        best = self._state.experiments.best() if self._state.experiments else None
        features = self._state.features
        return DatasetFingerprint(
            run_id=self._state.run_id,
            project=self._state.config.project,
            n_rows=profile.n_rows,
            n_columns=profile.n_columns,
            n_numeric=total(
                ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE
            ),
            n_categorical=total(
                ColumnKind.CATEGORICAL_NOMINAL,
                ColumnKind.CATEGORICAL_ORDINAL,
                ColumnKind.BOOLEAN,
            ),
            n_datetime=total(ColumnKind.DATETIME),
            n_text=total(ColumnKind.TEXT),
            missing_fraction=profile.missing_cell_fraction,
            duplicate_fraction=profile.duplicate_fraction,
            task_type=self._state.task_type,
            target_kind=profile.target.kind if profile.target else None,
            imbalance_ratio=profile.target.imbalance_ratio if profile.target else None,
            column_names=[c.name for c in profile.columns],
            primary_metric=self._state.primary_metric,
            best_score=self._best_score(),
            best_family=best.family if best else None,
            winning_feature_ops=(
                [d.op.value for d in features.decisions] if features else []
            ),
        )

    # -- self-improvement --------------------------------------------------

    def _reconcile(self) -> PlanReconciliation:
        reconciliation = reconcile_plan(self._state.plan)
        for note in reconciliation.notes:
            if note not in self._reported_notes:
                self._reported_notes.add(note)
                self._bus.log(f"plan reconciliation: {note}")
        for planned, reason in reconciliation.unmapped:
            if self._state.step(planned.step_id) is not None:
                continue
            self._state.upsert_step(
                planned.step_id,
                title=planned.title,
                agent=planned.agent,
                status=StepStatus.SKIPPED,
                summary=reason,
            )
            self._bus.emit(
                EventKind.STEP_SKIPPED,
                f"planned step '{planned.step_id}' skipped: {reason}",
                agent=planned.agent,
                step_id=planned.step_id,
            )
            self._state.add_warning(
                f"the plan contained step '{planned.step_id}' ({planned.title}) which "
                f"maps to no executor; it was skipped"
            )
        return reconciliation

    def _replan_decision(self) -> ReplanDecision:
        return self._replan_policy.decide(
            self._state.evaluation,
            replans_done=self._state.replans,
            min_acceptable_score=self._state.config.min_acceptable_score,
            best_score=self._best_score(),
            higher_is_better=self._higher_is_better(),
            time_remaining=self._state.time_remaining,
        )

    def _replan(self, decision: ReplanDecision) -> None:
        """Revise the plan and rewind to the step the verdict pointed at."""
        state = self._state
        previous_score = self._best_score()
        snapshot = self._snapshot_outcome()
        prior_plan = state.plan
        restart = decision.restart_step_id or "clean"

        if prior_plan is not None:
            state.plan_history.append(prior_plan)
        state.replans += 1
        state.status = RunStatus.REPLANNING
        self._bus.emit(
            EventKind.REPLAN_TRIGGERED,
            decision.reason,
            step_id=restart,
            payload={
                "replan": state.replans,
                "restart_from": restart,
                "previous_score": previous_score,
                "recommended_action": (
                    str(state.evaluation.recommended_action) if state.evaluation else ""
                ),
            },
        )

        # Re-plan *before* rewinding: the Planner must still see the evaluation
        # that rejected this model, and the reset clears it.
        state.plan = None
        self._reset_step_record("plan")
        try:
            self._run_step(require_step("plan"))
        except StepFailedError as exc:
            # Planning is critical on the first pass — without a plan there is
            # nothing to run. On a replan there already is a trained, evaluated
            # model, and failing the whole run would throw it away to punish an
            # optional second attempt. Keep the model, keep the old plan, stop
            # looping.
            state.plan = prior_plan
            state.status = RunStatus.RUNNING
            self._loop_locked = True
            state.add_warning(
                f"replan #{state.replans} could not produce a revised plan ({exc}); "
                "the run kept the plan and the model it already had"
            )
            return
        new_plan = state.plan
        if new_plan is None:
            state.plan = prior_plan
            state.add_warning(
                "the replan produced no new plan; the previous plan was reused"
            )
        elif prior_plan is not None:
            if new_plan.revision <= prior_plan.revision:
                new_plan.revision = prior_plan.revision + 1
            if not new_plan.revision_reason:
                new_plan.revision_reason = decision.reason

        self._reset_from(restart)
        state.status = RunStatus.RUNNING
        self._pending_comparison = (snapshot, previous_score)

    def _note_shipping_verdict(self, decision: ReplanDecision) -> None:
        verdict = self._state.evaluation
        if verdict is None or verdict.acceptable:
            return
        self._state.add_warning(
            f"the shipped model was graded {verdict.overall_grade} and judged "
            f"unacceptable by evaluation: {verdict.verdict_rationale[:300]} "
            f"({decision.reason})"
        )

    def _snapshot_outcome(self) -> dict[str, Any]:
        """Capture the current modelling outcome.

        References, not copies, for the frames and fitted objects: executors under
        pandas copy-on-write rebind ``state.working_df``/``feature_frame`` rather
        than mutating them in place, so holding the old objects is enough to
        restore the old outcome. The containers in :data:`_COPIED_SLOTS` are the
        exception — those are appended to in place, so they are shallow-copied.
        """
        snapshot: dict[str, Any] = {}
        for slot in _OUTCOME_SLOTS:
            value = getattr(self._state, slot)
            snapshot[slot] = copy.copy(value) if slot in _COPIED_SLOTS else value
        return snapshot

    def _keep_better(
        self, snapshot: dict[str, Any], previous_score: float | None
    ) -> bool:
        """Keep whichever of the two attempts scored better. Returns True if new."""
        new_score = self._best_score()
        metric = self._state.primary_metric
        higher = self._higher_is_better()

        if previous_score is None:
            return True
        if new_score is None:
            for slot, value in snapshot.items():
                setattr(self._state, slot, value)
            self._state.add_warning(
                f"replan #{self._state.replans} produced no scored model; the previous "
                f"model ({metric}={previous_score:.6g}) was restored and shipped"
            )
            return False

        improved = new_score > previous_score if higher else new_score < previous_score
        if improved:
            self._bus.log(
                f"replan #{self._state.replans} improved {metric} from "
                f"{previous_score:.6g} to {new_score:.6g}"
            )
            return True

        for slot, value in snapshot.items():
            setattr(self._state, slot, value)
        self._state.add_warning(
            f"replan #{self._state.replans} produced a worse model "
            f"({metric}={new_score:.6g} vs {previous_score:.6g}); the earlier model "
            "was restored and is the one reported"
        )
        return False

    def _reset_from(self, step_id: str) -> None:
        start = step_index(step_id)
        if start < 0:
            return
        for definition in canonical_steps():
            if not definition.planned or step_index(definition.step_id) < start:
                continue
            self._reset_step_record(definition.step_id)
            for slot in _RESET_SLOTS.get(definition.step_id, ()):
                factory = _SLOT_DEFAULTS.get(slot)
                setattr(self._state, slot, factory() if factory else None)
            for key in _RESET_EXTRA_KEYS.get(definition.step_id, ()):
                self._state.extras.pop(key, None)

    def _reset_step_record(self, step_id: str) -> None:
        record = self._state.step(step_id)
        if record is None:
            return
        record.status = StepStatus.PENDING
        record.error = None
        record.summary = ""
        record.finished_at = None

    # -- scoring helpers ---------------------------------------------------

    def _higher_is_better(self) -> bool:
        log = self._state.experiments
        if log is not None and log.primary_metric:
            return bool(log.higher_is_better)
        module = _optional_module("automl_architect.execution.metrics")
        fn = getattr(module, "higher_is_better", None) if module else None
        if callable(fn):
            try:
                return bool(fn(self._state.primary_metric))
            except Exception:  # noqa: BLE001 - fall back to the common case
                logger.debug("higher_is_better() failed; assuming higher is better")
        return True

    def _best_score(self) -> float | None:
        higher = self._higher_is_better()
        candidates: list[float] = []

        log = self._state.experiments
        if log is not None:
            best = log.best()
            if best is not None and best.primary_score is not None:
                candidates.append(best.primary_score)
            else:
                scored = [
                    r.primary_score
                    for r in log.results
                    if r.primary_score is not None and not r.failed
                ]
                if scored:
                    candidates.append(max(scored) if higher else min(scored))

        tuning = self._state.tuning
        if tuning is not None and tuning.ran and tuning.best_score is not None:
            candidates.append(tuning.best_score)

        if not candidates:
            return None
        return max(candidates) if higher else min(candidates)

    # -- lazy resolution ---------------------------------------------------

    def _agent(self, name: AgentName) -> Any:
        """Instantiate an agent, raising :class:`ConfigurationError` if absent."""
        cached = self._agents.get(name)
        if cached is not None:
            return cached
        cls = resolve_agent_class(name)
        if cls is None:
            raise ConfigurationError(
                f"the '{name.value}' agent is not available in this installation"
            )
        try:
            agent = cls()
        except Exception as exc:  # noqa: BLE001 - usually a missing API credential
            raise ConfigurationError(
                f"the '{name.value}' agent could not be created: {exc}"
            ) from exc
        self._agents[name] = agent
        return agent

    def _agent_or_none(self, name: AgentName) -> Any | None:
        try:
            return self._agent(name)
        except ConfigurationError as exc:
            self._state.add_warning(str(exc))
            return None

    def _invoke(self, agent: Any, *, slot: str | None = None) -> Any:
        """Run an agent and make sure its result landed on the state."""
        value = agent.run(self._state)
        if slot is not None and getattr(self._state, slot, None) is None:
            setattr(self._state, slot, value)
            self._state.add_warning(
                f"{getattr(agent, 'title', type(agent).__name__)} did not write its "
                f"result to state.{slot}; the orchestrator stored it directly"
            )
        return value

    # -- step handlers -----------------------------------------------------

    def _h_ingest(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.raw_df is not None and state.ingestion is not None:
            return "reused the already-loaded dataset"

        from ..ingestion.router import load_source

        frame, result = load_source(state.config.source, max_rows=state.config.max_rows)

        state.raw_df = frame
        state.ingestion = result
        for problem in result.validation_errors:
            state.add_warning(f"ingestion validation error: {problem}")
        for problem in result.validation_warnings:
            state.add_warning(f"ingestion warning: {problem}")
        if result.truncated:
            state.add_warning(
                f"the source was truncated to {result.n_rows:,} rows by the "
                "max_rows limit, so all statistics describe that sample"
            )
        return f"loaded {result.n_rows:,} rows x {result.n_columns} columns"

    def _h_profile(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.profile is not None:
            return "reused the existing profile"
        if state.raw_df is None:
            raise ConfigurationError("cannot profile: no data was loaded")

        target = state.config.target_column
        if target is not None and target not in set(map(str, state.raw_df.columns)):
            state.add_warning(
                f"the configured target column '{target}' is not in the data; "
                "profiling without a target"
            )
            target = None

        from ..profiling.profiler import profile_dataframe

        kwargs: dict[str, Any] = {"target": target, "settings": self._settings}
        # Pass the ingestion's dataset id through when the profiler accepts it, so
        # the two records describe the same dataset instead of inventing two ids.
        extras = {
            "dataset_id": state.ingestion.dataset_id if state.ingestion else None,
            "cv_folds": state.config.cv_folds,
        }
        accepted = inspect.signature(profile_dataframe).parameters
        kwargs.update({k: v for k, v in extras.items() if k in accepted and v})

        profile = profile_dataframe(state.raw_df, **kwargs)
        state.profile = profile
        for issue in profile.quality_issues:
            if issue.severity in (Severity.CRITICAL, Severity.HIGH):
                state.add_warning(f"data quality [{issue.code}]: {issue.detail}")
        return (
            f"profiled {profile.n_columns} columns; "
            f"{profile.missing_cell_fraction:.2%} of cells missing, "
            f"{profile.n_duplicate_rows:,} duplicate rows"
        )

    def _h_understand(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.understanding is None:
            agent = self._agent_or_none(AgentName.DATASET)
            if agent is None:
                raise _StepSkipped(
                    "the Dataset Understanding Agent is unavailable; the run "
                    "continues without an exploratory narrative"
                )
            self._invoke(agent, slot="understanding")
        understanding = state.understanding
        if understanding is None:  # pragma: no cover - defensive
            raise _StepSkipped("no dataset understanding was produced")
        if understanding.data_readiness == "unusable":
            state.add_warning(
                "the Dataset Agent judged this data unusable: "
                f"{understanding.readiness_rationale[:300]}"
            )
        return f"{understanding.likely_domain}: {understanding.headline}"

    def _h_identify_problem(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.problem is None:
            self._invoke(self._agent(AgentName.PROBLEM), slot="problem")
        problem = state.problem
        if problem is None:
            raise ConfigurationError("the Problem Agent produced no task definition")
        if not problem.task_type.is_supported:
            state.add_warning(
                f"task type '{problem.task_type.value}' is not fully supported by the "
                "execution layer; training may be limited"
            )
        override = state.config.task_type_override
        if override is not None and override is not problem.task_type:
            state.add_warning(
                f"the operator forced task type '{override.value}' but the Problem "
                f"Agent concluded '{problem.task_type.value}'; the override wins"
            )
        return (
            f"{problem.task_type.value} on "
            f"'{problem.target_column or 'no target'}', "
            f"optimising {problem.primary_metric} (confidence {problem.confidence})"
        )

    def _h_plan(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.plan is None:
            self._invoke(self._agent(AgentName.PLANNER), slot="plan")
        plan = state.plan
        if plan is None:
            raise ConfigurationError("the Planner Agent produced no execution plan")
        return f"{len(plan.steps)} step plan (revision {plan.revision})"

    def _h_clean(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.cleaning is None:
            agent = self._agent_or_none(AgentName.CLEANING)
            if agent is None:
                state.working_df = state.raw_df
                raise _StepSkipped(
                    "the Cleaning Agent is unavailable; modelling proceeds on the "
                    "raw data"
                )
            self._invoke(agent, slot="cleaning")

        plan = state.cleaning
        if plan is None:  # pragma: no cover - defensive
            state.working_df = state.raw_df
            raise _StepSkipped("no cleaning plan was produced")

        self._request_cleaning_approval(plan, plan_step)

        from ..execution.cleaning_ops import apply_cleaning_plan

        frame = apply_cleaning_plan(state, plan)
        if state.working_df is None:
            state.working_df = frame if frame is not None else state.raw_df
        rows = int(len(state.working_df)) if state.working_df is not None else 0
        return (
            f"applied {len(plan.decisions)} cleaning decision(s); "
            f"{rows:,} rows x "
            f"{len(state.working_df.columns) if state.working_df is not None else 0} "
            "columns remain"
        )

    def _request_cleaning_approval(
        self, plan: CleaningPlan, plan_step: PlanStep | None
    ) -> None:
        destructive = [
            d
            for d in plan.decisions
            if d.destructive
            or d.action in _DESTRUCTIVE_CLEANING
            or d.strategy
            in (MissingStrategy.DROP_COLUMN, MissingStrategy.DROP_ROWS)
        ]
        columns = list(
            dict.fromkeys(
                list(plan.columns_to_drop)
                + [c for d in destructive for c in d.columns]
            )
        )
        if not destructive and not columns:
            return

        rows = 0
        profile = self._state.profile
        if profile is not None:
            actions = {d.action for d in destructive}
            if CleaningAction.DROP_DUPLICATE_ROWS in actions:
                rows += profile.n_duplicate_rows
            if CleaningAction.DROP_ROWS_MISSING_TARGET in actions and profile.target:
                rows += profile.target.n_missing

        self._approval_policy.check(
            self._state,
            step_id="clean",
            agent=AgentName.CLEANING,
            action_summary=(
                f"Cleaning will drop {len(columns)} column(s) and about {rows:,} "
                f"row(s). {plan.summary[:240]}"
            ),
            details=[
                f"{d.action.value} on {', '.join(d.columns) or 'the whole table'}: "
                f"{d.rationale}"
                for d in destructive[:20]
            ]
            + list(plan.drop_rationale[:10]),
            affected_columns=columns,
            affected_row_estimate=rows,
            severity=Severity.HIGH if columns else Severity.MEDIUM,
            destructive=True,
        )

    def _h_engineer_features(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.working_df is None:
            state.working_df = state.raw_df

        if state.features is None:
            agent = self._agent_or_none(AgentName.FEATURES)
            if agent is None:
                raise _StepSkipped(
                    "the Feature Engineering Agent is unavailable; modelling proceeds "
                    "on the cleaned columns as they are"
                )
            self._invoke(agent, slot="features")

        plan = state.features
        if plan is None:  # pragma: no cover - defensive
            raise _StepSkipped("no feature plan was produced")

        self._request_feature_approval(plan, plan_step)

        from ..execution.feature_ops import apply_feature_plan

        frame = apply_feature_plan(state, plan)
        if state.feature_frame is None:
            state.feature_frame = frame
        columns = (
            len(state.feature_frame.columns) if state.feature_frame is not None else 0
        )
        return f"applied {len(plan.decisions)} feature op(s); {columns} model columns"

    def _request_feature_approval(
        self, plan: FeaturePlan, plan_step: PlanStep | None
    ) -> None:
        dropping = [d for d in plan.decisions if d.op in _DESTRUCTIVE_FEATURE_OPS]
        if not dropping:
            return
        columns = list(
            dict.fromkeys(c for d in dropping for c in d.input_columns)
        )
        self._approval_policy.check(
            self._state,
            step_id="engineer_features",
            agent=AgentName.FEATURES,
            action_summary=(
                f"Feature engineering will remove or replace columns via "
                f"{', '.join(sorted({d.op.value for d in dropping}))}. "
                f"{plan.summary[:240]}"
            ),
            details=[f"{d.op.value}: {d.rationale}" for d in dropping[:20]],
            affected_columns=columns,
            severity=Severity.MEDIUM,
            destructive=True,
        )

    def _h_split(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.working_df is None:
            state.working_df = state.raw_df

        from ..execution.splitter import make_splits

        splits = make_splits(state)
        state.splits = splits
        sizes = splits.sizes()
        if sizes["train"] == 0:
            raise ConfigurationError(
                "the split produced an empty training set; there is nothing to train on"
            )
        return (
            f"{splits.strategy or 'split'}: train={sizes['train']:,} "
            f"validation={sizes['validation']:,} test={sizes['test']:,}"
        )

    def _h_select_models(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.model_selection is None:
            agent = self._agent_or_none(AgentName.MODEL_SELECTION)
            if agent is not None:
                self._invoke(agent, slot="model_selection")
        if state.model_selection is None:
            state.model_selection = self._fallback_selection()
            state.add_warning(
                "the Model Selection Agent produced nothing, so the executor's "
                "default families for this task were used instead"
            )
        selection = state.model_selection
        families = ", ".join(c.family.value for c in selection.candidates[:6])
        return f"{len(selection.candidates)} candidate(s): {families}"

    def _fallback_selection(self) -> ModelSelection:
        """A defensible default shortlist when the agent is unavailable."""
        families: list[Any] = []
        task = self._state.task_type
        module = _optional_module("automl_architect.execution.model_zoo")
        fn = getattr(module, "available_families", None) if module else None
        if callable(fn) and task is not None:
            try:
                families = list(fn(task))
            except Exception as exc:  # noqa: BLE001
                logger.debug("available_families failed: %s", exc)
        limit = max(1, self._state.config.max_experiments)
        candidates = [
            ModelCandidate(
                family=family,
                rank=index + 1,
                suitability="fair",
                rationale=(
                    "Fallback shortlist: the Model Selection Agent was unavailable, so "
                    "the execution layer's supported families for this task were used "
                    "in their default order. This is not a dataset-specific judgement."
                ),
                is_baseline=getattr(family, "value", "") == "baseline_dummy",
                tune_priority="low",
            )
            for index, family in enumerate(families[:limit])
        ]
        return ModelSelection(
            candidates=candidates,
            summary=(
                f"{len(candidates)} family(ies) selected mechanically because no "
                "reasoning agent was available."
            ),
            reasoning=(
                "No comparative argument is available: this shortlist was produced by "
                "the orchestrator's degradation path, not by a model-selection "
                "judgement over this dataset."
            ),
            validation_strategy=f"{self._state.config.cv_folds}-fold cross-validation",
            validation_rationale="Run-config default; no agent rationale available.",
        )

    def _h_run_experiments(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.experiments is None:
            agent = self._agent_or_none(AgentName.EXPERIMENT)
            if agent is not None:
                self._invoke(agent, slot="experiments")
        if state.experiments is None:
            from ..execution.trainer import run_experiments

            state.experiments = run_experiments(state)
            state.add_warning(
                "the Experiment Agent was unavailable; training ran but the "
                "leaderboard has no narrative interpretation"
            )

        log = state.experiments
        succeeded = [r for r in log.results if not r.failed]
        for failure in (r for r in log.results if r.failed):
            state.add_warning(
                f"model family '{failure.family.value}' failed to train: "
                f"{failure.error or 'unknown error'}"
            )
        if not succeeded:
            raise NoViableModelError(
                f"all {len(log.results)} candidate model(s) failed to train; there is "
                "no model to evaluate or report on"
            )
        best = log.best() or succeeded[0]
        score = self._best_score()
        if score is not None:
            self._bus.metric(log.primary_metric or state.primary_metric, score)
        return (
            f"{len(succeeded)}/{len(log.results)} model(s) trained; best is "
            f"{best.family.value}"
            + (f" at {log.primary_metric or state.primary_metric}={score:.6g}" if score is not None else "")
        )

    def _h_tune(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        definition = require_step("tune")
        if state.tuning_decision is None:
            agent = self._agent_or_none(AgentName.TUNING)
            if agent is None:
                state.tuning = TuningResult(
                    ran=False, skipped_reason="the Tuning Agent was unavailable"
                )
                raise _StepSkipped("the Tuning Agent is unavailable")
            self._invoke(agent, slot="tuning_decision")

        decision = state.tuning_decision
        if decision is None:  # pragma: no cover - defensive
            raise _StepSkipped("no tuning decision was produced")
        if not decision.worthwhile:
            state.tuning = TuningResult(
                ran=False, method=decision.method, skipped_reason=decision.rationale
            )
            raise _StepSkipped(f"tuning judged not worthwhile: {decision.rationale}")

        allowance = self._budget_policy.allowance_for(state, definition)
        applied = decision
        if decision.timeout_seconds > allowance:
            applied = decision.model_copy(
                update={"timeout_seconds": max(5, int(allowance))}
            )
            state.add_warning(
                f"tuning timeout reduced from {decision.timeout_seconds}s to "
                f"{applied.timeout_seconds}s to protect the remaining time budget"
            )

        from ..execution.tuner import run_tuning

        result = run_tuning(state, applied)
        state.tuning = result
        if result.error:
            state.add_warning(f"tuning reported an error: {result.error}")
        if not result.ran:
            raise _StepSkipped(result.skipped_reason or "tuning did not run")

        # Refitting and promoting the tuned model belongs to the tuner: it is the
        # only component that knows whether the tuned candidate beat the whole
        # leaderboard or merely its own untuned self. Refitting here would
        # overwrite state.best_model even when it deliberately declined to.
        gain = f"{result.improvement:+.6g}" if result.improvement is not None else "n/a"
        return (
            f"{result.method.value} over {result.n_trials_completed} trial(s); "
            f"best={result.best_score if result.best_score is not None else 'n/a'} "
            f"(change {gain})"
        )

    def _h_explain(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.explainability is None:
            agent = self._agent_or_none(AgentName.EXPLAIN)
            if agent is not None:
                self._invoke(agent, slot="explainability")
        if state.explainability is None:
            from ..execution.explainer import compute_explanations

            state.explainability = compute_explanations(state)
            state.add_warning(
                "the Explainability Agent was unavailable; attributions were computed "
                "but not narrated"
            )
        report = state.explainability
        if not report.shap_available:
            state.add_warning(
                "SHAP values were unavailable, so feature attributions fall back to "
                "permutation importance"
            )
        top = ", ".join(a.feature for a in report.global_attributions[:5])
        return f"{len(report.global_attributions)} attribution(s); top: {top or 'none'}"

    def _h_evaluate(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.evaluation is None:
            agent = self._agent_or_none(AgentName.EVALUATION)
            if agent is not None:
                self._invoke(agent, slot="evaluation")
        if state.evaluation is None:
            self._compute_diagnostics_only()
            state.add_warning(
                "the Evaluation Agent was unavailable, so no quality gate ran and the "
                "self-improvement loop is disabled for this run"
            )
            raise _StepSkipped("the Evaluation Agent is unavailable")

        verdict = state.evaluation
        score = self._best_score()
        floor = state.config.min_acceptable_score
        if floor is not None and score is not None:
            below = score < floor if self._higher_is_better() else score > floor
            if below:
                state.add_warning(
                    f"best {state.primary_metric}={score:.6g} misses the operator's "
                    f"minimum acceptable score of {floor:.6g}"
                )
        return (
            f"grade {verdict.overall_grade}, "
            f"{'acceptable' if verdict.acceptable else 'not acceptable'}, "
            f"action '{verdict.recommended_action}'"
        )

    def _compute_diagnostics_only(self) -> None:
        """Compute diagnostics for the report even when no agent interprets them."""
        module = _optional_module("automl_architect.execution.diagnostics")
        fn = getattr(module, "compute_diagnostics", None) if module else None
        if not callable(fn):
            return
        try:
            self._state.extras["diagnostics"] = fn(self._state)
        except Exception as exc:  # noqa: BLE001
            logger.debug("diagnostics computation failed: %s", exc)

    def _h_insights(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.insights is None:
            agent = self._agent_or_none(AgentName.INSIGHT)
            if agent is None:
                raise _StepSkipped("the Business Insight Agent is unavailable")
            self._invoke(agent, slot="insights")
        report = state.insights
        if report is None:  # pragma: no cover - defensive
            raise _StepSkipped("no insights were produced")
        return f"{len(report.insights)} business insight(s)"

    def _h_visualise(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.visualization_plan is None:
            agent = self._agent_or_none(AgentName.VISUALIZATION)
            if agent is None:
                raise _StepSkipped("the Visualization Agent is unavailable")
            self._invoke(agent, slot="visualization_plan")

        plan = state.visualization_plan
        if plan is None:  # pragma: no cover - defensive
            raise _StepSkipped("no visualisation plan was produced")

        rendered_plan = plan
        cap = self._budget_policy.chart_budget(state)
        if cap is not None and len(plan.charts) > cap:
            priority = {"high": 0, "medium": 1, "low": 2}
            kept = sorted(plan.charts, key=lambda c: priority.get(c.priority, 3))[:cap]
            dropped = [c.title for c in plan.charts if c not in kept]
            rendered_plan = plan.model_copy(update={"charts": kept})
            state.add_warning(
                f"only {cap} of {len(plan.charts)} charts were rendered because "
                f"{state.time_remaining:.0f}s of budget remain; dropped the "
                f"lowest-priority ones: {dropped[:6]}"
            )

        from ..reporting.charts import render_charts

        bundle = render_charts(state, rendered_plan)
        state.visualizations = bundle
        for artifact in bundle.artifacts:
            path = artifact.html_path or artifact.png_path or artifact.json_path
            if artifact.rendered and path:
                self._bus.artifact(path, kind="chart")
            elif artifact.error:
                state.add_warning(
                    f"chart '{artifact.spec.title}' failed to render: {artifact.error}"
                )
        drawn = sum(1 for a in bundle.artifacts if a.rendered)
        return f"rendered {drawn}/{len(bundle.artifacts)} chart(s)"

    def _h_report(self, plan_step: PlanStep | None = None) -> str:
        state = self._state
        if state.report is None:
            agent = self._agent_or_none(AgentName.REPORT)
            if agent is None:
                raise _StepSkipped("the Report Agent is unavailable")
            self._invoke(agent, slot="report")

        report = state.report
        if report is None:  # pragma: no cover - defensive
            raise _StepSkipped("no report was produced")

        from ..reporting.writer import write_report

        bundle = write_report(state, report, list(state.config.report_formats))
        state.report_bundle = bundle
        for label, path in (
            ("markdown", bundle.markdown_path),
            ("html", bundle.html_path),
            ("pdf", bundle.pdf_path),
            ("pptx", bundle.pptx_path),
            ("json", bundle.json_path),
        ):
            if path:
                self._bus.artifact(path, kind=label)
        for warning in bundle.warnings:
            state.add_warning(f"report writer: {warning}")
        written = [
            label
            for label, path in (
                ("markdown", bundle.markdown_path),
                ("html", bundle.html_path),
                ("pdf", bundle.pdf_path),
                ("pptx", bundle.pptx_path),
                ("json", bundle.json_path),
            )
            if path
        ]
        return f"wrote {', '.join(written) or 'no'} report format(s)"


__all__ = [
    "Orchestrator",
    "resolve_agent_class",
    "state_from_summary",
]
