"""The canonical pipeline graph.

The division of labour here is the point of the module. The Planner Agent decides
**which** steps run and in **what order**; this graph owns **how** each step
executes — which agent reasons, which executor applies the decision, whether the
step may be skipped, and what it depends on. An agent therefore cannot invent an
unexecutable step: :func:`reconcile_plan` maps every planned step onto a
definition here before anything is dispatched, and a planned step that maps to no
definition is recorded as skipped with a warning instead of silently vanishing or
crashing the run.

The mapping is deliberately forgiving. A planner writing ``handle_missing_values``
or ``drop_leaky_columns`` is describing the *cleaning* step at a finer grain than
the executor layer works at, so several planned steps legitimately collapse onto
one definition. Resolution tries, in order: the literal step id, a curated alias,
a keyword signature, and finally the ``agent`` field — which is the most reliable
signal of the four because it is a typed enum the schema forces the planner to
fill, but also the coarsest.

Definitions carry the *name* of the deterministic executor rather than a
callable. Binding lives in :mod:`automl_architect.orchestrator.engine`, which
holds a ``step_id -> bound handler`` dispatch table; keeping this module free of
executor imports is what lets the graph be inspected (by the API, the report, or
a test) without importing pandas or scikit-learn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..core.schemas import AgentName, ExecutionPlan, PlanStep

StepKind = Literal["executor", "agent", "hybrid"]

#: The step whose verdict gates the self-improvement loop. Everything at or
#: before it may be re-run by a replan; everything after it ships once.
EVALUATION_STEP_ID = "evaluate"


@dataclass(frozen=True, slots=True)
class StepDefinition:
    """How one pipeline step executes.

    Attributes:
        step_id: Canonical slug. Stable across runs; used as the dispatch key.
        title: Human-readable label for the UI, events, and step records.
        kind: ``executor`` (pure deterministic code), ``agent`` (reason then
            apply), or ``hybrid`` (compute first, then interpret).
        agent: The owning agent, or ``None`` for pure executor steps.
        executor: Dotted name of the deterministic function that does the work,
            for the audit trail. ``None`` when the step is reasoning only.
        depends_on: Step ids that must run before this one *if they run at all*.
            Dependencies on steps absent from a sequence are ignored rather than
            treated as unsatisfiable.
        skippable: Whether the pipeline is still coherent without this step. A
            non-skippable step missing from a plan is injected back in.
        destructive: Whether the step can drop rows or columns. Gates human
            approval; the handler refines this at runtime from the actual plan.
        critical: Whether failure should abort the run. Non-critical failures
            degrade to a warning and the run continues to a report.
        budget_optional: Whether the budget policy may drop this step when the
            wall clock runs short.
        planned: Whether the Planner is allowed to schedule it. The prologue
            (ingest through plan) runs before a plan exists, so those steps are
            fixed.
        aliases: Alternative step ids treated as exact matches.
        keywords: Substrings that identify this step in a planner-invented slug
            or title.
        estimated_seconds: Rough cost, used for budget messaging only.
    """

    step_id: str
    title: str
    kind: StepKind = "executor"
    agent: AgentName | None = None
    executor: str | None = None
    depends_on: tuple[str, ...] = ()
    skippable: bool = True
    destructive: bool = False
    critical: bool = False
    budget_optional: bool = False
    planned: bool = True
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    estimated_seconds: int = 30

    @property
    def label(self) -> str:
        """``step_id`` plus title, for log lines."""
        return f"{self.step_id} ({self.title})"


CANONICAL_STEPS: tuple[StepDefinition, ...] = (
    StepDefinition(
        step_id="ingest",
        title="Load the dataset",
        kind="executor",
        executor="ingestion.router.load_source",
        skippable=False,
        critical=True,
        planned=False,
        estimated_seconds=20,
        aliases=("load", "load_data", "read_data", "ingestion", "acquire_data"),
        keywords=("ingest", "load", "read_csv", "connect", "fetch", "acquire"),
    ),
    StepDefinition(
        step_id="profile",
        title="Profile the data",
        kind="executor",
        executor="profiling.profiler.profile_dataframe",
        depends_on=("ingest",),
        skippable=False,
        critical=True,
        planned=False,
        estimated_seconds=30,
        aliases=("profiling", "compute_profile", "data_profile"),
        keywords=("profile", "profiling", "describe_data", "summary_statistics"),
    ),
    StepDefinition(
        step_id="understand",
        title="Understand the dataset",
        kind="agent",
        agent=AgentName.DATASET,
        depends_on=("profile",),
        skippable=True,
        planned=False,
        estimated_seconds=60,
        aliases=("understanding", "dataset_understanding", "eda", "explore_data"),
        keywords=("understand", "exploratory", "eda", "explore", "data_review"),
    ),
    StepDefinition(
        step_id="identify_problem",
        title="Identify the problem",
        kind="agent",
        agent=AgentName.PROBLEM,
        depends_on=("profile",),
        skippable=False,
        critical=True,
        planned=False,
        estimated_seconds=45,
        aliases=("problem", "problem_identification", "define_problem", "framing"),
        keywords=(
            "identify_problem",
            "problem_definition",
            "problem_framing",
            "task_type",
            "define_task",
            "frame_problem",
            "metric_selection",
            "choose_metric",
            "target_selection",
        ),
    ),
    StepDefinition(
        step_id="plan",
        title="Plan the analysis",
        kind="agent",
        agent=AgentName.PLANNER,
        depends_on=("identify_problem",),
        skippable=False,
        critical=True,
        planned=False,
        estimated_seconds=60,
        aliases=("planning", "execution_plan", "build_plan"),
        keywords=("plan", "planning", "strategy", "roadmap"),
    ),
    StepDefinition(
        step_id="clean",
        title="Clean the data",
        kind="agent",
        agent=AgentName.CLEANING,
        executor="execution.cleaning_ops.apply_cleaning_plan",
        depends_on=("plan",),
        skippable=True,
        destructive=True,
        estimated_seconds=60,
        aliases=(
            "cleaning",
            "data_cleaning",
            "clean_data",
            "preprocess",
            "preprocessing",
            "handle_missing_values",
            "handle_missing",
            "drop_duplicates",
            "handle_outliers",
        ),
        keywords=(
            "clean",
            "missing",
            "impute",
            "outlier",
            "duplicate",
            "dtype",
            "cast",
            "whitespace",
            "normalise_categor",
            "normalize_categor",
            "leakage",
            "constant_column",
            "data_quality",
        ),
    ),
    StepDefinition(
        step_id="engineer_features",
        title="Engineer features",
        kind="agent",
        agent=AgentName.FEATURES,
        executor="execution.feature_ops.apply_feature_plan",
        depends_on=("clean",),
        skippable=True,
        destructive=True,
        estimated_seconds=90,
        aliases=(
            "features",
            "feature_engineering",
            "feature_selection",
            "encode_categoricals",
            "scale_features",
            "transform_features",
        ),
        keywords=(
            "feature",
            "encode",
            "encoding",
            "scale",
            "scaling",
            "standardis",
            "standardiz",
            "normalis",
            "normaliz",
            "lag",
            "rolling",
            "interaction",
            "polynomial",
            "binning",
            "discretis",
            "tfidf",
            "text_vector",
            "pca",
            "dimensionality",
            "transform",
            "derive",
            "aggregate",
        ),
    ),
    StepDefinition(
        step_id="split",
        title="Split train/validation/test",
        kind="executor",
        executor="execution.splitter.make_splits",
        depends_on=("engineer_features",),
        skippable=False,
        # Training cannot happen without partitions, and a model scored on its
        # own training rows is worse than no model at all.
        critical=True,
        estimated_seconds=10,
        aliases=("splits", "train_test_split", "make_splits", "holdout"),
        keywords=("split", "holdout", "hold_out", "train_test", "partition"),
    ),
    StepDefinition(
        step_id="select_models",
        title="Select candidate models",
        kind="agent",
        agent=AgentName.MODEL_SELECTION,
        executor="execution.model_zoo.available_families",
        depends_on=("split",),
        skippable=False,
        estimated_seconds=45,
        aliases=("model_selection", "choose_models", "candidates", "shortlist_models"),
        keywords=(
            "select_model",
            "model_selection",
            "candidate_model",
            "choose_model",
            "choose_algorithm",
            "algorithm_selection",
            "shortlist",
        ),
    ),
    StepDefinition(
        step_id="run_experiments",
        title="Train and score candidates",
        kind="hybrid",
        agent=AgentName.EXPERIMENT,
        executor="execution.trainer.run_experiments",
        depends_on=("select_models",),
        skippable=False,
        critical=True,
        estimated_seconds=180,
        aliases=("experiments", "train_models", "training", "fit_models", "benchmark"),
        keywords=(
            "experiment",
            "train",
            "fit_model",
            "leaderboard",
            "cross_valid",
            "benchmark",
            "model_compar",
        ),
    ),
    StepDefinition(
        step_id="tune",
        title="Tune hyperparameters",
        kind="agent",
        agent=AgentName.TUNING,
        executor="execution.tuner.run_tuning",
        depends_on=("run_experiments",),
        skippable=True,
        budget_optional=True,
        estimated_seconds=240,
        aliases=("tuning", "hyperparameter_tuning", "optimise", "hpo"),
        keywords=(
            "tune",
            "tuning",
            "hyperparam",
            "optuna",
            "grid_search",
            "random_search",
            "bayesian",
            "hpo",
            "optimis",
            "optimiz",
        ),
    ),
    StepDefinition(
        step_id="explain",
        title="Explain the model",
        kind="hybrid",
        agent=AgentName.EXPLAIN,
        executor="execution.explainer.compute_explanations",
        depends_on=("run_experiments", "tune"),
        skippable=True,
        budget_optional=True,
        estimated_seconds=90,
        aliases=("explainability", "explanations", "interpretability", "shap"),
        keywords=("explain", "shap", "importance", "interpret", "attribution", "xai"),
    ),
    StepDefinition(
        step_id=EVALUATION_STEP_ID,
        title="Evaluate the model",
        kind="hybrid",
        agent=AgentName.EVALUATION,
        executor="execution.diagnostics.compute_diagnostics",
        depends_on=("run_experiments", "tune", "explain"),
        skippable=False,
        estimated_seconds=90,
        aliases=("evaluation", "assess", "quality_gate", "diagnostics"),
        keywords=(
            "evaluat",
            "assess",
            "diagnos",
            "calibration",
            "fairness",
            "residual",
            "quality_gate",
            "validate_model",
        ),
    ),
    StepDefinition(
        step_id="insights",
        title="Derive business insights",
        kind="agent",
        agent=AgentName.INSIGHT,
        depends_on=(EVALUATION_STEP_ID,),
        skippable=True,
        budget_optional=True,
        estimated_seconds=60,
        aliases=("insight", "business_insights", "recommendations"),
        keywords=("insight", "business", "recommendation", "executive", "stakeholder"),
    ),
    StepDefinition(
        step_id="visualise",
        title="Visualise the results",
        kind="agent",
        agent=AgentName.VISUALIZATION,
        executor="reporting.charts.render_charts",
        depends_on=(EVALUATION_STEP_ID,),
        skippable=True,
        budget_optional=True,
        estimated_seconds=90,
        aliases=("visualize", "visualisation", "visualization", "charts", "plots"),
        keywords=("visual", "chart", "plot", "dashboard", "figure"),
    ),
    StepDefinition(
        step_id="report",
        title="Write the report",
        kind="agent",
        agent=AgentName.REPORT,
        executor="reporting.writer.write_report",
        depends_on=(EVALUATION_STEP_ID, "insights", "visualise"),
        skippable=False,
        # The report is the deliverable: never dropped for budget, but a failure
        # to render it must not erase the run's other results either.
        budget_optional=False,
        estimated_seconds=90,
        aliases=("reporting", "final_report", "write_report", "deliverable"),
        keywords=("report", "deliverable", "document", "write_up", "presentation"),
    ),
)

_BY_ID: dict[str, StepDefinition] = {s.step_id: s for s in CANONICAL_STEPS}
_ORDER: dict[str, int] = {s.step_id: i for i, s in enumerate(CANONICAL_STEPS)}

#: Fixed steps the orchestrator runs before a plan exists.
PROLOGUE_STEP_IDS: tuple[str, ...] = tuple(
    s.step_id for s in CANONICAL_STEPS if not s.planned
)
#: Steps the Planner is allowed to schedule, in canonical order.
PLANNABLE_STEP_IDS: tuple[str, ...] = tuple(
    s.step_id for s in CANONICAL_STEPS if s.planned
)
#: Plannable steps that are injected when a plan omits them.
REQUIRED_STEP_IDS: tuple[str, ...] = tuple(
    s.step_id for s in CANONICAL_STEPS if s.planned and not s.skippable
)


def canonical_steps() -> list[StepDefinition]:
    """Every step definition, in canonical execution order."""
    return list(CANONICAL_STEPS)


def get_step(step_id: str) -> StepDefinition | None:
    """Look up a definition by exact canonical id."""
    return _BY_ID.get(step_id)


def require_step(step_id: str) -> StepDefinition:
    """Look up a definition, raising ``KeyError`` when the id is unknown."""
    try:
        return _BY_ID[step_id]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(f"unknown canonical step '{step_id}'") from exc


def step_index(step_id: str) -> int:
    """Canonical position, or ``-1`` for an unknown id."""
    return _ORDER.get(step_id, -1)


def step_for_agent(agent: AgentName | None) -> StepDefinition | None:
    """The definition an agent owns, if it owns exactly one."""
    if agent is None:
        return None
    owned = [s for s in CANONICAL_STEPS if s.agent is agent]
    return owned[0] if len(owned) == 1 else None


def normalise_id(text: str) -> str:
    """Lower-case, collapse punctuation to underscores, trim.

    Planner-authored slugs arrive as ``Handle Missing Values``,
    ``handle-missing-values``, or ``handle_missing_values`` with equal
    likelihood; normalising first means the alias tables only need one spelling.
    """
    lowered = re.sub(r"[^0-9a-z]+", "_", str(text).lower())
    return lowered.strip("_")


_ALIAS_TO_ID: dict[str, str] = {}
for _step in CANONICAL_STEPS:
    _ALIAS_TO_ID[normalise_id(_step.step_id)] = _step.step_id
    for _alias in _step.aliases:
        _ALIAS_TO_ID.setdefault(normalise_id(_alias), _step.step_id)


#: Work planners legitimately propose that this platform does not perform.
#: Without this table the ``agent`` fallback would quietly absorb such a step
#: into whatever that agent owns — a "deploy to Kubernetes" step folded into
#: report writing would look executed while nothing was deployed. Mapping them to
#: nothing instead makes the gap visible in the run record.
UNSUPPORTED_INTENTS: dict[str, str] = {
    "deploy": "this platform recommends a deployment pattern but does not deploy models",
    "kubernetes": "container orchestration is outside this platform's scope",
    "docker": "container builds are outside this platform's scope",
    "serve": "model serving is outside this platform's scope",
    "endpoint": "provisioning inference endpoints is outside this platform's scope",
    "monitor": "production monitoring is recommended in the report but not operated here",
    "retrain": "scheduled retraining is recommended in the report but not performed here",
    "schedule": "job scheduling is outside this platform's scope",
    "cron": "job scheduling is outside this platform's scope",
    "ab_test": "online experimentation is outside this platform's scope",
    "a_b_test": "online experimentation is outside this platform's scope",
    "collect": "acquiring more data requires a human; the platform analyses what it is given",
    "annotate": "labelling data requires a human",
    "labelling": "labelling data requires a human",
    "labeling": "labelling data requires a human",
    "sign_off": "human review happens through the approval gate, not as a pipeline step",
    "handover": "handover is not a pipeline step",
}


def unsupported_intent(raw_id: str) -> str | None:
    """Explain why a planned step id describes work this platform cannot do."""
    slug = normalise_id(raw_id)
    if slug in _ALIAS_TO_ID:
        return None
    for keyword, reason in UNSUPPORTED_INTENTS.items():
        if keyword in slug:
            return reason
    return None


def resolve_step_id(
    raw_id: str,
    *,
    agent: AgentName | None = None,
    title: str = "",
) -> str | None:
    """Map a planner-authored step onto a canonical step id.

    Resolution order is exact id or alias, then a check for work the platform
    does not do, then a keyword signature over id and title, then the declared
    owning agent. The agent comes last because it is the coarsest signal: every
    agent owns exactly one graph step, so it would match anything.

    Args:
        raw_id: The ``step_id`` the planner emitted.
        agent: The planner's declared owning agent, used as the final fallback.
        title: The planner's human title, searched for keywords.

    Returns:
        A canonical step id, or ``None`` when nothing plausibly matches.
    """
    slug = normalise_id(raw_id)
    exact = _ALIAS_TO_ID.get(slug)
    if exact:
        return exact
    if unsupported_intent(slug) is not None:
        return None

    haystack = f"{slug}_{normalise_id(title)}"
    # Canonical order is the tie-break: 'train_test_split' must resolve to the
    # split step, not to training, so earlier definitions win.
    for step in CANONICAL_STEPS:
        if any(keyword in haystack for keyword in step.keywords):
            return step.step_id

    owned = step_for_agent(agent)
    return owned.step_id if owned else None


@dataclass(slots=True)
class ResolvedStep:
    """One executable step, with the planned step(s) that asked for it."""

    definition: StepDefinition
    plan_step: PlanStep | None = None
    injected: bool = False
    merged: list[PlanStep] = field(default_factory=list)

    @property
    def step_id(self) -> str:
        return self.definition.step_id

    @property
    def title(self) -> str:
        if self.plan_step and self.plan_step.title:
            return self.plan_step.title
        return self.definition.title


@dataclass(slots=True)
class PlanReconciliation:
    """The plan, expressed in steps the orchestrator can actually run."""

    steps: list[ResolvedStep] = field(default_factory=list)
    unmapped: list[tuple[PlanStep, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps]

    def head(self, boundary: str = EVALUATION_STEP_ID) -> list[ResolvedStep]:
        """Steps at or before ``boundary`` — the replannable part of the run."""
        limit = step_index(boundary)
        return [s for s in self.steps if step_index(s.step_id) <= limit]

    def tail(self, boundary: str = EVALUATION_STEP_ID) -> list[ResolvedStep]:
        """Steps after ``boundary`` — run once, against the final model."""
        limit = step_index(boundary)
        return [s for s in self.steps if step_index(s.step_id) > limit]


def _enforce_dependencies(
    steps: list[ResolvedStep],
) -> tuple[list[ResolvedStep], list[str]]:
    """Stable topological sort that keeps the plan's order wherever it is legal."""
    present = {s.step_id for s in steps}
    remaining = list(steps)
    ordered: list[ResolvedStep] = []
    done: set[str] = set()
    notes: list[str] = []

    while remaining:
        for position, candidate in enumerate(remaining):
            deps = [d for d in candidate.definition.depends_on if d in present]
            if all(d in done for d in deps):
                if position != 0:
                    blocked = remaining[0].step_id
                    notes.append(
                        f"reordered: '{candidate.step_id}' ran before '{blocked}' "
                        "because the plan's order would have broken a dependency"
                    )
                ordered.append(candidate)
                done.add(candidate.step_id)
                remaining.pop(position)
                break
        else:  # pragma: no cover - only reachable if the graph gains a cycle
            remaining.sort(key=lambda s: step_index(s.step_id))
            notes.append(
                "dependency cycle detected; falling back to canonical order for "
                f"{[s.step_id for s in remaining]}"
            )
            ordered.extend(remaining)
            break

    return ordered, notes


def default_sequence() -> list[ResolvedStep]:
    """Every plannable step, in canonical order, marked as injected."""
    return [
        ResolvedStep(definition=_BY_ID[step_id], injected=True)
        for step_id in PLANNABLE_STEP_IDS
    ]


def reconcile_plan(
    plan: ExecutionPlan | None,
    *,
    inject_required: bool = True,
) -> PlanReconciliation:
    """Turn an agent's :class:`ExecutionPlan` into a runnable step sequence.

    Args:
        plan: The Planner's output. ``None`` yields the canonical pipeline.
        inject_required: Whether to add back non-skippable steps the plan
            forgot. Off only for inspection.

    Returns:
        A :class:`PlanReconciliation`: the ordered executable steps, the planned
        steps that mapped to no executor (to be recorded as skipped), and notes
        describing every adjustment made.
    """
    if plan is None or not plan.steps:
        recon = PlanReconciliation(steps=default_sequence())
        recon.notes.append(
            "no execution plan was available, so the canonical pipeline was used"
        )
        return recon

    resolved: dict[str, ResolvedStep] = {}
    unmapped: list[tuple[PlanStep, str]] = []
    notes: list[str] = []

    for planned in plan.ordered():
        canonical = resolve_step_id(
            planned.step_id, agent=planned.agent, title=planned.title
        )
        if canonical is None:
            unmapped.append(
                (
                    planned,
                    unsupported_intent(planned.step_id)
                    or "no executor in the step graph implements this step",
                )
            )
            continue

        definition = _BY_ID[canonical]
        if not definition.planned:
            notes.append(
                f"plan step '{planned.step_id}' maps to '{canonical}', which the "
                "orchestrator runs before planning; ignoring the duplicate"
            )
            continue

        existing = resolved.get(canonical)
        if existing is None:
            resolved[canonical] = ResolvedStep(
                definition=definition, plan_step=planned, merged=[planned]
            )
            if normalise_id(planned.step_id) != canonical:
                notes.append(
                    f"plan step '{planned.step_id}' mapped to graph step '{canonical}'"
                )
        else:
            existing.merged.append(planned)
            notes.append(
                f"plan step '{planned.step_id}' merged into '{canonical}', which "
                "already covers it"
            )

    steps = list(resolved.values())

    if inject_required:
        for step_id in REQUIRED_STEP_IDS:
            if step_id in resolved:
                continue
            injected = ResolvedStep(definition=_BY_ID[step_id], injected=True)
            steps.append(injected)
            notes.append(
                f"required step '{step_id}' was missing from the plan and was "
                "inserted at its canonical position"
            )

    # Injected steps have no plan order of their own, so seed the sort with
    # canonical position for them and plan position for everything else.
    def _priority(item: ResolvedStep) -> tuple[int, int]:
        if item.plan_step is not None:
            return (0, item.plan_step.order)
        return (0, step_index(item.step_id))

    steps.sort(key=_priority)
    ordered, order_notes = _enforce_dependencies(steps)
    notes.extend(order_notes)

    return PlanReconciliation(steps=ordered, unmapped=unmapped, notes=notes)


def describe_graph() -> str:
    """Render the graph as text, for docs, prompts, and audit output."""
    lines = ["Canonical pipeline:"]
    for index, step in enumerate(CANONICAL_STEPS, start=1):
        flags = [
            "required" if not step.skippable else "skippable",
            "critical" if step.critical else "non-critical",
        ]
        if step.destructive:
            flags.append("destructive")
        if step.budget_optional:
            flags.append("budget-optional")
        owner = step.agent.value if step.agent else "executor"
        lines.append(
            f"{index:>2}. {step.step_id:<18} {step.title:<32} "
            f"owner={owner:<16} [{', '.join(flags)}]"
        )
    return "\n".join(lines)


__all__ = [
    "CANONICAL_STEPS",
    "EVALUATION_STEP_ID",
    "PLANNABLE_STEP_IDS",
    "PROLOGUE_STEP_IDS",
    "REQUIRED_STEP_IDS",
    "UNSUPPORTED_INTENTS",
    "PlanReconciliation",
    "ResolvedStep",
    "StepDefinition",
    "StepKind",
    "canonical_steps",
    "default_sequence",
    "describe_graph",
    "get_step",
    "normalise_id",
    "reconcile_plan",
    "require_step",
    "resolve_step_id",
    "step_for_agent",
    "step_index",
    "unsupported_intent",
]
