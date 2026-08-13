"""Planning Agent — the brain of the run.

Everything after this is execution. The planner writes the whole strategy before
a single value is imputed, which is what makes the run auditable: the reason each
step exists is recorded before its outcome is known, so nobody can retrofit a
justification after seeing the score.

Two responsibilities live here that are easy to miss:

*   **Adaptation.** A plan that would suit any dataset is a plan nobody thought
    about. The prompt is fed deterministic *triggers* computed from the profile —
    nothing missing means no imputation step, leakage found means a removal step,
    a monotonic timestamp means a time-ordered split — so "adapt to this dataset"
    is a checkable instruction rather than an aspiration.
*   **Replanning.** When evaluation rejects a model the planner runs again, and
    the failure mode is a reshuffle of the same plan. The replan prompt therefore
    carries the previous plan, the measured scores, the evaluation's stated
    weaknesses, and everything already tried, with an explicit definition of what
    "materially different" means.

:meth:`PlannerAgent.postprocess` repairs plan structure rather than trusting it:
slugs are normalised, dangling and forward dependencies are dropped, the
pre-plan agents are removed so the plan cannot recurse into itself, and the
mandatory terminal steps (evaluation, insight, report) are appended if absent.
"""

from __future__ import annotations

import re

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ColumnKind,
    ExecutionPlan,
    PlanStep,
    Severity,
    TaskType,
)
from ..core.state import RunState

#: Agents that have already run by the time the plan exists. A plan step naming
#: one of these is at best wasted work and at worst (the planner itself) a loop.
PRE_PLAN_AGENTS = frozenset({AgentName.DATASET, AgentName.PROBLEM, AgentName.PLANNER})

#: Every run must end by judging the model, extracting meaning, and reporting.
TERMINAL_AGENTS: tuple[AgentName, ...] = (
    AgentName.EVALUATION,
    AgentName.INSIGHT,
    AgentName.REPORT,
)

#: Step-id slugs the orchestrator's canonical graph recognises without
#: reconciliation. Suggested to the model and used for every step this module
#: synthesises, so a repaired plan lines up with the executable graph exactly.
CANONICAL_STEP_IDS: dict[AgentName, str] = {
    AgentName.CLEANING: "clean",
    AgentName.FEATURES: "engineer_features",
    AgentName.MODEL_SELECTION: "select_models",
    AgentName.EXPERIMENT: "run_experiments",
    AgentName.TUNING: "tune",
    AgentName.EXPLAIN: "explain",
    AgentName.EVALUATION: "evaluate",
    AgentName.INSIGHT: "insights",
    AgentName.VISUALIZATION: "visualise",
    AgentName.REPORT: "report",
}

_TERMINAL_DEFAULTS: dict[AgentName, tuple[str, str, str, str]] = {
    AgentName.EVALUATION: (
        CANONICAL_STEP_IDS[AgentName.EVALUATION],
        "Evaluate the winning model",
        "Judge whether the best model is fit to recommend: bias/variance, "
        "calibration, confidence intervals, fairness slices, residual behaviour, "
        "and error analysis.",
        "Mandatory terminal step, appended by the orchestrator because the plan "
        "omitted it. No model may be recommended without passing the quality gate, "
        "and this step is what can reject the model and trigger a replan.",
    ),
    AgentName.INSIGHT: (
        CANONICAL_STEP_IDS[AgentName.INSIGHT],
        "Translate the model into business insight",
        "Convert the measured performance and feature attributions into decisions "
        "a non-technical owner can act on.",
        "Mandatory terminal step, appended by the orchestrator because the plan "
        "omitted it. A model nobody can act on has no value regardless of its score.",
    ),
    AgentName.REPORT: (
        CANONICAL_STEP_IDS[AgentName.REPORT],
        "Write the final report",
        "Assemble the full narrative, the evidence, and a deployment "
        "recommendation into the deliverable.",
        "Mandatory terminal step, appended by the orchestrator because the plan "
        "omitted it. The report is the run's only durable output.",
    ),
}

#: Used when the model returns nothing usable, so the run degrades instead of dying.
_MINIMAL_PIPELINE: tuple[tuple[AgentName, str, str, str], ...] = (
    (
        AgentName.CLEANING,
        CANONICAL_STEP_IDS[AgentName.CLEANING],
        "Repair the data quality issues the profiler measured",
        "Fallback pipeline: the planner returned no usable steps, so a standard "
        "cleaning pass runs to make the table modellable.",
    ),
    (
        AgentName.FEATURES,
        "engineer_features",
        "Encode and derive the model-ready feature matrix",
        "Fallback pipeline: encoding and scaling are required before any estimator "
        "can be fitted.",
    ),
    (
        AgentName.MODEL_SELECTION,
        "select_models",
        "Choose the candidate model families and the validation strategy",
        "Fallback pipeline: candidates and a validation scheme must be chosen "
        "before training.",
    ),
    (
        AgentName.EXPERIMENT,
        "run_experiments",
        "Train and score every candidate",
        "Fallback pipeline: the leaderboard is what every later judgement rests on.",
    ),
)

AGENT_ROSTER = """\
## The agents you may assign work to

Each step names exactly one of these. Nothing else exists — a step assigned to \
an agent that is not on this list is dropped before execution.

-   `cleaning` — imputation (mean/median/mode/sentinel/interpolate/KNN), \
duplicate removal, dropping constant or leaking columns, dtype casts, datetime \
parsing, category normalisation, outlier clipping or removal. Applied by real \
pandas code.
-   `features` — datetime decomposition, cyclical encoding, lags, rolling and \
expanding windows, differences, interactions and ratios, log/sqrt/Box-Cox \
transforms, binning, categorical encoding (one-hot, ordinal, target, frequency, \
hash), TF-IDF and text length, geo distance, group aggregates, scaling, \
PCA/SVD, and feature selection.
-   `model_selection` — chooses the model families to try, their initial \
parameters, and the validation strategy (stratified k-fold, grouped k-fold, \
expanding-window time series split, or a plain holdout).
-   `experiment` — trains and scores every selected candidate and produces the \
leaderboard. Deterministic; the agent interprets the results.
-   `tuning` — decides whether hyperparameter search is worth the compute, then \
which method and search space.
-   `explain` — SHAP and permutation importance, partial dependence, and the \
plain-language account of what drives the predictions.
-   `evaluation` — the quality gate. Bias/variance, calibration, bootstrap \
confidence intervals, fairness slices, residual analysis, learning curve, error \
analysis. It can reject the model, which sends the run back to you.
-   `insight` — business-level findings and recommended actions.
-   `visualization` — the chart specifications for the report and dashboard.
-   `report` — the final document plus a deployment recommendation.

`dataset`, `problem`, and `planner` have already run. Do not assign steps to them.

**One step per agent.** Each step you write causes exactly one invocation of that \
agent, and the agent's output *replaces* its previous output. Two `features` \
steps therefore do not compose — the second overwrites the first. When an agent \
must handle several concerns, give it one step whose `objective` enumerates them.

**Use these `step_id` slugs**, which the executor graph recognises directly: \
`clean`, `engineer_features`, `select_models`, `run_experiments`, `tune`, \
`explain`, `evaluate`, `insights`, `visualise`, `report`. Train/test splitting is \
performed by the orchestrator between feature engineering and model selection — \
you specify the strategy in the `select_models` step rather than writing a step \
for it.
"""

INSTRUCTIONS_CORE = """\
You are the lead data scientist planning this project. You write the strategy \
before any computation happens, and a deterministic orchestrator executes it \
step by step, invoking the agent you name and applying its decisions with real \
library code. The plan is also the audit trail: each step's rationale is recorded \
before its outcome is known, so it has to be a reason, not a label.

{roster}

## Your method

1.  **Start from the problem, not from a template.** The task type, target, \
metric, and time structure decided by the problem agent constrain everything. A \
forecasting problem and an imbalanced binary classification problem share almost \
no steps beyond their names.
2.  **Read the triggers.** The user turn lists conditions measured from the real \
data. Each one either creates a step, changes a step's objective, or is \
explicitly not worth acting on. Silence is not an option: if nothing is missing, \
there is no imputation work, and saying so in \
`dataset_specific_adaptations` is a decision, not an omission.
3.  **Choose the validation and split strategy, and name it.** Repeated entity \
keys demand a grouped split. A time order demands a time-ordered split — and \
any lag, rolling, or expanding feature is invalid without one. A small row count \
argues for cross-validation over a single holdout. An imbalanced target argues \
for stratification. State the strategy in the `model_selection` step's objective \
so the agent that owns it cannot pick something else by accident.
4.  **Budget the run.** The time budget, the maximum number of models, and the \
tuning/explainability switches are hard constraints. Sum your \
`estimated_seconds`; if the total exceeds the budget, cut scope deliberately \
rather than letting the orchestrator cut it for you. Mark genuinely expendable \
steps `optional`.
5.  **Order the work and wire the dependencies.** `depends_on` may only \
reference `step_id`s that exist and that come earlier in the order. Cleaning \
before feature engineering, features before model selection, training before \
evaluation.
6.  **Write the fallback.** `fallback_strategy` must name the concrete change \
you would make if evaluation rejects the model: which knob, in which direction, \
and what evidence would trigger it. "Try other models" is not a fallback.

## Marking a step destructive

Set `destructive=True` when the step drops rows or columns: deduplication, \
leakage-column removal, dropping constant or mostly-missing columns, removing \
outlier rows, dropping correlated features. Imputation, scaling, and encoding are \
not destructive — they change values, not the shape of the evidence.

This flag gates human approval, so it cuts both ways. Under-marking lets \
irreversible data loss happen unreviewed; over-marking stalls the run waiting \
for a human to approve a scaling step. Mark exactly what deletes information.

## dataset_specific_adaptations is the part I will actually read

Give me at least three, and make each one cite something concrete from this \
dataset — a column name, a measured fraction, a class ratio, a row count — and \
say what you did about it. Two examples of the difference:

Good: "`TotalCharges` is 11 rows missing (0.16%) and is a numeric column stored \
as text; the cleaning step parses it and drops those 11 rows rather than \
imputing, because 11 of 7,043 rows costs nothing and imputing a charge total \
invents revenue that never existed."

Bad: "We will handle missing values appropriately and engineer relevant \
features." — true of every dataset ever profiled, and therefore worthless.

## Failure modes of this specific job

-   Emitting the same generic seven-step pipeline regardless of what was \
measured.
-   Planning lag or rolling features without a time-ordered split. Those \
features leak the future into training unless the split respects time order, and \
the resulting score is fiction.
-   Planning target encoding without demanding out-of-fold fitting in the \
step's objective. Fitted on the full training set, it leaks the target directly \
into a feature.
-   Planning resampling (SMOTE, undersampling) before the split. Any \
class-balancing must happen inside the training fold only; done first, the \
synthetic neighbours of test rows end up in training.
-   Spending the budget on tuning when the baseline gap is tiny, or on twelve \
model families when the row count only supports two.
-   Dropping every column the leakage scan flagged without judging each one. A \
high association can be a genuinely strong feature; the question is whether the \
value is knowable at prediction time.
-   Planning a step whose success cannot be checked. `success_criteria` should \
be observable — "no column above 20% missing remains", "at least three families \
scored on the holdout".

Prefer six well-argued steps to twelve hedged ones. Every step must earn its \
place with a reason specific to this dataset.
"""

REPLAN_INSTRUCTIONS = """\
## You are REPLANNING, and this is a different job

A previous plan ran to completion and the evaluation agent rejected the result. \
Your first task is diagnosis, not authoring: read the measured scores, the \
bias/variance verdict, and the stated weaknesses, and name the mechanism that \
failed. Then change that mechanism.

A materially different plan changes at least one of these:

-   **the feature set** — different transforms, different encodings, aggressive \
selection, or dropping the features that turned out to be noise;
-   **the model families** — a different capacity class, not a different member \
of the same one;
-   **the validation or split strategy** — because a suspiciously good training \
score against a poor holdout score is often a split problem, not a model problem;
-   **the target treatment** — a log transform, a reframed threshold, a \
different class weighting.

Reordering the same steps, renaming them, or adding one more model family is a \
reshuffle. It will produce the same verdict and burn the remaining budget.

Match the change to the diagnosis:

-   **Overfitting** (train far above validation): reduce capacity — regularise, \
cut the feature count, drop correlated features, increase the fold count, prefer \
a simpler family. Do not add features.
-   **Underfitting** (both scores low): add signal — interactions, domain \
ratios, richer encodings of high-cardinality categories, a higher-capacity \
family. Do not regularise harder.
-   **Leakage suspected** (a score close to perfect, or one feature dominating \
implausibly): remove the offending column and re-run. Say which column and why \
it cannot be known at prediction time.
-   **Poor calibration with acceptable ranking**: keep the model and add \
calibration, or move the primary metric to one that reflects what is actually \
consumed.
-   **Data-limited** (wide confidence intervals, tiny minority class): say so. \
More model complexity cannot fix too few rows, and the honest plan says the \
ceiling is the data. Recommending `collect_more_data` in your summary is a \
legitimate outcome.

Do not re-try anything already recorded as tried and failed. Set \
`revision_reason` to name the specific evaluation finding you are responding to, \
and make `summary` open by stating what you are changing and why.
"""

MEMORY_FRAMING = """\
## Prior runs on similar data

The user turn may include evidence from previous runs on structurally similar \
datasets. Treat it as a colleague's experience report: informative, not \
binding. Weigh it against what was measured *here*. If it points somewhere this \
dataset's measurements do not support, ignore it and say so in \
`dataset_specific_adaptations` — deferring to precedent over evidence is exactly \
the failure this framing exists to prevent.
"""


def _pct(fraction: float | None) -> str:
    return "n/a" if fraction is None else f"{fraction * 100:.2f}%"


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5g}"


def _slug(text: str, fallback: str) -> str:
    """Normalise a step id to a stable lowercase slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned[:60] or fallback


def plan_triggers(state: RunState) -> list[str]:
    """Measured conditions the plan must visibly respond to.

    Computed deterministically so that "adapt to this dataset" becomes a
    checkable instruction: each line either creates a step, alters a step's
    objective, or is consciously ignored.

    Args:
        state: The run blackboard, after profiling and problem definition.

    Returns:
        Human-readable trigger lines, ready to drop into the prompt.
    """
    profile = state.profile
    if profile is None:
        return []

    lines: list[str] = []
    n_rows = profile.n_rows

    missing_cols = [c for c in profile.columns if c.n_missing > 0]
    if not missing_cols:
        lines.append(
            "- MISSINGNESS: none. Every cell is populated, so there is no "
            "imputation work to plan. Do not include an imputation step."
        )
    else:
        worst = sorted(missing_cols, key=lambda c: c.missing_fraction, reverse=True)[:6]
        rendered = ", ".join(f"`{c.name}` {_pct(c.missing_fraction)}" for c in worst)
        heavy = [c.name for c in missing_cols if c.missing_fraction > 0.4]
        lines.append(
            f"- MISSINGNESS: {len(missing_cols)} of {profile.n_columns} columns have "
            f"gaps (worst: {rendered}). Imputation is in scope."
        )
        if heavy:
            lines.append(
                f"- HEAVY MISSINGNESS: {heavy[:8]} exceed 40% missing. Decide per "
                "column between dropping it and encoding the missingness itself as "
                "a feature; imputing 40%+ of a column mostly invents data."
            )

    if profile.n_duplicate_rows:
        lines.append(
            f"- DUPLICATES: {profile.n_duplicate_rows:,} duplicate rows "
            f"({_pct(profile.duplicate_fraction)}). Left in place they appear on both "
            "sides of the split and inflate every score. A dedupe step is "
            "destructive."
        )
    else:
        lines.append("- DUPLICATES: none. No dedupe step needed.")

    if profile.leakage_findings:
        severe = [
            f.column
            for f in profile.leakage_findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        lines.append(
            f"- LEAKAGE CANDIDATES: {len(profile.leakage_findings)} flagged"
            + (f", {len(severe)} at high or critical severity ({severe[:6]})" if severe else "")
            + ". Plan an explicit removal step, judged column by column on whether "
            "the value is knowable at prediction time. Dropping columns is "
            "destructive."
        )
    else:
        lines.append("- LEAKAGE: no candidates flagged by the association scan.")

    if profile.constant_columns:
        lines.append(
            f"- CONSTANT COLUMNS: {profile.constant_columns[:8]} carry no "
            "information. Dropping them is destructive but free."
        )

    if profile.temporal_columns:
        details = []
        for name in profile.temporal_columns[:4]:
            col = profile.column(name)
            if col is None:
                details.append(f"`{name}`")
                continue
            details.append(
                f"`{name}` (freq={col.inferred_frequency or 'irregular'}, "
                f"monotonic={str(col.is_monotonic).lower()})"
            )
        lines.append(
            f"- TIME STRUCTURE: {', '.join(details)}. "
            "Temporal feature work (decomposition, cyclical encoding, and — only "
            "with a time-ordered split — lags and rolling windows) is available, and "
            "the split must respect time order."
        )
    else:
        lines.append(
            "- TIME STRUCTURE: no temporal column. No date decomposition, no lag or "
            "rolling features, and a time-ordered split is impossible."
        )

    high_card = [
        c
        for c in profile.columns
        if c.kind
        in (ColumnKind.CATEGORICAL_NOMINAL, ColumnKind.CATEGORICAL_ORDINAL)
        and c.n_unique > 15
    ]
    if high_card:
        rendered = ", ".join(f"`{c.name}` ({c.n_unique:,})" for c in high_card[:6])
        lines.append(
            f"- HIGH-CARDINALITY CATEGORIES: {rendered}. One-hot encoding these "
            "explodes the feature count; target, frequency, or hash encoding is the "
            "alternative, and target encoding must be specified as out-of-fold."
        )

    if profile.text_columns:
        lines.append(
            f"- FREE TEXT: {profile.text_columns[:6]}. TF-IDF or length/token "
            "features are available; so is deciding the text is not worth the "
            "dimensionality."
        )
    if profile.geo_columns:
        lines.append(
            f"- GEO COLUMNS: {profile.geo_columns[:6]}. Distance or region features "
            "are available."
        )
    if profile.identifier_columns:
        lines.append(
            f"- IDENTIFIERS: {profile.identifier_columns[:6]}. These must not reach "
            "the model as features; a tree will memorise them."
        )
    if profile.highly_correlated_pairs:
        pair = profile.highly_correlated_pairs[0]
        lines.append(
            f"- MULTICOLLINEARITY: {len(profile.highly_correlated_pairs)} highly "
            f"correlated pairs (e.g. `{pair.left}` <-> `{pair.right}` at "
            f"{pair.coefficient:.3f}). Relevant to linear models and to feature "
            "attribution stability; dropping one of a pair is destructive."
        )

    target = profile.target
    if target is not None and target.is_imbalanced:
        ratio = (
            f"{target.imbalance_ratio:.2f}"
            if target.imbalance_ratio is not None
            else "unquantified"
        )
        lines.append(
            f"- CLASS IMBALANCE: ratio {ratio} on "
            f"`{target.name}`. Stratify the split, consider class weights, and note "
            "that any resampling must happen inside the training fold only — never "
            "before the split."
        )
    if target is not None and target.n_missing:
        lines.append(
            f"- TARGET GAPS: {target.n_missing:,} rows have no `{target.name}` "
            "value. They cannot be trained or scored on; dropping them is "
            "destructive and unavoidable."
        )
    if target is not None and target.skewness is not None and abs(target.skewness) > 1.5:
        lines.append(
            f"- TARGET SKEW: skewness {target.skewness:.2f} on `{target.name}`. A "
            "log or Box-Cox transform of the target is worth considering, but "
            "commits the report to back-transforming before quoting any figure."
        )

    n_features = max(profile.n_columns - 1, 1)
    if n_rows < 1000:
        lines.append(
            f"- SMALL DATA: {n_rows:,} rows. A single holdout will be noisy; prefer "
            "cross-validation, favour lower-variance models, and be sceptical of "
            "hyperparameter tuning gains at this size."
        )
    if n_rows / n_features < 10:
        lines.append(
            f"- WIDE SHAPE: {n_rows:,} rows against {n_features} candidate features "
            f"({n_rows / n_features:.1f} rows per feature). Dimensionality reduction "
            "or feature selection is close to mandatory."
        )
    if n_rows > 500_000:
        lines.append(
            f"- LARGE DATA: {n_rows:,} rows. Histogram-based boosting and "
            "subsampling matter more than model variety; a twelve-family sweep will "
            "not fit the time budget."
        )
    return lines


class PlannerAgent(BaseAgent[ExecutionPlan]):
    """Writes — and, after a rejection, rewrites — the whole execution strategy."""

    name = AgentName.PLANNER
    title = "Planning Agent"
    output_model = ExecutionPlan
    effort: Effort = "max"
    max_tokens = 32_000

    # -- prompts -----------------------------------------------------------

    @staticmethod
    def is_replanning(state: RunState) -> bool:
        """Whether a previous plan was executed and judged unacceptable."""
        return state.evaluation is not None and not state.evaluation.acceptable

    def instructions(self, state: RunState) -> str:
        """The system prompt: roster, method, and — on a replan — the pivot rules."""
        parts = [INSTRUCTIONS_CORE.format(roster=AGENT_ROSTER)]
        if self.is_replanning(state):
            parts.append(REPLAN_INSTRUCTIONS)
        if state.memory is not None and state.memory.has_precedent:
            parts.append(MEMORY_FRAMING)
        return "\n".join(parts)

    def build_prompt(self, state: RunState) -> str:
        """The user turn: the framing, the triggers, the budget, and the question."""
        lines: list[str] = ["## YOUR TASK", ""]
        lines.extend(self._problem_block(state))
        lines.extend(self._understanding_block(state))
        lines.extend(self._trigger_block(state))
        lines.extend(self._budget_block(state))
        lines.extend(self._memory_block(state))
        lines.extend(self._replan_block(state))

        if self.is_replanning(state):
            lines.extend(
                [
                    "### Produce",
                    "A revised plan that changes the mechanism the evaluation "
                    "blamed. Open `summary` with what you are changing and why, "
                    "set `revision_reason` to the specific finding you are "
                    "answering, and make sure a reader can tell this plan apart "
                    "from the previous one at a glance.",
                ]
            )
        else:
            lines.extend(
                [
                    "### Produce",
                    "The execution plan for this dataset: ordered steps with real "
                    "dependencies, each owned by one agent on the roster and each "
                    "justified by something measured here; the adaptations that "
                    "make this plan specific to this data; the risks you are "
                    "accepting; and a fallback that names the concrete change to "
                    "make if evaluation rejects the model.",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _problem_block(state: RunState) -> list[str]:
        problem = state.problem
        if problem is None:
            return [
                "### Framing",
                "No problem definition is available; plan conservatively and make "
                "the first step establish one.",
                "",
            ]
        lines = [
            "### The framing you are planning for",
            f"- task: {problem.task_type.value} (confidence: {problem.confidence})",
            f"- target: `{problem.target_column or 'none'}`",
            f"- primary metric: {problem.primary_metric}"
            + (
                f" (secondary: {', '.join(problem.secondary_metrics)})"
                if problem.secondary_metrics
                else ""
            ),
            f"- metric reasoning: {problem.metric_rationale}",
            f"- business objective: {problem.business_objective}",
        ]
        if problem.positive_class:
            lines.append(f"- positive class: '{problem.positive_class}'")
        if problem.temporal_column:
            lines.append(
                f"- temporal column: `{problem.temporal_column}` — the split must "
                "respect it"
            )
        if problem.group_column:
            lines.append(
                f"- group column: `{problem.group_column}` — rows repeat per entity, "
                "so the split must be grouped"
            )
        if problem.horizon:
            lines.append(f"- forecast horizon: {problem.horizon} periods")
        for constraint in problem.constraints[:6]:
            lines.append(f"- constraint: {constraint}")
        if not problem.task_type.is_supported:
            fallback = state.extras.get("task_type_fallback") or "the nearest supported task"
            lines.append(
                f"- NOTE: {problem.task_type.value} has no end-to-end execution "
                f"path; the executors will run it as {fallback}. Plan for what will "
                "actually run, and say so in the adaptations."
            )
        lines.append("")
        return lines

    @staticmethod
    def _understanding_block(state: RunState) -> list[str]:
        understanding = state.understanding
        if understanding is None:
            return []
        lines = [
            "### What the dataset agent found",
            f"- grain: {understanding.grain}",
            f"- readiness: {understanding.data_readiness} — "
            f"{understanding.readiness_rationale}",
        ]
        for finding in understanding.key_findings[:6]:
            lines.append(f"- finding: {finding}")
        for risk in understanding.risks[:6]:
            lines.append(f"- risk: {risk}")
        hazards = [
            a
            for a in understanding.column_assessments
            if a.role.value in ("leakage_suspect", "identifier") or a.concerns
        ]
        if hazards:
            lines.append("- columns flagged as hazards:")
            for assessment in hazards[:10]:
                concerns = "; ".join(assessment.concerns[:2]) or assessment.notes
                lines.append(f"  - `{assessment.name}` [{assessment.role.value}]: {concerns}")
        lines.append("")
        return lines

    @staticmethod
    def _trigger_block(state: RunState) -> list[str]:
        triggers = plan_triggers(state)
        if not triggers:
            return []
        return [
            "### Plan triggers, measured from this dataset",
            "",
            "Each line below is a fact, not a suggestion. Respond to every one: "
            "create a step, change a step's objective, or state in "
            "`dataset_specific_adaptations` that you deliberately did neither.",
            "",
            *triggers,
            "",
        ]

    @staticmethod
    def _budget_block(state: RunState) -> list[str]:
        config = state.config
        remaining = state.time_remaining or config.time_budget_seconds
        lines = [
            "### Budget and switches",
            f"- wall-clock remaining: about {remaining:.0f}s of the "
            f"{config.time_budget_seconds}s budget",
            f"- maximum models to train: {config.max_experiments}",
            f"- cross-validation folds configured: {config.cv_folds}",
            f"- hyperparameter tuning: {'enabled' if config.enable_tuning else 'DISABLED — do not plan a tuning step'}",
            f"- explainability: {'enabled' if config.enable_explainability else 'DISABLED — do not plan an explain step'}",
            f"- human approval before destructive steps: "
            f"{'required, so each destructive step will pause the run' if config.require_approval else 'not required'}",
            f"- replans still available: {max(config.max_replans - state.replans, 0)}",
        ]
        if config.fairness_attributes:
            lines.append(
                f"- fairness attributes to audit: {config.fairness_attributes} — the "
                "evaluation step must slice on these"
            )
        if config.min_acceptable_score is not None:
            lines.append(
                f"- the run fails below {config.min_acceptable_score} on "
                f"{state.primary_metric}"
            )
        lines.append("")
        return lines

    @staticmethod
    def _memory_block(state: RunState) -> list[str]:
        memory = state.memory
        if memory is None or not memory.has_precedent:
            return []
        lines = ["### Evidence from previous runs on similar data", ""]
        if memory.narrative:
            lines.append(memory.narrative)
            lines.append("")
        for run in memory.similar_runs[:5]:
            lines.append(
                f"- run {run.run_id}: similarity {run.similarity:.2f}, "
                f"{run.task_type.value if run.task_type else 'unknown task'}, best "
                f"{run.best_family.value if run.best_family else 'n/a'} at "
                f"{_score(run.best_score)} {run.primary_metric}"
                + (f" — {run.why_similar}" if run.why_similar else "")
            )
        if memory.recommended_families:
            lines.append(
                "- families that worked before: "
                + ", ".join(f.value for f in memory.recommended_families)
            )
        if memory.recommended_feature_ops:
            lines.append(
                "- feature operations that worked before: "
                + ", ".join(op.value for op in memory.recommended_feature_ops)
            )
        for caution in memory.cautions[:5]:
            lines.append(f"- caution: {caution}")
        lines.append("")
        lines.append(
            "This is precedent, not instruction. Follow it only where this "
            "dataset's measurements agree."
        )
        lines.append("")
        return lines

    def _replan_block(self, state: RunState) -> list[str]:
        if not self.is_replanning(state):
            return []
        evaluation = state.evaluation
        assert evaluation is not None  # guarded by is_replanning
        lines = [
            "### The previous attempt, and why it was rejected",
            "",
            f"- grade: {evaluation.overall_grade}; recommended action: "
            f"{evaluation.recommended_action}",
            f"- verdict: {evaluation.verdict_rationale}",
        ]
        bv = evaluation.bias_variance
        lines.append(
            f"- fit diagnosis: {bv.verdict} (train={_score(bv.train_score)}, "
            f"validation={_score(bv.validation_score)}, test={_score(bv.test_score)}, "
            f"gap={_score(bv.gap)}) — {bv.detail}"
        )
        if evaluation.calibration.applicable:
            lines.append(
                f"- calibration: brier={_score(evaluation.calibration.brier_score)}, "
                f"ECE={_score(evaluation.calibration.expected_calibration_error)} — "
                f"{evaluation.calibration.verdict}"
            )
        for interval in evaluation.confidence_intervals[:4]:
            lines.append(
                f"- {interval.metric} 95% CI: {_score(interval.point_estimate)} "
                f"[{_score(interval.lower)}, {_score(interval.upper)}]"
            )
        for weakness in evaluation.weaknesses[:8]:
            lines.append(f"- weakness: {weakness}")
        for improvement in evaluation.specific_improvements[:8]:
            lines.append(f"- suggested improvement: {improvement}")
        if evaluation.residual_notes:
            lines.append(f"- residuals: {evaluation.residual_notes}")
        if evaluation.error_analysis:
            lines.append(f"- error analysis: {'; '.join(evaluation.error_analysis[:4])}")
        if evaluation.action_rationale:
            lines.append(f"- action rationale: {evaluation.action_rationale}")
        lines.append("")

        lines.extend(self._leaderboard_block(state))
        lines.extend(self._tried_block(state))
        lines.extend(self._previous_plan_block(state))
        return lines

    @staticmethod
    def _leaderboard_block(state: RunState) -> list[str]:
        log = state.experiments
        if log is None or not log.results:
            return []
        ranked = sorted(
            (r for r in log.results if not r.failed and r.primary_score is not None),
            key=lambda r: r.primary_score,  # type: ignore[arg-type,return-value]
            reverse=log.higher_is_better,
        )
        lines = [
            f"### Measured scores from the previous run ({log.primary_metric}, "
            f"{'higher' if log.higher_is_better else 'lower'} is better)",
            "",
        ]
        for result in ranked[:6]:
            cv = (
                f", cv mean={sum(result.cv_scores) / len(result.cv_scores):.5g}"
                if result.cv_scores
                else ""
            )
            lines.append(
                f"- {result.family.value}"
                + (f" ({result.label})" if result.label else "")
                + f": {_score(result.primary_score)}{cv}, "
                f"{result.n_features_in} features, {result.train_seconds:.1f}s"
                + (" [baseline]" if result.is_baseline else "")
                + (" [tuned]" if result.tuned else "")
            )
        failures = [r for r in log.results if r.failed]
        for result in failures[:4]:
            lines.append(f"- {result.family.value}: FAILED — {result.error}")
        if log.leaderboard_notes:
            lines.append(f"- notes: {log.leaderboard_notes}")
        lines.append("")
        return lines

    @staticmethod
    def _tried_block(state: RunState) -> list[str]:
        lines: list[str] = []
        if state.applied_cleaning:
            lines.append(f"- cleaning applied: {'; '.join(state.applied_cleaning[:12])}")
        if state.applied_features:
            lines.append(f"- features built: {'; '.join(state.applied_features[:14])}")
        if state.dropped_columns:
            lines.append(f"- columns dropped: {state.dropped_columns[:14]}")
        if state.splits.strategy:
            lines.append(
                f"- split strategy used: {state.splits.strategy} "
                f"({state.splits.sizes()}) — {state.splits.rationale}"
            )
        tuning = state.tuning
        if tuning is not None:
            if tuning.ran:
                lines.append(
                    f"- tuning: {tuning.method.value} on "
                    f"{tuning.family.value if tuning.family else 'n/a'}, "
                    f"{tuning.n_trials_completed} trials, "
                    f"{_score(tuning.baseline_score)} -> {_score(tuning.best_score)} "
                    f"(improvement {_score(tuning.improvement)})"
                )
            else:
                lines.append(f"- tuning: skipped ({tuning.skipped_reason or 'no reason recorded'})")
        explain = state.explainability
        if explain is not None and explain.global_attributions:
            top = ", ".join(
                f"{a.feature} {a.importance:.3f}"
                for a in explain.global_attributions[:6]
            )
            lines.append(f"- top attributions last time: {top}")
        if not lines:
            return []
        return ["### Already tried — do not simply repeat it", "", *lines, ""]

    @staticmethod
    def _previous_plan_block(state: RunState) -> list[str]:
        plan = state.plan
        if plan is None:
            return []
        lines = [
            f"### The plan being replaced (revision {plan.revision})",
            "",
            f"- summary: {plan.summary}",
            f"- fallback it named: {plan.fallback_strategy}",
            "- steps:",
        ]
        for step in plan.ordered():
            lines.append(
                f"  {step.order}. [{step.agent.value}] {step.step_id} — "
                f"{step.title}: {step.objective}"
                + (" (destructive)" if step.destructive else "")
            )
        for adaptation in plan.dataset_specific_adaptations[:8]:
            lines.append(f"- it claimed adaptation: {adaptation}")
        lines.append("")
        return lines

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: ExecutionPlan, state: RunState) -> ExecutionPlan:
        """Repair plan structure so the orchestrator can execute it blindly."""
        steps = self._normalise_ids(value.steps)
        steps = self._drop_invalid_agents(steps, state)
        steps = self._dedupe_steps(steps, state)
        if not any(step.agent not in TERMINAL_AGENTS for step in steps):
            # Terminal steps alone cannot produce a model to judge; graft a
            # standard body on rather than failing the run.
            steps = self._dedupe_steps(self._minimal_body(state) + steps, state)
        steps, synthesised = self._append_terminals(steps, state)
        steps = self._reorder(steps)
        self._renumber(steps, synthesised)
        self._prune_dependencies(steps, state)
        self._sanity_check(steps, state)
        value.steps = steps

        value.revision = (state.plan.revision + 1) if state.plan is not None else 0
        if self.is_replanning(state) and not value.revision_reason:
            evaluation = state.evaluation
            assert evaluation is not None
            weakness = evaluation.weaknesses[0] if evaluation.weaknesses else evaluation.verdict_rationale
            value.revision_reason = (
                f"Evaluation graded the previous model {evaluation.overall_grade} and "
                f"recommended '{evaluation.recommended_action}': {weakness}"
            )
        elif not self.is_replanning(state):
            value.revision_reason = None
        return value

    @staticmethod
    def _normalise_ids(steps: list[PlanStep]) -> list[PlanStep]:
        """Slugify step ids and remap the dependencies that referenced them."""
        remap: dict[str, str] = {}
        for index, step in enumerate(steps, start=1):
            new_id = _slug(step.step_id or step.title, fallback=f"step_{index}")
            if new_id != step.step_id:
                remap[step.step_id] = new_id
                step.step_id = new_id
        if remap:
            for step in steps:
                step.depends_on = [remap.get(dep, dep) for dep in step.depends_on]
        return steps

    def _drop_invalid_agents(
        self, steps: list[PlanStep], state: RunState
    ) -> list[PlanStep]:
        """Remove steps the orchestrator cannot dispatch, and self-referential ones."""
        kept: list[PlanStep] = []
        for step in steps:
            if not isinstance(step.agent, AgentName):
                state.add_warning(
                    f"{self.title}: step '{step.step_id}' names unknown agent "
                    f"'{step.agent}'; dropping it."
                )
                continue
            if step.agent in PRE_PLAN_AGENTS:
                state.add_warning(
                    f"{self.title}: step '{step.step_id}' assigns work to "
                    f"'{step.agent.value}', which runs before planning; dropping it."
                )
                continue
            kept.append(step)
        return kept

    def _dedupe_steps(self, steps: list[PlanStep], state: RunState) -> list[PlanStep]:
        """Keep the first step of any duplicated id; ids are the dependency keys."""
        seen: set[str] = set()
        kept: list[PlanStep] = []
        for step in steps:
            if step.step_id in seen:
                state.add_warning(
                    f"{self.title}: duplicate step_id '{step.step_id}'; keeping only "
                    "the first occurrence so dependencies stay unambiguous."
                )
                continue
            seen.add(step.step_id)
            kept.append(step)
        return kept

    @staticmethod
    def _reorder(steps: list[PlanStep]) -> list[PlanStep]:
        """Sort by the model's order, then force the terminal agents to the tail."""
        ordered = sorted(enumerate(steps), key=lambda pair: (pair[1].order, pair[0]))
        body = [step for _, step in ordered if step.agent not in TERMINAL_AGENTS]
        tail: list[PlanStep] = []
        for agent in TERMINAL_AGENTS:
            tail.extend(step for _, step in ordered if step.agent is agent)
        return body + tail

    def _append_terminals(
        self, steps: list[PlanStep], state: RunState
    ) -> tuple[list[PlanStep], set[str]]:
        """Guarantee the run ends by judging, interpreting, and reporting.

        Returns the steps plus the ids of the ones synthesised here, whose
        dependencies can only be wired once the final order is known.
        """
        present = {step.agent for step in steps}
        synthesised: set[str] = set()
        for agent in TERMINAL_AGENTS:
            if agent in present:
                continue
            step_id, title, objective, rationale = _TERMINAL_DEFAULTS[agent]
            self._free_step_id(step_id, steps, state)
            state.add_warning(
                f"{self.title}: the plan omitted the '{agent.value}' step; appending "
                "the mandatory one."
            )
            synthesised.add(step_id)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    order=len(steps) + 1,
                    title=title,
                    agent=agent,
                    objective=objective,
                    rationale=rationale,
                    success_criteria="The step produced its typed output.",
                    estimated_seconds=45,
                )
            )
        return steps, synthesised

    def _free_step_id(
        self, step_id: str, steps: list[PlanStep], state: RunState
    ) -> None:
        """Vacate ``step_id`` so a mandatory terminal step can own it uniquely.

        Reached when the model gave a non-terminal step one of the reserved
        terminal slugs — a `features` step called ``evaluate`` — *and* omitted the
        terminal agent. Appending the terminal step would then produce two steps
        sharing an id, and step ids are the dependency keys and the key
        ``RunState.step()`` looks up, so status updates would land on the wrong
        record. The squatter is renamed (never the mandatory step, whose canonical
        slug is what the executor graph dispatches on) and its dependents remapped.
        """
        squatter = next((s for s in steps if s.step_id == step_id), None)
        if squatter is None:
            return
        taken = {s.step_id for s in steps}
        preferred = CANONICAL_STEP_IDS.get(squatter.agent, f"{step_id}_step")
        new_id = preferred
        suffix = 2
        while new_id in taken:
            new_id = f"{preferred}_{suffix}"
            suffix += 1
        state.add_warning(
            f"{self.title}: step '{step_id}' is owned by "
            f"'{squatter.agent.value}' but '{step_id}' is the reserved id of the "
            f"mandatory step being appended; renaming it to '{new_id}'."
        )
        squatter.step_id = new_id
        for step in steps:
            step.depends_on = [
                new_id if dep == step_id else dep for dep in step.depends_on
            ]

    @staticmethod
    def _renumber(steps: list[PlanStep], synthesised: set[str]) -> None:
        """Assign a dense 1..N order and chain the steps this class invented."""
        for position, step in enumerate(steps, start=1):
            step.order = position
            if step.step_id in synthesised and not step.depends_on and position > 1:
                step.depends_on = [steps[position - 2].step_id]

    def _minimal_body(self, state: RunState) -> list[PlanStep]:
        """Standard pre-terminal pipeline, for when the plan has no body at all."""
        state.add_warning(
            f"{self.title}: returned no executable steps before the terminal ones; "
            "substituting a minimal standard pipeline so the run can continue with "
            "reduced ambition."
        )
        steps: list[PlanStep] = []
        previous: str | None = None
        for agent, step_id, objective, rationale in _MINIMAL_PIPELINE:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    order=len(steps) + 1,
                    title=objective,
                    agent=agent,
                    objective=objective,
                    rationale=rationale,
                    depends_on=[previous] if previous else [],
                    estimated_seconds=60,
                )
            )
            previous = step_id
        return steps

    def _prune_dependencies(self, steps: list[PlanStep], state: RunState) -> None:
        """Drop edges that point nowhere or forward; both deadlock a linear walk."""
        position = {step.step_id: index for index, step in enumerate(steps)}
        dangling: list[str] = []
        forward: list[str] = []
        for index, step in enumerate(steps):
            kept: list[str] = []
            for dep in step.depends_on:
                if dep == step.step_id:
                    continue
                target = position.get(dep)
                if target is None:
                    dangling.append(f"{step.step_id}->{dep}")
                    continue
                if target >= index:
                    forward.append(f"{step.step_id}->{dep}")
                    continue
                if dep not in kept:
                    kept.append(dep)
            step.depends_on = kept
        if dangling:
            state.add_warning(
                f"{self.title}: dropped {len(dangling)} dependency reference(s) to "
                f"non-existent steps {dangling[:6]}."
            )
        if forward:
            state.add_warning(
                f"{self.title}: dropped {len(forward)} dependency reference(s) that "
                f"pointed at later steps {forward[:6]}."
            )

    def _sanity_check(self, steps: list[PlanStep], state: RunState) -> None:
        """Warn about plans that are structurally legal but operationally suspect."""
        counts: dict[AgentName, int] = {}
        for step in steps:
            counts[step.agent] = counts.get(step.agent, 0) + 1
            if step.estimated_seconds < 0:
                step.estimated_seconds = 0
        repeated = [agent.value for agent, count in counts.items() if count > 1]
        if repeated:
            state.add_warning(
                f"{self.title}: {repeated} appear in more than one step; each "
                "invocation overwrites the previous output for that agent."
            )
        total = sum(step.estimated_seconds for step in steps)
        if total > state.config.time_budget_seconds:
            state.add_warning(
                f"{self.title}: the plan's own estimate ({total}s) exceeds the "
                f"{state.config.time_budget_seconds}s budget; later steps may be cut."
            )
        if not state.config.enable_tuning and any(
            step.agent is AgentName.TUNING for step in steps
        ):
            state.add_warning(
                f"{self.title}: planned a tuning step although tuning is disabled; "
                "it will be skipped at execution time."
            )
        if not state.config.enable_explainability and any(
            step.agent is AgentName.EXPLAIN for step in steps
        ):
            state.add_warning(
                f"{self.title}: planned an explainability step although "
                "explainability is disabled; it will be skipped at execution time."
            )
        task = state.task_type
        if (
            task is not None
            and task is not TaskType.TIME_SERIES_FORECASTING
            and state.problem is not None
            and state.problem.temporal_column is None
            and state.profile is not None
            and not state.profile.temporal_columns
        ):
            # Lag/rolling work is impossible without a time order; the features
            # agent will be told the same thing, but a warning here makes a
            # mis-scoped plan visible before it runs.
            for step in steps:
                if step.agent is AgentName.FEATURES and re.search(
                    r"\blag|rolling|window|seasonal", step.objective, re.IGNORECASE
                ):
                    state.add_warning(
                        f"{self.title}: step '{step.step_id}' asks for time-based "
                        "features but no temporal column exists; they cannot be built."
                    )

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: ExecutionPlan) -> None:
        """Publish the plan, archiving the one it replaces."""
        previous = state.plan
        if previous is not None and previous is not value:
            already = any(p.revision == previous.revision for p in state.plan_history)
            if not already:
                state.plan_history.append(previous)
        state.plan = value

    def decision_summary(self, value: ExecutionPlan) -> str:
        """One line for the event stream."""
        destructive = sum(1 for step in value.steps if step.destructive)
        agents = " -> ".join(step.agent.value for step in value.ordered())
        head = (
            f"revision {value.revision}: {len(value.steps)} steps "
            f"({destructive} destructive), "
            f"{len(value.dataset_specific_adaptations)} adaptations"
        )
        if value.revision_reason:
            head += f"; because {value.revision_reason}"
        return f"{head} | {agents}"


__all__ = [
    "AGENT_ROSTER",
    "CANONICAL_STEP_IDS",
    "INSTRUCTIONS_CORE",
    "MEMORY_FRAMING",
    "PRE_PLAN_AGENTS",
    "REPLAN_INSTRUCTIONS",
    "TERMINAL_AGENTS",
    "PlannerAgent",
    "plan_triggers",
]
