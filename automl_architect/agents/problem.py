"""Problem Identification Agent.

Everything downstream is conditioned on two decisions made here: the task type
and the primary metric. A wrong task type sends the run to the wrong model zoo
and the wrong scorer; a wrong metric optimises the wrong thing and the run can
still report a good number while being useless (accuracy on a 95/5 target being
the canonical case). So this agent reasons at ``xhigh`` effort, and its output
passes through a grounding pass that is unusually strict:

*   Operator overrides win absolutely — a human who forced the task type or the
    metric is not being asked for a second opinion.
*   Every column reference is checked against the real schema.
*   A metric from the wrong family for the task is replaced rather than passed
    to the scorer, because ``rmse`` on a classifier is a crash, not a debate.
*   An unsupported-but-honest diagnosis is kept as diagnosed, with the execution
    fallback recorded on ``state.extras`` so the report can say what happened.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    ProblemDefinition,
    TaskType,
)
from ..core.state import RunState

#: ``state.extras`` keys written when the diagnosed task cannot be executed.
EXTRA_TASK_FALLBACK = "task_type_fallback"
EXTRA_TASK_FALLBACK_REASON = "task_type_fallback_reason"

#: Above this many distinct target values, "class labels" stops being a coherent
#: reading of the column. Used only to reject an impossible classification
#: diagnosis, never to propose one.
_MAX_CLASS_LABELS = 50

_CLASSIFICATION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "brier_score",
    "cohen_kappa",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "log_loss",
    "matthews_corrcoef",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "roc_auc",
    "roc_auc_ovo",
    "roc_auc_ovr",
    "top_k_accuracy",
}
_REGRESSION_METRICS = {
    "explained_variance",
    "mae",
    "mape",
    "medae",
    "mse",
    "msle",
    "r2",
    "rmse",
    "rmsle",
    "smape",
}
_CLUSTERING_METRICS = {"silhouette", "calinski_harabasz", "davies_bouldin"}

#: Common spellings the model may emit, normalised to what ``metrics.py`` knows.
_METRIC_ALIASES = {
    "auc": "roc_auc",
    "auroc": "roc_auc",
    "roc-auc": "roc_auc",
    "roc_auc_score": "roc_auc",
    "pr_auc": "average_precision",
    "auprc": "average_precision",
    "average_precision_score": "average_precision",
    "f1_score": "f1",
    "f_1": "f1",
    "f1-macro": "f1_macro",
    "root_mean_squared_error": "rmse",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "mean_absolute_percentage_error": "mape",
    "r_squared": "r2",
    "r2_score": "r2",
    "neg_root_mean_squared_error": "rmse",
    "neg_mean_absolute_error": "mae",
    "logloss": "log_loss",
    "cross_entropy": "log_loss",
    "silhouette_score": "silhouette",
}

INSTRUCTIONS = """\
You are the problem-framing lead. Before anyone cleans a column or trains a \
model, you decide what question this dataset answers and how success is \
measured. Both decisions are structural: the task type selects the model zoo, \
the loss, and the validation scheme, and the primary metric is what every later \
comparison and the final accept/reject verdict are computed against. Getting \
either wrong produces a run that looks healthy and is worthless.

## Your method, in this order

1.  **Fix the target.** If the operator named one, it is the target and there is \
nothing to decide. Otherwise take the best candidate from the dataset \
understanding and satisfy yourself that it is an *outcome* rather than an \
attribute, that it is populated enough to model, and that it is not a \
restatement of another column. For a genuinely unsupervised framing there is no \
target — say so rather than inventing one.
2.  **Type the task from the target's measurements.**
    -   Exactly two distinct non-null values, in any dtype — including 0/1 \
integers, 0.0/1.0 floats, and yes/no strings — is binary classification.
    -   A categorical or low-cardinality target (a handful of distinct labels, \
or an object dtype whose distinct count is a small fraction of the row count) is \
multiclass classification. If the classes have a natural order (a 1-5 rating, a \
risk band), it is still multiclass here, but say in your rationale that the \
target is ordinal so later agents do not treat the classes as interchangeable.
    -   A continuous numeric target with many distinct values is regression. \
Integer counts with a wide range are regression too; note the count shape in \
`constraints` if the distribution is zero-inflated or Poisson-like.
    -   Time-series forecasting requires *all* of: a usable temporal column that \
orders the rows, a regular or near-regular frequency, a grain of one row per \
period (per series, if there is a series key), and a question about future \
values of the target. A date column on cross-sectional data does **not** make \
this a forecasting problem — it makes it a classification or regression problem \
that needs a time-aware split, which is a different decision.
    -   Reach for the exotic task types only when the data genuinely demands \
them, and expect to justify it: multilabel needs multiple simultaneous labels \
per row, ranking needs a query grouping, survival needs a duration plus a \
censoring indicator.
3.  **Watch the traps.** A float column holding only 0.0 and 1.0 is a binary \
label, not a regression target. A near-unique numeric column is an identifier, \
not a target. A target with a large missing fraction may not be usable at all — \
say so and state what has to happen. A column whose association with the target \
is near-perfect is leakage; framing the problem around it produces a model that \
predicts the past.
4.  **Choose the primary metric against the measured balance and shape**, using \
the rules below. Then choose secondary metrics that show what the primary one \
hides.
5.  **Set the supporting fields.** For binary classification, `positive_class` \
is the class the business cares about detecting — usually the minority, \
event-bearing class, and it must be one of the measured class values, spelled \
exactly. Set `temporal_column` whenever a time order exists, even for \
non-forecasting tasks, because the splitter needs it to avoid training on the \
future. Set `group_column` when rows repeat per entity, because otherwise the \
same entity lands on both sides of the split and every score is inflated. Set \
`horizon` only for forecasting, and derive it from the observed frequency and \
span rather than picking a round number.
6.  **Record the alternatives you rejected** and set `confidence` honestly. Low \
confidence with a clear explanation is far more useful than false certainty.

## Metric selection rules

Classification:

-   Near-balanced (imbalance ratio below roughly 1.5): accuracy is defensible \
and readable. `roc_auc` is still preferable when the model's output will be a \
score rather than a hard label.
-   Imbalanced (imbalance ratio at or above roughly 3, or a minority class under \
roughly 10%): **accuracy is a trap.** On a 95/5 target, a model that always \
predicts the majority class scores 0.95 and has zero value. Choose `roc_auc` \
when both classes matter and the model ranks; `average_precision` when the \
minority class *is* the point and its prevalence is low, because precision-recall \
is not flattered by the large true-negative mass; `f1` when a single hard \
threshold will be shipped and precision and recall trade off directly.
-   Multiclass: `f1_macro` when every class matters equally regardless of \
frequency, accuracy only when the classes are near-balanced, `roc_auc_ovr` for \
probability ranking, `log_loss` when calibrated probabilities are the product.
-   Whatever you choose, add `balanced_accuracy` or per-class F1 to the \
secondaries on an imbalanced target so the failure is visible in the report.

Regression:

-   Modest skew (absolute skewness below about 1) and few outliers: `rmse`. It \
matches the squared-error objective most learners optimise and penalises the \
large misses that usually cost the most.
-   Heavy tail or a material outlier fraction (roughly above 5%): `mae`, which \
does not let a handful of extreme rows dominate the score. The alternative is \
modelling `log(target)` with `rmse` — legitimate when the target is strictly \
positive and multiplicative in nature, but be explicit that the metric then \
reports in log units and needs back-transforming before anyone quotes it.
-   Target spanning orders of magnitude where relative error is what matters: \
`mape` or `smape`, but only when the target is strictly positive and never near \
zero, otherwise the metric explodes.
-   `r2` is a good reporting metric for scale-free quality and a poor \
optimisation target when the variance is dominated by outliers.

Forecasting: `mae` or `rmse` on a held-out future window; `smape` when series \
have very different scales. Always name the seasonal-naive baseline you expect \
to beat.

Never choose a metric from the wrong family for the task; it will be rejected \
before the first model trains.

## Failure modes of this specific job

-   Accuracy on an imbalanced target. The single most common way a run of this \
kind produces a confident, useless model.
-   Declaring forecasting because a timestamp exists.
-   Setting `positive_class` to the majority class, or to a value that is not \
one of the measured class labels.
-   Leaving `group_column` empty when the grain is one row per entity-period. \
The split then leaks the entity across the boundary and every score is optimistic.
-   Optimising `r2` on a heavy-tailed target and then being surprised that the \
model chases the tail.
-   Treating an ordinal target as unordered without saying so.
-   Contradicting an operator override. If the operator forced a task type or a \
metric, adopt it and, if you disagree, put the disagreement in \
`alternatives_considered` and `constraints` — never in the decision itself.

## Rationale quality

Good: "`Churn` has exactly two distinct values, 'No' (5,174 rows, 73.5%) and \
'Yes' (1,869 rows, 26.5%), so this is binary classification. The 2.77 imbalance \
ratio is mild but real, so accuracy would be beaten by a majority-class stub at \
73.5%; `roc_auc` scores the ranking of churn risk, which is what a retention \
team actually consumes." — names the measured cardinality, the measured balance, \
the concrete failure of the rejected metric, and the business use.

Bad: "The target is categorical so this is a classification problem, and we will \
use accuracy since it is the standard classification metric." — cites no measured \
value and would be written identically for a 99/1 target.

State the evidence, then the conclusion. One paragraph of real reasoning beats \
five sentences of category names.
"""


def _pct(fraction: float | None) -> str:
    return "n/a" if fraction is None else f"{fraction * 100:.2f}%"


def _num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.4g}"


def _normalise_metric(name: str) -> str:
    """Fold a metric name to the spelling the metrics module recognises."""
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return _METRIC_ALIASES.get(cleaned, cleaned)


def _metric_family(metric: str) -> str | None:
    if metric in _CLASSIFICATION_METRICS:
        return "classification"
    if metric in _REGRESSION_METRICS:
        return "regression"
    if metric in _CLUSTERING_METRICS:
        return "clustering"
    return None


def _task_family(task: TaskType) -> str:
    if task.is_classification:
        return "classification"
    if task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
        return "clustering"
    return "regression"


def default_metric_for(task: TaskType, profile: DatasetProfile | None) -> str:
    """Pick a defensible metric when the agent's choice cannot be used.

    Args:
        task: The task type execution will actually run.
        profile: Measured profile, used to read the class balance when present.

    Returns:
        A metric name from the family that matches ``task``.
    """
    family = _task_family(task)
    if family == "clustering":
        return "silhouette"
    if family == "classification":
        target = profile.target if profile else None
        if target is not None and target.is_imbalanced:
            return "average_precision" if task is TaskType.BINARY_CLASSIFICATION else "f1_macro"
        if task is TaskType.BINARY_CLASSIFICATION:
            return "roc_auc"
        return "f1_macro"
    target = profile.target if profile else None
    if target is not None and target.skewness is not None and abs(target.skewness) > 2:
        return "mae"
    return "rmse"


def nearest_supported_task(
    task: TaskType, profile: DatasetProfile | None
) -> TaskType:
    """Map an unexecutable task type onto the closest one the executors handle.

    The honest diagnosis is preserved on the :class:`ProblemDefinition`; this is
    only what the training layer will actually run.

    Args:
        task: The diagnosed task type.
        profile: Measured profile, used to type the target when the mapping
            depends on it.

    Returns:
        A ``TaskType`` for which ``is_supported`` is true.
    """
    if task.is_supported:
        return task
    if task is TaskType.MULTILABEL_CLASSIFICATION:
        return TaskType.MULTICLASS_CLASSIFICATION
    if task is TaskType.SURVIVAL_ANALYSIS:
        return TaskType.REGRESSION
    return _task_from_target(profile)


def _task_from_target(profile: DatasetProfile | None) -> TaskType:
    """Infer a supported task from the measured target alone."""
    target = profile.target if profile else None
    if target is None:
        return TaskType.CLUSTERING
    if target.kind in (
        ColumnKind.NUMERIC_CONTINUOUS,
        ColumnKind.DATETIME,
    ):
        return TaskType.REGRESSION
    n_classes = target.n_classes
    if n_classes is None:
        return TaskType.REGRESSION
    if n_classes <= 2:
        return TaskType.BINARY_CLASSIFICATION
    if n_classes <= 50:
        return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _column_line(col: ColumnProfile, n_rows: int) -> str:
    bits = [
        f"- `{col.name}`: kind={col.kind.value} dtype={col.dtype}",
        f"unique={col.n_unique:,}",
        f"missing={col.n_missing:,} ({_pct(col.missing_fraction)})",
    ]
    if col.n_unique and n_rows:
        bits.append(f"rows_per_distinct_value={n_rows / max(col.n_unique, 1):.1f}")
    if col.mean is not None:
        bits.append(
            f"min={_num(col.minimum)} mean={_num(col.mean)} max={_num(col.maximum)} "
            f"skew={_num(col.skewness)}"
        )
    if col.outliers and col.outliers.n_outliers:
        bits.append(f"outliers={_pct(col.outliers.fraction)}")
    if col.top_values:
        rendered = ", ".join(
            f"{v.value!r}:{v.count:,}({_pct(v.fraction)})" for v in col.top_values[:6]
        )
        bits.append(f"values=[{rendered}]")
    if col.looks_like_id:
        bits.append("ID_LIKE")
    return " | ".join(bits)


class ProblemAgent(BaseAgent[ProblemDefinition]):
    """Decides the task type, the target, and the metric that defines success."""

    name = AgentName.PROBLEM
    title = "Problem Identification Agent"
    output_model = ProblemDefinition
    effort: Effort = "xhigh"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: task typing and metric selection, with the traps."""
        return INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        """The user turn: target candidates, time structure, and the question."""
        lines: list[str] = ["## YOUR TASK", ""]
        lines.extend(self._understanding_block(state))
        lines.extend(self._target_block(state))
        lines.extend(self._temporal_block(state))
        lines.extend(self._grouping_block(state))
        lines.extend(self._override_block(state))
        lines.extend(
            [
                "### Decide",
                "1. The task type, with the measured evidence that forces it.",
                "2. The target column (or none, if the honest framing is "
                "unsupervised), plus `positive_class` if the task is binary.",
                "3. The primary metric, argued against the measured class balance "
                "or target shape, and the secondary metrics that expose what it "
                "hides.",
                "4. `temporal_column` if any time order exists, and "
                "`group_column` if rows repeat per entity — both drive how the "
                "data is split, and omitting them silently inflates every score.",
                "5. The alternative framings you rejected, and why.",
                "6. The business objective: what a decision-maker does with this "
                "model, stated concretely enough to be wrong.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _understanding_block(state: RunState) -> list[str]:
        understanding = state.understanding
        if understanding is None:
            return [
                "No dataset-understanding pass is available, so reason from the "
                "measured facts in the context above alone.",
                "",
            ]
        lines = [
            "### What the dataset agent concluded",
            f"- headline: {understanding.headline}",
            f"- domain: {understanding.likely_domain}",
            f"- grain (one row = ): {understanding.grain}",
            f"- readiness: {understanding.data_readiness} — {understanding.readiness_rationale}",
        ]
        if understanding.suggested_target_columns:
            lines.append(
                "- ranked target candidates: "
                + ", ".join(f"`{c}`" for c in understanding.suggested_target_columns[:8])
            )
        for risk in understanding.risks[:5]:
            lines.append(f"- risk raised: {risk}")
        suspects = [
            a.name
            for a in understanding.column_assessments
            if a.role.value == "leakage_suspect"
        ]
        if suspects:
            lines.append(
                "- flagged as leakage suspects: "
                + ", ".join(f"`{c}`" for c in suspects[:10])
            )
        lines.append("")
        lines.append(
            "That is a colleague's read, not ground truth. Where it conflicts with "
            "the measurements, the measurements win."
        )
        lines.append("")
        return lines

    def _target_block(self, state: RunState) -> list[str]:
        profile = state.profile
        if profile is None:
            return []
        if profile.target is not None:
            target = profile.target
            if target.class_counts:
                detail = (
                    "its class counts and imbalance ratio are in the context above. "
                    "Read the balance off those numbers when you argue the metric — "
                    "the majority-class score is the bar any hard-label metric has to "
                    "clear."
                )
            else:
                detail = (
                    "its mean, spread, skewness, and outlier fraction are in the "
                    "context above. Argue the metric from that shape, not from "
                    "convention."
                )
            return [
                "### Target",
                f"The profiler measured `{target.name}` (kind={target.kind.value}) as "
                f"the target; {detail}",
                "",
            ]

        candidates: list[str] = []
        if state.understanding:
            candidates.extend(state.understanding.suggested_target_columns)
        if state.config.target_column:
            candidates.insert(0, state.config.target_column)
        seen: set[str] = set()
        ordered = [c for c in candidates if not (c in seen or seen.add(c))]

        lines = ["### Candidate targets, measured", ""]
        if not ordered:
            lines.append(
                "No candidate was nominated. Identify the target yourself from the "
                "column statistics above, or conclude that the honest framing is "
                "unsupervised."
            )
            lines.append("")
            return lines
        lines.append(
            "No target was configured, so the profiler measured no class balance. "
            "These are the raw statistics for the nominated candidates:"
        )
        for name in ordered[:6]:
            col = profile.column(name)
            if col is not None:
                lines.append(_column_line(col, profile.n_rows))
        lines.append("")
        return lines

    @staticmethod
    def _temporal_block(state: RunState) -> list[str]:
        profile = state.profile
        if profile is None:
            return []
        if not profile.temporal_columns:
            return [
                "### Time structure",
                "No temporal column was detected. Time-series forecasting is "
                "therefore off the table, and a time-ordered split is impossible; "
                "leave `temporal_column` and `horizon` null.",
                "",
            ]
        lines = ["### Time structure (decides forecasting vs a time-aware split)"]
        for name in profile.temporal_columns[:6]:
            col = profile.column(name)
            if col is None:
                continue
            lines.append(
                f"- `{name}`: range={col.min_timestamp} .. {col.max_timestamp} | "
                f"freq={col.inferred_frequency or 'irregular'} | "
                f"gaps={col.n_gaps if col.n_gaps is not None else 'n/a'} | "
                f"monotonic={str(col.is_monotonic).lower()} | "
                f"distinct={col.n_unique:,} of {profile.n_rows:,} rows | "
                f"missing={_pct(col.missing_fraction)}"
            )
        lines.append("")
        lines.append(
            "One row per timestamp with a regular frequency supports forecasting. "
            "Many rows per timestamp, or an irregular one, means this is a "
            "cross-sectional problem that merely needs a time-ordered split."
        )
        lines.append("")
        return lines

    @staticmethod
    def _grouping_block(state: RunState) -> list[str]:
        profile = state.profile
        if profile is None or profile.n_rows <= 0:
            return []
        candidates: list[tuple[str, float, int]] = []
        for col in profile.columns:
            if col.kind not in (
                ColumnKind.CATEGORICAL_NOMINAL,
                ColumnKind.CATEGORICAL_ORDINAL,
                ColumnKind.IDENTIFIER,
                ColumnKind.NUMERIC_DISCRETE,
            ):
                continue
            if col.n_unique < 2 or col.n_unique >= profile.n_rows:
                continue
            per_group = profile.n_rows / col.n_unique
            if per_group < 2:
                continue
            candidates.append((col.name, per_group, col.n_unique))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[2], reverse=True)
        lines = ["### Repeated-entity candidates (for `group_column`)"]
        for name, per_group, n_unique in candidates[:8]:
            lines.append(
                f"- `{name}`: {n_unique:,} distinct values, "
                f"{per_group:.1f} rows each on average"
            )
        lines.append("")
        lines.append(
            "High rows-per-value on an entity-like column means the grain repeats "
            "per entity; that demands a grouped split. High rows-per-value on an "
            "ordinary low-cardinality category does not."
        )
        lines.append("")
        return lines

    @staticmethod
    def _override_block(state: RunState) -> list[str]:
        config = state.config
        lines: list[str] = []
        if config.target_column:
            lines.append(
                f"- The operator set the target to `{config.target_column}`. Use it."
            )
        if config.task_type_override:
            lines.append(
                f"- The operator forced task_type = {config.task_type_override.value}. "
                "Emit exactly that. If the data argues otherwise, record the "
                "disagreement in `alternatives_considered`."
            )
        if config.primary_metric_override:
            lines.append(
                f"- The operator forced primary_metric = {config.primary_metric_override}. "
                "Emit exactly that, and use `metric_rationale` to explain what it "
                "will and will not reveal on this target."
            )
        if config.min_acceptable_score is not None:
            lines.append(
                f"- The run is only acceptable at a primary-metric score of "
                f"{config.min_acceptable_score} or better; that bar should shape "
                "which metric you consider meaningful."
            )
        if not lines:
            return []
        return ["### Non-negotiable operator settings", *lines, ""]

    # -- grounding ---------------------------------------------------------

    def postprocess(
        self, value: ProblemDefinition, state: RunState
    ) -> ProblemDefinition:
        """Apply overrides, validate columns, and record any execution fallback."""
        config = state.config
        profile = state.profile

        if config.task_type_override and value.task_type is not config.task_type_override:
            state.add_warning(
                f"{self.title}: diagnosed {value.task_type.value} but the operator "
                f"forced {config.task_type_override.value}; the override wins."
            )
            value.alternatives_considered = [
                *value.alternatives_considered,
                f"agent's own diagnosis: {value.task_type.value} (superseded by the "
                "operator override)",
            ]
            value.task_type = config.task_type_override

        value.target_column = self._checked_column(
            value.target_column, state, field="target_column"
        )
        value.temporal_column = self._checked_column(
            value.temporal_column, state, field="temporal_column"
        )
        value.group_column = self._checked_column(
            value.group_column, state, field="group_column"
        )

        self._repair_target(value, state)
        self._repair_task_type(value, state, profile)
        self._repair_positive_class(value, state, profile)
        self._repair_temporal(value, state, profile)
        self._repair_metric(value, state, profile)
        self._record_fallback(value, state, profile)
        return value

    def _checked_column(
        self, name: str | None, state: RunState, *, field: str
    ) -> str | None:
        if not name:
            return None
        kept = self.keep_known_columns([name], state, context=field)
        return kept[0] if kept else None

    def _repair_target(self, value: ProblemDefinition, state: RunState) -> None:
        """A supervised task without a target cannot run; substitute the best known one."""
        if value.target_column or not value.task_type.is_supervised:
            return
        fallbacks: list[str] = []
        if state.config.target_column:
            fallbacks.append(state.config.target_column)
        if state.profile and state.profile.target:
            fallbacks.append(state.profile.target.name)
        if state.understanding:
            fallbacks.extend(state.understanding.suggested_target_columns)
        for candidate in fallbacks:
            resolved = self._checked_column(candidate, state, field="target_column")
            if resolved:
                state.add_warning(
                    f"{self.title}: returned no target for the supervised task "
                    f"{value.task_type.value}; falling back to `{resolved}`."
                )
                value.target_column = resolved
                return
        state.add_warning(
            f"{self.title}: no usable target column for supervised task "
            f"{value.task_type.value}; training will not be able to proceed."
        )

    def _repair_task_type(
        self,
        value: ProblemDefinition,
        state: RunState,
        profile: DatasetProfile | None,
    ) -> None:
        """Reject a classification diagnosis the measured target cannot support.

        The metric guard below stops ``rmse`` reaching a classifier, but the
        mirror case is just as fatal and was unguarded: a *classification* task
        declared on a continuous target sends ``roc_auc`` to the scorer, which
        raises ``ValueError: multi_class must be in ('ovo', 'ovr')`` mid
        cross-validation and fails the whole run.

        Deliberately narrow. It only fires where the measurement makes the
        declared task arithmetically impossible — a continuous, many-valued
        target cannot carry two class labels — and it never argues with an
        operator override, because a human who forced the task type is not asking
        for a second opinion. Ambiguous cases (a numeric target stored as text, an
        ordinal rating, a 0/1 column framed as regression) are left alone: they
        train, and second-guessing them would overrule a legitimate framing.
        """
        if state.config.task_type_override:
            return
        if not value.task_type.is_classification or not value.target_column:
            return
        column = profile.column(value.target_column) if profile else None
        if column is None:
            return

        target = profile.target if profile else None
        measured_classes = (
            target.n_classes
            if target is not None and target.name == value.target_column
            else None
        )
        distinct = measured_classes if measured_classes is not None else column.n_unique
        if distinct <= 0:
            return

        continuous = column.kind is ColumnKind.NUMERIC_CONTINUOUS
        if continuous and distinct > _MAX_CLASS_LABELS:
            corrected = TaskType.REGRESSION
        elif value.task_type is TaskType.BINARY_CLASSIFICATION and distinct > 2:
            corrected = (
                TaskType.REGRESSION
                if continuous or distinct > _MAX_CLASS_LABELS
                else TaskType.MULTICLASS_CLASSIFICATION
            )
        else:
            return

        state.add_warning(
            f"{self.title}: diagnosed {value.task_type.value} but `"
            f"{value.target_column}` is measured as {column.kind.value} with "
            f"{distinct:,} distinct values, which cannot be class labels; "
            f"re-typing the task as {corrected.value}."
        )
        value.alternatives_considered = [
            *value.alternatives_considered,
            f"agent's own diagnosis: {value.task_type.value} (contradicted by the "
            f"measured target and corrected to {corrected.value})",
        ]
        value.rationale = (
            f"{value.rationale}\n\n[Grounding note: the agent diagnosed "
            f"{value.task_type.value}, but `{value.target_column}` is "
            f"{column.kind.value} with {distinct:,} distinct values, so the task "
            f"was corrected to {corrected.value}.]"
        )
        value.task_type = corrected

    def _repair_positive_class(
        self,
        value: ProblemDefinition,
        state: RunState,
        profile: DatasetProfile | None,
    ) -> None:
        """Keep ``positive_class`` a real, measured label of a binary target."""
        if value.task_type is not TaskType.BINARY_CLASSIFICATION:
            if value.positive_class:
                state.add_warning(
                    f"{self.title}: positive_class='{value.positive_class}' is "
                    f"meaningless for {value.task_type.value}; clearing it."
                )
                value.positive_class = None
            return

        target = profile.target if profile else None
        measured = [c.value for c in target.class_counts] if target else []
        if not measured:
            return
        if value.positive_class in measured:
            return
        # Default to the least frequent class: on an imbalanced binary target the
        # rare, event-bearing class is what anybody is trying to detect.
        minority = min(target.class_counts, key=lambda c: c.count).value  # type: ignore[union-attr]
        if value.positive_class:
            state.add_warning(
                f"{self.title}: positive_class='{value.positive_class}' is not one "
                f"of the measured labels {measured}; using the minority class "
                f"'{minority}' instead."
            )
        else:
            state.add_warning(
                f"{self.title}: no positive_class was set for a binary task; "
                f"defaulting to the minority class '{minority}'."
            )
        value.positive_class = minority

    def _repair_temporal(
        self,
        value: ProblemDefinition,
        state: RunState,
        profile: DatasetProfile | None,
    ) -> None:
        """Forecasting without a time index is not executable; degrade explicitly."""
        if value.task_type is not TaskType.TIME_SERIES_FORECASTING:
            if value.horizon is not None:
                value.horizon = None
            return

        if not value.temporal_column and profile and profile.temporal_columns:
            substitute = profile.temporal_columns[0]
            state.add_warning(
                f"{self.title}: forecasting was diagnosed without a temporal "
                f"column; using the detected `{substitute}`."
            )
            value.temporal_column = substitute

        if not value.temporal_column:
            state.extras[EXTRA_TASK_FALLBACK] = _task_from_target(profile).value
            state.extras[EXTRA_TASK_FALLBACK_REASON] = (
                "time_series_forecasting was diagnosed but no temporal column "
                "exists, so execution will treat this as a tabular problem."
            )
            state.add_warning(
                f"{self.title}: forecasting requires a temporal column and none "
                "exists; execution will fall back to "
                f"{state.extras[EXTRA_TASK_FALLBACK]}."
            )
            return

        if value.horizon is not None and value.horizon < 1:
            state.add_warning(
                f"{self.title}: horizon={value.horizon} is not a usable forecast "
                "length; clearing it so the executor picks a default."
            )
            value.horizon = None

    def _repair_metric(
        self,
        value: ProblemDefinition,
        state: RunState,
        profile: DatasetProfile | None,
    ) -> None:
        """Normalise metric spellings and reject cross-family metrics."""
        override = state.config.primary_metric_override
        if override:
            normalised = _normalise_metric(override)
            if _normalise_metric(value.primary_metric) != normalised:
                state.add_warning(
                    f"{self.title}: chose '{value.primary_metric}' but the operator "
                    f"forced '{override}'; the override wins."
                )
            value.primary_metric = normalised
        else:
            value.primary_metric = _normalise_metric(value.primary_metric)
            wanted = _task_family(value.task_type)
            family = _metric_family(value.primary_metric)
            if family is not None and family != wanted:
                replacement = default_metric_for(value.task_type, profile)
                state.add_warning(
                    f"{self.title}: '{value.primary_metric}' is a {family} metric "
                    f"but the task is {value.task_type.value}; using "
                    f"'{replacement}' instead."
                )
                value.metric_rationale = (
                    f"{value.metric_rationale}\n\n[Grounding note: the agent's choice "
                    f"of '{value.primary_metric}' does not apply to a "
                    f"{value.task_type.value} task and was replaced with "
                    f"'{replacement}'.]"
                )
                value.primary_metric = replacement

        secondaries: list[str] = []
        for metric in value.secondary_metrics:
            normalised = _normalise_metric(metric)
            if normalised == value.primary_metric or normalised in secondaries:
                continue
            secondaries.append(normalised)
        value.secondary_metrics = secondaries

    def _record_fallback(
        self,
        value: ProblemDefinition,
        state: RunState,
        profile: DatasetProfile | None,
    ) -> None:
        """Keep an unsupported diagnosis honest, but tell execution what to run.

        ``ProblemDefinition`` has no field for "what we will actually train", and
        inventing one would break the shared contract, so the mapping lives on
        ``state.extras`` where the runner and the report can both read it.
        """
        if value.task_type.is_supported:
            return
        substitute = nearest_supported_task(value.task_type, profile)
        state.extras[EXTRA_TASK_FALLBACK] = substitute.value
        state.extras[EXTRA_TASK_FALLBACK_REASON] = (
            f"{value.task_type.value} has no end-to-end execution path in this "
            f"build; the closest executable framing is {substitute.value}."
        )
        state.add_warning(
            f"{self.title}: diagnosed {value.task_type.value}, which the execution "
            f"layer cannot train end-to-end; it will fall back to "
            f"{substitute.value}. The diagnosis is preserved in the report."
        )
        if _metric_family(value.primary_metric) != _task_family(substitute):
            replacement = default_metric_for(substitute, profile)
            state.extras["primary_metric_fallback"] = replacement
            state.add_warning(
                f"{self.title}: metric '{value.primary_metric}' cannot be scored "
                f"under the {substitute.value} fallback; execution will use "
                f"'{replacement}'."
            )

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: ProblemDefinition) -> None:
        """Publish the problem definition onto the blackboard."""
        state.problem = value

    def decision_summary(self, value: ProblemDefinition) -> str:
        """One line for the event stream."""
        bits: list[Any] = [
            f"{value.task_type.value} on `{value.target_column or 'no target'}`",
            f"metric={value.primary_metric}",
            f"confidence={value.confidence}",
        ]
        if value.positive_class:
            bits.append(f"positive='{value.positive_class}'")
        if value.temporal_column:
            bits.append(f"time=`{value.temporal_column}`")
        if value.group_column:
            bits.append(f"group=`{value.group_column}`")
        if value.horizon:
            bits.append(f"horizon={value.horizon}")
        return ", ".join(str(b) for b in bits)


__all__ = [
    "EXTRA_TASK_FALLBACK",
    "EXTRA_TASK_FALLBACK_REASON",
    "INSTRUCTIONS",
    "ProblemAgent",
    "default_metric_for",
    "nearest_supported_task",
]
