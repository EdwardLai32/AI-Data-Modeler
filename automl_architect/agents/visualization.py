"""Visualization Agent.

Chart selection is a judgement about questions, not a rendering exercise. A
dashboard of every plot the library can draw is worse than six plots that each
answer something a reader was going to ask, because the reader has to work out
which ones matter. So this agent is asked for a prioritised shortlist where every
spec carries the question it answers.

The postprocessing is defensive in two directions. Charts whose kind cannot exist
for this task (a ROC curve on a regression run) are dropped with a warning rather
than handed to the renderer to fail on, and the handful of charts that any reader
of this task type will expect are synthesised if the agent left them out.
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ChartKind,
    ChartSpec,
    TaskType,
    VisualizationPlan,
)
from ..core.state import RunState
from . import _deliver_support as support

#: Charts that describe data, features, or model comparison — valid for any task.
_UNIVERSAL: frozenset[ChartKind] = frozenset(
    {
        ChartKind.CORRELATION_HEATMAP,
        ChartKind.HISTOGRAM,
        ChartKind.BOX,
        ChartKind.SCATTER,
        ChartKind.BAR,
        ChartKind.LINE,
        ChartKind.FEATURE_IMPORTANCE,
        ChartKind.SHAP_SUMMARY,
        ChartKind.LEARNING_CURVE,
        ChartKind.PREDICTION_DISTRIBUTION,
        ChartKind.MISSINGNESS,
        ChartKind.LEADERBOARD,
        ChartKind.PARTIAL_DEPENDENCE,
    }
)

#: Need discrete predicted labels or class probabilities to exist at all.
_CLASSIFICATION_ONLY: frozenset[ChartKind] = frozenset(
    {
        ChartKind.ROC_CURVE,
        ChartKind.PR_CURVE,
        ChartKind.CONFUSION_MATRIX,
        ChartKind.CALIBRATION_CURVE,
    }
)

#: Need a continuous target, so that ``y_true - y_pred`` is meaningful.
_CONTINUOUS_TARGET_ONLY: frozenset[ChartKind] = frozenset(
    {ChartKind.RESIDUALS, ChartKind.RESIDUAL_HISTOGRAM}
)

_CONTINUOUS_TASKS: frozenset[TaskType] = frozenset(
    {TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING, TaskType.SURVIVAL_ANALYSIS}
)

_LABELLED_TASKS: frozenset[TaskType] = frozenset(
    {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
    }
)


def allowed_chart_kinds(task: TaskType | None) -> frozenset[ChartKind]:
    """Chart kinds that can be rendered for a task type.

    Args:
        task: The run's task type, or ``None`` when it has not been identified.

    Returns:
        The renderable kinds. Every kind is allowed when the task is unknown —
        gating on a guess would drop legitimate charts.
    """
    if task is None:
        return frozenset(ChartKind)
    allowed = set(_UNIVERSAL)
    if task.is_classification:
        allowed |= _CLASSIFICATION_ONLY
    if task in _LABELLED_TASKS:
        allowed.add(ChartKind.CLASS_BALANCE)
    if task in _CONTINUOUS_TASKS:
        allowed |= _CONTINUOUS_TARGET_ONLY
    if task is TaskType.TIME_SERIES_FORECASTING:
        allowed.add(ChartKind.TIME_SERIES_FORECAST)
    return frozenset(allowed)


def essential_chart_kinds(task: TaskType | None) -> tuple[ChartKind, ...]:
    """Charts a reader of this task type will look for regardless of the agent."""
    essentials: list[ChartKind] = [ChartKind.LEADERBOARD, ChartKind.FEATURE_IMPORTANCE]
    if task is None:
        return tuple(essentials)
    if task.is_classification:
        essentials += [ChartKind.ROC_CURVE, ChartKind.CONFUSION_MATRIX]
    if task is TaskType.REGRESSION:
        essentials.append(ChartKind.RESIDUALS)
    if task is TaskType.TIME_SERIES_FORECASTING:
        essentials.append(ChartKind.TIME_SERIES_FORECAST)
    return tuple(essentials)


_ESSENTIAL_TITLES: dict[ChartKind, tuple[str, str]] = {
    ChartKind.LEADERBOARD: (
        "Model leaderboard",
        "Which model won, and by how much over the others and the baseline — the "
        "first question anyone asks of a modelling run.",
    ),
    ChartKind.FEATURE_IMPORTANCE: (
        "Feature importance",
        "Which inputs the winning model actually relies on, which is what any "
        "recommendation about the process has to rest on.",
    ),
    ChartKind.ROC_CURVE: (
        "ROC curve",
        "How the true-positive rate trades against the false-positive rate across "
        "thresholds, so an operating point can be chosen deliberately.",
    ),
    ChartKind.CONFUSION_MATRIX: (
        "Confusion matrix",
        "Where the errors land: which class is confused with which, at the chosen "
        "operating threshold.",
    ),
    ChartKind.RESIDUALS: (
        "Residuals versus predicted",
        "Whether the error is unbiased across the prediction range, or the model "
        "systematically over- or under-predicts in part of it.",
    ),
    ChartKind.TIME_SERIES_FORECAST: (
        "Forecast versus actuals",
        "Whether the forecast tracks the holdout period, and where it diverges.",
    ),
}


class VisualizationAgent(BaseAgent[VisualizationPlan]):
    """Chooses the charts that answer a question about this specific run."""

    name = AgentName.VISUALIZATION
    title = "Visualization Agent"
    output_model = VisualizationPlan
    effort: Effort = "medium"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: chart-selection method and the task gating rules."""
        return """\
## YOUR ROLE

You are the analyst who builds the figure set for a modelling read-out. You have
seen enough dashboards to know the failure mode: twenty charts, no argument. Your
output is a prioritised shortlist where each chart earns its place by answering a
question someone will actually ask about THIS run.

## THE METHOD

1.  **Start from the questions, not the chart types.** Write down what a reader
    of this run needs to know: is the model any good, what drives it, where does
    it fail, what does the data look like, is the target balanced, did the
    engineered features matter. Then choose the one chart that answers each. If
    two charts answer the same question, keep the clearer one.
2.  **Gate on the task type before anything else.** Requesting a chart the task
    cannot produce is a wasted slot and a rendering failure:
    - ROC, precision-recall, confusion matrix, and calibration curves require
      classification. Never request them for regression or forecasting.
    - Residual plots and residual histograms require a continuous target. Never
      request them for classification.
    - Forecast-versus-actual plots require a time-series task with a temporal
      column.
    - Class-balance charts require discrete labels.
3.  **Gate on what exists.** Only name columns that appear in the column list you
    were given. A SHAP summary is only worth requesting if SHAP was available; a
    learning curve only if the diagnostics recorded one. Prefer the highest-signal
    columns — the top attributions and the strongest target correlations — over an
    arbitrary sample of the schema.
4.  **Order by decision value.** `priority=high` for the four or five charts that
    carry the argument; `medium` for supporting detail; `low` for context a reader
    might want but will usually skip. The report renders in priority order.
5.  **Say what each chart is for.** `rationale` states the question the chart
    answers and what a reader should conclude from each possible shape of it —
    "if the residual band widens at high predictions, the model is
    heteroscedastic and the high end needs a separate treatment". A chart nobody
    has a question for is noise; if you cannot write the question, drop the chart.

## SPECIFICS THAT MATTER

*   Ask for **8 to 14 charts**, prioritised. Not everything possible.
*   Univariate charts (histogram, box, bar) are for the columns that matter — the
    target, the top drivers, a column with an obvious quality problem — not a
    tour of the schema.
*   For a scatter, put the columns in `columns` as ``[x, y]`` order; for a
    histogram or box, one column; for a heatmap, either the specific columns to
    include or an empty list to mean "the numeric block".
*   Use `parameters` for rendering knobs the renderer can honour: `bins`,
    `top_n`, `normalize`, `log_x`, `color_by`, `n_classes`. Keep them few and
    obvious.
*   Titles are read by humans: "Churn rate by contract type", not
    "bar chart of contract_type".

## THE DASHBOARD NARRATIVE

`dashboard_narrative` is a guided tour, not a list of captions. Walk the reader
through the figures in the order you prioritised them, in a few short paragraphs:
what the first chart establishes, what the next one adds, where the story turns,
and what the last one leaves them able to decide. Reference the charts by their
titles. If the model is weak, the tour should say where the figures show that,
rather than presenting them as a success story."""

    def build_prompt(self, state: RunState) -> str:
        """The user turn: what can be plotted, from what, and for whom."""
        task = state.task_type
        allowed = sorted(k.value for k in allowed_chart_kinds(task))
        columns = sorted(support.real_column_names(state))

        drivers: list[str] = []
        if state.explainability:
            drivers = [
                a.feature
                for a in (
                    state.explainability.global_attributions
                    or state.explainability.permutation_importance
                )
            ][:12]

        correlations: list[str] = []
        if state.profile:
            correlations = [
                f"`{p.left}` vs `{p.right}`: {support.num(p.coefficient, 3)}"
                for p in state.profile.target_correlations[:10]
            ]

        experiments = state.experiments
        n_ok = len(support.ranked_results(experiments))
        temporal = state.problem.temporal_column if state.problem else None

        sections = [
            "# WHAT CAN BE PLOTTED FOR THIS RUN",
            "",
            "## Framing",
            support.run_facts_text(state),
            f"chart kinds renderable for this task (use ONLY these): {allowed}",
            "",
            "## Model evidence available",
            f"- experiments that trained successfully: {n_ok}"
            f"{' (a leaderboard chart is meaningful)' if n_ok > 1 else ''}",
            f"- feature attributions available: {'yes' if drivers else 'no'}",
            f"- SHAP available: "
            f"{str(bool(state.explainability and state.explainability.shap_available)).lower()}",
            f"- partial-dependence artifacts: "
            f"{len(state.explainability.partial_dependence_paths) if state.explainability else 0}",
            f"- temporal column: {f'`{temporal}`' if temporal else 'none'}",
            "",
            "## Top drivers (best candidates for univariate and dependence charts)",
            support.bullets([f"`{d}`" for d in drivers], 12) if drivers else "- (none measured)",
            "",
            "## Strongest target correlations",
            support.bullets(correlations, 10) if correlations else "- (none measured)",
            "",
            "## Target shape",
            support.target_text(state.profile),
            "",
            "## Evaluation findings the figures should make visible",
            support.evaluation_text(state.evaluation),
            "",
            "## Every column that exists (do not name anything outside this list)",
            ", ".join(f"`{c}`" for c in columns) if columns else "(no columns available)",
        ]

        if state.insights:
            sections += [
                "",
                "## Business insights the figures need to support",
                support.bullets(
                    [i.headline for i in state.insights.insights], 8
                ),
            ]

        sections += [
            "",
            "# YOUR TASK",
            "",
            "Specify the figure set for this run's report and dashboard: 8-14 charts, "
            "each with the question it answers, prioritised so the top few carry the "
            "argument on their own. Then write the dashboard narrative as a guided "
            "tour of them.",
        ]
        return "\n".join(sections)

    # -- grounding ---------------------------------------------------------

    def postprocess(
        self, value: VisualizationPlan, state: RunState
    ) -> VisualizationPlan:
        """Drop impossible charts, dedupe, and backfill the expected ones.

        Order matters: filter columns first (a spec may lose all of its columns),
        then gate on task type, then dedupe, then add essentials. Adding
        essentials last means a synthesised chart cannot be removed by a later
        pass.

        Args:
            value: The plan as returned by the model.
            state: The run blackboard.

        Returns:
            A plan whose every spec the renderer can attempt.
        """
        task = state.task_type
        allowed = allowed_chart_kinds(task)
        task_label = task.value if task else "unknown task"

        kept: list[ChartSpec] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        dropped_kinds: list[str] = []
        dropped_empty: list[str] = []
        duplicates = 0

        for spec in value.charts:
            if spec.kind not in allowed:
                dropped_kinds.append(f"{spec.kind.value} ('{support.clip(spec.title, 60)}')")
                continue

            spec.columns = support.filter_columns(
                state,
                spec.columns,
                agent_title=self.title,
                context=f"chart '{support.clip(spec.title, 60)}'",
            )
            if not spec.columns and self._needs_columns(spec.kind):
                dropped_empty.append(f"{spec.kind.value} ('{support.clip(spec.title, 60)}')")
                continue

            key = (spec.kind.value, tuple(spec.columns))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            kept.append(spec)

        if dropped_kinds:
            state.add_warning(
                f"{self.title} requested {len(dropped_kinds)} chart(s) that cannot be "
                f"rendered for a {task_label} run and were dropped: "
                f"{dropped_kinds[:6]}{'...' if len(dropped_kinds) > 6 else ''}"
            )
        if dropped_empty:
            state.add_warning(
                f"{self.title} requested {len(dropped_empty)} chart(s) whose columns "
                f"do not exist in this dataset; dropped: {dropped_empty[:6]}"
            )
        if duplicates:
            state.add_warning(
                f"{self.title} produced {duplicates} duplicate chart spec(s) "
                "(same kind and columns); the duplicates were removed."
            )

        present = {spec.kind for spec in kept}
        added: list[str] = []
        for kind in essential_chart_kinds(task):
            if kind in present:
                continue
            title, rationale = _ESSENTIAL_TITLES[kind]
            kept.append(
                ChartSpec(
                    kind=kind,
                    title=title,
                    columns=[],
                    rationale=(
                        f"{rationale} Added by the orchestrator because this figure "
                        f"is expected in any {task_label} read-out and the agent "
                        "did not request it."
                    ),
                    priority="high",
                )
            )
            present.add(kind)
            added.append(kind.value)
        if added:
            state.add_warning(
                f"{self.title} omitted essential chart(s) for a {task_label} run; "
                f"added {added}."
            )

        order = {"high": 0, "medium": 1, "low": 2}
        kept.sort(key=lambda s: order.get(s.priority, 1))
        value.charts = kept
        return value

    @staticmethod
    def _needs_columns(kind: ChartKind) -> bool:
        """Whether a chart is meaningless without column references.

        Model-diagnostic charts read predictions and the leaderboard off the run,
        not columns of the frame, so an empty ``columns`` list is correct for
        them and must not be treated as a hallucination.
        """
        return kind in {
            ChartKind.HISTOGRAM,
            ChartKind.BOX,
            ChartKind.SCATTER,
            ChartKind.BAR,
            ChartKind.LINE,
        }

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: VisualizationPlan) -> None:
        """Record the visualization plan on the blackboard."""
        state.visualization_plan = value

    def decision_summary(self, value: VisualizationPlan) -> str:
        """One line for the event stream."""
        high = sum(1 for c in value.charts if c.priority == "high")
        kinds = sorted({c.kind.value for c in value.charts})
        return (
            f"{len(value.charts)} chart(s) planned ({high} high priority): "
            f"{kinds[:8]}{'...' if len(kinds) > 8 else ''}"
        )


__all__ = ["VisualizationAgent", "allowed_chart_kinds", "essential_chart_kinds"]
