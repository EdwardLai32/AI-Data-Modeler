"""Text renderers shared by the delivery agents.

The insight, visualization, report, and Q&A agents all reason over the same tail
of a run — the leaderboard, the recorded decisions and their rationales, the
feature attributions, the evaluation verdict, the event log. Rendering that
material once here keeps each agent module about its *prompt*, which is the part
that carries the product value, and guarantees the four agents describe the same
run the same way.

Two conventions hold throughout:

*   Nothing here calls the model or mutates :class:`RunState`. Every function
    maps typed run state to compact prompt text.
*   Numbers are never invented or rounded into something they are not. A missing
    measurement renders as ``n/a`` so an agent can see the gap instead of
    inferring a value.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence

from ..core.schemas import (
    CleaningPlan,
    DatasetProfile,
    EvaluationVerdict,
    ExecutionPlan,
    ExperimentLog,
    ExperimentResult,
    ExplainabilityReport,
    FeaturePlan,
    InsightReport,
    Param,
    RunEvent,
    TuningDecision,
    TuningResult,
    VisualizationPlan,
    params_to_dict,
)
from ..core.state import RunState

MAX_TABLE_ROWS = 25
MAX_EVENTS = 80


# ---------------------------------------------------------------------------
# Scalar formatting
# ---------------------------------------------------------------------------


def num(value: float | int | None, digits: int = 4) -> str:
    """Format a measurement for a prompt, or ``n/a`` when it was not measured."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        if value != value:  # NaN
            return "n/a"
        if value in (float("inf"), float("-inf")):
            return "inf"
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.{digits}g}"


def pct(fraction: float | None, digits: int = 2) -> str:
    """Render a 0-1 fraction as a percentage."""
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.{digits}f}%"


def mb(n_bytes: int | float | None) -> str:
    """Render a byte count in megabytes."""
    if not n_bytes:
        return "n/a"
    return f"{float(n_bytes) / (1024 * 1024):.3f} MB"


def clip(text: str | None, limit: int = 240) -> str:
    """Collapse whitespace and truncate, so one long field cannot flood a prompt."""
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def bullets(items: Iterable[str] | None, limit: int = 12, indent: str = "- ") -> str:
    """Render an iterable of strings as a bullet list, capped at ``limit``."""
    values = [clip(i, 300) for i in (items or []) if str(i).strip()]
    if not values:
        return f"{indent}(none recorded)"
    lines = [f"{indent}{v}" for v in values[:limit]]
    if len(values) > limit:
        lines.append(f"{indent}… and {len(values) - limit} more")
    return "\n".join(lines)


def params_text(params: Sequence[Param] | None, limit: int = 10) -> str:
    """Render ``list[Param]`` as ``k=v`` pairs."""
    data = params_to_dict(list(params or []))
    if not data:
        return "defaults"
    items = list(data.items())[:limit]
    rendered = ", ".join(f"{k}={v!r}" for k, v in items)
    if len(data) > limit:
        rendered += f", … (+{len(data) - limit})"
    return rendered


def _md_cell(text: str) -> str:
    """Flatten and escape a cell so it cannot break out of its table row."""
    return clip(text, 400).replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Build a markdown table. Returns an empty string when there are no rows."""
    if not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_md_cell(str(cell)) for cell in row) + " |" for row in rows
    ]
    return "\n".join([head, rule, *body])


# ---------------------------------------------------------------------------
# Column grounding
# ---------------------------------------------------------------------------


def real_column_names(state: RunState) -> set[str]:
    """Every column name that genuinely exists somewhere in this run.

    The union matters: an agent may legitimately reference a raw column that
    cleaning dropped, the target (absent from the feature frame), or an
    engineered feature (absent from the raw frame). Filtering against only the
    most-processed frame would reject all three.

    Args:
        state: The run blackboard.

    Returns:
        Set of known column and feature names. Empty when nothing is loaded yet,
        which callers should treat as "cannot verify" rather than "all invalid".
    """
    names: set[str] = set()
    frames = (
        state.raw_df,
        state.working_df,
        state.feature_frame,
        state.splits.X_train,
        state.splits.X_test,
    )
    for frame in frames:
        columns = getattr(frame, "columns", None)
        if columns is None:
            continue
        try:
            names.update(str(c) for c in columns)
        except TypeError:  # pragma: no cover - exotic frame-likes
            continue
    if state.profile:
        names.update(c.name for c in state.profile.columns)
    names.update(str(n) for n in (state.feature_names or []))
    if state.target:
        names.add(str(state.target))
    return names


def filter_columns(
    state: RunState,
    columns: Sequence[str],
    *,
    agent_title: str,
    context: str = "",
) -> list[str]:
    """Keep only column references that exist somewhere in this run.

    ``BaseAgent.keep_known_columns`` checks against the most-processed frame
    alone, which rejects legitimate references to the target and to raw columns
    that cleaning dropped. The delivery agents describe the whole run, so they
    check against the union instead — while keeping the same discipline of
    logging every rejection.

    Args:
        state: The run blackboard.
        columns: Agent-supplied column names.
        agent_title: Used in the warning message.
        context: Where the reference appeared, for the warning message.

    Returns:
        The subset of ``columns`` that really exists, order preserved.
    """
    known = real_column_names(state)
    if not known:
        return list(columns)
    kept = [c for c in columns if c in known]
    unknown = [c for c in columns if c not in known]
    if unknown:
        state.add_warning(
            f"{agent_title} referenced {len(unknown)} unknown column(s) "
            f"{unknown[:8]}{'...' if len(unknown) > 8 else ''}"
            f"{f' in {context}' if context else ''}; ignoring them."
        )
    return kept


def known_reference_names(state: RunState) -> set[str]:
    """Names an agent may cite in prose: columns, features, metrics, families.

    Used to spot hallucinated back-ticked references without flagging legitimate
    non-column citations such as ``roc_auc`` or ``lightgbm``.
    """
    names = real_column_names(state)
    if state.explainability:
        names.update(a.feature for a in state.explainability.global_attributions)
        names.update(a.feature for a in state.explainability.permutation_importance)
    log = state.experiments
    if log:
        names.add(log.primary_metric)
        for result in log.results:
            names.add(result.family.value)
            names.add(result.label)
            names.update(m.name for m in result.metrics)
    if state.problem:
        names.add(state.problem.primary_metric)
        names.update(state.problem.secondary_metrics)
        names.add(state.problem.task_type.value)
    names.add(state.primary_metric)
    return {n for n in names if n}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def ranked_results(log: ExperimentLog | None) -> list[ExperimentResult]:
    """Successful experiments ordered best-first by the primary metric."""
    if log is None:
        return []
    scored = [r for r in log.results if not r.failed and r.primary_score is not None]
    return sorted(
        scored,
        key=lambda r: r.primary_score,  # type: ignore[arg-type,return-value]
        reverse=bool(log.higher_is_better),
    )


def best_experiment(state: RunState) -> ExperimentResult | None:
    """The winning experiment: the recorded best, else the top-ranked one."""
    log = state.experiments
    if log is None:
        return None
    recorded = log.best()
    if recorded is not None:
        return recorded
    ranked = ranked_results(log)
    return ranked[0] if ranked else None


def baseline_experiment(log: ExperimentLog | None) -> ExperimentResult | None:
    """The baseline run, which is what a lift claim must be measured against."""
    if log is None:
        return None
    return next((r for r in log.results if r.is_baseline and not r.failed), None)


def _cv_text(result: ExperimentResult) -> str:
    scores = [s for s in result.cv_scores if s == s]
    if not scores:
        return "n/a"
    if len(scores) == 1:
        return num(scores[0])
    return f"{num(statistics.fmean(scores))} ± {num(statistics.pstdev(scores), 3)}"


def metrics_text(result: ExperimentResult, limit: int = 8) -> str:
    """All measured metrics for one experiment as ``name=value`` pairs."""
    if not result.metrics:
        return "no metrics recorded"
    shown = result.metrics[:limit]
    return ", ".join(f"{m.name}={num(m.value)}" for m in shown)


def _label(result: ExperimentResult) -> str:
    tags = []
    if result.is_baseline:
        tags.append("baseline")
    if result.tuned:
        tags.append("tuned")
    suffix = f" ({', '.join(tags)})" if tags else ""
    return f"{result.label or result.family.value}{suffix}"


def leaderboard_markdown(log: ExperimentLog | None, limit: int = MAX_TABLE_ROWS) -> str:
    """The leaderboard as a real markdown table, for the report deliverable."""
    if log is None or not log.results:
        return "_No experiments were recorded for this run._"
    metric = log.primary_metric or "primary metric"
    rows: list[list[str]] = []
    for rank, result in enumerate(ranked_results(log)[:limit], start=1):
        rows.append(
            [
                str(rank),
                _label(result),
                num(result.primary_score),
                _cv_text(result),
                f"{num(result.train_seconds, 3)}s",
                f"{num(result.predict_seconds, 3)}s",
                mb(result.model_size_bytes),
                str(result.n_features_in or "n/a"),
            ]
        )
    table = md_table(
        [
            "#",
            "Model",
            f"{metric}",
            "CV mean ± sd",
            "Train",
            "Predict",
            "Size",
            "Features",
        ],
        rows,
    )
    failures = [r for r in log.results if r.failed]
    parts = [table or "_No model completed training._"]
    direction = "higher is better" if log.higher_is_better else "lower is better"
    parts.append(f"\n_Ranked by {metric} ({direction})._")
    if failures:
        parts.append("")
        parts.append("Families that failed to fit:")
        parts.append(
            "\n".join(
                f"- `{r.family.value}`: {clip(r.error, 200) or 'unspecified error'}"
                for r in failures[:10]
            )
        )
    if log.leaderboard_notes:
        parts.append("")
        parts.append(clip(log.leaderboard_notes, 600))
    return "\n".join(parts)


def leaderboard_text(log: ExperimentLog | None, limit: int = 15) -> str:
    """Compact leaderboard for prompts (cheaper than the markdown table)."""
    if log is None or not log.results:
        return "(no experiments recorded)"
    metric = log.primary_metric or "primary"
    lines = [
        f"metric={metric} ({'higher' if log.higher_is_better else 'lower'} is better)"
    ]
    for rank, result in enumerate(ranked_results(log)[:limit], start=1):
        lines.append(
            f"{rank}. {_label(result)}: {metric}={num(result.primary_score)} "
            f"cv={_cv_text(result)} | {metrics_text(result)} | "
            f"train={num(result.train_seconds, 3)}s "
            f"predict={num(result.predict_seconds, 3)}s "
            f"size={mb(result.model_size_bytes)} "
            f"n_features={result.n_features_in} | params: {params_text(result.params, 6)}"
        )
    for result in [r for r in log.results if r.failed][:8]:
        lines.append(f"x. {result.family.value}: FAILED — {clip(result.error, 160)}")
    return "\n".join(lines)


def winner_vs_baseline_text(state: RunState) -> str:
    """The winning model against the baseline, with the lift spelled out."""
    log = state.experiments
    winner = best_experiment(state)
    if winner is None:
        return "(no winning model was recorded)"
    metric = log.primary_metric if log else state.primary_metric
    lines = [
        f"winner: {_label(winner)} ({winner.family.value})",
        f"  {metric}={num(winner.primary_score)} cv={_cv_text(winner)}",
        f"  all metrics: {metrics_text(winner, 10)}",
        f"  params: {params_text(winner.params, 10)}",
        f"  cost profile: train={num(winner.train_seconds, 3)}s "
        f"predict={num(winner.predict_seconds, 3)}s "
        f"peak_memory={num(winner.peak_memory_mb, 3)}MB "
        f"size={mb(winner.model_size_bytes)} n_features_in={winner.n_features_in}",
    ]
    base = baseline_experiment(log)
    if base is None or base.primary_score is None or winner.primary_score is None:
        lines.append("  baseline: none recorded, so lift cannot be quantified")
        return "\n".join(lines)
    higher = bool(log.higher_is_better) if log else True
    delta = winner.primary_score - base.primary_score
    gain = delta if higher else -delta
    denominator = abs(base.primary_score) or None
    relative = f"{gain / denominator * 100:.1f}%" if denominator else "n/a"
    lines.append(
        f"  baseline ({_label(base)}): {metric}={num(base.primary_score)}"
    )
    lines.append(
        f"  lift over baseline: {num(gain)} absolute ({relative} relative); "
        f"{'the model beats' if gain > 0 else 'the model does NOT beat'} the baseline"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def attributions_text(
    report: ExplainabilityReport | None, limit: int = 15
) -> str:
    """Top feature attributions with direction and method."""
    if report is None:
        return "(no explainability report was produced)"
    lines: list[str] = []
    if report.global_attributions:
        lines.append(f"global attributions (method={report.global_attributions[0].method}):")
        for rank, attr in enumerate(report.global_attributions[:limit], start=1):
            lines.append(
                f"  {rank}. {attr.feature}: importance={num(attr.importance)} "
                f"direction={attr.direction}"
            )
    if report.permutation_importance:
        lines.append("permutation importance (holdout):")
        for rank, attr in enumerate(report.permutation_importance[:limit], start=1):
            lines.append(
                f"  {rank}. {attr.feature}: importance={num(attr.importance)} "
                f"direction={attr.direction}"
            )
    lines.append(f"shap available: {str(report.shap_available).lower()}")
    if report.plain_language_explanations:
        lines.append("plain-language explanations already recorded:")
        lines.append(bullets(report.plain_language_explanations, 10, "  - "))
    if report.counterfactuals:
        lines.append("counterfactuals computed:")
        for cf in report.counterfactuals[:5]:
            lines.append(
                f"  - {clip(cf.description, 180)} "
                f"[{params_text(cf.changed_features, 5)}] "
                f"{cf.original_prediction} -> {cf.new_prediction}"
            )
    if report.narrative:
        lines.append(f"explainability narrative: {clip(report.narrative, 900)}")
    if report.method_notes:
        lines.append(f"method notes: {clip(report.method_notes, 400)}")
    return "\n".join(lines) if lines else "(explainability report is empty)"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluation_text(verdict: EvaluationVerdict | None) -> str:
    """The quality gate's findings, verbatim enough to argue from."""
    if verdict is None:
        return "(evaluation has not run; treat model quality as unverified)"
    bv = verdict.bias_variance
    cal = verdict.calibration
    lines = [
        f"acceptable: {str(verdict.acceptable).lower()} | grade: {verdict.overall_grade}",
        f"verdict rationale: {clip(verdict.verdict_rationale, 900)}",
        f"recommended action: {verdict.recommended_action} — "
        f"{clip(verdict.action_rationale, 400)}",
        f"bias/variance: verdict={bv.verdict} train={num(bv.train_score)} "
        f"valid={num(bv.validation_score)} test={num(bv.test_score)} gap={num(bv.gap)}"
        f"{' — ' + clip(bv.detail, 300) if bv.detail else ''}",
        f"calibration: applicable={str(cal.applicable).lower()} "
        f"brier={num(cal.brier_score)} ece={num(cal.expected_calibration_error)} "
        f"{clip(cal.verdict, 240)}",
        f"drift risk: {verdict.drift_risk} — {clip(verdict.drift_rationale, 300)}",
    ]
    if verdict.generalisation_notes:
        lines.append(f"generalisation: {clip(verdict.generalisation_notes, 400)}")
    if verdict.confidence_intervals:
        rendered = "; ".join(
            f"{ci.metric}={num(ci.point_estimate)} "
            f"[{num(ci.lower)}, {num(ci.upper)}] @{pct(ci.level, 0)} ({ci.method})"
            for ci in verdict.confidence_intervals[:8]
        )
        lines.append(f"confidence intervals: {rendered}")
    if verdict.fairness_slices:
        lines.append("fairness slices (metric vs overall):")
        for slice_ in verdict.fairness_slices[:12]:
            lines.append(
                f"  - {slice_.attribute}={slice_.slice_value} n={slice_.n_rows} "
                f"{slice_.metric_name}={num(slice_.metric_value)} "
                f"delta={num(slice_.delta_vs_overall)}"
            )
    if verdict.fairness_notes:
        lines.append(f"fairness notes: {clip(verdict.fairness_notes, 400)}")
    if verdict.residual_notes:
        lines.append(f"residuals: {clip(verdict.residual_notes, 400)}")
    if verdict.learning_curve_notes:
        lines.append(f"learning curve: {clip(verdict.learning_curve_notes, 400)}")
    if verdict.error_analysis:
        lines.append("error analysis:")
        lines.append(bullets(verdict.error_analysis, 8, "  - "))
    lines.append("weaknesses:")
    lines.append(bullets(verdict.weaknesses, 10, "  - "))
    if verdict.specific_improvements:
        lines.append("improvements the evaluator suggested:")
        lines.append(bullets(verdict.specific_improvements, 8, "  - "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decisions with their recorded rationale (the audit trail)
# ---------------------------------------------------------------------------


def cleaning_text(plan: CleaningPlan | None) -> str:
    """Cleaning decisions and rationales, compact form."""
    if plan is None or not plan.decisions:
        return "(no cleaning plan was recorded)"
    lines = [f"summary: {clip(plan.summary, 500)}"]
    for decision in plan.decisions:
        strategy = f" strategy={decision.strategy.value}" if decision.strategy else ""
        lines.append(
            f"- {decision.action.value}{strategy} "
            f"columns={decision.columns or 'ALL'} "
            f"params=[{params_text(decision.parameters, 6)}] "
            f"destructive={str(decision.destructive).lower()}\n"
            f"    rationale: {clip(decision.rationale, 400)}"
            + (
                f"\n    expected impact: {clip(decision.expected_impact, 240)}"
                if decision.expected_impact
                else ""
            )
        )
    if plan.columns_to_drop:
        lines.append(f"columns dropped: {plan.columns_to_drop}")
        lines.append(bullets(plan.drop_rationale, 8, "  - "))
    if plan.skipped_considerations:
        lines.append("deliberately not applied:")
        lines.append(bullets(plan.skipped_considerations, 8, "  - "))
    return "\n".join(lines)


def cleaning_markdown(plan: CleaningPlan | None, applied: Sequence[str] | None = None) -> str:
    """Cleaning decision log as a markdown table plus the applied-op record."""
    if plan is None or not plan.decisions:
        return "_No cleaning transformations were recorded for this run._"
    rows = [
        [
            decision.action.value,
            ", ".join(f"`{c}`" for c in decision.columns) or "whole table",
            decision.strategy.value if decision.strategy else "—",
            decision.rationale,
            decision.expected_impact or "—",
        ]
        for decision in plan.decisions
    ]
    parts = [
        clip(plan.summary, 800),
        "",
        md_table(["Action", "Columns", "Strategy", "Rationale", "Expected impact"], rows),
    ]
    if plan.columns_to_drop:
        parts += ["", f"**Columns dropped:** {', '.join(f'`{c}`' for c in plan.columns_to_drop)}"]
        if plan.drop_rationale:
            parts += ["", bullets(plan.drop_rationale, 12)]
    if plan.skipped_considerations:
        parts += ["", "**Considered and deliberately skipped:**", "", bullets(plan.skipped_considerations, 12)]
    if applied:
        parts += ["", "**Operations actually applied by the executor:**", "", bullets(list(applied), 20)]
    return "\n".join(parts)


def features_text(plan: FeaturePlan | None) -> str:
    """Feature-engineering decisions and rationales, compact form."""
    if plan is None or not plan.decisions:
        return "(no feature plan was recorded)"
    lines = [f"summary: {clip(plan.summary, 500)}"]
    for decision in plan.decisions:
        lines.append(
            f"- {decision.op.value} inputs={decision.input_columns} "
            f"out_hint={decision.output_name_hint or '-'} "
            f"priority={decision.priority} params=[{params_text(decision.parameters, 6)}]\n"
            f"    rationale: {clip(decision.rationale, 400)}"
            + (f"\n    hypothesis: {clip(decision.hypothesis, 240)}" if decision.hypothesis else "")
            + (f"\n    risk: {clip(decision.risk, 240)}" if decision.risk else "")
        )
    if plan.dimensionality_strategy:
        lines.append(f"dimensionality strategy: {clip(plan.dimensionality_strategy, 300)}")
    if plan.selection_strategy:
        lines.append(f"selection strategy: {clip(plan.selection_strategy, 300)}")
    return "\n".join(lines)


def features_markdown(
    plan: FeaturePlan | None, applied: Sequence[str] | None = None
) -> str:
    """Feature list as a markdown table, rationale and risk included."""
    if plan is None or not plan.decisions:
        return "_No engineered features were recorded for this run._"
    rows = [
        [
            decision.op.value,
            ", ".join(f"`{c}`" for c in decision.input_columns) or "—",
            decision.output_name_hint or "—",
            decision.priority,
            decision.rationale,
            decision.risk or "—",
        ]
        for decision in plan.decisions
    ]
    parts = [
        clip(plan.summary, 800),
        "",
        md_table(
            ["Operation", "Inputs", "Output", "Priority", "Rationale", "Leakage / overfit risk"],
            rows,
        ),
    ]
    if plan.dimensionality_strategy:
        parts += ["", f"**Dimensionality strategy:** {clip(plan.dimensionality_strategy, 500)}"]
    if plan.selection_strategy:
        parts += ["", f"**Selection strategy:** {clip(plan.selection_strategy, 500)}"]
    if applied:
        parts += ["", "**Transformations actually applied:**", "", bullets(list(applied), 25)]
    return "\n".join(parts)


def plan_text(plan: ExecutionPlan | None, history: Sequence[ExecutionPlan] | None = None) -> str:
    """The execution plan, its adaptations, and any revisions that followed."""
    if plan is None:
        return "(no execution plan was recorded)"
    lines = [
        f"revision: {plan.revision}"
        + (f" (reason: {clip(plan.revision_reason, 300)})" if plan.revision_reason else ""),
        f"strategy: {clip(plan.summary, 900)}",
        "steps:",
    ]
    for step in plan.ordered():
        lines.append(
            f"  {step.order}. [{step.agent.value}] {step.title} "
            f"(optional={str(step.optional).lower()}, destructive={str(step.destructive).lower()})\n"
            f"      objective: {clip(step.objective, 200)}\n"
            f"      rationale: {clip(step.rationale, 240)}"
        )
    if plan.dataset_specific_adaptations:
        lines.append("dataset-specific adaptations:")
        lines.append(bullets(plan.dataset_specific_adaptations, 10, "  - "))
    if plan.risks:
        lines.append("planner-identified risks:")
        lines.append(bullets(plan.risks, 10, "  - "))
    if plan.fallback_strategy:
        lines.append(f"fallback strategy: {clip(plan.fallback_strategy, 400)}")
    prior = [p for p in (history or []) if p.revision != plan.revision]
    if prior:
        lines.append("earlier plan revisions:")
        for earlier in prior[-3:]:
            lines.append(
                f"  - revision {earlier.revision}: {clip(earlier.revision_reason or earlier.summary, 260)}"
            )
    return "\n".join(lines)


def tuning_text(decision: TuningDecision | None, result: TuningResult | None) -> str:
    """The tuning decision and what actually came of it."""
    if decision is None and result is None:
        return "(hyperparameter tuning was not considered)"
    lines: list[str] = []
    if decision is not None:
        lines.append(
            f"decision: worthwhile={str(decision.worthwhile).lower()} "
            f"method={decision.method.value} target={decision.target_family.value if decision.target_family else 'n/a'} "
            f"n_trials={decision.n_trials} timeout={decision.timeout_seconds}s"
        )
        lines.append(f"  rationale: {clip(decision.rationale, 400)}")
        if decision.expected_gain:
            lines.append(f"  expected gain: {clip(decision.expected_gain, 240)}")
        if decision.search_space:
            rendered = "; ".join(
                f"{e.name}[{e.kind}]"
                + (f" {num(e.low)}..{num(e.high)}" if e.low is not None else "")
                + (f" choices={e.choices}" if e.choices else "")
                for e in decision.search_space[:12]
            )
            lines.append(f"  search space: {rendered}")
    if result is not None:
        lines.append(
            f"outcome: ran={str(result.ran).lower()} method={result.method.value} "
            f"family={result.family.value if result.family else 'n/a'} "
            f"trials_completed={result.n_trials_completed} seconds={num(result.seconds, 3)}"
        )
        lines.append(
            f"  best_score={num(result.best_score)} baseline_score={num(result.baseline_score)} "
            f"improvement={num(result.improvement)}"
        )
        if result.best_params:
            lines.append(f"  best params: {params_text(result.best_params, 12)}")
        if result.skipped_reason:
            lines.append(f"  skipped because: {clip(result.skipped_reason, 240)}")
        if result.error:
            lines.append(f"  error: {clip(result.error, 240)}")
    return "\n".join(lines)


def splits_text(state: RunState) -> str:
    """How the data was partitioned — the precondition for trusting any score."""
    splits = state.splits
    sizes = splits.sizes()
    return (
        f"split strategy: {splits.strategy or 'not recorded'} "
        f"(train={sizes['train']:,} validation={sizes['validation']:,} test={sizes['test']:,})\n"
        f"split rationale: {clip(splits.rationale, 400) or 'not recorded'}\n"
        f"validation strategy chosen by model selection: "
        f"{clip(state.model_selection.validation_strategy, 240) if state.model_selection else 'n/a'}"
    )


def target_text(profile: DatasetProfile | None) -> str:
    """Target distribution or class balance, whichever applies."""
    if profile is None or profile.target is None:
        return "(no target summary was measured)"
    target = profile.target
    lines = [f"target `{target.name}` (kind={target.kind.value}, missing={target.n_missing:,})"]
    if target.n_classes is not None:
        lines.append(f"  classes: {target.n_classes}")
    if target.class_counts:
        lines.append(
            "  distribution: "
            + ", ".join(
                f"{c.value!r}={c.count:,} ({pct(c.fraction)})" for c in target.class_counts[:12]
            )
        )
    if target.imbalance_ratio is not None:
        lines.append(
            f"  imbalance ratio (majority/minority): {num(target.imbalance_ratio, 3)} "
            f"-> {'IMBALANCED' if target.is_imbalanced else 'reasonably balanced'}"
        )
    if target.mean is not None:
        lines.append(
            f"  mean={num(target.mean)} std={num(target.std)} skew={num(target.skewness, 3)}"
        )
    return "\n".join(lines)


def temporal_span_text(profile: DatasetProfile | None) -> str:
    """The dataset's time coverage, which is what a retraining cadence rests on."""
    if profile is None:
        return "(no profile available)"
    spans: list[str] = []
    for name in profile.temporal_columns:
        col = profile.column(name)
        if col is None:
            continue
        spans.append(
            f"`{name}`: {col.min_timestamp or '?'} .. {col.max_timestamp or '?'} "
            f"freq={col.inferred_frequency or 'unknown'} gaps={col.n_gaps if col.n_gaps is not None else 'n/a'}"
        )
    if not spans:
        return "no datetime column was detected, so the data has no measurable temporal span"
    return "\n".join(f"- {s}" for s in spans[:6])


def insights_text(report: InsightReport | None) -> str:
    """Business insights already produced, for the report and Q&A agents."""
    if report is None:
        return "(no business insights were produced)"
    lines = [f"executive summary: {clip(report.executive_summary, 900)}"]
    for insight in report.insights:
        lines.append(
            f"- [{insight.audience}, confidence={insight.confidence}] {clip(insight.headline, 200)}\n"
            f"    detail: {clip(insight.detail, 400)}\n"
            f"    evidence: {clip(insight.supporting_evidence, 240)}\n"
            f"    action: {clip(insight.recommended_action, 240)}"
            + (f"\n    expected value: {clip(insight.expected_value, 200)}" if insight.expected_value else "")
        )
    if report.key_drivers_plain_language:
        lines.append("key drivers in plain language:")
        lines.append(bullets(report.key_drivers_plain_language, 10, "  - "))
    if report.caveats:
        lines.append("caveats:")
        lines.append(bullets(report.caveats, 10, "  - "))
    if report.suggested_next_experiments:
        lines.append("suggested next experiments:")
        lines.append(bullets(report.suggested_next_experiments, 8, "  - "))
    return "\n".join(lines)


def charts_text(plan: VisualizationPlan | None) -> str:
    """Planned charts, so the report can reference figures that will exist."""
    if plan is None or not plan.charts:
        return "(no charts were planned)"
    lines = [
        f"- {c.kind.value} | \"{clip(c.title, 120)}\" | columns={c.columns or '-'} "
        f"| priority={c.priority}"
        for c in plan.charts
    ]
    if plan.dashboard_narrative:
        lines.append(f"dashboard narrative: {clip(plan.dashboard_narrative, 600)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event log & executor bundles
# ---------------------------------------------------------------------------


def events_text(events: Sequence[RunEvent] | None, limit: int = MAX_EVENTS) -> str:
    """The tail of the run's event log, rendered compactly.

    This is the Q&A agent's primary evidence base: "why did accuracy drop?" is
    answered by replaying what happened, not by re-deriving it.
    """
    if not events:
        return "(no events recorded)"
    tail = list(events)[-limit:]
    lines: list[str] = []
    for event in tail:
        agent = f" {event.agent.value}" if event.agent else ""
        step = f" step={event.step_id}" if event.step_id else ""
        timing = (
            f" ({event.duration_seconds:.2f}s)" if event.duration_seconds is not None else ""
        )
        lines.append(
            f"#{event.sequence} [{event.kind.value}]{agent}{step}: "
            f"{clip(event.message, 200)}{timing}"
        )
    header = (
        f"(showing the most recent {len(tail)} of {len(events)} events, oldest first)"
    )
    return "\n".join([header, *lines])


def executor_bundles_text(state: RunState, char_budget: int = 4000) -> str:
    """Render any executor bundle parked on ``state.extras`` that can self-describe.

    ``execution/diagnostics.py`` exposes a ``to_prompt()`` on its bundle; rather
    than guessing the key it was stored under, pick up anything in ``extras``
    that offers the same protocol.
    """
    parts: list[str] = []
    used = 0
    for key, value in sorted(state.extras.items()):
        renderer = getattr(value, "to_prompt", None)
        if not callable(renderer):
            continue
        try:
            text = str(renderer())
        except Exception as exc:  # noqa: BLE001 - a bad renderer must not break a prompt
            text = f"(failed to render: {exc})"
        remaining = char_budget - used
        if remaining <= 0:
            break
        snippet = text[:remaining]
        used += len(snippet)
        parts.append(f"### {key}\n{snippet}")
    return "\n\n".join(parts)


def warnings_text(state: RunState, limit: int = 15) -> str:
    """Degradations recorded during the run — reduced capability, not failure."""
    if not state.warnings:
        return "(no warnings recorded)"
    return bullets(state.warnings[-limit:], limit)


def run_facts_text(state: RunState) -> str:
    """One-line-per-fact orientation block: task, target, metric, budget, status."""
    task = state.task_type.value if state.task_type else "unknown"
    problem = state.problem
    lines = [
        f"task: {task}",
        f"target: `{state.target}`" if state.target else "target: none (unsupervised)",
        f"primary metric: {state.primary_metric}",
    ]
    if problem:
        lines.append(f"positive class: {problem.positive_class or 'n/a'}")
        lines.append(f"business objective: {clip(problem.business_objective, 600)}")
        if problem.constraints:
            lines.append("stated constraints:")
            lines.append(bullets(problem.constraints, 8, "  - "))
    lines.append(
        f"elapsed: {state.elapsed_seconds:.0f}s of a {state.config.time_budget_seconds}s budget"
    )
    return "\n".join(lines)


__all__ = [name for name in dir() if not name.startswith("_")]
