"""The Evaluation Agent: the quality gate and the self-improvement trigger.

Every other agent in this pipeline pushes forward. This one is allowed to say
no, and its ``recommended_action`` is control flow: ``retry_feature_engineering``,
``retry_model_selection``, and ``retry_cleaning`` send the orchestrator back to
the planner for a real second pass. That is why the prompt spends its length on
two things — how to read each diagnostic, and the discipline of only asking for a
retry when a specific change can be named.

The numbers come from ``execution.diagnostics.compute_diagnostics``: train,
validation and test scores, the generalisation gap, calibration, bootstrap
confidence intervals, residual statistics, learning-curve points, fairness
slices, and concrete error examples. :meth:`EvaluationAgent.postprocess` copies
the measured values back over whatever the model returned, keeping the agent's
*judgements* (the verdict labels and prose) but never its arithmetic, and clamps
the recommendation when a retry could not actually run — a retry that cannot
happen is a false statement about what comes next.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    BiasVarianceDiagnosis,
    CalibrationDiagnosis,
    ConfidenceInterval,
    EvaluationVerdict,
    FairnessSlice,
)
from ..core.state import RunState
from .experiment import (
    baseline_result,
    best_result,
    metric_higher_is_better,
    render_leaderboard,
    render_margin_analysis,
)

logger = logging.getLogger(__name__)

#: Where the computed diagnostics bundle is cached on ``state.extras`` so the
#: report and visualization agents can reuse it without recomputing.
DIAGNOSTICS_EXTRA_KEY = "diagnostics"

#: Recommendations that make the orchestrator replan and re-run the pipeline.
RETRY_ACTIONS = frozenset(
    {"retry_feature_engineering", "retry_model_selection", "retry_cleaning"}
)


def _g(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number:.{digits}g}"


# ---------------------------------------------------------------------------
# Diagnostics access
# ---------------------------------------------------------------------------


def get_diagnostics(state: RunState, *, refresh: bool = False) -> Any | None:
    """Compute (once) the deterministic diagnostics bundle for this run.

    The bundle is cached under ``state.extras[DIAGNOSTICS_EXTRA_KEY]`` because
    :meth:`EvaluationAgent.build_prompt` and :meth:`EvaluationAgent.postprocess`
    must see the same numbers, and the report agent wants them again later.

    Args:
        state: Live run state.
        refresh: Recompute even when a cached bundle exists.

    Returns:
        The ``DiagnosticsBundle`` from the execution layer, or ``None`` when it
        could not be produced. Failure is degraded to ``None`` plus a run
        warning; it never aborts the run.
    """
    if not refresh and state.extras.get(DIAGNOSTICS_EXTRA_KEY) is not None:
        return state.extras[DIAGNOSTICS_EXTRA_KEY]

    try:
        from ..execution.diagnostics import compute_diagnostics
    except Exception as exc:  # noqa: BLE001 - executor may be unavailable
        state.add_warning(
            f"Evaluation Agent: the diagnostics executor could not be imported "
            f"({exc}); the verdict will rest on the leaderboard alone."
        )
        return None

    try:
        bundle = compute_diagnostics(state)
    except Exception as exc:  # noqa: BLE001 - reduced capability, not a crash
        logger.exception("compute_diagnostics failed")
        state.add_warning(
            f"Evaluation Agent: diagnostics computation failed "
            f"({type(exc).__name__}: {exc}); the verdict will rest on the "
            "leaderboard alone."
        )
        return None

    state.extras[DIAGNOSTICS_EXTRA_KEY] = bundle
    return bundle


def _render_bundle_fallback(bundle: Any) -> str:
    """Render the documented bundle attributes when ``to_prompt`` is absent."""
    lines: list[str] = ["## MEASURED MODEL DIAGNOSTICS", ""]

    bias_variance = getattr(bundle, "bias_variance", None)
    if bias_variance is not None:
        lines.append("#### Fit (train vs validation vs test)")
        lines.append(
            f"- train={_g(getattr(bias_variance, 'train_score', None))} "
            f"validation={_g(getattr(bias_variance, 'validation_score', None))} "
            f"test={_g(getattr(bias_variance, 'test_score', None))} "
            f"gap={_g(getattr(bias_variance, 'gap', None))}"
        )
        detail = getattr(bias_variance, "detail", "")
        verdict = getattr(bias_variance, "verdict", "")
        if verdict or detail:
            lines.append(f"- executor read: {verdict} {detail}".rstrip())

    calibration = getattr(bundle, "calibration", None)
    if calibration is not None and getattr(calibration, "applicable", False):
        lines.append("")
        lines.append("#### Calibration")
        lines.append(
            f"- brier={_g(getattr(calibration, 'brier_score', None))} "
            f"expected_calibration_error="
            f"{_g(getattr(calibration, 'expected_calibration_error', None))}"
        )
        note = getattr(calibration, "verdict", "")
        if note:
            lines.append(f"- executor read: {note}")

    intervals = getattr(bundle, "confidence_intervals", None) or []
    if intervals:
        lines.append("")
        lines.append("#### Bootstrap confidence intervals")
        for interval in intervals:
            lines.append(
                f"- {getattr(interval, 'metric', '?')}: "
                f"{_g(getattr(interval, 'point_estimate', None))} "
                f"[{_g(getattr(interval, 'lower', None))}, "
                f"{_g(getattr(interval, 'upper', None))}] at "
                f"{_g(getattr(interval, 'level', None), 3)} "
                f"({getattr(interval, 'method', 'bootstrap')})"
            )

    residuals = getattr(bundle, "residual_stats", None) or {}
    if residuals:
        lines.append("")
        lines.append("#### Residual statistics")
        lines.append(
            "- " + ", ".join(f"{name}={_g(value)}" for name, value in residuals.items())
        )

    curve = getattr(bundle, "learning_curve", None) or {}
    if curve:
        lines.append("")
        lines.append("#### Learning curve")
        for name, series in curve.items():
            rendered = ", ".join(_g(point) for point in list(series)[:12])
            lines.append(f"- {name}: [{rendered}]")

    fairness = getattr(bundle, "fairness", None) or []
    if fairness:
        lines.append("")
        lines.append("#### Fairness slices")
        for slice_ in fairness:
            lines.append(
                f"- {getattr(slice_, 'attribute', '?')}="
                f"{getattr(slice_, 'slice_value', '?')} "
                f"(n={getattr(slice_, 'n_rows', 0):,}): "
                f"{getattr(slice_, 'metric_name', '?')}="
                f"{_g(getattr(slice_, 'metric_value', None))} "
                f"delta_vs_overall={_g(getattr(slice_, 'delta_vs_overall', None))}"
            )

    examples = getattr(bundle, "error_examples", None) or []
    if examples:
        lines.append("")
        lines.append("#### Error examples")
        for example in list(examples)[:12]:
            lines.append(f"- {example}")

    notes = getattr(bundle, "notes", None) or []
    if notes:
        lines.append("")
        lines.append("#### Diagnostics that could not be computed")
        for note in list(notes)[:12]:
            lines.append(f"- {note}")

    if len(lines) <= 2:
        return "## MEASURED MODEL DIAGNOSTICS\n\n_the diagnostics bundle was empty_"
    return "\n".join(lines)


def render_diagnostics(bundle: Any | None) -> str:
    """Render the diagnostics bundle as prompt text.

    Prefers the bundle's own ``to_prompt()``; falls back to reading the
    documented attributes directly so a partially-implemented bundle still
    yields usable facts.

    Args:
        bundle: The diagnostics bundle, or ``None``.

    Returns:
        Prompt-ready text.
    """
    if bundle is None:
        return (
            "## MEASURED MODEL DIAGNOSTICS\n\n"
            "No diagnostics could be computed for this run. Judge only from the "
            "leaderboard and split sizes above, state explicitly that the "
            "generalisation, calibration, fairness and confidence-interval "
            "evidence is missing, and let that missing evidence cap how confident "
            "your verdict can be."
        )
    renderer = getattr(bundle, "to_prompt", None)
    if callable(renderer):
        try:
            text = renderer()
            if isinstance(text, str) and text.strip():
                return text
        except Exception:  # noqa: BLE001 - fall back rather than fail the step
            logger.exception("DiagnosticsBundle.to_prompt failed; using fallback")
    return _render_bundle_fallback(bundle)


def _effective_best_score(
    state: RunState, bundle: Any | None
) -> tuple[float | None, str]:
    """The score the acceptance gate should judge, and what it is.

    Prefers the holdout test score, because that is the honest generalisation
    estimate; falls back to validation, then to the best cross-validated
    leaderboard score.
    """
    bias_variance = getattr(bundle, "bias_variance", None)
    for attribute, label in (
        ("test_score", "holdout test score"),
        ("validation_score", "validation score"),
    ):
        value = getattr(bias_variance, attribute, None) if bias_variance else None
        if value is not None:
            return float(value), label

    top = best_result(state.experiments)
    if top is not None and top.primary_score is not None:
        return float(top.primary_score), "best cross-validated leaderboard score"
    return None, "no measured score"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_INSTRUCTIONS = """\
You are the reviewer who signs off on a model, or refuses to. You did not build \
it and you have no stake in it passing. Every number you are given was measured \
by deterministic code on real partitions; your job is to read them the way an \
experienced practitioner does, decide whether this model is fit to recommend, \
and — if it is not — say exactly what to change.

HOW TO READ EACH DIAGNOSTIC:

-   Train vs validation vs test. A high train score with a much lower holdout \
score is variance: the model memorised. All three low is bias: either the model \
class is too rigid or the feature set carries no signal — distinguish those two, \
because they lead to different retries. A test score meaningfully *above* \
validation is a warning, not good news: usually a small test partition or a \
split that leaked.
-   Score against the baseline. This is the question that matters most and the \
one most often skipped. A model that barely beats its baseline learned almost \
nothing, whatever its absolute number looks like: 0.91 accuracy against a 90/10 \
class split is worthless, and 0.68 ROC AUC against 0.50 is real. Judge the lift, \
then judge the absolute level.
-   Confidence intervals. Compare the interval width against the gaps you care \
about. If the interval spans the best-vs-baseline gap, the dataset cannot support \
the claim that the model works. If it spans the best-vs-runner-up gap, model \
choice was arbitrary and you should say so instead of ranking confidently. Wide \
intervals are a statement about sample size, not about the model.
-   Calibration. Brier score and expected calibration error tell you whether the \
predicted probabilities mean what they say. A model can rank well and still be \
badly calibrated; that is fine for prioritising a call list, and disqualifying \
for anything that multiplies a probability by a cost or applies a fixed \
threshold. Say which use is safe.
-   Residuals (regression). Check bias in the mean residual, heteroscedasticity \
across the range, and skew or heavy tails. Systematic under-prediction of large \
values is a business failure even when aggregate error looks acceptable.
-   Learning curve. Still improving at the largest training size means more data \
will help and the honest recommendation may be to collect it. Flat and low means \
more rows will not help — the problem is the features or the model class. Train \
and validation curves diverging widely and early means regularise or simplify.
-   Fairness slices. A slice with a large delta against the overall metric and \
enough rows to be real is a genuine problem even when the aggregate looks fine, \
and it must appear in the verdict rather than only in a footnote. A slice of a \
dozen rows is noise; say that it is noise rather than treating it as evidence.
-   Error examples. Use them for the mechanism. Concentrated errors in one \
region, one category, or one time period is a specific, fixable finding; \
uniformly spread errors point at irreducible noise.

THE DECISION:

`acceptable` means: fit to recommend for the stated business objective, given the \
operator's minimum score if one was set. Not "better than nothing" and not \
"beat the baseline by any margin".

`overall_grade` rubric: A — decisive lift over baseline, tight interval, honest \
generalisation gap, calibrated where relevant, no material slice disparity. \
B — solid and usable with one clear caveat. C — works but marginal or fragile: \
narrow lift, wide interval, or a real slice gap. D — barely beats baseline, or \
badly overfit, or unusable for the stated objective without rework. F — no better \
than the baseline, or actively misleading.

`recommended_action` IS CONTROL FLOW. Choosing retry_feature_engineering, \
retry_model_selection, or retry_cleaning makes the orchestrator revise the plan \
and re-run that part of the pipeline for real, spending the remaining time \
budget. Choose one only when you can name a specific change and argue why it \
should help. Map the diagnosis to the action:

-   High variance with plausible leakage or dirty inputs -> retry_cleaning, and \
name the column and the treatment.
-   High bias, or importance concentrated in one crude feature, or a clear \
untapped structure (temporal, interaction, high-cardinality categorical) -> \
retry_feature_engineering, and name the operation and the columns.
-   The winning family's inductive bias plainly mismatches the data (a linear \
model on visibly non-linear structure, a deep tree ensemble on 300 rows, a model \
that cannot use the dominant feature type) -> retry_model_selection, and name \
the families to try instead and why.
-   The learning curve is still rising and the interval is wide -> \
collect_more_data. This does not replan; it is advice to the operator.
-   The model is unusable or would cause harm -> reject.
-   Nothing specific to change, or the remaining headroom is smaller than the \
noise -> accept, and record the caveats honestly.

`specific_improvements` must be executable instructions, each naming a column, \
an operation, or a parameter. If your suggestion could be pasted into any \
project, it is not specific enough.

A good improvement: "Target-encode `merchant_id` out of fold instead of one-hot \
encoding it: its 3,412 levels currently produce 78% of the 4,380-column feature \
matrix, and the tree models cannot find useful splits in columns that are \
positive for 0.03% of rows."

A bad improvement: "Try more feature engineering to improve performance." — no \
column, no operation, no mechanism, no reason to expect a change.

DISCIPLINE:

-   Do not order a retry to chase a decimal. Each replan re-runs the pipeline and \
consumes the budget that reporting needs. Retry only when you expect a step \
change, and say what size of change you expect.
-   Do not invent numbers. The measured diagnostic values are restored over \
whatever you return, so quote them exactly and spend your output on judgement.
-   Do not soften. If the model is not usable, an honest F with a precise reason \
is far more valuable to the operator than a generous C.
-   Be explicit when evidence is missing. A verdict reached without a test score, \
without calibration, or without fairness slices must say so and be less \
confident accordingly.
"""


class EvaluationAgent(BaseAgent[EvaluationVerdict]):
    """Judges the trained model and decides whether the pipeline tries again."""

    name = AgentName.EVALUATION
    title = "Evaluation Agent"
    output_model = EvaluationVerdict
    effort: Effort = "max"
    max_tokens = 16_000

    # -- prompt -------------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        return _INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        bundle = get_diagnostics(state)
        log = state.experiments
        top = best_result(log)
        baseline = baseline_result(log)
        higher = (
            log.higher_is_better
            if log is not None
            else metric_higher_is_better(state.primary_metric)
        )
        sizes = state.splits.sizes()

        parts: list[str] = ["## MODEL UNDER REVIEW", ""]
        if top is not None:
            parts.append(
                f"- family: `{top.family.value}`"
                + (f" [{top.label}]" if top.label else "")
                + (" (hyperparameter-tuned)" if top.tuned else "")
            )
            parts.append(
                f"- measured {state.primary_metric}: {_g(top.primary_score)} "
                f"({'higher' if higher else 'lower'} is better)"
            )
            if (
                baseline is not None
                and baseline.primary_score is not None
                and top.primary_score is not None
            ):
                # Both scores must be measured: substituting 0.0 for a missing
                # winner score would put an invented lift in front of the model.
                delta = (
                    top.primary_score - baseline.primary_score
                    if higher
                    else baseline.primary_score - top.primary_score
                )
                parts.append(
                    f"- baseline (`{baseline.family.value}`): "
                    f"{_g(baseline.primary_score)} -> lift {delta:+.4g}"
                )
            parts.append(
                f"- training cost: {top.train_seconds:.2f}s; features in: "
                f"{top.n_features_in or 'unrecorded'}"
            )
        else:
            parts.append("- no model was trained successfully")
        parts.append(
            f"- task: {state.task_type.value if state.task_type else 'unknown'}; "
            f"target `{state.target or 'unknown'}`"
        )
        if state.problem:
            parts.append(f"- business objective: {state.problem.business_objective}")
            if state.problem.constraints:
                parts.append(f"- stated constraints: {state.problem.constraints}")
        parts.append("")

        parts.append("### Split provenance")
        parts.append(
            f"- train={sizes['train']:,} validation={sizes['validation']:,} "
            f"test={sizes['test']:,}"
        )
        parts.append(f"- strategy: {state.splits.strategy or 'unrecorded'}")
        if state.splits.rationale:
            parts.append(f"- why: {state.splits.rationale}")
        parts.append("")

        parts.append(render_leaderboard(log, limit=8))
        parts.append("")
        parts.append(render_margin_analysis(log))
        parts.append("")

        if state.tuning is not None:
            tuning = state.tuning
            if tuning.ran:
                parts.append("### Tuning outcome (measured)")
                parts.append(
                    f"- {tuning.method.value} on "
                    f"`{tuning.family.value if tuning.family else 'unknown'}`: "
                    f"{tuning.n_trials_completed} trials in {tuning.seconds:.0f}s"
                )
                parts.append(
                    f"- best {state.primary_metric}: {_g(tuning.best_score)} vs "
                    f"pre-tuning {_g(tuning.baseline_score)} "
                    f"(improvement {_g(tuning.improvement)})"
                )
            else:
                parts.append(
                    "### Tuning was not run\n- reason: "
                    f"{tuning.skipped_reason or tuning.error or 'not recorded'}"
                )
            parts.append("")

        pipeline = self._pipeline_context(state)
        if pipeline:
            parts.append("### What the pipeline already did (your retry must differ)")
            parts.extend(pipeline)
            parts.append("")

        if state.explainability and (
            state.explainability.global_attributions
            or state.explainability.permutation_importance
        ):
            attributions = (
                state.explainability.global_attributions
                or state.explainability.permutation_importance
            )
            parts.append("### Where the model's importance sits")
            parts.append(
                "- "
                + ", ".join(
                    f"`{item.feature}` {item.importance * 100:.0f}%"
                    for item in attributions[:8]
                )
            )
            parts.append("")

        parts.append(render_diagnostics(bundle))
        parts.append("")

        parts.append("### Gate settings and what your recommendation will do")
        if state.config.min_acceptable_score is not None:
            score, label = _effective_best_score(state, bundle)
            parts.append(
                f"- operator minimum acceptable {state.primary_metric}: "
                f"{state.config.min_acceptable_score}. The {label} is {_g(score)}; "
                "failing this forces acceptable=false regardless of your verdict."
            )
        else:
            parts.append(
                "- the operator set no minimum score, so acceptability is your "
                "judgement against the business objective."
            )
        replans_left = max(0, state.config.max_replans - state.replans)
        if not state.config.enable_self_improvement:
            parts.append(
                "- self-improvement is DISABLED for this run: a retry_* "
                "recommendation cannot execute and will be recorded as 'accept'. "
                "Put your ideas in specific_improvements instead."
            )
        elif replans_left <= 0:
            parts.append(
                f"- the replan budget is exhausted ({state.replans} of "
                f"{state.config.max_replans} used): a retry_* recommendation cannot "
                "execute and will be recorded as 'accept'. Put your ideas in "
                "specific_improvements instead."
            )
        else:
            parts.append(
                f"- replans remaining: {replans_left} of {state.config.max_replans}. "
                "A retry_* recommendation WILL re-run that stage for real."
            )
        parts.append(
            f"- time remaining in the run budget: {state.time_remaining:.0f}s of "
            f"{state.config.time_budget_seconds}s"
            + (
                " - a retry must be finishable in that time."
                if replans_left > 0
                else ""
            )
        )
        if state.config.fairness_attributes:
            parts.append(
                f"- fairness attributes the operator asked about: "
                f"{state.config.fairness_attributes}"
            )
        if state.warnings:
            parts.append(
                f"- {len(state.warnings)} warning(s) were raised earlier in this run; "
                "the most recent: " + state.warnings[-1]
            )
        parts.append("")

        parts.append(
            "Deliver the verdict. Judge fit, lift over baseline, interval width, "
            "calibration, residuals, learning curve, and slice disparity; grade the "
            "model; then choose recommended_action knowing it drives what the "
            "orchestrator does next, and make every entry in specific_improvements "
            "concrete enough to execute."
        )
        return "\n".join(parts)

    @staticmethod
    def _pipeline_context(state: RunState) -> list[str]:
        """Summarise upstream decisions so a retry can propose something new."""
        lines: list[str] = []
        if state.cleaning and state.cleaning.summary:
            lines.append(f"- cleaning: {state.cleaning.summary}")
        if state.applied_cleaning:
            lines.append(
                f"- cleaning steps applied: {', '.join(state.applied_cleaning[:12])}"
            )
        if state.dropped_columns:
            lines.append(f"- columns dropped: {state.dropped_columns[:15]}")
        if state.features and state.features.decisions:
            rendered = ", ".join(
                f"{d.op.value}({', '.join(d.input_columns[:3]) or 'all'})"
                for d in state.features.decisions[:12]
            )
            lines.append(f"- feature operations applied: {rendered}")
        if state.features and state.features.selection_strategy:
            lines.append(
                f"- feature selection: {state.features.selection_strategy}"
            )
        if state.feature_names:
            lines.append(f"- final feature count: {len(state.feature_names)}")
        if state.model_selection and state.model_selection.excluded_families:
            lines.append(
                f"- model families deliberately excluded: "
                f"{state.model_selection.excluded_families}"
            )
        return lines

    # -- grounding ----------------------------------------------------------

    def postprocess(
        self, value: EvaluationVerdict, state: RunState
    ) -> EvaluationVerdict:
        """Restore measured diagnostics and clamp promises that cannot be kept."""
        verdict = value.model_copy(deep=True)
        bundle = state.extras.get(DIAGNOSTICS_EXTRA_KEY)

        self._restore_measurements(verdict, value, bundle, state)
        self._enforce_score_gate(verdict, state, bundle)
        self._clamp_recommendation(verdict, value, state)
        return verdict

    def _restore_measurements(
        self,
        verdict: EvaluationVerdict,
        original: EvaluationVerdict,
        bundle: Any | None,
        state: RunState,
    ) -> None:
        """Numbers come from the executor; the labels stay the agent's call."""
        if bundle is None:
            # With no measurements there is nothing to copy, and anything the
            # model returned in these fields would be fabricated.
            if original.confidence_intervals or original.fairness_slices:
                state.add_warning(
                    f"{self.title}: discarded model-supplied confidence intervals "
                    "and fairness slices; diagnostics were not computed, so those "
                    "numbers had no measured source."
                )
            verdict.confidence_intervals = []
            verdict.fairness_slices = []
            return

        measured_bv = getattr(bundle, "bias_variance", None)
        if isinstance(measured_bv, BiasVarianceDiagnosis):
            grounded = measured_bv.model_copy(deep=True)
            # The executor measures the scores; the agent judges what they mean,
            # so its verdict/detail survive unless it declined to give one.
            authored_verdict = original.bias_variance.verdict
            if authored_verdict != "inconclusive":
                grounded.verdict = authored_verdict
            if original.bias_variance.detail.strip():
                grounded.detail = original.bias_variance.detail
            disagrees = (
                authored_verdict != "inconclusive"
                and measured_bv.verdict != "inconclusive"
                and authored_verdict != measured_bv.verdict
            )
            if disagrees:
                # A reviewer may legitimately overrule a threshold rule, but the
                # disagreement itself belongs in the audit trail.
                grounded.detail = (
                    f"{grounded.detail} "
                    f"[Executor's rule-based read was '{measured_bv.verdict}': "
                    f"{measured_bv.detail}]"
                ).strip()
                state.add_warning(
                    f"{self.title}: judged fit as '{authored_verdict}' where the "
                    f"measured rule said '{measured_bv.verdict}' "
                    f"(gap={_g(measured_bv.gap)}); the agent's reading was kept and "
                    "the executor's recorded alongside it."
                )
            verdict.bias_variance = grounded

        measured_cal = getattr(bundle, "calibration", None)
        if isinstance(measured_cal, CalibrationDiagnosis):
            grounded_cal = measured_cal.model_copy(deep=True)
            if original.calibration.verdict.strip():
                grounded_cal.verdict = original.calibration.verdict
            verdict.calibration = grounded_cal

        intervals = getattr(bundle, "confidence_intervals", None)
        if isinstance(intervals, list):
            verdict.confidence_intervals = [
                item.model_copy(deep=True)
                for item in intervals
                if isinstance(item, ConfidenceInterval)
            ]

        slices = getattr(bundle, "fairness", None)
        if isinstance(slices, list):
            verdict.fairness_slices = [
                item.model_copy(deep=True)
                for item in slices
                if isinstance(item, FairnessSlice)
            ]

    def _enforce_score_gate(
        self, verdict: EvaluationVerdict, state: RunState, bundle: Any | None
    ) -> None:
        """An explicit operator threshold overrides the agent's generosity."""
        threshold = state.config.min_acceptable_score
        if threshold is None:
            return
        score, label = _effective_best_score(state, bundle)
        if score is None:
            return
        higher = (
            state.experiments.higher_is_better
            if state.experiments is not None
            else metric_higher_is_better(state.primary_metric)
        )
        passes = score >= threshold if higher else score <= threshold
        if passes:
            return

        note = (
            f"The {label} of {_g(score)} fails the operator's minimum acceptable "
            f"{state.primary_metric} of {threshold}, so this model is recorded as "
            "not acceptable."
        )
        if verdict.acceptable:
            state.add_warning(f"{self.title}: {note}")
        verdict.acceptable = False
        verdict.verdict_rationale = f"{note} {verdict.verdict_rationale}".strip()
        if verdict.overall_grade in ("A", "B"):
            # A passing grade beside a failed hard threshold reads as approval.
            previous_grade = verdict.overall_grade
            verdict.overall_grade = "C"
            verdict.verdict_rationale += (
                f" (Grade lowered from {previous_grade} to C because the "
                "operator's hard threshold was not met.)"
            )

    def _clamp_recommendation(
        self,
        verdict: EvaluationVerdict,
        original: EvaluationVerdict,
        state: RunState,
    ) -> None:
        """Never promise a retry the orchestrator is not allowed to run."""
        if original.recommended_action not in RETRY_ACTIONS:
            return

        reason: str | None = None
        if not state.config.enable_self_improvement:
            reason = "self-improvement is disabled for this run"
        elif state.replans >= state.config.max_replans:
            reason = (
                f"the replan budget is exhausted ({state.replans} of "
                f"{state.config.max_replans} used)"
            )
        if reason is None:
            return

        verdict.recommended_action = "accept"
        verdict.action_rationale = (
            f"Recorded as 'accept' because {reason}, so the requested "
            f"'{original.recommended_action}' could not execute; the verdict itself "
            f"(acceptable={str(verdict.acceptable).lower()}) stands and the "
            "suggested changes are retained as recommended future work. "
            f"[Agent's original reasoning: {original.action_rationale.strip()}]"
        ).strip()
        state.add_warning(
            f"{self.title}: requested '{original.recommended_action}' was clamped "
            f"to 'accept' because {reason}."
        )

    # -- state --------------------------------------------------------------

    def apply(self, state: RunState, value: EvaluationVerdict) -> None:
        state.evaluation = value

    def decision_summary(self, value: EvaluationVerdict) -> str:
        gap = value.bias_variance.gap
        pieces = [
            f"Grade {value.overall_grade}",
            f"acceptable={str(value.acceptable).lower()}",
            f"action={value.recommended_action}",
        ]
        if gap is not None:
            pieces.append(f"train-holdout gap={_g(gap)}")
        if value.bias_variance.verdict != "inconclusive":
            pieces.append(value.bias_variance.verdict)
        return "Verdict: " + ", ".join(pieces)


__all__ = [
    "DIAGNOSTICS_EXTRA_KEY",
    "RETRY_ACTIONS",
    "EvaluationAgent",
    "get_diagnostics",
    "render_diagnostics",
]
