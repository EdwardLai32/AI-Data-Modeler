"""The Experiment Agent: measure first, interpret second.

This is the purest expression of the platform's split. Every number on the
leaderboard is produced by :func:`automl_architect.execution.trainer.run_experiments`
— real fits, real cross-validation, real wall-clock timings. The agent's only
job is the sentence a senior practitioner writes underneath the table: which
model won, whether the margin survives the fold-to-fold noise, and what the win
cost in training time.

:meth:`ExperimentAgent.postprocess` rebuilds the returned :class:`ExperimentLog`
from the computed one field by field, so the model *physically cannot* alter a
measured score even if it tries. That is not defensive paranoia about this
particular model; it is what makes the leaderboard citable in an audit.

The rendering helpers here (:func:`render_leaderboard`,
:func:`render_margin_analysis`) are shared with the Tuning and Evaluation agents,
which reason over the same table from different angles.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from ..core.agent import BaseAgent, HybridAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ExperimentLog,
    ExperimentResult,
)
from ..core.state import RunState

logger = logging.getLogger(__name__)

#: Metric-name fragments whose scores improve as they fall. Used only as a
#: fallback when ``execution.metrics`` is unavailable.
_LOWER_IS_BETTER_TOKENS = (
    "error",
    "loss",
    "rmse",
    "mse",
    "mae",
    "mape",
    "smape",
    "brier",
    "deviance",
    "perplexity",
    "distance",
    "aic",
    "bic",
)


# ---------------------------------------------------------------------------
# Shared formatting / leaderboard helpers
# ---------------------------------------------------------------------------


def _g(value: float | int | None, digits: int = 4) -> str:
    """Format a possibly-missing number compactly, never as a bare ``None``."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    if number in (float("inf"), float("-inf")):
        return "inf"
    return f"{number:.{digits}g}"


def _secs(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def metric_higher_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is a better model.

    Delegates to the execution layer's authoritative table when it is
    importable, and falls back to a name heuristic otherwise so agents keep
    working before/without the executor package.

    Args:
        metric: Metric name, e.g. ``"roc_auc"`` or ``"rmse"``.

    Returns:
        True when higher scores are better.
    """
    try:  # pragma: no cover - depends on sibling module availability
        from ..execution.metrics import higher_is_better as _authoritative

        return bool(_authoritative(metric))
    except Exception:  # noqa: BLE001 - degrade to the heuristic, never crash
        low = (metric or "").lower()
        return not any(token in low for token in _LOWER_IS_BETTER_TOKENS)


def _score_of(result: ExperimentResult, metric: str) -> float | None:
    if result.primary_score is not None:
        return result.primary_score
    return result.metric(metric)


def _cv_stats(result: ExperimentResult) -> tuple[float | None, float | None, int]:
    scores = [s for s in result.cv_scores if s is not None]
    if not scores:
        return None, None, 0
    mean = statistics.fmean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std, len(scores)


def rank_results(log: ExperimentLog) -> list[ExperimentResult]:
    """Successful results ordered best-first on the primary metric.

    Args:
        log: A populated experiment log.

    Returns:
        Successful results, best first; rows with no score are appended last.
    """
    metric = log.primary_metric
    scored: list[tuple[ExperimentResult, float]] = []
    unscored: list[ExperimentResult] = []
    for result in log.results:
        if result.failed:
            continue
        score = _score_of(result, metric)
        if score is None:
            unscored.append(result)
        else:
            scored.append((result, score))
    scored.sort(key=lambda pair: pair[1], reverse=bool(log.higher_is_better))
    return [result for result, _ in scored] + unscored


def successful_results(log: ExperimentLog | None) -> list[ExperimentResult]:
    """Every result that trained and scored without raising."""
    if log is None:
        return []
    return [r for r in log.results if not r.failed]


def baseline_result(log: ExperimentLog | None) -> ExperimentResult | None:
    """The baseline row, which every other score must be read against."""
    if log is None:
        return None
    return next((r for r in log.results if r.is_baseline and not r.failed), None)


def best_result(log: ExperimentLog | None) -> ExperimentResult | None:
    """The winning row, preferring the executor's own ``best_experiment_id``."""
    if log is None:
        return None
    chosen = log.best()
    if chosen is not None and not chosen.failed:
        return chosen
    ranked = rank_results(log)
    return ranked[0] if ranked else None


def _relative_gain(best: float, baseline: float, higher_is_better: bool) -> float | None:
    if baseline == 0:
        return None
    delta = (best - baseline) if higher_is_better else (baseline - best)
    return delta / abs(baseline)


def render_leaderboard(log: ExperimentLog | None, *, limit: int = 14) -> str:
    """Render the measured leaderboard as compact prompt text.

    Args:
        log: The computed experiment log.
        limit: Maximum number of successful rows to render.

    Returns:
        Markdown-ish text with one line per experiment, plus a failures section.
    """
    if log is None or not log.results:
        return "No experiments were run."

    metric = log.primary_metric or "primary_metric"
    direction = "higher is better" if log.higher_is_better else "lower is better"
    lines: list[str] = [
        f"### Measured leaderboard (primary metric: {metric}, {direction})",
    ]

    baseline = baseline_result(log)
    baseline_score = _score_of(baseline, metric) if baseline else None
    ranked = rank_results(log)
    winner = best_result(log)
    best_id = winner.experiment_id if winner is not None else None

    for position, result in enumerate(ranked[:limit], start=1):
        score = _score_of(result, metric)
        mean, std, folds = _cv_stats(result)
        bits = [
            f"{position}. `{result.family.value}`"
            + (f" [{result.label}]" if result.label else ""),
            f"{metric}={_g(score)}",
        ]
        if mean is not None:
            bits.append(f"cv={_g(mean)} +/- {_g(std, 3)} over {folds} folds")
        if baseline_score is not None and score is not None and not result.is_baseline:
            delta = (
                score - baseline_score
                if log.higher_is_better
                else baseline_score - score
            )
            bits.append(f"vs_baseline={delta:+.4g}")
        others = [
            f"{m.name}={_g(m.value)}" for m in result.metrics if m.name != metric
        ][:6]
        if others:
            bits.append("also: " + " ".join(others))
        bits.append(
            f"train={_secs(result.train_seconds)} predict={_secs(result.predict_seconds)}"
        )
        if result.n_features_in:
            bits.append(f"n_features={result.n_features_in}")
        if result.peak_memory_mb:
            bits.append(f"peak_mem={result.peak_memory_mb:.1f}MB")
        flags = []
        if result.is_baseline:
            flags.append("BASELINE")
        if result.tuned:
            flags.append("TUNED")
        if result.experiment_id == best_id:
            flags.append("SELECTED_BEST")
        if flags:
            bits.append("|".join(flags))
        lines.append(" | ".join(bits))

    hidden = max(0, len(ranked) - limit)
    if hidden:
        lines.append(f"_({hidden} further successful rows omitted for brevity.)_")

    failures = [r for r in log.results if r.failed]
    if failures:
        lines.append("")
        lines.append("### Candidates that failed to train")
        for result in failures:
            lines.append(
                f"- `{result.family.value}`"
                + (f" [{result.label}]" if result.label else "")
                + f": {result.error or 'no error message recorded'}"
            )
    return "\n".join(lines)


def render_margin_analysis(log: ExperimentLog | None) -> str:
    """Render the deltas that decide whether the winner really won.

    The ratio of the top-two margin to the pooled cross-validation standard
    deviation is the single most useful number for judging a leaderboard, so it
    is computed here rather than left to the model's arithmetic.

    Args:
        log: The computed experiment log.

    Returns:
        Prompt text, or a short note when there is nothing to compare.
    """
    if log is None:
        return "No margin analysis available: no experiments were run."
    ranked = rank_results(log)
    if not ranked:
        return "No margin analysis available: no candidate trained successfully."

    metric = log.primary_metric or "primary_metric"
    hib = bool(log.higher_is_better)
    lines = ["### Margin analysis (computed, not estimated)"]

    top = ranked[0]
    top_score = _score_of(top, metric)
    _, top_std, top_folds = _cv_stats(top)

    baseline = baseline_result(log)
    baseline_score = _score_of(baseline, metric) if baseline else None
    if baseline_score is not None and top_score is not None:
        delta = top_score - baseline_score if hib else baseline_score - top_score
        relative = _relative_gain(top_score, baseline_score, hib)
        lines.append(
            f"- best (`{top.family.value}`) vs baseline (`{baseline.family.value}`): "  # type: ignore[union-attr]
            f"{_g(top_score)} vs {_g(baseline_score)} -> {delta:+.4g}"
            + (f" ({relative * 100:+.1f}% relative)" if relative is not None else "")
        )
        if top_std:
            lines.append(
                f"- that baseline margin is {abs(delta) / top_std:.2f}x the winner's "
                f"CV standard deviation ({_g(top_std, 3)} over {top_folds} folds)"
            )
    else:
        lines.append("- no baseline score is available to measure lift against")

    non_baseline = [r for r in ranked if not r.is_baseline]
    if len(non_baseline) >= 2:
        first, second = non_baseline[0], non_baseline[1]
        s1, s2 = _score_of(first, metric), _score_of(second, metric)
        _, std1, _ = _cv_stats(first)
        _, std2, _ = _cv_stats(second)
        if s1 is not None and s2 is not None:
            margin = abs(s1 - s2)
            pooled = None
            if std1 is not None and std2 is not None and (std1 or std2):
                pooled = math.sqrt((std1**2 + std2**2) / 2.0)
            line = (
                f"- top two non-baseline candidates: `{first.family.value}` {_g(s1)} vs "
                f"`{second.family.value}` {_g(s2)} -> margin {margin:.4g}"
            )
            if pooled:
                ratio = margin / pooled
                verdict = (
                    "WITHIN NOISE (treat as a tie)"
                    if ratio < 1.0
                    else "outside one pooled CV std"
                )
                line += f"; pooled CV std {_g(pooled, 3)} -> {ratio:.2f}x -> {verdict}"
            else:
                line += "; no CV spread recorded, so the margin cannot be tested"
            lines.append(line)

    spreads = [
        (r.family.value, _cv_stats(r)[1])
        for r in non_baseline[:5]
        if _cv_stats(r)[1] is not None
    ]
    if spreads:
        lines.append(
            "- fold-to-fold spread of the leading candidates: "
            + ", ".join(f"{name} +/- {_g(std, 3)}" for name, std in spreads)
        )

    costs = [(r.family.value, r.train_seconds) for r in non_baseline[:5]]
    if costs:
        lines.append(
            "- training cost of the leading candidates: "
            + ", ".join(f"{name} {_secs(sec)}" for name, sec in costs)
        )
    return "\n".join(lines)


def _fallback_notes(log: ExperimentLog) -> str:
    """Deterministic leaderboard notes, used when no LLM call is worthwhile.

    Prefers the trainer's own rendered leaderboard, which is a real measured
    table, over a summary sentence assembled here.
    """
    survivors = successful_results(log)
    if survivors and log.leaderboard_notes.strip():
        return log.leaderboard_notes
    if not survivors:
        failures = [r for r in log.results if r.failed]
        if failures:
            reasons = "; ".join(
                f"{r.family.value}: {r.error or 'unknown error'}" for r in failures[:5]
            )
            return (
                f"No candidate trained successfully ({len(failures)} attempted). "
                f"Recorded failures: {reasons}."
            )
        return "No experiments were run, so there is no leaderboard to compare."
    top = best_result(log)
    metric = log.primary_metric or "primary metric"
    score = _score_of(top, metric) if top else None
    return (
        f"{len(survivors)} candidate(s) trained. Best measured: "
        f"{top.family.value if top else 'unknown'} at {metric}={_g(score)}. "
        "Narrative comparison was skipped for this run."
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


_INSTRUCTIONS = """\
You are the experiment analyst on a modelling team. Training has already \
happened: a deterministic executor fitted every selected candidate, \
cross-validated it, and timed it. You did not choose the candidates and you are \
not choosing the winner — the executor already selected it by the primary \
metric. Your job is the paragraph a senior practitioner writes underneath a \
leaderboard, the one that tells a reader whether the table means what it \
appears to mean.

METHOD — read the table in this order:

1.  Find the baseline. Every other score is only meaningful as a delta over it. \
A 0.91 accuracy against a 0.90 majority-class baseline is a model that learned \
almost nothing; a 0.72 ROC AUC against 0.50 is a model that learned a lot.
2.  Test the winner's margin against the noise. The margin analysis gives you \
the top-two gap as a multiple of the pooled cross-validation standard \
deviation. Below roughly 1x, say plainly that the ordering is within noise and \
that the choice between those two families is effectively a tie — then say \
which one you would keep on grounds other than the score (fewer parameters, \
faster inference, better calibration, easier explanation).
3.  Look at the fold-to-fold spread itself. A wide spread means small data, an \
unlucky split, or heterogeneous subgroups; it caps how much any downstream \
comparison can claim.
4.  Price the win. Compare the winner's training time against the runner-up's \
and the baseline's. A gradient-boosting model that takes 40x the training time \
for +0.004 is a bad trade for a batch job and a worse one for retraining daily; \
say so.
5.  Read the failures. A family that failed to fit is evidence about the \
feature matrix — sparse input, remaining NaNs, non-numeric columns, singular \
design — not just a missing row. Name the likely cause when the error text \
supports it.
6.  Flag anything that looks too good. A near-perfect score on a non-trivial \
task usually means leakage, not skill.

FAILURE MODES TO AVOID, SPECIFIC TO THIS JOB:

-   Declaring a winner on a margin smaller than the CV standard deviation.
-   Reading accuracy as success on an imbalanced target, where the baseline \
already scores what looks like a good number.
-   Treating one holdout number as truth when the CV spread is wide.
-   Ignoring training cost, model size, or prediction latency because the score \
column is the only one that looks like a result.
-   Praising a model whose score is suspiciously close to perfect.
-   Restating rows of the table as prose. The table is already in the prompt; \
add the judgement it lacks.

OUTPUT DISCIPLINE: you author `leaderboard_notes` and nothing else. Every \
numeric field and the results list are overwritten with the measured values \
after you reply, so leave them at their defaults rather than echoing them. \
Quote figures inside your prose exactly as given — never round a score into a \
different number, and never introduce a figure that is not in the prompt.

A good note: "Hist gradient boosting leads at ROC AUC 0.8731, but only 0.0042 \
ahead of random forest — 0.33x the pooled CV standard deviation of 0.0128, so \
the two are indistinguishable on this 4,120-row dataset. Both are decisively \
above the stratified-dummy baseline of 0.5001 (+0.373), so there is real signal. \
Random forest trained in 2.4s against boosting's 18.7s; if retraining cadence \
matters, the tie should break toward the forest. SVM failed on the 3,412 one-hot \
columns, which is consistent with its O(n^2) kernel on a wide sparse matrix."

A bad note: "Gradient boosting performed best with strong results. Random forest \
also did well. Overall the models show good performance and the best one should \
be selected for deployment." — no numbers, no baseline, no noise test, no cost, \
no judgement.
"""


class ExperimentAgent(HybridAgent[ExperimentLog]):
    """Runs every candidate for real, then narrates the leaderboard.

    ``compute`` delegates to ``execution.trainer.run_experiments``; the agent
    contributes only ``leaderboard_notes``. Numeric fields are restored verbatim
    in :meth:`postprocess`.
    """

    name = AgentName.EXPERIMENT
    title = "Experiment Agent"
    output_model = ExperimentLog
    effort: Effort = "medium"
    max_tokens = 16_000

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm)
        self._computed: ExperimentLog | None = None

    # -- deterministic half -------------------------------------------------

    def compute(self, state: RunState) -> None:
        """Train and cross-validate every selected candidate.

        Any failure degrades to an empty log with a warning rather than
        aborting the run, so the orchestrator can still report what happened.
        """
        self._computed = None
        empty = ExperimentLog(
            primary_metric=state.primary_metric,
            higher_is_better=metric_higher_is_better(state.primary_metric),
        )
        try:
            from ..execution.trainer import run_experiments
        except Exception as exc:  # noqa: BLE001 - executor may be unavailable
            state.add_warning(
                f"{self.title}: the training executor could not be imported "
                f"({exc}); no experiments were run."
            )
            self._computed = empty
            state.experiments = empty
            return

        try:
            log = run_experiments(state)
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the run
            logger.exception("run_experiments failed")
            state.add_warning(
                f"{self.title}: training failed ({type(exc).__name__}: {exc}); "
                "continuing with an empty leaderboard."
            )
            self._computed = empty
            state.experiments = empty
            return

        if not isinstance(log, ExperimentLog):
            state.add_warning(
                f"{self.title}: the training executor returned "
                f"{type(log).__name__} instead of ExperimentLog; ignoring it."
            )
            log = empty
        if not log.primary_metric:
            log.primary_metric = state.primary_metric
        self._computed = log
        state.experiments = log

    # -- reasoning half -----------------------------------------------------

    def instructions(self, state: RunState) -> str:
        return _INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        log = self._computed or state.experiments
        parts: list[str] = ["## EXPERIMENT RESULTS", ""]

        sizes = state.splits.sizes()
        context = [
            f"- task: {state.task_type.value if state.task_type else 'unknown'}",
            f"- target: `{state.target or 'none'}`",
            f"- primary metric: {state.primary_metric}",
            f"- split sizes: train={sizes['train']:,} validation={sizes['validation']:,} "
            f"test={sizes['test']:,}",
            f"- split strategy: {state.splits.strategy or 'unrecorded'}",
            f"- cross-validation folds configured: {state.config.cv_folds}",
            f"- engineered feature count: {len(state.feature_names) or 'unrecorded'}",
            f"- time remaining in the run budget: {state.time_remaining:.0f}s "
            f"of {state.config.time_budget_seconds}s",
        ]
        if state.model_selection and state.model_selection.validation_strategy:
            context.append(
                f"- validation strategy chosen upstream: "
                f"{state.model_selection.validation_strategy}"
            )
        parts.extend(context)
        parts.append("")
        parts.append(render_leaderboard(log))
        parts.append("")
        parts.append(render_margin_analysis(log))

        if state.model_selection and state.model_selection.candidates:
            parts.append("")
            parts.append("### What the model-selection agent expected")
            for candidate in state.model_selection.candidates[:8]:
                parts.append(
                    f"- `{candidate.family.value}` (rank {candidate.rank}, "
                    f"{candidate.suitability}): {candidate.rationale}"
                )

        parts.append("")
        parts.append(
            "Write `leaderboard_notes`: the analyst's comparison of these measured "
            "results. Name the winner and the size of its lift over the baseline, "
            "state whether the margin over the runner-up survives the "
            "cross-validation spread, price the win in training time, and account "
            "for any failed candidate. Be concrete and quote the figures above "
            "exactly. Two to four tight paragraphs; leave every other field alone."
        )
        return "\n".join(parts)

    # -- grounding ----------------------------------------------------------

    def postprocess(self, value: ExperimentLog, state: RunState) -> ExperimentLog:
        """Restore every measured field, keeping only the authored notes."""
        computed = self._computed or state.experiments
        if computed is None:
            return value

        if value.results and (
            len(value.results) != len(computed.results)
            or value.best_experiment_id != computed.best_experiment_id
        ):
            state.add_warning(
                f"{self.title}: the model returned its own leaderboard rows; "
                "the measured results were restored verbatim."
            )

        notes = (value.leaderboard_notes or "").strip() or _fallback_notes(computed)
        return ExperimentLog(
            results=[r.model_copy(deep=True) for r in computed.results],
            best_experiment_id=computed.best_experiment_id,
            primary_metric=computed.primary_metric,
            higher_is_better=computed.higher_is_better,
            leaderboard_notes=notes,
        )

    def apply(self, state: RunState, value: ExperimentLog) -> None:
        """Publish the grounded log and record the winning score as a metric."""
        state.experiments = value
        top = best_result(value)
        score = _score_of(top, value.primary_metric) if top else None
        if score is not None:
            state.bus.metric(value.primary_metric, float(score))

    def decision_summary(self, value: ExperimentLog) -> str:
        top = best_result(value)
        if top is None:
            return "No candidate trained successfully"
        score = _score_of(top, value.primary_metric)
        baseline = baseline_result(value)
        baseline_score = (
            _score_of(baseline, value.primary_metric) if baseline else None
        )
        trained = len(successful_results(value))
        failed = sum(1 for r in value.results if r.failed)
        lift = ""
        if score is not None and baseline_score is not None:
            delta = (
                score - baseline_score
                if value.higher_is_better
                else baseline_score - score
            )
            lift = f", {delta:+.4g} vs baseline"
        return (
            f"Best: {top.family.value} {value.primary_metric}={_g(score)}{lift} "
            f"({trained} trained, {failed} failed)"
        )

    # -- execution ----------------------------------------------------------

    def run(self, state: RunState) -> ExperimentLog:
        """Train, then interpret — skipping the LLM when nothing was measured."""
        self.compute(state)
        computed = self._computed or ExperimentLog(
            primary_metric=state.primary_metric,
            higher_is_better=metric_higher_is_better(state.primary_metric),
        )
        if not successful_results(computed):
            # Narrating an empty leaderboard would be prose about nothing, so
            # skip the call and record the measured failure instead.
            log = computed.model_copy(deep=True)
            log.leaderboard_notes = _fallback_notes(computed)
            self.apply(state, log)
            state.bus.decision(self.name, self.decision_summary(log))
            return log
        # compute() already ran; call BaseAgent.run directly so HybridAgent.run
        # does not train everything a second time.
        return BaseAgent.run(self, state)


__all__ = [
    "ExperimentAgent",
    "baseline_result",
    "best_result",
    "metric_higher_is_better",
    "rank_results",
    "render_leaderboard",
    "render_margin_analysis",
    "successful_results",
]
