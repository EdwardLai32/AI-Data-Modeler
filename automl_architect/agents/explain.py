"""The Explainability Agent: measured attributions, business-readable language.

The execution layer does the arithmetic — SHAP where the library and the model
support it, permutation importance always, partial dependence for the top
features. This agent turns that into the two things a stakeholder can actually
use: a narrative that says which drivers matter and in which direction, and a
list of plain sentences of the form "customer tenure contributes approximately
31% of churn prediction importance".

Two grounding rules are enforced in code rather than trusted to the prompt.
Every importance value, direction, and file path is copied verbatim from the
computed report, and the importances are renormalised to sum to 1.0 so the
percentages in the narrative and the numbers in the charts cannot disagree. And
if SHAP did not run, the narrative is not allowed to imply that it did — an
explanation that misstates its own method is worse than no explanation, because
it is trusted more than it deserves.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from ..core.agent import BaseAgent, HybridAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ExplainabilityReport,
    FeatureAttribution,
)
from ..core.state import RunState
from .experiment import best_result

logger = logging.getLogger(__name__)

#: Appended when the agent forgets to say it. Importance is association under
#: one fitted model, and stakeholders reliably read it as causation.
IMPORTANCE_CAVEAT = (
    "These contributions describe what the fitted model relies on, not what "
    "causes the outcome: intervening on a high-importance feature will not "
    "necessarily move the result by the amount its share suggests."
)

_DIRECTION_PHRASE = {
    "increases": "higher values push the prediction upward",
    "decreases": "higher values push the prediction downward",
    "mixed": "its effect changes direction across the range, so there is no single sign",
    "unknown": "the direction of its effect was not measured",
}


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.1f}%"


def _g(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number:.{digits}g}"


def renormalise_attributions(
    items: list[FeatureAttribution],
) -> list[FeatureAttribution]:
    """Return attributions sorted best-first with importances summing to 1.0.

    Absolute values are used, because permutation importance can come back
    slightly negative for an irrelevant feature and a negative share makes the
    percentage phrasing nonsensical.

    Args:
        items: Attributions as measured by the execution layer.

    Returns:
        A new list; the input is not mutated.
    """
    if not items:
        return []
    total = sum(abs(item.importance) for item in items)
    out: list[FeatureAttribution] = []
    for item in items:
        copy = item.model_copy(deep=True)
        copy.importance = abs(item.importance) / total if total > 0 else 0.0
        out.append(copy)
    out.sort(key=lambda a: a.importance, reverse=True)
    return out


def _method_label(report: ExplainabilityReport) -> str:
    """The dominant attribution method actually used, for honest narration."""
    methods = Counter(
        item.method
        for item in (*report.global_attributions, *report.permutation_importance)
        if item.method
    )
    if methods:
        return methods.most_common(1)[0][0]
    return "shap" if report.shap_available else "permutation importance"


def render_attributions(
    items: list[FeatureAttribution], *, limit: int = 20
) -> str:
    """Render ranked attributions as prompt text with percentage shares.

    Renormalises first, so the percentages an agent reads are always the
    percentages that end up in the report even if the executor handed over raw
    importance magnitudes.

    Args:
        items: Attributions to render.
        limit: How many rows to show before summarising the tail.

    Returns:
        Prompt text, one ranked line per feature.
    """
    if not items:
        return "_none computed_"
    items = renormalise_attributions(items)
    lines: list[str] = []
    for position, item in enumerate(items[:limit], start=1):
        lines.append(
            f"{position}. `{item.feature}` - share {_pct(item.importance)} "
            f"(direction={item.direction}, method={item.method})"
        )
    if len(items) > limit:
        remainder = sum(abs(i.importance) for i in items[limit:])
        lines.append(
            f"_({len(items) - limit} further features hold {_pct(remainder)} "
            "of the total between them.)_"
        )
    return "\n".join(lines)


#: Phrases that show the author already made the association-not-causation point.
#: Matched as substrings, so each one must be specific enough that an ordinary
#: sentence cannot contain it by accident — in particular, a bare ``"caus"``
#: test is defeated by the word "because".
_CAUSATION_CAVEAT_MARKERS = (
    "causation",
    "causal",
    "causes",
    "cause the",
    "not cause",
    "correlation",
    "association",
    "associated with",
)


def _has_causation_caveat(lines: list[str]) -> bool:
    """Whether the authored explanations already disclaim causation."""
    joined = " ".join(lines).lower()
    return any(marker in joined for marker in _CAUSATION_CAVEAT_MARKERS)


_SHAP_HONEST_MARKERS = (
    "shap was not",
    "shap was unavailable",
    "shap is not available",
    "shap unavailable",
    "shap could not",
    "without shap",
    "no shap",
    "shap did not",
)


def _misstates_shap(text: str, report: ExplainabilityReport) -> bool:
    """Whether ``text`` implies SHAP ran when the measured report says otherwise."""
    if report.shap_available:
        return False
    low = (text or "").lower()
    if "shap" not in low:
        return False
    return not any(marker in low for marker in _SHAP_HONEST_MARKERS)


def _fallback_explanations(report: ExplainabilityReport) -> list[str]:
    """Deterministic plain-language sentences, from the measured shares alone."""
    items = report.global_attributions or report.permutation_importance
    method = _method_label(report)
    out: list[str] = []
    for item in items[:6]:
        sentence = (
            f"{item.feature} contributes approximately "
            f"{item.importance * 100:.0f}% of total prediction importance "
            f"(measured by {method})"
        )
        if item.direction in ("increases", "decreases"):
            sentence += f", and {_DIRECTION_PHRASE[item.direction]}"
        out.append(sentence + ".")
    out.append(IMPORTANCE_CAVEAT)
    return out


def _fallback_narrative(report: ExplainabilityReport) -> str:
    """Deterministic narrative for when no LLM call is worthwhile."""
    items = report.global_attributions or report.permutation_importance
    if not items:
        return (
            "No feature attributions could be computed for this run, so the "
            "model's behaviour is not explained here."
        )
    method = _method_label(report)
    top = ", ".join(
        f"{item.feature} ({item.importance * 100:.0f}%)" for item in items[:5]
    )
    return (
        f"Feature importance was measured with {method}. The leading "
        f"contributors are {top}. {IMPORTANCE_CAVEAT}"
    )


_INSTRUCTIONS = """\
You are the model-interpretation specialist. Someone has already computed the \
attributions with real code; nobody has yet said what they mean to the person \
who has to act on them. That translation is your only job, and it is judged by \
whether a non-technical stakeholder could repeat your sentences in a meeting \
without saying anything false.

METHOD:

1.  Read the ranked shares and convert them to percentage-of-importance \
phrasing. "Customer tenure contributes approximately 31% of churn prediction \
importance" is the register: the feature, a rounded percentage, and the outcome \
being predicted.
2.  State direction only where it was measured. The attribution list carries a \
direction field. Where it says increases or decreases, say so in plain words. \
Where it says mixed or unknown, say the feature matters without asserting a \
sign — an invented direction is the most damaging error you can make here, \
because it is the part a stakeholder will act on.
3.  Read the shape of the distribution, not just the ranking. If the top three \
features hold most of the importance, this is a few-driver problem and the \
narrative should be about those drivers. If the shares are flat across twenty \
features, say that the signal is diffuse — that usually means weak individual \
predictors, and it changes what the business should conclude.
4.  Cross-check the top drivers against the dataset facts you were given. A \
leading feature with heavy missingness is a fragile driver in production. A \
leading feature that looks like an identifier or a timestamp is a leakage \
signal, not an insight. A leading feature that is one level of a one-hot \
encoding tells you about that level, not the whole variable — say "customers on \
month-to-month contracts", not "contract type".
5.  Name the method honestly and explain its limits in one clause. SHAP \
attributes each prediction's deviation from the base value and gives direction. \
Permutation importance measures how much the metric degrades when one column is \
shuffled, gives no direction, and splits credit unpredictably between correlated \
columns — so when two correlated features both appear mid-table, their individual \
shares understate the pair. If SHAP was not available, say which method was used \
instead. Never write as though SHAP ran when the facts say it did not.
6.  Close with the causation caveat, in your own words. Importance is what this \
model leans on given these features; it is not a claim that changing the feature \
changes the outcome.

FAILURE MODES TO AVOID, SPECIFIC TO THIS JOB:

-   Causal language: "tenure drives churn", "reducing charges will retain \
customers". Say "associated with", "predictive of", "the model relies on".
-   Inventing or adjusting a number. Every percentage you write must be the \
share you were given, rounded.
-   Inventing a feature name, or tidying one. Use the names exactly as measured.
-   Asserting a direction the method did not measure.
-   Implying SHAP ran when it did not.
-   Treating a one-hot level as the whole variable.
-   Silence about correlated features when permutation importance is the method.
-   Writing counterfactuals. Do not author them at all: a counterfactual claims \
what the model would predict under a changed input, which requires re-running \
the model. You cannot do that, and a plausible-looking invented prediction is a \
fabrication.

OUTPUT DISCIPLINE: you author `narrative`, `plain_language_explanations`, and \
`method_notes`. Every attribution value, direction, availability flag, and file \
path is restored from the measured report after you reply, so leave those fields \
at their defaults. Write one plain-language sentence per meaningful driver \
(roughly three to six of them, highest share first), then the caveat. The \
narrative is several paragraphs for a technical reader; the plain-language list \
is for an executive.

A good plain-language line: "Contract type being month-to-month contributes \
approximately 24% of churn prediction importance, and shifts predictions toward \
churn — the single strongest signal the model uses."

A bad one: "Contract type is very important and causes customers to churn, so \
the business should change contracts to reduce churn by 24%." — causal, \
misreads the percentage as an effect size, and turns a share of importance into \
a business promise.
"""


class ExplainAgent(HybridAgent[ExplainabilityReport]):
    """Computes real attributions, then narrates them for humans."""

    name = AgentName.EXPLAIN
    title = "Explainability Agent"
    output_model = ExplainabilityReport
    effort: Effort = "high"
    max_tokens = 16_000

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm)
        self._computed: ExplainabilityReport | None = None

    # -- deterministic half -------------------------------------------------

    def compute(self, state: RunState) -> None:
        """Run SHAP / permutation importance / partial dependence for real."""
        self._computed = None
        empty = ExplainabilityReport()

        if not state.config.enable_explainability:
            empty.method_notes = (
                "Explainability was disabled for this run by the operator."
            )
            self._computed = empty
            state.explainability = empty
            state.bus.log("explainability disabled by configuration", agent=self.name)
            return

        try:
            from ..execution.explainer import compute_explanations
        except Exception as exc:  # noqa: BLE001 - executor may be unavailable
            state.add_warning(
                f"{self.title}: the explainability executor could not be imported "
                f"({exc}); the model will not be explained."
            )
            empty.method_notes = f"Explainability executor unavailable: {exc}"
            self._computed = empty
            state.explainability = empty
            return

        try:
            report = compute_explanations(state)
        except Exception as exc:  # noqa: BLE001 - reduced capability, not a crash
            logger.exception("compute_explanations failed")
            state.add_warning(
                f"{self.title}: attribution computation failed "
                f"({type(exc).__name__}: {exc}); continuing without explanations."
            )
            empty.method_notes = f"Attribution computation failed: {exc}"
            self._computed = empty
            state.explainability = empty
            return

        if not isinstance(report, ExplainabilityReport):
            state.add_warning(
                f"{self.title}: the explainability executor returned "
                f"{type(report).__name__} instead of ExplainabilityReport; "
                "ignoring it."
            )
            report = empty

        report.global_attributions = renormalise_attributions(
            report.global_attributions
        )
        report.permutation_importance = renormalise_attributions(
            report.permutation_importance
        )
        self._computed = report
        state.explainability = report

    # -- reasoning half -----------------------------------------------------

    def instructions(self, state: RunState) -> str:
        return _INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        report = self._computed or state.explainability or ExplainabilityReport()
        parts: list[str] = ["## MEASURED MODEL EXPLANATIONS", ""]

        top = best_result(state.experiments)
        parts.append("### The model being explained")
        if top is not None:
            score = top.primary_score
            parts.append(
                f"- family: `{top.family.value}`"
                + (f" [{top.label}]" if top.label else "")
                + (" (hyperparameter-tuned)" if top.tuned else "")
            )
            parts.append(
                f"- measured {state.primary_metric}: {_g(score)}; "
                f"features in: {top.n_features_in or 'unrecorded'}"
            )
        else:
            parts.append("- no winning model was recorded")
        parts.append(
            f"- task: {state.task_type.value if state.task_type else 'unknown'}; "
            f"predicting `{state.target or 'unknown target'}`"
        )
        if state.problem and state.problem.positive_class:
            parts.append(
                f"- positive class of interest: `{state.problem.positive_class}`"
            )
        if state.problem and state.problem.business_objective:
            parts.append(
                f"- business objective this must speak to: "
                f"{state.problem.business_objective}"
            )
        parts.append("")

        parts.append("### Attribution method actually used")
        parts.append(
            f"- SHAP available for this model: "
            f"{str(report.shap_available).lower()}"
            + (
                ""
                if report.shap_available
                else " -> do NOT write as though SHAP values were computed"
            )
        )
        parts.append(f"- dominant method recorded: {_method_label(report)}")
        if report.method_notes:
            parts.append(f"- executor notes: {report.method_notes}")
        if report.shap_summary_path:
            parts.append(f"- SHAP summary chart: {report.shap_summary_path}")
        if report.partial_dependence_paths:
            parts.append(
                f"- partial-dependence charts rendered: "
                f"{len(report.partial_dependence_paths)}"
            )
        parts.append("")

        parts.append("### Global attributions (normalised shares, best first)")
        parts.append(render_attributions(report.global_attributions))
        parts.append("")

        if report.permutation_importance:
            parts.append("### Permutation importance (normalised shares)")
            parts.append(render_attributions(report.permutation_importance, limit=15))
            parts.append("")

        cross_check = self._cross_check_lines(state, report)
        if cross_check:
            parts.append("### Cross-checks against the profiled columns")
            parts.extend(cross_check)
            parts.append("")

        parts.append(
            "Write `narrative`, `plain_language_explanations`, and `method_notes` "
            "for these measured attributions. Use the shares exactly as given, "
            "assert a direction only where one was measured, name the method "
            "honestly, and end with the causation caveat. Leave every numeric "
            "field and path alone."
        )
        return "\n".join(parts)

    def _cross_check_lines(
        self, state: RunState, report: ExplainabilityReport
    ) -> list[str]:
        """Flag top drivers whose profile makes them fragile or suspicious."""
        if state.profile is None:
            return []
        lines: list[str] = []
        items = renormalise_attributions(
            report.global_attributions or report.permutation_importance
        )
        for item in items[:8]:
            column = state.profile.column(item.feature)
            if column is None:
                continue
            notes: list[str] = []
            if column.missing_fraction >= 0.05:
                notes.append(f"{_pct(column.missing_fraction)} missing")
            if column.looks_like_id:
                notes.append("looks like an identifier (leakage risk)")
            if column.looks_like_datetime:
                notes.append("datetime-like")
            if column.is_near_zero_variance:
                notes.append("near-zero variance")
            if column.n_unique <= 2:
                notes.append("binary")
            if notes:
                lines.append(
                    f"- `{item.feature}` ({_pct(item.importance)} of importance): "
                    + "; ".join(notes)
                )
        if items and not lines:
            lines.append(
                "- no profiled concerns on the leading features (several may be "
                "engineered columns with no direct raw-column counterpart)"
            )
        return lines

    # -- grounding ----------------------------------------------------------

    def postprocess(
        self, value: ExplainabilityReport, state: RunState
    ) -> ExplainabilityReport:
        """Restore every measured value and enforce honest method language."""
        computed = self._computed or state.explainability
        if computed is None:
            return value

        out = value.model_copy(deep=True)
        out.global_attributions = renormalise_attributions(
            computed.global_attributions
        )
        out.permutation_importance = renormalise_attributions(
            computed.permutation_importance
        )
        out.shap_available = computed.shap_available
        out.shap_summary_path = computed.shap_summary_path
        out.partial_dependence_paths = list(computed.partial_dependence_paths)

        if value.counterfactuals and not computed.counterfactuals:
            state.add_warning(
                f"{self.title}: discarded {len(value.counterfactuals)} "
                "model-authored counterfactual(s); counterfactual predictions must "
                "come from re-running the estimator, not from narration."
            )
        out.counterfactuals = [c.model_copy(deep=True) for c in computed.counterfactuals]

        out.method_notes = self._merge_method_notes(computed, value.method_notes)
        out.narrative = self._ground_narrative(out, value.narrative, state)
        out.plain_language_explanations = self._ground_explanations(
            out, value.plain_language_explanations
        )
        return out

    @staticmethod
    def _merge_method_notes(computed: ExplainabilityReport, authored: str) -> str:
        """The executor's record of what it ran stays authoritative.

        The agent's commentary is appended only when it does not contradict that
        record — a note claiming SHAP ran when it did not would be the one place
        a reader goes to check the method.
        """
        measured = (computed.method_notes or "").strip()
        authored = (authored or "").strip()
        if _misstates_shap(authored, computed):
            authored = ""
        if not measured:
            if authored:
                return authored
            if computed.global_attributions or computed.permutation_importance:
                return f"Attributions computed by {_method_label(computed)}."
            return "No attribution method completed."
        if authored and authored not in measured:
            return f"{measured}\n\n{authored}"
        return measured

    def _ground_narrative(
        self, report: ExplainabilityReport, narrative: str, state: RunState
    ) -> str:
        """Prepend a correction when the narrative misstates its own method."""
        text = (narrative or "").strip()
        if not text:
            return _fallback_narrative(report)
        if _misstates_shap(text, report):
            state.add_warning(
                f"{self.title}: the narrative referred to SHAP, which was not "
                "available for this model; a correction was prepended."
            )
            return (
                f"Method note: SHAP was not available for this model, so the "
                f"attributions below come from {_method_label(report)}. "
                f"{text}"
            )
        return text

    @staticmethod
    def _ground_explanations(
        report: ExplainabilityReport, explanations: list[str]
    ) -> list[str]:
        """Keep authored sentences, but guarantee shares exist and a caveat closes."""
        cleaned = [line.strip() for line in explanations if line and line.strip()]
        if not cleaned:
            return _fallback_explanations(report)
        if not _has_causation_caveat(cleaned):
            cleaned.append(IMPORTANCE_CAVEAT)
        return cleaned

    def apply(self, state: RunState, value: ExplainabilityReport) -> None:
        state.explainability = value

    def decision_summary(self, value: ExplainabilityReport) -> str:
        items = value.global_attributions or value.permutation_importance
        if not items:
            return "No feature attributions were produced"
        top = ", ".join(
            f"{item.feature} ({item.importance * 100:.0f}%)" for item in items[:3]
        )
        method = _method_label(value)
        return f"Top drivers via {method}: {top}"

    # -- execution ----------------------------------------------------------

    def run(self, state: RunState) -> ExplainabilityReport:
        """Attribute, then narrate — skipping the LLM when nothing was measured."""
        self.compute(state)
        computed = self._computed or ExplainabilityReport()
        if not computed.global_attributions and not computed.permutation_importance:
            report = computed.model_copy(deep=True)
            report.narrative = _fallback_narrative(computed)
            report.plain_language_explanations = []
            self.apply(state, report)
            state.bus.decision(self.name, self.decision_summary(report))
            return report
        # compute() already ran; BaseAgent.run avoids HybridAgent.run's second call.
        return BaseAgent.run(self, state)


__all__ = [
    "IMPORTANCE_CAVEAT",
    "ExplainAgent",
    "render_attributions",
    "renormalise_attributions",
]
