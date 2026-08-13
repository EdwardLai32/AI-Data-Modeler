"""Business Insight Agent.

The pipeline up to this point produces numbers: a leaderboard, SHAP values, a
verdict. None of those are decisions a business can take. This agent's only job
is the translation — from "``tenure`` carries 0.31 of the attribution mass" to
"accounts churn hardest in their first four months, so onboarding is where
retention spend earns the most" — and to refuse the translation where the
measured evidence does not support it.

The postprocessing here exists because that translation is exactly where an LLM
is most tempted to overclaim. Two guards run: references to features that do not
exist are demoted to plain prose, and any insight asserted with high confidence
while the evaluation gate judged the model unacceptable is downgraded rather than
deleted, so the reviewer sees both the claim and the reason to distrust it.
"""

from __future__ import annotations

import re

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import AgentName, InsightReport
from ..core.state import RunState
from . import _deliver_support as support

#: Back-ticked spans are how agents cite a column or metric. Anything cited this
#: way is checkable, which is the point of encouraging the convention.
_BACKTICKED = re.compile(r"`([^`\n]{1,64})`")


class InsightAgent(BaseAgent[InsightReport]):
    """Turns measured model behaviour into actions a decision-maker can take."""

    name = AgentName.INSIGHT
    title = "Business Insight Agent"
    output_model = InsightReport
    effort: Effort = "high"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: the translation method, and what it forbids."""
        return """\
## YOUR ROLE

You are the analytics lead who presents model results to the people who fund and
act on them: a VP of operations, a head of marketing, a product director. You
have twenty years of reading model output and knowing which parts of it change a
decision and which parts are trivia. You do not run models; you decide what they
mean.

## THE METHOD

Work through this procedure in order. Do not skip to writing insights.

1.  **Establish what the model actually knows.** Read the winning model's score
    against the baseline. The lift over baseline — not the absolute score — is
    the measure of how much signal exists in this data. A 0.91 accuracy that a
    majority-class baseline already reaches is worth nothing; a 0.68 AUC against
    a 0.50 baseline is real, exploitable signal. Everything you claim downstream
    is capped by this number.
2.  **Read the attributions as mechanisms, not rankings.** For each of the top
    features, ask what real-world behaviour it stands for and which direction it
    pushes the outcome. `direction=decreases` on a tenure feature is a story
    about early-life churn. `direction=mixed` means the relationship is
    non-monotonic and you must NOT state a simple "more X means more Y" claim.
    Where direction is `unknown`, describe the feature's importance without
    asserting a direction.
3.  **Find the intervention.** An insight that names no action is a fact, not an
    insight. For each mechanism, ask: who can act on this, at what point in the
    process, and what would change? If nobody can act on a feature — a customer's
    signup year, a random id — it is context, not an insight. Say so and move on.
4.  **Size the opportunity from measured quantities only.** Base rates, class
    counts, and slice sizes are given to you. Use them: "the minority class is
    1,842 of 14,300 rows, so a 10-point recall improvement reaches roughly 180
    additional at-risk accounts". Never multiply by a revenue figure, contract
    value, or margin you were not given. If no measured quantity supports a size,
    leave `expected_value` empty rather than inventing an amount.
5.  **Check every claim against the evaluation verdict.** The verdict's
    weaknesses list is a list of things you may not claim. If calibration is
    poor, do not present predicted probabilities as risk percentages. If a
    fairness slice is small or degraded, do not recommend acting on that slice.
    If the model is judged unacceptable, your insights are about what the data
    suggests and what to investigate — not about deploying anything.
6.  **Cut ruthlessly.** Four insights a decision-maker will remember beat twelve
    that blur together. Merge overlapping findings. Drop anything whose action is
    "monitor it" unless monitoring is genuinely the decision.

## WHAT IS FORBIDDEN

*   **Restating technical output.** "Income has importance 0.41" is the input to
    your job, not the output of it. So is "the model achieved an F1 of 0.72" as a
    headline. Convert every number into the behaviour it describes and the choice
    it informs.
*   **Causal language for correlational evidence.** The model measured
    association. Write "accounts with X are substantially more likely to churn",
    not "X causes churn". Recommend the intervention as a test where the causal
    direction is unverified.
*   **Claims beyond the measured scores.** No projected revenue, no ROI, no
    "industry benchmarks", no comparisons to models that were not run.
*   **Hedged mush.** "The model may possibly indicate some relationship" helps
    nobody. State the finding plainly and put the uncertainty in the confidence
    field and the caveats, where a reader can weigh it.

## RATIONALE AND EVIDENCE QUALITY

`supporting_evidence` must name the specific measurement the claim rests on, with
its value, so a reviewer can check it.

Good: "Permutation importance ranks `contract_type` first at 0.24 with
direction=decreases for long-term contracts; month-to-month accounts are 55% of
rows (7,865 of 14,300), so the affected population is large."

Bad: "The model shows that contract type is important for churn." — names no
value, no direction, no population size; nothing here is checkable.

## CONFIDENCE AND AUDIENCE

Set `confidence` from the evidence, not from your enthusiasm: `high` only when
the supporting metric is well above baseline, the attribution is corroborated by
two methods or a large margin, and the affected slice is large. `low` whenever
the claim rests on a small slice, a `mixed`/`unknown` direction, or a model the
evaluator flagged. Tag `audience` with the function that would actually own the
action, so the report can route it.

## CAVEATS ARE MANDATORY

Populate `caveats` with the real limits of this analysis, including: the model
reflects only the historical period it was trained on; association is not
causation and the recommended interventions are hypotheses to test; slices with
few rows are unreliable; and any specific weakness the evaluator recorded. A
caveat list that could be pasted into any report means you have not written the
caveats for this one.

Also populate `key_drivers_plain_language` — one sentence per driver, no jargon,
no numbers-as-headlines — and `suggested_next_experiments` with the concrete
follow-up work that would resolve your biggest uncertainty."""

    def build_prompt(self, state: RunState) -> str:
        """The user turn: outcome, drivers, verdict, and the target's shape."""
        sections = [
            "# WHAT THIS RUN PRODUCED",
            "",
            "## Objective and framing",
            support.run_facts_text(state),
            "",
            "## Model outcome versus baseline",
            support.winner_vs_baseline_text(state),
            "",
            "## Full leaderboard",
            support.leaderboard_text(state.experiments, limit=10),
            "",
            "## Tuning",
            support.tuning_text(state.tuning_decision, state.tuning),
            "",
            "## Feature attributions (what the model relies on, and which way)",
            support.attributions_text(state.explainability, limit=15),
            "",
            "## Evaluation verdict (the ceiling on what you may claim)",
            support.evaluation_text(state.evaluation),
            "",
            "## Target distribution / class balance (population sizes for any quantification)",
            support.target_text(state.profile),
            "",
            "## How the data was split",
            support.splits_text(state),
        ]

        bundles = support.executor_bundles_text(state, char_budget=3000)
        if bundles:
            sections += ["", "## Measured diagnostics", bundles]

        if state.warnings:
            sections += [
                "",
                "## Degradations recorded during this run (limits on your claims)",
                support.warnings_text(state, limit=10),
            ]

        sections += [
            "",
            "# YOUR TASK",
            "",
            "Write the business read-out of this run for the objective stated above.",
            "",
            "Produce the smallest set of insights that would genuinely change a "
            "decision — aim for four to six, not a survey of everything measurable. "
            "For each one: state the behaviour in the world, name the measurement it "
            "rests on with its value, give the action and who owns it, and quantify "
            "the opportunity only from the population counts and scores above.",
            "",
            "Lead your `executive_summary` with the outcome and the decision it "
            "enables — a reader who stops after that paragraph should still know "
            "what the model found, how much to trust it, and what to do next.",
        ]
        return "\n".join(sections)

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: InsightReport, state: RunState) -> InsightReport:
        """Demote unverifiable references and over-confident claims.

        Two repairs, both recorded rather than silent:

        *   A back-ticked name that is not a real column, engineered feature,
            metric, or model family is stripped of its back-ticks, so it reads as
            prose instead of masquerading as a checkable dataset reference.
        *   ``confidence="high"`` is downgraded to ``"low"`` when the evaluation
            gate judged the model unacceptable. Deleting the insight would hide
            the agent's reasoning; downgrading keeps it and flags it.

        Args:
            value: The report as returned by the model.
            state: The run blackboard, used for the ground truth to check against.

        Returns:
            The repaired report.
        """
        known = support.known_reference_names(state)
        unknown: set[str] = set()

        def scrub(text: str) -> str:
            if not text or not known:
                return text

            def replace(match: re.Match[str]) -> str:
                token = match.group(1)
                if token in known:
                    return match.group(0)
                unknown.add(token)
                return token

            return _BACKTICKED.sub(replace, text)

        value.executive_summary = scrub(value.executive_summary)
        value.key_drivers_plain_language = [
            scrub(d) for d in value.key_drivers_plain_language
        ]
        value.caveats = [scrub(c) for c in value.caveats]
        value.suggested_next_experiments = [
            scrub(s) for s in value.suggested_next_experiments
        ]
        for insight in value.insights:
            insight.headline = scrub(insight.headline)
            insight.detail = scrub(insight.detail)
            insight.supporting_evidence = scrub(insight.supporting_evidence)
            insight.recommended_action = scrub(insight.recommended_action)
            insight.expected_value = scrub(insight.expected_value)

        if unknown:
            shown = sorted(unknown)[:8]
            state.add_warning(
                f"{self.title} cited {len(unknown)} name(s) that are not dataset "
                f"columns, engineered features, metrics, or model families "
                f"{shown}{'...' if len(unknown) > 8 else ''}; they are rendered as "
                "plain text so they do not read as verified references."
            )

        verdict = state.evaluation
        if verdict is not None and not verdict.acceptable:
            downgraded = [i for i in value.insights if i.confidence == "high"]
            for insight in downgraded:
                insight.confidence = "low"
            if downgraded:
                note = (
                    f"{len(downgraded)} insight(s) were stated with high confidence, "
                    f"but the evaluation gate judged this model unacceptable "
                    f"(grade {verdict.overall_grade}, recommended action "
                    f"'{verdict.recommended_action}'). Their confidence has been "
                    "downgraded to low: treat them as hypotheses to test, not "
                    "findings to act on."
                )
                value.caveats.append(note)
                state.add_warning(f"{self.title}: {note}")

        if not value.insights:
            state.add_warning(
                f"{self.title} returned no insights; the report will fall back to "
                "the measured leaderboard and attributions for its recommendations."
            )
        return value

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: InsightReport) -> None:
        """Record the insight report on the blackboard."""
        state.insights = value

    def decision_summary(self, value: InsightReport) -> str:
        """One line for the event stream."""
        by_confidence = {"high": 0, "medium": 0, "low": 0}
        for insight in value.insights:
            by_confidence[insight.confidence] = by_confidence.get(insight.confidence, 0) + 1
        audiences = sorted({i.audience for i in value.insights})
        return (
            f"{len(value.insights)} business insight(s) "
            f"(high={by_confidence['high']}, medium={by_confidence['medium']}, "
            f"low={by_confidence['low']}) for {audiences or ['no audience']}; "
            f"{len(value.caveats)} caveat(s)"
        )


__all__ = ["InsightAgent"]
