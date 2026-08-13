"""Natural-language interface over a completed run.

"Why did accuracy decrease?" and "what happens if I remove income?" are the two
question shapes this agent has to handle, and they need opposite treatments. The
first is answerable: the run's recorded history contains the plan revision, the
leaderboard delta, and the events that surround the change, so the answer is a
retrieval-and-explanation task. The second is a counterfactual the run never
tested, and the only honest answer names what the evidence suggests and what
would have to be re-run to actually know.

The failure mode this module is written against is the plausible fabrication —
inventing a score for a model that was never trained, or a reason that appears
nowhere in the log. Hence the prompt's central rule and the ``evidence`` field:
every claim must be traceable to something the orchestrator recorded.

``state.extras['question']`` is the input; ``state.extras['answer']`` is the
output. The agent is invoked outside the plan graph, after a run finishes, so it
does not own a step.
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import AgentName, QuestionAnswer
from ..core.state import RunState
from . import _deliver_support as support

#: Key on ``state.extras`` the caller sets before invoking the agent.
QUESTION_KEY = "question"
#: Key on ``state.extras`` where the answer is stored.
ANSWER_KEY = "answer"

_NO_QUESTION = (
    "(no question was supplied; summarise what this run established and what it "
    "left unresolved)"
)


class QAAgent(BaseAgent[QuestionAnswer]):
    """Answers free-form questions about a finished run, from its record only.

    ``AgentName`` has no ``QA`` member and the schema module is frozen, so this
    agent reports under :attr:`AgentName.REPORT` — the closest existing identity,
    since Q&A is a delivery surface — with its own :attr:`title` for the event
    stream and UI.
    """

    name = AgentName.REPORT
    title = "Question Answering Agent"
    output_model = QuestionAnswer
    effort: Effort = "high"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: grounding rule, counterfactual policy, evidence."""
        return """\
## YOUR ROLE

You are the data scientist who ran this analysis, answering a colleague's question
about it. You have the complete record in front of you: the plan and its
revisions, every cleaning and feature decision with the reasoning recorded at the
time, the full leaderboard, the tuning outcome, the feature attributions, the
evaluation verdict, and the event log. You have nothing else, and you did not
memorise anything beyond it.

## THE CENTRAL RULE

Answer ONLY from the recorded history you were given. This is not a style
preference — the value of this interface is that a user can trust the answer
because it is traceable. Concretely:

*   Never state a score, a parameter value, a row count, or a timing that does not
    appear in the record. If the number the question needs was not measured, say
    which measurement is missing.
*   Never explain a decision with a reason that was not recorded. The rationales
    are in the record verbatim; quote or paraphrase them. If a decision has no
    recorded rationale, say that the reasoning was not captured rather than
    reconstructing a plausible one.
*   Never describe a model, feature, or split that does not appear in the record.
    If someone asks about a family that was excluded, the record says it was
    excluded and usually why — that is the answer.
*   If the record contradicts an assumption inside the question, correct the
    assumption first. "Accuracy did not decrease — it rose from 0.71 to 0.78
    between the baseline and the tuned model; what fell was precision, from 0.66
    to 0.61" is a better answer than answering the question as asked.

## HOW TO ANSWER A "WHY DID X HAPPEN" QUESTION

1.  Locate the change in the record: which two experiments, which plan revision,
    which cleaning or feature step sits between them.
2.  Read the event log around it. Warnings are the highest-signal entries — a
    family that failed to fit, a SHAP fallback, a dropped column — because they
    record exactly where the run degraded.
3.  State the mechanism, then the evidence, then the confidence. Where two
    explanations both fit the record, give both and say what would distinguish
    them.

## HOW TO ANSWER A COUNTERFACTUAL

Questions like "what if we removed this feature", "would a neural network do
better", "what if we had more data" are almost never answerable from the record,
and guessing at them is the single most damaging thing you can do here. The
required shape of the answer:

1.  **What the evidence suggests.** Reason from what was actually measured — the
    feature's attribution share, whether a correlated substitute exists, the
    learning-curve reading, the gap between families already on the leaderboard.
    Be specific about the direction and rough size the evidence points to.
2.  **Why it is not a measured result.** State plainly that the run did not test
    this.
3.  **What would answer it.** Name the concrete experiment: "refit the winning
    LightGBM configuration with `income` dropped and compare the 5-fold ROC-AUC;
    the correlated `credit_band` at 0.11 attribution would likely absorb part of
    the loss". That is a runnable instruction, which is what makes the answer
    useful.

Never present a counterfactual as though it had been run. Never fabricate a
number for it.

## EVIDENCE, CONFIDENCE, AND CAVEATS

`evidence` holds concrete references, one per entry, each naming the source: an
experiment and its metric value, a recorded rationale, an event sequence number,
a verdict field. "The leaderboard shows random_forest at roc_auc=0.871 versus the
dummy baseline at 0.500" is evidence. "The model performed well" is not.

`confidence` is `high` only when the record directly answers the question; `medium`
when the answer requires inference across several records; `low` for anything
counterfactual or resting on a degradation the log flagged.

`caveats` states the limits of the answer: what the record does not cover, which
measurements were absent, whether a warning in the log undermines the evidence,
and — for any counterfactual — that the result is unmeasured.

`suggested_followups` should be questions this record can actually answer, or
experiments that would answer this one.

## STYLE

Answer in prose, directly, starting with the answer rather than a restatement of
the question. Cite numbers inline with their source. Be brief when the record is
clear; be explicit about the gap when it is not. Do not pad with methodology the
colleague did not ask about."""

    def build_prompt(self, state: RunState) -> str:
        """The user turn: the whole recorded history, then the question."""
        question = self.question(state)
        sections = [
            "# THE RUN RECORD",
            "",
            "## Framing",
            support.run_facts_text(state),
            f"run id: {state.run_id} | status: {state.status.value}",
            "",
            "## Plan and revisions",
            support.plan_text(state.plan, state.plan_history),
            "",
            "## Cleaning decisions (with the rationale recorded at the time)",
            support.cleaning_text(state.cleaning),
            f"applied by the executor: {state.applied_cleaning or 'none recorded'}",
            f"columns dropped: {state.dropped_columns or 'none'}",
            "",
            "## Feature engineering decisions (with the rationale recorded at the time)",
            support.features_text(state.features),
            f"applied by the executor: {state.applied_features or 'none recorded'}",
            f"final feature names ({len(state.feature_names)}): "
            f"{state.feature_names[:60]}"
            f"{'...' if len(state.feature_names) > 60 else ''}",
            "",
            "## Data splitting",
            support.splits_text(state),
            "",
            "## Full experiment leaderboard",
            support.leaderboard_text(state.experiments, limit=30),
            "",
            "## Winner versus baseline",
            support.winner_vs_baseline_text(state),
            "",
            "## Hyperparameter tuning",
            support.tuning_text(state.tuning_decision, state.tuning),
            "",
            "## Feature attributions",
            support.attributions_text(state.explainability, limit=25),
            "",
            "## Evaluation verdict",
            support.evaluation_text(state.evaluation),
            "",
            "## Target distribution",
            support.target_text(state.profile),
        ]

        if state.insights:
            sections += [
                "",
                "## Business insights recorded",
                support.insights_text(state.insights),
            ]

        bundles = support.executor_bundles_text(state, char_budget=3500)
        if bundles:
            sections += ["", "## Measured diagnostics", bundles]

        sections += [
            "",
            "## Warnings recorded during the run",
            support.warnings_text(state, limit=25),
            "",
            "## Event log",
            support.events_text(state.bus.events, limit=80),
            "",
            "# THE QUESTION",
            "",
            question,
            "",
            "Answer it from the record above. If the record does not contain the "
            "answer, say what it does contain, what is missing, and what would have "
            "to be run to answer it.",
        ]
        return "\n".join(sections)

    # -- input / output ----------------------------------------------------

    @staticmethod
    def question(state: RunState) -> str:
        """The question to answer, read from ``state.extras['question']``."""
        raw = state.extras.get(QUESTION_KEY)
        text = str(raw).strip() if raw is not None else ""
        return text or _NO_QUESTION

    def postprocess(self, value: QuestionAnswer, state: RunState) -> QuestionAnswer:
        """Pin the echoed question and flag an answer with no evidence.

        The model sometimes rewrites the question into its own words; the stored
        record should carry what the user actually asked, so the field is reset
        from state.

        Args:
            value: The answer as returned by the model.
            state: The run blackboard.

        Returns:
            The repaired answer.
        """
        asked = self.question(state)
        if asked != _NO_QUESTION:
            value.question = asked

        if not value.evidence:
            note = (
                "No concrete references were attached to this answer, so it cannot "
                "be traced back to the run record; treat it as unverified."
            )
            value.caveats.append(note)
            if value.confidence == "high":
                value.confidence = "medium"
            state.add_warning(f"{self.title}: {note}")
        return value

    def apply(self, state: RunState, value: QuestionAnswer) -> None:
        """Store the answer on ``state.extras['answer']``."""
        state.extras[ANSWER_KEY] = value

    def decision_summary(self, value: QuestionAnswer) -> str:
        """One line for the event stream."""
        return (
            f"answered '{support.clip(value.question, 80)}' "
            f"(confidence={value.confidence}, {len(value.evidence)} evidence "
            f"reference(s), {len(value.caveats)} caveat(s))"
        )


__all__ = ["ANSWER_KEY", "QUESTION_KEY", "QAAgent"]
