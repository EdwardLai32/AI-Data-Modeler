"""Dataset Understanding Agent.

The first reasoning step of a run, and the one every later agent inherits. Its
output is not a statistics dump — the measured statistics are already in the
cached run context, and repeating them adds nothing. What is missing from the
profile, and what only a reasoner can supply, is *interpretation*: what this
table is, what one row means, which of the measured relationships are real,
which columns are hazards, and whether the data is fit to model at all.

Grounding here is cheap and pays for itself downstream: a hallucinated column
name in ``suggested_target_columns`` would otherwise become a ``KeyError`` three
steps later inside pandas, so :meth:`DatasetAgent.postprocess` filters every
column reference against the real schema before the value reaches the state.
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import AgentName, ColumnRole, DatasetUnderstanding
from ..core.state import RunState

#: Above this width, demanding one assessment per column costs more output
#: tokens than the marginal insight is worth, so coverage becomes prioritised.
WIDE_TABLE_COLUMNS = 80

INSTRUCTIONS = """\
You are the exploratory data analysis lead on this project: a senior data \
scientist who has opened a few thousand unfamiliar tables and knows that the \
first hour decides whether the next week is wasted. The deterministic profiler \
has already measured everything measurable. Your job is the part it cannot do — \
work out what this data *is*, and where it will hurt you.

## Your method, in this order

1.  **Establish the grain.** What does one row represent? One customer, one \
customer-month, one transaction, one sensor reading? Read it off the evidence: \
which column combination is unique, whether an entity id repeats, whether a \
timestamp advances per row. The grain determines whether aggregation, lag \
features, and grouped splitting are even meaningful, so get it right before \
anything else. If the grain is ambiguous or looks mixed, say so plainly — that \
is a finding, not a failure.
2.  **Name the domain and the likely decision.** Column names, category values, \
and units usually give away the business context. State what the dataset appears \
to be and what a model built on it would be used for.
3.  **Triage every column.** For each one, assign a role, judge its predictive \
potential, and record the concerns that a reviewer would want flagged. Base both \
on the measurements: cardinality, missingness, variance, correlation with the \
target, semantic flags.
4.  **Separate real relationships from artefacts.** A high correlation is not \
automatically signal. Interrogate the strong ones: is the pair a near-duplicate \
encoding of the same quantity, is one derived from the other, is the association \
carried by a handful of rows, is it an ID-like column that happens to be \
monotonic with time? Say which measured relationships you believe and which you \
distrust, with your reason for each.
5.  **Rank the target candidates.** If the operator named a target, that is the \
target — put it first and assess it. If not, rank the plausible outcome columns \
best-first: a column that represents an outcome rather than an attribute, that \
is populated, that is not a restatement of another column, and that a business \
would actually want predicted. Explain the ranking in your narrative.
6.  **Judge readiness from the measurements.** ``data_readiness`` is a \
conclusion drawn from measured missingness, duplication, leakage, and dtype \
quality — not an overall mood about the dataset.

## Assigning a column role

-   ``identifier`` — near-unique keys, hashes, row numbers, surrogate ids. \
High cardinality ratio plus an ID-like name or flag. These are never features; \
a tree will happily memorise them and the model will not generalise.
-   ``temporal_index`` — the timestamp that orders the data. Prefer the one that \
is monotonic or nearly so and has a regular inferred frequency.
-   ``group_key`` — an entity key that repeats across rows (customer id in a \
panel, store id in a demand table). Rows sharing this key must not straddle a \
train/test boundary.
-   ``weight`` — an explicit sample or exposure weight, if one exists.
-   ``leakage_suspect`` — a column that could only be known at or after the \
moment the outcome is known: a resolution date, a churn reason, a settled \
amount, a post-hoc status flag, a near-perfect association with the target. \
Judge each on the timeline, not just the correlation.
-   ``ignored`` — constants, near-zero variance, free-form notes with no reuse, \
or columns whose missingness is so high that nothing survives imputation.
-   ``feature`` / ``target`` — everything else.

## Predictive potential

Rate ``high`` only when there is measured support: a meaningful correlation with \
the target, a class-discriminating distribution, or a well-understood causal \
mechanism in this domain. Rate ``none`` for constants, identifiers, and columns \
that are entirely missing in practice. Do not distribute ``medium`` as a way of \
avoiding the judgement.

## data_readiness, tied to the numbers

-   ``ready`` — missingness under roughly 1% of cells, no material duplication, \
no leakage candidates above low severity, dtypes already usable.
-   ``needs_cleaning`` — routine work is required and will be sufficient: \
imputation, deduplication, dtype and datetime parsing, category normalisation, \
outlier clipping.
-   ``needs_major_work`` — one or more of: high or critical leakage findings; \
target absent, ambiguous, or substantially missing; important columns above \
roughly 30-40% missing; the grain is inconsistent; duplication is large enough \
to distort every estimate.
-   ``unusable`` — there is no plausible target, or every strong signal is \
leakage, or the row count cannot support any honest validation.

``readiness_rationale`` must cite the specific figures that put you in that \
band. "Roughly 8% of TotalCharges values are missing and 0.3% of rows duplicate" \
is a rationale. "The data is fairly messy" is not.

## Failure modes of this specific job — do not walk into them

-   Treating a high-cardinality identifier as a strong predictor because it \
correlates with the target. It is an index, and the correlation is an artefact.
-   Reading correlation as mechanism. Two columns computed from the same \
quantity will correlate perfectly and teach the model nothing.
-   Missing the leakage that is obvious in hindsight: a column that is only \
populated once the outcome has happened, or whose missingness pattern encodes \
the outcome.
-   Ignoring duplicated rows. They inflate apparent signal and make any holdout \
score optimistic, because the same row can appear on both sides of the split.
-   Assuming missingness is noise. Structurally missing values often *are* the \
signal (no contract end date because the customer has not left), and imputing \
them destroys it. Say when you suspect this.
-   Assessing the columns and forgetting the shape: a table with more columns \
than rows, or a class with a dozen examples, constrains everything that follows.

## Rationale quality

Good: "`tenure` correlates 0.35 with churn and its distribution is bimodal with \
a spike at 1 month, which matches the usual pattern of early-life cancellations; \
this should carry real signal." — names the measurement, adds a mechanism, draws \
a consequence.

Bad: "`tenure` is an important feature that should be included because customer \
tenure is generally predictive of churn." — cites nothing from this dataset and \
would read identically for any table with a tenure column.

## Output discipline

Write the narrative as a colleague would write it in a handover document: \
several paragraphs, concrete, opinionated, ordered from what the data is to what \
will bite you. Do not enumerate statistics back at me; I measured them. Prefer \
five findings that change what we do next over twenty that restate the profile. \
Every column reference must be a real column name, spelled exactly as given.
"""


class DatasetAgent(BaseAgent[DatasetUnderstanding]):
    """Turns measured statistics into an interpreted read on the dataset."""

    name = AgentName.DATASET
    title = "Dataset Understanding Agent"
    output_model = DatasetUnderstanding
    effort: Effort = "high"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: the EDA method and its failure modes."""
        return INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        """The user turn: what is new beyond the cached digest, then the question."""
        profile = state.profile
        n_columns = profile.n_columns if profile else 0
        lines: list[str] = ["## YOUR TASK", ""]

        if state.config.target_column:
            lines.append(
                f"The operator named `{state.config.target_column}` as the prediction "
                "target. Treat that as settled: assess it as the target, rank it first "
                "among the candidates, and focus the leakage question on what could "
                "contaminate it."
            )
        else:
            lines.append(
                "The operator did not name a target column. Ranking the plausible "
                "targets is therefore one of your deliverables, and the next agent "
                "will act on your ranking."
            )
        lines.append("")

        ingestion_notes = self._ingestion_notes(state)
        if ingestion_notes:
            lines.append("### Load-time notes (not in the profile above)")
            lines.extend(ingestion_notes)
            lines.append("")

        lines.append("### Coverage requirement")
        if n_columns and n_columns <= WIDE_TABLE_COLUMNS:
            lines.append(
                f"Assess all {n_columns} columns — one assessment each, no omissions. "
                "A column you skip is a column nobody downstream will question."
            )
        else:
            lines.append(
                f"This table has {n_columns} columns. Assess every column that could "
                "plausibly be a feature, the target, an identifier, a temporal index, "
                "a group key, or a leakage hazard. If you compress a routine group "
                "(for example a block of one-hot indicators), name that group in "
                "`risks` so the omission is on the record rather than accidental."
            )
        lines.append("")

        lines.append("### Deliver")
        lines.extend(
            [
                "1. What this dataset is, and what exactly one row represents.",
                "2. A column-by-column read: role, predictive potential, concerns.",
                "3. Which measured relationships you believe, and which look "
                "spurious or circular — with the reason for each verdict.",
                "4. The columns that are dangerous to model on, and why they are "
                "dangerous rather than merely imperfect.",
                "5. A ranked list of target candidates.",
                "6. A readiness verdict justified by the measured missingness, "
                "duplication, and leakage figures.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _ingestion_notes(state: RunState) -> list[str]:
        result = state.ingestion
        if result is None:
            return []
        notes: list[str] = []
        if result.truncated:
            notes.append(
                f"- The load was capped: {result.n_rows:,} rows were read out of a "
                "larger source, so rare categories and tail behaviour may be "
                "under-represented."
            )
        for message in result.validation_errors[:6]:
            notes.append(f"- Load error: {message}")
        for message in result.validation_warnings[:6]:
            notes.append(f"- Load warning: {message}")
        if result.load_seconds > 30:
            notes.append(
                f"- The source took {result.load_seconds:.0f}s to read, which is "
                "relevant to any retraining cadence we recommend later."
            )
        return notes

    # -- grounding ---------------------------------------------------------

    def postprocess(
        self, value: DatasetUnderstanding, state: RunState
    ) -> DatasetUnderstanding:
        """Drop invented column names and honour an operator-named target."""
        known = self.known_columns(state)

        seen: set[str] = set()
        assessments = []
        for assessment in value.column_assessments:
            if known and assessment.name not in known:
                continue
            if assessment.name in seen:
                continue
            seen.add(assessment.name)
            assessments.append(assessment)
        dropped = len(value.column_assessments) - len(assessments)
        if dropped:
            state.add_warning(
                f"{self.title}: discarded {dropped} column assessment(s) naming "
                "unknown or duplicated columns."
            )
        value.column_assessments = assessments

        candidates = self.keep_known_columns(
            value.suggested_target_columns, state, context="suggested_target_columns"
        )
        ranked: list[str] = []
        for name in candidates:
            if name not in ranked:
                ranked.append(name)

        # The operator's choice outranks the model's opinion, always.
        operator_target = state.config.target_column
        if operator_target and (not known or operator_target in known):
            ranked = [operator_target] + [n for n in ranked if n != operator_target]
        value.suggested_target_columns = ranked

        if known and not ranked and state.profile and state.profile.target is None:
            state.add_warning(
                f"{self.title}: produced no usable target candidates; the Problem "
                "Agent will have to infer the target from the profile alone."
            )

        unassessed = known - seen
        if unassessed and len(known) <= WIDE_TABLE_COLUMNS:
            preview = sorted(unassessed)[:8]
            state.add_warning(
                f"{self.title}: {len(unassessed)} column(s) received no assessment "
                f"{preview}{'...' if len(unassessed) > 8 else ''}."
            )
        return value

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: DatasetUnderstanding) -> None:
        """Publish the understanding onto the blackboard."""
        state.understanding = value

    def decision_summary(self, value: DatasetUnderstanding) -> str:
        """One line for the event stream."""
        suspects = sum(
            1
            for a in value.column_assessments
            if a.role is ColumnRole.LEAKAGE_SUSPECT
        )
        target = value.suggested_target_columns[0] if value.suggested_target_columns else "none"
        return (
            f"{value.data_readiness}: {value.grain} in {value.likely_domain}; "
            f"{len(value.column_assessments)} columns assessed, "
            f"{suspects} leakage suspect(s), top target candidate `{target}`"
        )


__all__ = ["INSTRUCTIONS", "WIDE_TABLE_COLUMNS", "DatasetAgent"]
