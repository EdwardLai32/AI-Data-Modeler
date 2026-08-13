"""Data Cleaning Agent.

The product requirement this module encodes: **no transformation is applied
without a reason tied to measured evidence**. The schema already forces a
``rationale`` string onto every decision; the prompt here is what makes that
string worth reading. It teaches the decision procedure a senior practitioner
actually follows — which statistic decides mean vs median, when missingness is
itself the signal, when an outlier is an error rather than the tail you are
being paid to model — and it shows a good rationale against a bad one so the
contrast is concrete rather than exhorted.

:meth:`CleaningAgent.postprocess` is the safety net. Three failures in
particular must never reach pandas: a column that does not exist, an imputation
aimed at the target (fabricated ground truth), and a "drop" decision that was
not marked destructive and so slipped past the human-approval gate.
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.schemas import (
    AgentName,
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    ColumnProfile,
    DatasetProfile,
    MissingStrategy,
    TaskType,
)
from ..core.state import RunState

# Actions that remove data. The orchestrator gates human approval on
# ``destructive``, so an unmarked drop is a silent bypass of that gate.
_DESTRUCTIVE_ACTIONS = frozenset(
    {
        CleaningAction.DROP_COLUMN,
        CleaningAction.DROP_DUPLICATE_ROWS,
        CleaningAction.DROP_ROWS_MISSING_TARGET,
        CleaningAction.REMOVE_OUTLIER_ROWS,
        CleaningAction.DROP_CONSTANT_COLUMN,
        CleaningAction.DROP_LEAKAGE_COLUMN,
    }
)
_DESTRUCTIVE_STRATEGIES = frozenset(
    {MissingStrategy.DROP_COLUMN, MissingStrategy.DROP_ROWS}
)
_COLUMN_DROPPING_ACTIONS = frozenset(
    {
        CleaningAction.DROP_COLUMN,
        CleaningAction.DROP_CONSTANT_COLUMN,
        CleaningAction.DROP_LEAKAGE_COLUMN,
    }
)


def _num(value: float | int | None, digits: int = 4) -> str:
    """Render a measured number for prompt text, or ``n/a`` when absent."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        if value != value:  # NaN
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.{digits}g}"


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.2f}%"


def _median(col: ColumnProfile) -> float | None:
    return col.quantiles.p50 if col.quantiles else None


def _mean_median_gap_sd(col: ColumnProfile) -> str:
    """How far the mean sits from the median, in standard deviations.

    This is the single most decision-relevant derived fact for imputation: it
    converts "is it skewed" into "would imputing the mean insert a value the
    bulk of rows do not resemble".
    """
    median = _median(col)
    if col.mean is None or median is None or not col.std:
        return "n/a"
    return f"{abs(col.mean - median) / col.std:.2f}"


def _missing_line(col: ColumnProfile) -> str:
    bits = [
        f"- `{col.name}` [{col.kind.value}]",
        f"missing={col.n_missing:,} ({_pct(col.missing_fraction)})",
        f"unique={col.n_unique:,}",
    ]
    if col.mean is not None or col.minimum is not None:
        bits.append(
            f"mean={_num(col.mean)} median={_num(_median(col))} std={_num(col.std)}"
        )
        bits.append(f"skew={_num(col.skewness, 3)}")
        bits.append(f"|mean-median|/std={_mean_median_gap_sd(col)}")
        if col.outliers and col.outliers.n_outliers:
            bits.append(
                f"iqr_outliers={col.outliers.n_outliers:,} ({_pct(col.outliers.fraction)})"
            )
        if col.negative_fraction:
            bits.append(f"negatives={_pct(col.negative_fraction)}")
    if col.top_values:
        top = col.top_values[0]
        bits.append(f"dominant_value={top.value!r} share={_pct(top.fraction)}")
    if col.looks_like_id:
        bits.append("flag=ID_LIKE")
    if col.is_constant:
        bits.append("flag=CONSTANT")
    if col.is_near_zero_variance:
        bits.append("flag=NEAR_ZERO_VARIANCE")
    if col.detected_semantic_type:
        bits.append(f"semantic={col.detected_semantic_type}")
    return " | ".join(bits)


def _outlier_line(col: ColumnProfile) -> str:
    outliers = col.outliers
    if outliers is None:
        return f"- `{col.name}`: no outlier summary computed"
    return (
        f"- `{col.name}`: {outliers.n_outliers:,} outliers ({_pct(outliers.fraction)}) "
        f"by {outliers.method}, bounds=[{_num(outliers.lower_bound)}, "
        f"{_num(outliers.upper_bound)}] | actual range=[{_num(col.minimum)}, "
        f"{_num(col.maximum)}] | skew={_num(col.skewness, 3)} "
        f"kurtosis={_num(col.kurtosis, 3)}"
    )


def _temporal_ordering_note(profile: DatasetProfile, temporal_column: str | None) -> str:
    """Whether forward/backward fill is even defensible on this table."""
    candidates = list(profile.temporal_columns)
    if temporal_column and temporal_column not in candidates:
        candidates.insert(0, temporal_column)
    if not candidates:
        return (
            "No temporal column was detected. Rows have no meaningful order, so "
            "forward_fill / backward_fill / interpolate would copy an arbitrary "
            "neighbour's value and are not available to you."
        )
    lines = ["Temporal columns that could establish row order:"]
    for name in candidates:
        col = profile.column(name)
        if col is None:
            continue
        lines.append(
            f"- `{name}`: range={col.min_timestamp} .. {col.max_timestamp} "
            f"freq={col.inferred_frequency or 'unknown'} gaps={col.n_gaps} "
            f"monotonic={str(col.is_monotonic).lower()}"
        )
    lines.append(
        "Order-dependent fills are only defensible if one of these is genuinely "
        "monotonic (or the frame is sorted by it before filling)."
    )
    return "\n".join(lines)


class CleaningAgent(BaseAgent[CleaningPlan]):
    """Decides which remediations this specific table needs, and argues each one."""

    name = AgentName.CLEANING
    title = "Data Cleaning Agent"
    output_model = CleaningPlan
    effort = "high"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The cleaning decision procedure, adapted to this run's task shape."""
        task = state.task_type
        target = state.target
        profile = state.profile

        situational: list[str] = []
        if target:
            situational.append(
                f"The prediction target is `{target}`. It is never a candidate for "
                "imputation, encoding, clipping, or dropping — it is the ground "
                "truth you are protecting."
            )
        if task and task.is_classification and profile and profile.target:
            summary = profile.target
            if summary.is_imbalanced:
                situational.append(
                    f"The target is imbalanced (majority/minority ratio "
                    f"{_num(summary.imbalance_ratio, 3)}). Any row-removing decision "
                    "risks deleting minority-class rows, which are the scarce and "
                    "expensive ones. Justify row removal against that cost, and never "
                    "remove rows to 'tidy' a distribution."
                )
        if task is TaskType.TIME_SERIES_FORECASTING:
            situational.append(
                "This is a forecasting problem. Rows are ordered observations, so "
                "duplicate 'identical rows' may be legitimate repeated measurements, "
                "and dropping rows punches holes in the series. Prefer filling or "
                "interpolating over deleting, and never reorder the data."
            )
        if profile and profile.duplicate_fraction > 0.5:
            situational.append(
                f"Over half the rows are exact duplicates "
                f"({_pct(profile.duplicate_fraction)}). That is more likely a grain "
                "misunderstanding or a bad join than dirty input — say what you think "
                "happened before recommending a mass deletion."
            )

        situational_block = ""
        if situational:
            situational_block = "\n## Specific to this run\n\n" + "\n\n".join(
                f"- {item}" for item in situational
            )

        return f"""\
You are a senior data engineer who has prepared several hundred production
modelling datasets. Your job is not to run a checklist. It is to decide, column
by column, which remediations *this* table actually needs, and to justify each
one from the statistics that were measured for you.

## Work in this order

1.  **The target first.** Rows whose label is missing are unusable for
    supervised learning: drop them with `drop_rows_missing_target`. Never impute
    a label. An imputed target is fabricated ground truth, and a model trained
    on it learns your imputer rather than the world. If the target's missing
    fraction is large, say so in the summary — it reduces the effective sample
    size, and every downstream agent needs to know.
2.  **Structural defects.** A constant column carries exactly zero information
    and can break variance-based transforms downstream; drop it. An
    identifier-like column (cardinality ratio near 1.0) is memorisable noise as
    a feature — drop it, but note if the id plausibly encodes something real,
    such as a sequential key that proxies for time, because that is a feature
    idea rather than a defect.
3.  **Leakage.** Judge every flagged leakage candidate on *mechanism*, not on
    its score. Ask one question: at the moment a prediction must be made, would
    this value exist yet? A `churn_reason`, a `closed_date`, a `final_amount`
    answers the target because it is recorded after the outcome. Drop those and
    explain the mechanism in the rationale. If a high-scoring column is
    genuinely legitimate — a strong, causally-prior predictor — keep it and
    explain why the association is not leakage. That judgement is the reason you
    are in this pipeline instead of a threshold.
4.  **Duplicates.** Before dropping them, say what a duplicate *means* at this
    grain. If one row is one customer, two identical rows are an integration
    artifact and should go. If one row is one sensor reading or one order-day,
    identical values can be a legitimate repeat, and deleting them distorts the
    distribution you are trying to model. Decide from the grain and the measured
    duplicate fraction.
5.  **Missing values,** column by column, using the distribution (procedure
    below).
6.  **Types and parsing.** Numbers stored as strings, dates stored as text,
    categories differing only by whitespace or case — fix these where the
    evidence shows them. `strip_whitespace` and `normalise_categories` are cheap
    and safe; but do not merge levels that are genuinely distinct just because
    they look similar.
7.  **Outliers last,** and only when they are errors (procedure below).

## Choosing an imputation strategy

The distribution decides, not convention:

-   Numeric, near-symmetric (|skew| below roughly 1, mean close to median):
    `mean`. It is the minimum-variance estimate and sits inside the bulk.
-   Numeric, skewed (|skew| above roughly 1) or with a visible mean-vs-median
    gap: `median`. The mean has been dragged out of the bulk by the tail, so
    imputing it inserts values that almost no real row resembles.
-   Numeric where missingness is plausibly informative — a field only populated
    for one segment, or a strong association between the missingness pattern and
    the target — prefer `leave_as_is` when a NaN-native learner is plausible
    (histogram gradient boosting, XGBoost and LightGBM all split on missing), or
    a `constant` sentinel, and record in the summary that the missingness itself
    is signal. Averaging it away destroys a real predictor.
-   Categorical with one dominant level (top value above roughly 60-70% and few
    levels): `mode` is defensible; you are betting on a strong base rate, so
    say what that base rate is.
-   Categorical spread across levels, or where absence likely means something:
    `missing_category`. Inventing the mode fabricates signal and shifts the
    distribution of a real predictor.
-   Genuinely time-ordered rows with a monotonic timestamp: `forward_fill`, and
    only then. Forward filling an unordered table copies an arbitrary
    neighbour's value.
-   `interpolate` for smooth numeric series indexed by time. `knn` or
    `iterative` only when several correlated columns are jointly missing and the
    row count is comfortably large — both are expensive, and both can smear
    structure if applied without care.
-   Missingness above roughly 40-50%: imputing is no longer estimating, it is
    manufacturing the majority of the column. Prefer `drop_column`, or keep the
    missingness as its own binary indicator when the pattern looks informative.
    Choose one explicitly and say which, and why.

## Outliers

Clip, do not delete, by default. Act only when the values are *implausible*
rather than a genuine heavy tail. A long right tail in income, claim size, or
session duration is the signal you are being paid to model; deleting those rows
throws away exactly the cases that matter and quietly flatters your metrics.
Remove rows only for values that cannot physically exist — a negative age, a
future timestamp, a 400% percentage — and say what makes them impossible.
Prefer `clip_outliers` with bounds drawn from the measured quantiles, and prefer
leaving a real tail alone so feature engineering can apply a log transform
instead.

## What is not your job

Scaling, encoding, binning, and log transforms belong to the feature
engineering agent. Do not propose them here. Splitting belongs to the executor.
You choose *which statistic* an imputer uses; you never state its value, and you
never compute anything — the executor fits every imputer on the training
partition only, so your choice cannot leak by itself.

## skipped_considerations is graded

Populate it. List the remediations you deliberately declined and why: the 3%
missing column you left alone because 3% barely moves an estimate; the heavy
tail you refused to clip because it is real; the duplicate rows you kept because
they are repeated measurements. A plan with an empty `skipped_considerations` is
a plan that did not consider anything, and it will be read that way.

## Rationale standard

Good: "skewness of 3.42 with 4.1% IQR outliers puts the mean (41.2) roughly 0.31
standard deviations above the median (33.0), so the median imputes a value far
more rows actually resemble."

Bad: "median is standard practice for numeric columns." — it cites no evidence
from this dataset, it would read identically for any table on earth, and it
therefore cannot be audited or challenged.

Every rationale must name at least one measured figure from the facts you were
given. Group several columns into one decision only when the *same* evidence
applies to all of them; if the reason differs, the decision differs.

Prefer few, well-argued decisions over an exhaustive sweep. Ten defensible
transformations beat forty reflexive ones, and every unnecessary transformation
is a place where the pipeline can silently distort the data.
{situational_block}
"""

    def build_prompt(self, state: RunState) -> str:
        profile = state.profile
        lines: list[str] = []
        add = lines.append

        add("# CLEANING BRIEF")
        add("")
        add(
            "The measured dataset facts are already in your context. Below is the "
            "cleaning-relevant subset, plus the derived quantities that decide "
            "imputation strategy. Do not restate them; use them."
        )
        add("")

        add("## Problem framing")
        problem = state.problem
        if problem:
            add(f"- task: {problem.task_type.value}")
            add(f"- target: `{problem.target_column or 'none (unsupervised)'}`")
            add(f"- primary metric: {problem.primary_metric}")
            if problem.temporal_column:
                add(f"- temporal column: `{problem.temporal_column}`")
            if problem.group_column:
                add(f"- group / entity key: `{problem.group_column}`")
            add(f"- business objective: {problem.business_objective}")
            if problem.constraints:
                add(f"- constraints: {'; '.join(problem.constraints)}")
        else:
            add(
                f"- task: {state.task_type.value if state.task_type else 'not yet determined'}"
            )
            add(f"- target: `{state.target or 'unknown'}`")
        add("")

        if profile is None:
            add(
                "No dataset profile is available, which means you have no measured "
                "evidence. Return a minimal plan and say plainly in the summary that "
                "cleaning decisions could not be grounded."
            )
            add("")
            add("## Your task")
            add(
                "Produce a cleaning plan whose every decision cites measured evidence, "
                "and populate skipped_considerations."
            )
            return "\n".join(lines)

        add("## Grain and integrity")
        add(f"- rows: {profile.n_rows:,} | columns: {profile.n_columns:,}")
        add(
            f"- exact duplicate rows: {profile.n_duplicate_rows:,} "
            f"({_pct(profile.duplicate_fraction)})"
        )
        add(
            f"- missing cells overall: {profile.total_missing_cells:,} "
            f"({_pct(profile.missing_cell_fraction)})"
        )
        if state.understanding:
            add(f"- grain (one row is): {state.understanding.grain}")
            add(f"- likely domain: {state.understanding.likely_domain}")
        add("")

        if profile.target:
            target_summary = profile.target
            add("## Target integrity")
            add(
                f"- `{target_summary.name}`: {target_summary.n_missing:,} missing "
                f"label(s) out of {profile.n_rows:,} rows"
            )
            if target_summary.imbalance_ratio is not None:
                add(
                    f"- class imbalance (majority/minority): "
                    f"{_num(target_summary.imbalance_ratio, 3)} "
                    f"({'IMBALANCED' if target_summary.is_imbalanced else 'reasonably balanced'})"
                )
            if target_summary.skewness is not None:
                add(f"- target skewness: {_num(target_summary.skewness, 3)}")
            add("")

        missing_cols = sorted(
            (c for c in profile.columns if c.n_missing > 0),
            key=lambda c: c.missing_fraction,
            reverse=True,
        )
        add(f"## Columns with missing values ({len(missing_cols)})")
        if missing_cols:
            add(
                "`|mean-median|/std` is how far the mean sits from the median in "
                "standard deviations — the direct evidence for mean vs median."
            )
            for col in missing_cols[:60]:
                add(_missing_line(col))
            if len(missing_cols) > 60:
                add(f"- ... and {len(missing_cols) - 60} more, all with lower missingness")
        else:
            add("None. No imputation decisions are warranted.")
        add("")

        outlier_cols = sorted(
            (c for c in profile.columns if c.outliers and c.outliers.n_outliers > 0),
            key=lambda c: c.outliers.fraction if c.outliers else 0.0,
            reverse=True,
        )
        add(f"## Columns with IQR outliers ({len(outlier_cols)})")
        if outlier_cols:
            add(
                "Decide error vs genuine tail from the bound-versus-range comparison "
                "and from what the column measures."
            )
            for col in outlier_cols[:25]:
                add(_outlier_line(col))
        else:
            add("None flagged.")
        add("")

        add("## Structural candidates")
        add(f"- constant columns: {profile.constant_columns or 'none'}")
        near_zero = [c.name for c in profile.columns if c.is_near_zero_variance]
        add(f"- near-zero-variance columns: {near_zero or 'none'}")
        add(f"- identifier-like columns: {profile.identifier_columns or 'none'}")
        add("")

        add("## Row ordering")
        add(_temporal_ordering_note(profile, state.problem.temporal_column if state.problem else None))
        add("")

        if profile.leakage_findings:
            add("## Leakage candidates awaiting your judgement")
            for finding in profile.leakage_findings:
                add(
                    f"- `{finding.column}`: score={_num(finding.score, 4)} "
                    f"severity={finding.severity.value} ({finding.method}) — {finding.reason}"
                )
            add(
                "For each: keep or drop, and give the mechanism. A score is not a "
                "rationale."
            )
            add("")

        if state.understanding and state.understanding.risks:
            add("## Risks the dataset review already flagged")
            for risk in state.understanding.risks[:10]:
                add(f"- {risk}")
            add("")

        if state.plan:
            cleaning_steps = [
                s for s in state.plan.ordered() if s.agent is AgentName.CLEANING
            ]
            if cleaning_steps:
                add("## What the plan asked of you")
                for step in cleaning_steps:
                    add(f"- {step.title}: {step.objective} (because: {step.rationale})")
                add("")

        add("## Your task")
        add(
            "Produce the cleaning plan for this table. Work the order given in your "
            "instructions: target, structure, leakage, duplicates, missing values, "
            "types, outliers. Every decision must cite a measured figure from the "
            "facts above. Mark every decision that removes rows or columns as "
            "destructive, keep `columns_to_drop` consistent with those decisions, and "
            "populate `skipped_considerations` with what you deliberately left alone "
            "and why. Fewer, better-argued decisions are the goal."
        )
        return "\n".join(lines)

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: CleaningPlan, state: RunState) -> CleaningPlan:
        """Filter to real columns and make the plan safe to execute.

        Repairs, in order: unknown column references, imputation aimed at the
        target, any decision that would delete the target column, missing
        ``destructive`` flags, and a ``columns_to_drop`` list inconsistent with
        the decisions that actually drop columns.
        """
        target = state.target
        kept: list[CleaningDecision] = []
        seen: set[tuple[str, tuple[str, ...], str]] = set()

        for decision in value.decisions:
            columns = list(decision.columns)
            updates: dict[str, object] = {}

            if columns:
                filtered = self.keep_known_columns(
                    columns, state, context=f"cleaning action {decision.action.value}"
                )
                if not filtered:
                    state.add_warning(
                        f"{self.title}: dropped a {decision.action.value} decision "
                        "because none of its columns exist in the data."
                    )
                    continue

                if target and target in filtered:
                    if decision.action is CleaningAction.IMPUTE_MISSING:
                        filtered = [c for c in filtered if c != target]
                        state.add_warning(
                            f"{self.title}: refused to impute the target column "
                            f"`{target}` — an imputed label is fabricated ground "
                            "truth. Rows with a missing target are dropped instead."
                        )
                    elif decision.action in _COLUMN_DROPPING_ACTIONS:
                        filtered = [c for c in filtered if c != target]
                        state.add_warning(
                            f"{self.title}: refused to drop the target column "
                            f"`{target}` via {decision.action.value}."
                        )
                    if not filtered:
                        continue

                if filtered != columns:
                    updates["columns"] = filtered
                columns = filtered

            destructive = (
                decision.action in _DESTRUCTIVE_ACTIONS
                or decision.strategy in _DESTRUCTIVE_STRATEGIES
            )
            if destructive and not decision.destructive:
                updates["destructive"] = True

            repaired = decision.model_copy(update=updates) if updates else decision

            signature = (
                repaired.action.value,
                tuple(sorted(repaired.columns)),
                repaired.strategy.value if repaired.strategy else "",
            )
            if signature in seen:
                state.add_warning(
                    f"{self.title}: removed a duplicate {repaired.action.value} "
                    f"decision for columns {list(repaired.columns) or ['<table>']}."
                )
                continue
            seen.add(signature)
            kept.append(repaired)

        # columns_to_drop is what the executor and the approval gate read, so it
        # must be the union of every column any decision actually removes.
        implied: list[str] = []
        for decision in kept:
            drops_columns = (
                decision.action in _COLUMN_DROPPING_ACTIONS
                or decision.strategy is MissingStrategy.DROP_COLUMN
            )
            if drops_columns:
                for name in decision.columns:
                    # `strategy=drop_column` on an action that is not itself a
                    # column-dropping action bypasses the target guard above, so
                    # re-assert it here: columns_to_drop is read directly by the
                    # executor and must never name the target.
                    if name != target and name not in implied:
                        implied.append(name)

        declared = self.keep_known_columns(
            list(value.columns_to_drop), state, context="columns_to_drop"
        )
        if target and target in declared:
            declared = [c for c in declared if c != target]
            state.add_warning(
                f"{self.title}: removed the target `{target}` from columns_to_drop."
            )

        unexplained = [c for c in declared if c not in implied]
        if unexplained:
            # The column still gets dropped — the agent asked for it — but an
            # audit trail with a drop and no decision behind it is a gap worth
            # surfacing rather than silently accepting.
            state.add_warning(
                f"{self.title}: columns_to_drop lists {unexplained} with no "
                "corresponding drop decision; the drop will happen but is only "
                "explained by drop_rationale."
            )

        columns_to_drop = list(declared)
        rationale = list(value.drop_rationale)
        for name in implied:
            if name not in columns_to_drop:
                columns_to_drop.append(name)
                source = next(
                    (d for d in kept if name in d.columns), None
                )
                if source is not None:
                    rationale.append(f"`{name}`: {source.rationale}")

        return value.model_copy(
            update={
                "decisions": kept,
                "columns_to_drop": columns_to_drop,
                "drop_rationale": rationale,
            }
        )

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: CleaningPlan) -> None:
        """Record the plan. Applying it to the dataframe is the executor's job."""
        state.cleaning = value

    def decision_summary(self, value: CleaningPlan) -> str:
        actions = ", ".join(
            sorted({d.action.value for d in value.decisions})
        ) or "no transformations"
        return (
            f"{len(value.decisions)} cleaning decision(s) [{actions}]; "
            f"{len(value.columns_to_drop)} column(s) to drop; "
            f"{len(value.skipped_considerations)} consideration(s) deliberately skipped"
        )
