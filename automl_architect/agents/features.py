"""Feature Engineering Agent.

A feature without a stated mechanism is a lottery ticket. The schema reflects
that: ``FeatureDecision`` carries both a ``hypothesis`` (why this quantity
should relate to the target) and a ``risk`` (how it could leak or overfit), and
the prompt below is written to make those fields carry real content rather than
restating the operation's name.

Two grounding jobs happen in :meth:`FeatureAgent.postprocess`. First, unknown
columns are filtered out — a cleaning step may have already dropped a column the
agent is still reasoning about. Second, time-dependent operations (lag, rolling,
diff, expanding) are removed when the data has no temporal order at all: those
ops read from other rows, and without a time-ordered split they silently import
future information into the training set. That is the single most common way a
tabular pipeline reports a score it cannot reproduce in production.
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.schemas import (
    AgentName,
    ColumnKind,
    ColumnProfile,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    TaskType,
)
from ..core.state import RunState

#: Operations that read values from *other rows*. Meaningless — and leaky —
#: unless the rows have a genuine order and the split respects it.
_TEMPORAL_OPS = frozenset(
    {FeatureOp.LAG, FeatureOp.ROLLING, FeatureOp.DIFF, FeatureOp.EXPANDING}
)

#: Operations that combine columns and are meaningless with a single input. A
#: ratio can lose its denominator to column filtering, which would otherwise
#: reach the executor as an undefined operation.
_MIN_INPUT_COLUMNS = {
    FeatureOp.RATIO: 2,
    FeatureOp.INTERACTION: 2,
    FeatureOp.GEO_DISTANCE: 2,
}

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _num(value: float | int | None, digits: int = 4) -> str:
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


def _numeric_line(col: ColumnProfile) -> str:
    bits = [
        f"- `{col.name}`",
        f"min={_num(col.minimum)} max={_num(col.maximum)}",
        f"mean={_num(col.mean)} std={_num(col.std)}",
        f"skew={_num(col.skewness, 3)}",
    ]
    # The minimum is what decides legality of log/sqrt/boxcox, so state the
    # conclusion rather than making the model re-derive it.
    if col.minimum is not None:
        if col.minimum > 0:
            bits.append("all_positive=true (log/sqrt/boxcox all legal)")
        elif col.minimum == 0:
            bits.append("has_zeros=true (log1p legal, boxcox is not)")
        else:
            bits.append(
                f"negatives={_pct(col.negative_fraction)} (log/sqrt/boxcox illegal)"
            )
    if col.n_unique <= 20:
        bits.append(f"only {col.n_unique} distinct values (behaves categorically)")
    if col.missing_fraction:
        bits.append(f"missing={_pct(col.missing_fraction)}")
    return " | ".join(bits)


def _categorical_line(col: ColumnProfile, n_rows: int) -> str:
    bits = [
        f"- `{col.name}` [{col.kind.value}]",
        f"cardinality={col.n_unique:,}",
        f"card_ratio={_num(col.cardinality_ratio, 3)}",
    ]
    if col.n_unique and n_rows:
        bits.append(f"~{n_rows / max(col.n_unique, 1):.0f} rows per level")
    if col.top_values:
        rendered = ", ".join(
            f"{v.value!r}:{_pct(v.fraction)}" for v in col.top_values[:5]
        )
        bits.append(f"top=[{rendered}]")
    if col.missing_fraction:
        bits.append(f"missing={_pct(col.missing_fraction)}")
    return " | ".join(bits)


def _datetime_line(col: ColumnProfile) -> str:
    return (
        f"- `{col.name}`: {col.min_timestamp} .. {col.max_timestamp} | "
        f"freq={col.inferred_frequency or 'unknown'} | gaps={col.n_gaps} | "
        f"monotonic={str(col.is_monotonic).lower()} | unique={col.n_unique:,}"
    )


def _text_line(col: ColumnProfile) -> str:
    return (
        f"- `{col.name}`: mean_chars={_num(col.mean_string_length, 3)} "
        f"max_chars={col.max_string_length} "
        f"mean_tokens={_num(col.mean_token_count, 3)} unique={col.n_unique:,}"
    )


def _temporal_columns(state: RunState) -> list[str]:
    """Every column that could establish row order, problem definition first."""
    names: list[str] = []
    if state.problem and state.problem.temporal_column:
        names.append(state.problem.temporal_column)
    if state.profile:
        for name in state.profile.temporal_columns:
            if name not in names:
                names.append(name)
        for col in state.profile.columns:
            if col.kind is ColumnKind.DATETIME and col.name not in names:
                names.append(col.name)
    return names


def _is_time_ordered(state: RunState) -> bool:
    """Whether row-relative features are defensible at all on this dataset."""
    if state.task_type is TaskType.TIME_SERIES_FORECASTING:
        return True
    return bool(_temporal_columns(state))


class FeatureAgent(BaseAgent[FeaturePlan]):
    """Proposes engineered features, each with a mechanism and a named risk."""

    name = AgentName.FEATURES
    title = "Feature Engineering Agent"
    output_model = FeaturePlan
    effort = "high"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The feature-selection method, adapted to this run's task and ordering."""
        task = state.task_type
        time_ordered = _is_time_ordered(state)

        situational: list[str] = []
        if time_ordered:
            temporal = _temporal_columns(state)
            situational.append(
                f"This dataset has temporal structure ({', '.join(f'`{c}`' for c in temporal[:4])}). "
                "Lag, rolling, diff, and expanding features are therefore on the table "
                "— but each one must state in its `risk` field that it is only valid "
                "under a time-ordered split, because computing them across a random "
                "split hands the training set information from the future."
            )
        else:
            situational.append(
                "No temporal column exists and this is not a forecasting problem. Lag, "
                "rolling, diff, and expanding features have no defined meaning here — "
                "row order is arbitrary — and they will be discarded if you propose "
                "them. Do not."
            )
        if task and task.is_classification and state.profile and state.profile.target:
            summary = state.profile.target
            if summary.is_imbalanced:
                situational.append(
                    f"The target is imbalanced (ratio {_num(summary.imbalance_ratio, 3)}). "
                    "Target encoding on a rare positive class is especially unstable: "
                    "a level with a handful of rows gets an encoded value that is "
                    "nearly the label itself. Require smoothing and out-of-fold "
                    "fitting, or prefer frequency encoding."
                )
        if task is TaskType.TIME_SERIES_FORECASTING:
            situational.append(
                "This is a forecasting problem, so lags and rolling windows are the "
                "core feature set rather than an embellishment. Say which horizon each "
                "lag serves, and never propose a rolling window whose span reaches "
                "past the prediction point."
            )
        if state.profile and state.profile.n_rows < 2_000:
            situational.append(
                f"With only {state.profile.n_rows:,} rows, every feature you add eats "
                "degrees of freedom. Feature count is a cost here, not an achievement: "
                "a wide, thin matrix overfits before it generalises."
            )

        situational_block = "\n\n".join(f"- {item}" for item in situational)

        return f"""\
You are a feature engineer who has watched more models die of leakage than of
underfitting. Every feature you propose must come with a *mechanism*: a sentence
about the world that explains why this quantity should carry information about
the target. "It might help" is not a hypothesis. Your output is a small set of
argued features, not a catalogue of transforms.

## How each operation earns its place

**Datetime.** `date_decompose` when a timestamp exists and calendar effects are
plausible for this domain — day-of-week for retail traffic, month for seasonal
demand, hour for anything human-driven. `cyclical_encode` (sin/cos) on top when
the field genuinely wraps: hour 23 sits next to hour 0, December next to January,
and an ordinal encoding tells the model the opposite. Decompose only the
granularities the measured time range can support: extracting month from a
three-week window produces a near-constant.

**Row-relative (lag, rolling, diff, expanding).** Only for forecasting or data
that is genuinely time-ordered. These read values from neighbouring rows, so
under a random split they carry the future into the training set and inflate
validation scores that production will never reproduce. If you propose one, its
`risk` field must name that condition explicitly.

**Categorical encoding, decided by cardinality.** `one_hot_encode` for low
cardinality (roughly under 15 levels, more if rows are plentiful) — it is exact,
interpretable, and costs one column per level. `frequency_encode` is the safe
default above that: it uses no target information at all, so it cannot leak, and
it captures the real signal that rare levels behave differently from common ones.
`target_encode` is the highest-signal option for high cardinality **and the
easiest way to destroy a model**: fitted on the full training set it memorises
the label per level, and validation looks excellent until the day it is deployed.
If you propose it, the `risk` field must state that it requires out-of-fold
fitting with smoothing, and `input_columns` must list only the categorical
column(s) being encoded — the target is implied by the operation and must never
appear as an input. `ordinal_encode` only for genuinely ordered levels (small <
medium < large); imposing an order on nominal categories invents a false
distance. `hash_encode` when cardinality is extreme and you accept the loss of
per-level interpretability.

**Numeric shape.** `log_transform` for right-skewed, non-negative quantities
(counts, money, durations) where a multiplicative effect is more plausible than
an additive one; check the minimum first, because log needs positivity and
`boxcox` needs strict positivity. `sqrt_transform` for milder skew and for count
data. `quantile_transform` when the shape is pathological and you care more about
rank than value. Skip all of these for tree ensembles, which are invariant to
monotone transforms — say so if the downstream models are trees, because then the
transform buys nothing and costs interpretability.

**Interactions, ratios, polynomials.** Only where a domain mechanism makes the
combination meaningful. A ratio like balance-to-limit, price-per-square-metre, or
spend-per-tenure-month is a real quantity a human would compute; the product of
two arbitrary columns is noise with extra variance, and polynomial expansion on a
wide frame is a combinatorial overfitting engine. State the mechanism or do not
propose it. Guard ratios whose denominator can be zero and say so in `risk`.

**Text.** `text_tfidf` only for genuine free text (multi-token, high uniqueness);
on a short categorical string it is a worse one-hot. `text_length` is a cheap,
robust proxy that is often most of the signal.

**Scaling.** `standard_scale` / `minmax_scale` / `robust_scale` matter for
distance- and margin-based learners (KNN, SVM) and for regularised linear models,
and are irrelevant to trees. Prefer `robust_scale` when heavy tails are present.
Fitting happens on the training partition only; the executor handles that, so
your job is choosing which, not preventing the leak.

**Dimensionality.** `pca` / `svd` only when width is actually a problem — p
approaching n, or genuine multicollinearity you have been shown. Say the
explainability cost out loud: after PCA the report can no longer tell a business
reader which real-world quantity drives the prediction, which for many
stakeholders is worse than a slightly lower score. `drop_correlated` and
`variance_threshold` are the cheaper, interpretable alternatives and should be
your first reach. `select_k_best` must be fitted inside the cross-validation
fold; selecting features on the full dataset leaks.

**Aggregations.** `aggregate_by_group` is powerful and dangerous: an aggregate
computed over all rows of an entity includes the row being predicted. Say how it
is restricted (train-only, or excluding the current row) in `risk`.

## Naming, when one operation feeds another

Your plan is applied in dependency order, so a later operation may consume a
column an earlier one creates — scaling a log-transformed income, for instance.
For that reference to resolve, you have to name the column the way it will
actually exist:

-   **Set `output_name_hint` on the producing decision, and reuse that exact
    string** in the consuming decision's `input_columns`. This is the reliable
    route, and the only one fully under your control.
-   If you leave `output_name_hint` empty, the executor derives a default, and
    it always keeps the full source column name as the prefix:
    `date_decompose` on `signup_date` yields `signup_date_year`,
    `signup_date_month`, `signup_date_dayofweek` (**not** `signup_year`);
    `log_transform` on `annual_income` yields `annual_income_log1p`;
    `ratio` of `a` over `b` yields `a_per_b`; `lag` yields `<column>_lag<N>`.
-   A reference that matches nothing when its turn comes is dropped with a
    warning and the operation proceeds with whatever remains. Nothing breaks,
    but the feature you intended silently does not exist — so if a scaling or
    selection step is meant to cover an engineered column, name it exactly.

## Discipline

-   Ten well-argued features beat sixty speculative ones. The feature count is
    not the score. A long list signals that you were generating, not choosing.
-   Order by `priority`. High priority means you would keep it if you could keep
    only three.
-   Every decision needs a populated `risk`. If a feature truly carries no
    leakage or overfitting risk, say what makes it safe — "computed within-row
    from a single column, so it cannot see other rows or the label" is a real
    answer. An empty `risk` reads as an unexamined feature.
-   Use `dimensionality_strategy` and `selection_strategy` to state a stance on
    the width of the resulting matrix and how features will be pruned. These are
    not optional decoration; downstream agents read them.
-   Do not repeat cleaning work: imputation, deduplication, and outlier handling
    already happened. Do not propose features on columns the cleaning plan
    dropped.

## Rationale and hypothesis standard

Good — hypothesis: "support tickets are logged more often in the weeks before a
cancellation, so a 30-day rolling ticket count should rise ahead of churn."
rationale: "`ticket_count` correlates 0.31 with the target at row level; a
rolling window should sharpen that by capturing acceleration rather than level."
risk: "reads from prior rows, so it is only valid under the expanding-window
split; a random split would leak future tickets into training."

Bad — hypothesis: "rolling features capture temporal patterns." rationale:
"rolling means are a standard time-series feature." That names no column, no
measured value, and no mechanism; it would be identical on any dataset, and it
cannot be checked by a reviewer or falsified by a result.
{situational_block}
"""

    def build_prompt(self, state: RunState) -> str:
        profile = state.profile
        lines: list[str] = []
        add = lines.append

        add("# FEATURE ENGINEERING BRIEF")
        add("")
        add(
            "The measured dataset facts are already in your context. Below is what is "
            "new: the problem framing, what cleaning already changed, and the column "
            "inventory grouped by the decision each type calls for."
        )
        add("")

        add("## Problem framing")
        problem = state.problem
        if problem:
            add(f"- task: {problem.task_type.value}")
            add(f"- target: `{problem.target_column or 'none (unsupervised)'}`")
            add(f"- primary metric: {problem.primary_metric} ({problem.metric_rationale})")
            if problem.temporal_column:
                add(f"- temporal column: `{problem.temporal_column}`")
            if problem.group_column:
                add(f"- group / entity key: `{problem.group_column}`")
            if problem.horizon:
                add(f"- forecast horizon: {problem.horizon} period(s)")
            add(f"- business objective: {problem.business_objective}")
            if problem.constraints:
                add(f"- constraints: {'; '.join(problem.constraints)}")
        else:
            add(
                f"- task: {state.task_type.value if state.task_type else 'not yet determined'}"
            )
            add(f"- target: `{state.target or 'unknown'}`")
        add(
            f"- row-relative features (lag/rolling/diff/expanding) permitted: "
            f"{str(_is_time_ordered(state)).lower()}"
        )
        add("")

        if state.cleaning:
            add("## What cleaning already did")
            add(f"- summary: {state.cleaning.summary}")
            if state.cleaning.columns_to_drop:
                add(
                    f"- columns dropped (do not reference these): "
                    f"{state.cleaning.columns_to_drop}"
                )
            imputed = sorted(
                {
                    col
                    for d in state.cleaning.decisions
                    if d.action.value == "impute_missing"
                    for col in d.columns
                }
            )
            if imputed:
                add(
                    f"- columns whose missing values were imputed (their distributions "
                    f"are now slightly narrower than the profile shows): {imputed}"
                )
            add("")
        elif state.dropped_columns:
            add(f"## Columns already dropped\n{state.dropped_columns}")
            add("")

        if profile is None:
            add("## Your task")
            add(
                "No dataset profile is available, so you have no measured evidence. "
                "Return a minimal, conservative plan and say plainly in the summary "
                "that feature decisions could not be grounded."
            )
            return "\n".join(lines)

        available = self.known_columns(state)
        target = state.target
        usable = [
            c
            for c in profile.columns
            if (not available or c.name in available) and c.name != target
        ]

        numeric = [
            c
            for c in usable
            if c.kind
            in {ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE}
        ]
        categorical = [
            c
            for c in usable
            if c.kind
            in {
                ColumnKind.CATEGORICAL_NOMINAL,
                ColumnKind.CATEGORICAL_ORDINAL,
                ColumnKind.BOOLEAN,
            }
        ]
        datetimes = [c for c in usable if c.kind is ColumnKind.DATETIME]
        texts = [c for c in usable if c.kind is ColumnKind.TEXT or c.looks_like_text]
        geos = [c for c in usable if c.kind is ColumnKind.GEO or c.looks_like_geo]

        add("## Feature budget")
        add(f"- rows available: {profile.n_rows:,}")
        add(f"- candidate feature columns: {len(usable)}")
        if usable:
            add(
                f"- rows per candidate column: "
                f"{profile.n_rows / max(len(usable), 1):.0f} "
                "(under ~10 after expansion is overfitting territory)"
            )
        one_hot_cost = sum(c.n_unique for c in categorical if c.n_unique <= 30)
        if one_hot_cost:
            add(
                f"- one-hot encoding every low-cardinality categorical would add "
                f"roughly {one_hot_cost} columns"
            )
        add("")

        add(f"## Numeric columns ({len(numeric)}) — shape transforms, ratios")
        if numeric:
            for col in numeric[:45]:
                add(_numeric_line(col))
            if len(numeric) > 45:
                add(f"- ... and {len(numeric) - 45} more")
        else:
            add("None.")
        add("")

        add(f"## Categorical columns ({len(categorical)}) — encoding decided by cardinality")
        if categorical:
            for col in sorted(categorical, key=lambda c: c.n_unique, reverse=True)[:45]:
                add(_categorical_line(col, profile.n_rows))
            if len(categorical) > 45:
                add(f"- ... and {len(categorical) - 45} more")
        else:
            add("None.")
        add("")

        add(f"## Datetime columns ({len(datetimes)}) — decomposition, cyclical encoding")
        if datetimes:
            for col in datetimes:
                add(_datetime_line(col))
        else:
            add("None detected.")
        add("")

        if texts:
            add(f"## Free-text columns ({len(texts)}) — tfidf vs length")
            for col in texts:
                add(_text_line(col))
            add("")

        if geos:
            add(f"## Geographic columns ({len(geos)})")
            for col in geos:
                add(f"- `{col.name}` (semantic={col.detected_semantic_type or 'geo-like'})")
            add("")

        if profile.target_correlations:
            add("## Existing signal: correlation with the target")
            add(
                "A feature that improves on one of these has to beat the raw column it "
                "is derived from, so start where signal already exists."
            )
            for pair in profile.target_correlations[:15]:
                add(
                    f"- `{pair.left}` vs `{pair.right}`: {_num(pair.coefficient, 4)} "
                    f"({pair.method})"
                )
            add("")

        if profile.highly_correlated_pairs:
            add("## Redundancy: highly correlated feature pairs")
            add(
                "These argue for `drop_correlated` or a ratio that captures the "
                "relationship in one column, not for adding both."
            )
            for pair in profile.highly_correlated_pairs[:15]:
                add(
                    f"- `{pair.left}` <-> `{pair.right}`: {_num(pair.coefficient, 4)} "
                    f"({pair.method})"
                )
            add("")

        if state.model_selection and state.model_selection.candidates:
            families = ", ".join(
                c.family.value for c in state.model_selection.candidates[:6]
            )
            add("## Downstream models already chosen")
            add(
                f"- {families}. Match the features to them: monotone transforms and "
                "scaling are wasted on tree ensembles, while linear, SVM, and KNN "
                "models need scaling and explicit interactions."
            )
            add("")

        if state.memory and state.memory.recommended_feature_ops:
            ops = ", ".join(op.value for op in state.memory.recommended_feature_ops)
            add("## Precedent from similar past runs")
            add(f"- operations that helped before: {ops}")
            add(
                "- treat this as a prior, not an instruction; the evidence in front of "
                "you outranks it."
            )
            add("")

        if state.plan:
            feature_steps = [
                s for s in state.plan.ordered() if s.agent is AgentName.FEATURES
            ]
            if feature_steps:
                add("## What the plan asked of you")
                for step in feature_steps:
                    add(f"- {step.title}: {step.objective} (because: {step.rationale})")
                add("")

        add("## Your task")
        add(
            "Propose the feature set for this dataset. Each decision needs a mechanism "
            "in `hypothesis`, measured evidence in `rationale`, and its leakage or "
            "overfitting exposure in `risk`. Order by `priority`. State a stance in "
            "`dimensionality_strategy` and `selection_strategy`. Choose few features "
            "you can defend over many you cannot — and if the raw columns are already "
            "adequate, say that instead of manufacturing work."
        )
        return "\n".join(lines)

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: FeaturePlan, state: RunState) -> FeaturePlan:
        """Strip target inputs and gate temporal ops.

        Row-relative operations are removed when nothing establishes row order,
        because their computed values would be arbitrary and their validation
        scores unreproducible.

        Column existence is deliberately *not* enforced here — see the note in
        the loop below. A plan's later ops routinely consume columns its earlier
        ops create, so the only layer that can judge a column reference is the
        executor, once the creating ops have run.
        """
        time_ordered = _is_time_ordered(state)
        target = state.target
        kept: list[FeatureDecision] = []
        # Names this plan's own decisions will create. Consulted, not enforced —
        # the agent may also correctly predict an executor-derived default name
        # it never declared as a hint.
        produced: set[str] = {
            d.output_name_hint for d in value.decisions if d.output_name_hint
        }

        for decision in value.decisions:
            if decision.op in _TEMPORAL_OPS and not time_ordered:
                state.add_warning(
                    f"{self.title}: dropped a {decision.op.value} feature on "
                    f"{list(decision.input_columns) or ['<frame>']} — the task is not "
                    "temporal and no datetime column exists, so row-relative features "
                    "have no defined order and would leak under a random split."
                )
                continue

            columns = list(decision.input_columns)
            updates: dict[str, object] = {}

            # A lag/rolling/diff of the forecast series is the *point* of a
            # forecasting feature set, not a leak: the value comes from an
            # earlier timestamp, and ``execution.feature_ops`` treats an empty
            # input list for these ops as "autoregressive on the target" for
            # exactly this reason. Stripping it would silently delete the most
            # valuable feature in the plan.
            autoregressive = (
                decision.op in _TEMPORAL_OPS
                and state.task_type is TaskType.TIME_SERIES_FORECASTING
            )

            if target and target in columns and not autoregressive:
                if decision.op is FeatureOp.TARGET_ENCODE:
                    state.add_warning(
                        f"{self.title}: dropped a target_encode decision that listed "
                        f"the target `{target}` among its input columns; the target is "
                        "implied by the operation and listing it makes the intended "
                        "input ambiguous."
                    )
                    continue
                columns = [c for c in columns if c != target]
                state.add_warning(
                    f"{self.title}: removed the target `{target}` from the inputs of a "
                    f"{decision.op.value} feature — a feature built from the label "
                    "leaks it directly."
                )

            # Deliberately NOT filtered against the source schema here.
            #
            # Feature engineering is sequential: `standard_scale` legitimately
            # consumes `annual_income_log1p`, which a `log_transform` earlier in
            # the same plan creates. At postprocess time that column does not
            # exist yet, so a schema check at this layer would strip the very
            # references that make a multi-stage plan work — and it did, until
            # a live run surfaced it.
            #
            # `execution.feature_ops._resolve_columns` performs the same check
            # against the real frame *during* execution, after `_OP_ORDER` has
            # run the creating ops. It has the information this layer cannot
            # have, and it already warns and skips. So the job here is to
            # observe, not to prune.
            unresolved = [
                c
                for c in columns
                if c not in self.known_columns(state) and c not in produced
            ]
            if unresolved:
                state.bus.log(
                    f"{self.title}: {decision.op.value} references "
                    f"{unresolved[:6]} which no earlier decision declares; the "
                    "executor will resolve them against the real frame and skip "
                    "any that never materialise.",
                    agent=self.name,
                )

            required = _MIN_INPUT_COLUMNS.get(decision.op, 0)
            if required and len(columns) < required:
                state.add_warning(
                    f"{self.title}: dropped a {decision.op.value} feature — it needs "
                    f"at least {required} input columns and only {len(columns)} "
                    f"were supplied ({columns})."
                )
                continue

            if decision.output_name_hint:
                produced.add(decision.output_name_hint)

            if columns != list(decision.input_columns):
                updates["input_columns"] = columns
            if updates:
                decision = decision.model_copy(update=updates)
            kept.append(decision)

        # Stable sort keeps the agent's own ordering inside each priority band.
        kept.sort(key=lambda d: _PRIORITY_ORDER.get(d.priority, 1))

        return value.model_copy(update={"decisions": kept})

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: FeaturePlan) -> None:
        """Record the plan. Materialising the features is the executor's job."""
        state.features = value

    def decision_summary(self, value: FeaturePlan) -> str:
        ops = ", ".join(sorted({d.op.value for d in value.decisions})) or "none"
        high = sum(1 for d in value.decisions if d.priority == "high")
        return (
            f"{len(value.decisions)} feature decision(s) ({high} high priority) "
            f"[{ops}]; expected feature delta {value.expected_feature_count_delta:+d}"
        )
