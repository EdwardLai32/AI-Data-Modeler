"""Deterministic execution of a :class:`~automl_architect.core.schemas.FeaturePlan`.

Feature engineering is where a pipeline either earns its score or fakes it, so
this module is organised around leakage discipline rather than around convenience.

Three rules decide where each operation runs:

*   **Target-derived features are computed out-of-fold.** Target encoding fitted
    on the full column writes the answer into the feature, which is the single
    most common cause of a validation score that collapses in production. Here it
    is K-fold out-of-fold with smoothing toward the prior, and when a training
    partition already exists the mapping is fitted on that partition alone.
*   **Anything fitted from the feature distribution is deferred to a
    transformer.** Scaling, binning, power transforms, PCA/SVD, univariate
    selection, and variance filtering all learn statistics. Learning them before
    the split contaminates the test set, so they are assembled into an unfitted
    sklearn pipeline on ``state.preprocessor`` and fitted by the trainer on train
    only.
*   **Time-ordered features must know the order.** Lags, rolling windows, diffs,
    and expanding statistics sort by the temporal column and are bounded by the
    group key; without both, a "lag" is an arbitrary neighbouring row and an
    entity's history bleeds into the next entity's.

The frame this module produces (``state.feature_frame``) intentionally keeps the
target, group, and temporal columns alongside the engineered features, because
:func:`~automl_architect.execution.splitter.make_splits` needs them to draw the
split. ``state.feature_names`` is the authoritative list of X columns.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.errors import ExecutionError
from ..core.schemas import (
    AgentName,
    EventKind,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    TaskType,
    params_to_dict,
)
from ..core.state import RunState

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_FEATURE_COLUMNS",
    "MAX_ONEHOT_CARDINALITY",
    "MAX_POLYNOMIAL_DEGREE",
    "SafeDimensionReduction",
    "TFIDF_MAX_FEATURES",
    "apply_feature_plan",
    "transformed_feature_names",
]

#: One-hot beyond this many levels produces more columns than rows of signal.
MAX_ONEHOT_CARDINALITY = 50
#: Hard cap on TF-IDF width: a text column must not silently dominate the matrix.
TFIDF_MAX_FEATURES = 300
#: Guard against an op (polynomial, interactions) exploding the frame.
MAX_FEATURE_COLUMNS = 600
MAX_POLYNOMIAL_DEGREE = 3
#: Consolidate the frame's blocks after this many engineered columns.
_DEFRAGMENT_EVERY = 32
#: Default smoothing weight for target encoding, in pseudo-observations.
DEFAULT_TARGET_SMOOTHING = 10.0
DEFAULT_TARGET_ENCODE_FOLDS = 5

_CYCLE_PERIODS = {
    "month": 12,
    "day": 31,
    "dayofweek": 7,
    "weekday": 7,
    "dayofyear": 366,
    "hour": 24,
    "minute": 60,
    "second": 60,
    "quarter": 4,
    "weekofyear": 52,
    "week": 52,
}

_EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# Deferred transformer specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    """How one column should be transformed *inside* the fitted pipeline.

    Columns sharing an identical spec collapse into a single
    :class:`~sklearn.compose.ColumnTransformer` branch, which keeps the
    transformed feature names short and the pipeline readable.
    """

    kind: str
    params: tuple[tuple[str, Any], ...] = ()
    per_column: bool = False

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def wants_1d(self) -> bool:
        """Text vectorisers take a 1-D column; every other transformer takes 2-D."""
        return self.kind == "tfidf"


def _spec(kind: str, per_column: bool = False, **options: Any) -> _Spec:
    return _Spec(
        kind=kind,
        params=tuple(sorted((k, v) for k, v in options.items())),
        per_column=per_column,
    )


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Mutable working state for one feature-engineering pass."""

    state: RunState
    frame: pd.DataFrame
    target: str | None
    task: TaskType | None
    temporal: str | None
    group: str | None
    features: list[str] = field(default_factory=list)
    specs: dict[str, _Spec] = field(default_factory=dict)
    post_steps: list[tuple[str, Any]] = field(default_factory=list)
    generated: set[str] = field(default_factory=set)
    encodings: dict[str, Any] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    capped: bool = False
    _inserted: int = 0

    # -- naming -----------------------------------------------------------

    def unique_name(self, base: str) -> str:
        """A deterministic, collision-free column name."""
        slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(base)).strip("_") or "feature"
        candidate = slug
        suffix = 2
        taken = set(map(str, self.frame.columns)) | self.generated
        while candidate in taken:
            candidate = f"{slug}_{suffix}"
            suffix += 1
        self.generated.add(candidate)
        return candidate

    # -- feature list -----------------------------------------------------

    def add(self, name: str, values: Any, *, spec: _Spec | None = None) -> str | None:
        """Attach an engineered column to the frame and the feature list."""
        if len(self.features) >= MAX_FEATURE_COLUMNS:
            if not self.capped:
                self.capped = True
                self.state.add_warning(
                    f"features: reached the {MAX_FEATURE_COLUMNS}-column cap; "
                    f"further generated features were skipped"
                )
            return None
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        self.frame[name] = series.reindex(self.frame.index)
        self._inserted += 1
        if self._inserted % _DEFRAGMENT_EVERY == 0:
            # Each insertion appends a block; later ops read engineered columns
            # back, so they must be visible immediately, and pandas warns once
            # the block count grows. A periodic consolidating copy is the cost of
            # keeping the frame both live and unfragmented.
            self.frame = self.frame.copy()
        if name not in self.features:
            self.features.append(name)
        if spec is not None:
            self.specs[name] = spec
        return name

    def consume(self, column: str, reason: str) -> None:
        """Keep a column in the frame but stop using it as a feature."""
        if column in self.features:
            self.features.remove(column)
        self.excluded[column] = reason
        self.specs.pop(column, None)

    def exclude(self, column: str, reason: str) -> None:
        self.consume(column, reason)

    def record(self, op: FeatureOp, message: str, rationale: str = "") -> None:
        line = f"{op.value}: {message}"
        if rationale:
            line = f"{line} — {_trim(rationale)}"
        self.state.applied_features.append(line)
        self.state.bus.emit(
            EventKind.LOG,
            line,
            agent=AgentName.FEATURES,
            payload={"op": op.value, "detail": message},
        )

    def skip(self, op: FeatureOp, reason: str) -> None:
        self.state.add_warning(f"features: skipped {op.value} — {reason}")


def _trim(text: str, limit: int = 180) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _is_numeric(series: pd.Series) -> bool:
    return bool(pdt.is_numeric_dtype(series)) and not pdt.is_bool_dtype(series)


def _is_stringish(series: pd.Series) -> bool:
    if _is_categorical(series):
        return True
    return bool(pdt.is_string_dtype(series) or pdt.is_object_dtype(series))


def _numeric_view(ctx: _Ctx, column: str) -> pd.Series | None:
    series = ctx.frame[column]
    if _is_numeric(series):
        return series.astype("float64")
    if pdt.is_bool_dtype(series):
        return series.astype("float64")
    return None


def _text_view(series: pd.Series) -> pd.Series:
    if _is_categorical(series):
        return series.astype("object").astype("str")
    return series.astype("str")


def _category_key(series: pd.Series) -> pd.Series:
    """A stable string key for grouping/mapping, with nulls made explicit.

    Nulls become a real level rather than being dropped, so the same value maps
    to the same code at fit time and at scoring time.
    """
    return _text_view(series).where(series.notna(), other="__MISSING__").astype("object")


def _as_datetime(ctx: _Ctx, column: str) -> pd.Series | None:
    series = ctx.frame[column]
    if pdt.is_datetime64_any_dtype(series):
        return series
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            return None
    if int(parsed.notna().sum()) == 0:
        return None
    return parsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_feature_plan(state: RunState, plan: FeaturePlan) -> Any:
    """Apply every decision in ``plan``, building the frame and the preprocessor.

    Each operation is wrapped individually: a failure warns and is skipped so one
    bad instruction costs one feature rather than the run.

    Args:
        state: The run blackboard. Reads ``working_df`` (falling back to
            ``raw_df``), ``profile``, ``problem``, and ``splits`` when a training
            partition already exists. Writes ``feature_frame``, ``feature_names``,
            ``preprocessor``, ``applied_features``, and several ``extras`` keys.
        plan: The Feature Engineering Agent's decisions.

    Returns:
        The engineered :class:`pandas.DataFrame`, also assigned to
        ``state.feature_frame``. It retains the target, group, and temporal
        columns so the splitter can use them; ``state.feature_names`` lists the
        columns that are actually model inputs.

    Raises:
        ExecutionError: If no dataframe has been loaded yet.
    """
    source = state.working_df if state.working_df is not None else state.raw_df
    if source is None:
        raise ExecutionError("apply_feature_plan: no dataframe on the run state")

    frame = source.copy()
    if frame.index.has_duplicates:
        frame = frame.reset_index(drop=True)
        state.add_warning("features: duplicate index labels found; index was reset")

    ctx = _Ctx(
        state=state,
        frame=frame,
        target=state.target,
        task=state.task_type,
        temporal=_temporal_column(state, frame),
        group=_group_column(state, frame),
    )
    ctx.features = _initial_features(ctx)
    n_initial = len(ctx.features)

    decisions = [d for d in plan.decisions if isinstance(d, FeatureDecision)]
    decisions.sort(key=lambda d: _OP_ORDER.get(d.op, 50))

    for decision in decisions:
        handler = _HANDLERS.get(decision.op)
        if handler is None:
            ctx.skip(decision.op, "no executor is registered for this operation")
            continue
        columns = _resolve_columns(ctx, decision)
        params = params_to_dict(decision.parameters)
        try:
            handler(ctx, decision, columns, params)
        except Exception as exc:  # noqa: BLE001 - one op must not abort the run
            logger.exception("feature op %s failed", decision.op)
            state.add_warning(
                f"features: {decision.op.value} on {columns or 'table'} failed "
                f"({exc}); skipped"
            )

    _finalise(ctx)

    state.feature_frame = ctx.frame
    state.feature_names = list(ctx.features)
    state.extras["feature_encodings"] = ctx.encodings
    state.extras["feature_excluded_columns"] = ctx.excluded
    state.bus.emit(
        EventKind.LOG,
        f"feature engineering complete: {n_initial} -> {len(ctx.features)} model input "
        f"column(s), {len(state.applied_features)} operation(s) applied",
        agent=AgentName.FEATURES,
        payload={
            "n_features_before": n_initial,
            "n_features_after": len(ctx.features),
            "n_frame_columns": int(ctx.frame.shape[1]),
        },
    )
    return ctx.frame


# ---------------------------------------------------------------------------
# Column bookkeeping
# ---------------------------------------------------------------------------


def _temporal_column(state: RunState, frame: pd.DataFrame) -> str | None:
    """The time axis for sequential ops.

    Falls back to any datetime column, unlike the splitter's stricter rule: a lag
    computed along the wrong timestamp is a weak feature that selection will
    discard, whereas a split along the wrong timestamp corrupts every score.
    """
    problem = state.problem
    if problem and problem.temporal_column and problem.temporal_column in frame.columns:
        return problem.temporal_column
    if state.profile:
        for name in state.profile.temporal_columns:
            if name in frame.columns:
                return name
    for name in frame.columns:
        if pdt.is_datetime64_any_dtype(frame[name]):
            return str(name)
    return None


def _group_column(state: RunState, frame: pd.DataFrame) -> str | None:
    problem = state.problem
    if problem and problem.group_column and problem.group_column in frame.columns:
        return problem.group_column
    return None


def _initial_features(ctx: _Ctx) -> list[str]:
    """Everything that is a candidate model input before any op runs."""
    state = ctx.state
    frame = ctx.frame
    features: list[str] = []
    identifiers = set(state.profile.identifier_columns) if state.profile else set()
    constants = set(state.profile.constant_columns) if state.profile else set()
    leakage = (
        {f.column for f in state.profile.leakage_findings if f.severity.value in ("high", "critical")}
        if state.profile
        else set()
    )

    for name in map(str, frame.columns):
        if ctx.target and name == ctx.target:
            ctx.excluded[name] = "target"
            continue
        if name in identifiers:
            ctx.excluded[name] = "identifier: unique per row, so it carries no signal"
            continue
        if name in constants:
            ctx.excluded[name] = "constant: zero variance"
            continue
        if name in leakage:
            ctx.excluded[name] = "flagged as high-severity leakage by the profiler"
            continue
        if name == ctx.group:
            ctx.excluded[name] = (
                "group key: held out entity-wise, so its raw levels never generalise "
                "(an encoding of it can still be added explicitly)"
            )
            continue
        if pdt.is_datetime64_any_dtype(frame[name]):
            ctx.excluded[name] = (
                "raw datetime: not a numeric input; decompose it into calendar parts"
            )
            continue
        features.append(name)
    return features


def _resolve_columns(ctx: _Ctx, decision: FeatureDecision) -> list[str]:
    """Filter agent-supplied column names against the real frame."""
    known = set(map(str, ctx.frame.columns))
    named = [c for c in decision.input_columns if c in known]
    unknown = [c for c in decision.input_columns if c not in known]
    if unknown:
        ctx.state.add_warning(
            f"features: {decision.op.value} referenced unknown column(s) "
            f"{unknown[:8]}{'...' if len(unknown) > 8 else ''}; ignoring them"
        )
    return named


def _fit_index(ctx: _Ctx) -> pd.Index:
    """Rows any target-aware statistic may be fitted on.

    If the split has already been drawn, only training rows qualify — that is the
    difference between an honest encoding and one that has read the test set.
    """
    train = ctx.state.splits.X_train
    if train is not None and len(train):
        shared = ctx.frame.index.intersection(pd.Index(train.index))
        if len(shared) >= max(10, int(0.05 * len(ctx.frame))):
            return shared
    return ctx.frame.index


def _target_values(ctx: _Ctx) -> pd.Series | None:
    """The target as floats, suitable for mean-based encodings.

    Multiclass targets are refused rather than silently collapsed: a single mean
    over more than two labels encodes label *ordering* that does not exist.
    """
    target = ctx.target
    if not target or target not in ctx.frame.columns:
        return None
    y = ctx.frame[target]
    task = ctx.task
    if (task and task.is_classification) or _is_stringish(y) or pdt.is_bool_dtype(y):
        levels = pd.unique(y.dropna())
        if len(levels) > 2:
            return None
        positive = getattr(ctx.state.problem, "positive_class", None)
        text = _text_view(y)
        if positive is not None and str(positive) in set(text.dropna()):
            chosen = str(positive)
        else:
            chosen = str(sorted(map(str, levels))[-1]) if len(levels) else None
        if chosen is None:
            return None
        return (text == chosen).astype("float64").where(y.notna())
    if _is_numeric(y):
        return y.astype("float64")
    return None


# ---------------------------------------------------------------------------
# Temporal ordering
# ---------------------------------------------------------------------------


def _ordered_index(ctx: _Ctx) -> pd.Index:
    """Frame index sorted by group then time, so shifts mean what they say."""
    keys: list[str] = []
    if ctx.group and ctx.group in ctx.frame.columns:
        keys.append(ctx.group)
    if ctx.temporal and ctx.temporal in ctx.frame.columns:
        keys.append(ctx.temporal)
    if not keys:
        return ctx.frame.index
    return ctx.frame.sort_values(keys, kind="stable").index


def _time_ordered(ctx: _Ctx, index: pd.Index) -> pd.Index:
    """``index`` in ascending time order, for splitters that assume row order."""
    if not ctx.temporal or ctx.temporal not in ctx.frame.columns:
        return index
    return ctx.frame.loc[index, ctx.temporal].sort_values(kind="stable").index


def _sequential_setup(
    ctx: _Ctx, op: FeatureOp, columns: list[str]
) -> tuple[pd.Index, pd.Series | None, list[str]] | None:
    """Shared preflight for lag/rolling/diff/expanding."""
    if not ctx.temporal:
        ctx.skip(
            op,
            "no temporal column is available, so row order carries no time meaning "
            "and a shifted value would be an arbitrary neighbouring row",
        )
        return None
    if not columns and ctx.target and ctx.target in ctx.frame.columns:
        # An autoregressive feature on the target is the default intent of these
        # ops in a forecasting problem.
        columns = [ctx.target]
    usable = [c for c in columns if _numeric_view(ctx, c) is not None]
    if not usable:
        ctx.skip(op, "no numeric input columns")
        return None
    order = _ordered_index(ctx)
    groups = (
        _category_key(ctx.frame.loc[order, ctx.group])
        if ctx.group and ctx.group in ctx.frame.columns
        else None
    )
    return order, groups, usable


def _grouped_apply(
    series: pd.Series, groups: pd.Series | None, func: Any
) -> pd.Series:
    if groups is None:
        return func(series)
    return series.groupby(groups, sort=False, observed=True).transform(func)


def _autoregressive_shift(ctx: _Ctx, column: str) -> int:
    """1 for the target, 0 otherwise.

    A rolling mean of a feature may include the current row — that value is known
    at prediction time. A rolling mean of the *target* may not: it would contain
    the answer.
    """
    return 1 if ctx.target and column == ctx.target else 0


# ---------------------------------------------------------------------------
# Operations: calendar
# ---------------------------------------------------------------------------


def _op_date_decompose(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    columns = columns or ([ctx.temporal] if ctx.temporal else [])
    if not columns:
        ctx.skip(FeatureOp.DATE_DECOMPOSE, "no datetime column to decompose")
        return
    requested = params.get("parts") or params.get("components")
    for column in columns:
        dt = _as_datetime(ctx, column)
        if dt is None:
            ctx.skip(FeatureOp.DATE_DECOMPOSE, f"'{column}' does not parse as a datetime")
            continue
        sub_daily = bool((dt.dt.floor("D") != dt).any())
        parts = (
            [str(p) for p in requested]
            if isinstance(requested, list)
            else ["year", "month", "day", "dayofweek", "quarter", "weekofyear", "is_weekend"]
            + (["hour"] if sub_daily else [])
        )
        made: list[str] = []
        for part in parts:
            values = _calendar_part(dt, part)
            if values is None:
                continue
            name = ctx.unique_name(f"{column}_{part}")
            if ctx.add(name, values, spec=_spec("numeric")) is not None:
                made.append(name)
        if made:
            ctx.record(
                FeatureOp.DATE_DECOMPOSE,
                f"decomposed '{column}' into {made}",
                decision.rationale,
            )


def _calendar_part(dt: pd.Series, part: str) -> pd.Series | None:
    accessor = dt.dt
    try:
        if part == "weekofyear" or part == "week":
            iso = accessor.isocalendar()
            return iso["week"].astype("float64")
        if part == "is_weekend":
            return (accessor.dayofweek >= 5).astype("float64").where(dt.notna())
        if part in ("is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end"):
            return getattr(accessor, part).astype("float64").where(dt.notna())
        if part == "epoch_days":
            return (accessor.floor("D").astype("int64") / 86_400_000_000_000).astype(
                "float64"
            )
        value = getattr(accessor, part)
    except (AttributeError, ValueError, TypeError):
        return None
    if not pdt.is_numeric_dtype(value):
        return None
    return value.astype("float64")


def _op_cyclical_encode(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    columns = columns or ([ctx.temporal] if ctx.temporal else [])
    if not columns:
        ctx.skip(FeatureOp.CYCLICAL_ENCODE, "no input column")
        return
    part = str(params.get("part") or params.get("component") or "month")
    for column in columns:
        dt = _as_datetime(ctx, column)
        if dt is not None:
            values = _calendar_part(dt, part)
            period = float(params.get("period") or _CYCLE_PERIODS.get(part, 12))
            label = f"{column}_{part}"
        else:
            values = _numeric_view(ctx, column)
            if values is None:
                ctx.skip(
                    FeatureOp.CYCLICAL_ENCODE,
                    f"'{column}' is neither a datetime nor numeric",
                )
                continue
            span = float(np.nanmax(values.to_numpy()) - np.nanmin(values.to_numpy()) + 1)
            period = float(params.get("period") or span or 1.0)
            label = str(column)
        if values is None:
            ctx.skip(FeatureOp.CYCLICAL_ENCODE, f"cannot extract '{part}' from '{column}'")
            continue
        angle = 2.0 * math.pi * values.astype("float64") / max(period, 1e-9)
        sin_name = ctx.unique_name(f"{label}_sin")
        cos_name = ctx.unique_name(f"{label}_cos")
        ctx.add(sin_name, np.sin(angle), spec=_spec("numeric"))
        ctx.add(cos_name, np.cos(angle), spec=_spec("numeric"))
        ctx.record(
            FeatureOp.CYCLICAL_ENCODE,
            f"encoded '{label}' on a {period:g}-unit cycle as {sin_name}/{cos_name}, so "
            f"the last and first positions sit adjacent",
            decision.rationale,
        )


# ---------------------------------------------------------------------------
# Operations: sequential
# ---------------------------------------------------------------------------


def _op_lag(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    setup = _sequential_setup(ctx, FeatureOp.LAG, columns)
    if setup is None:
        return
    order, groups, usable = setup
    lags = _int_list(params.get("lags") or params.get("periods") or params.get("lag") or 1)
    for column in usable:
        series = ctx.frame.loc[order, column].astype("float64")
        for lag in lags:
            step = max(1, abs(int(lag)))
            values = _grouped_apply(series, groups, lambda s, k=step: s.shift(k))
            name = ctx.unique_name(f"{column}_lag{step}")
            ctx.add(name, values.reindex(ctx.frame.index), spec=_spec("numeric"))
        ctx.record(
            FeatureOp.LAG,
            f"lagged '{column}' by {lags} ordered on '{ctx.temporal}'"
            + (f" within each '{ctx.group}'" if groups is not None else ""),
            decision.rationale,
        )


def _op_rolling(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    setup = _sequential_setup(ctx, FeatureOp.ROLLING, columns)
    if setup is None:
        return
    order, groups, usable = setup
    windows = _int_list(params.get("windows") or params.get("window") or 3)
    stats = _str_list(params.get("stats") or params.get("agg") or "mean")
    for column in usable:
        shift = int(params.get("shift", _autoregressive_shift(ctx, column)))
        series = ctx.frame.loc[order, column].astype("float64")
        made: list[str] = []
        for window in windows:
            size = max(2, abs(int(window)))
            for stat in stats:
                if not hasattr(pd.Series([1.0]).rolling(2), stat):
                    continue

                def _roll(chunk: pd.Series, w: int = size, s: str = stat) -> pd.Series:
                    base = chunk.shift(shift) if shift else chunk
                    return getattr(base.rolling(w, min_periods=1), s)()

                values = _grouped_apply(series, groups, _roll)
                name = ctx.unique_name(f"{column}_roll{size}_{stat}")
                if ctx.add(name, values.reindex(ctx.frame.index), spec=_spec("numeric")):
                    made.append(name)
        if made:
            detail = f"rolling {stats} over windows {windows} on '{column}'"
            if shift:
                detail += (
                    f", shifted {shift} period(s) because it derives from the target and "
                    f"must not include the current row"
                )
            ctx.record(FeatureOp.ROLLING, detail, decision.rationale)


def _op_diff(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    setup = _sequential_setup(ctx, FeatureOp.DIFF, columns)
    if setup is None:
        return
    order, groups, usable = setup
    periods = _int_list(params.get("periods") or params.get("lags") or 1)
    for column in usable:
        shift = _autoregressive_shift(ctx, column)
        series = ctx.frame.loc[order, column].astype("float64")
        for period in periods:
            step = max(1, abs(int(period)))

            def _diff(chunk: pd.Series, k: int = step) -> pd.Series:
                base = chunk.shift(shift) if shift else chunk
                return base.diff(k)

            values = _grouped_apply(series, groups, _diff)
            name = ctx.unique_name(f"{column}_diff{step}")
            ctx.add(name, values.reindex(ctx.frame.index), spec=_spec("numeric"))
        ctx.record(
            FeatureOp.DIFF,
            f"differenced '{column}' at {periods} period(s) ordered on '{ctx.temporal}'",
            decision.rationale,
        )


def _op_expanding(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    setup = _sequential_setup(ctx, FeatureOp.EXPANDING, columns)
    if setup is None:
        return
    order, groups, usable = setup
    stats = _str_list(params.get("stats") or params.get("agg") or "mean")
    for column in usable:
        shift = int(params.get("shift", _autoregressive_shift(ctx, column)))
        series = ctx.frame.loc[order, column].astype("float64")
        made: list[str] = []
        for stat in stats:
            if not hasattr(pd.Series([1.0]).expanding(), stat):
                continue

            def _expand(chunk: pd.Series, s: str = stat) -> pd.Series:
                base = chunk.shift(shift) if shift else chunk
                return getattr(base.expanding(min_periods=1), s)()

            values = _grouped_apply(series, groups, _expand)
            name = ctx.unique_name(f"{column}_expanding_{stat}")
            if ctx.add(name, values.reindex(ctx.frame.index), spec=_spec("numeric")):
                made.append(name)
        if made:
            ctx.record(
                FeatureOp.EXPANDING,
                f"expanding {stats} on '{column}' (history to date"
                + (f", per '{ctx.group}'" if groups is not None else "")
                + ")",
                decision.rationale,
            )


# ---------------------------------------------------------------------------
# Operations: arithmetic
# ---------------------------------------------------------------------------


def _op_interaction(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    if len(columns) < 2:
        ctx.skip(FeatureOp.INTERACTION, "needs at least two input columns")
        return
    numeric = {c: _numeric_view(ctx, c) for c in columns}
    if all(v is not None for v in numeric.values()):
        product = None
        for series in numeric.values():
            product = series if product is None else product * series
        name = ctx.unique_name(decision.output_name_hint or "_x_".join(columns))
        ctx.add(name, product, spec=_spec("numeric"))
        ctx.record(
            FeatureOp.INTERACTION,
            f"multiplied {columns} into '{name}'",
            decision.rationale,
        )
        return

    combined = None
    for column in columns:
        key = _category_key(ctx.frame[column])
        combined = key if combined is None else combined.str.cat(key, sep="|")
    name = ctx.unique_name(decision.output_name_hint or "_x_".join(columns))
    levels = int(pd.Series(combined).nunique())
    if levels > MAX_ONEHOT_CARDINALITY:
        ctx.skip(
            FeatureOp.INTERACTION,
            f"crossing {columns} produces {levels} levels, above the "
            f"{MAX_ONEHOT_CARDINALITY}-level one-hot limit",
        )
        return
    ctx.add(name, combined, spec=_spec("onehot"))
    ctx.record(
        FeatureOp.INTERACTION,
        f"crossed {columns} into the {levels}-level categorical '{name}'",
        decision.rationale,
    )


def _op_ratio(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    numerator = params.get("numerator")
    denominator = params.get("denominator")
    pair = [c for c in (numerator, denominator) if c in ctx.frame.columns]
    if len(pair) != 2:
        pair = columns[:2]
    if len(pair) != 2:
        ctx.skip(FeatureOp.RATIO, "needs a numerator and a denominator column")
        return
    top = _numeric_view(ctx, pair[0])
    bottom = _numeric_view(ctx, pair[1])
    if top is None or bottom is None:
        ctx.skip(FeatureOp.RATIO, f"{pair} are not both numeric")
        return
    safe = bottom.where(bottom != 0.0)  # 0/0 becomes NaN, imputed in the pipeline
    name = ctx.unique_name(decision.output_name_hint or f"{pair[0]}_per_{pair[1]}")
    ctx.add(name, top / safe, spec=_spec("numeric"))
    n_zero = int((bottom == 0.0).sum())
    ctx.record(
        FeatureOp.RATIO,
        f"'{name}' = {pair[0]} / {pair[1]}"
        + (f" ({n_zero} zero denominator(s) left missing for imputation)" if n_zero else ""),
        decision.rationale,
    )


def _op_polynomial(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    degree = min(MAX_POLYNOMIAL_DEGREE, max(2, int(params.get("degree", 2))))
    usable = [c for c in columns if _numeric_view(ctx, c) is not None]
    if not usable:
        ctx.skip(FeatureOp.POLYNOMIAL, "no numeric input columns")
        return
    made: list[str] = []
    for column in usable:
        series = _numeric_view(ctx, column)
        assert series is not None
        for power in range(2, degree + 1):
            name = ctx.unique_name(f"{column}_pow{power}")
            if ctx.add(name, series**power, spec=_spec("numeric")):
                made.append(name)
    if bool(params.get("interactions", len(usable) > 1)):
        for i, left in enumerate(usable):
            for right in usable[i + 1 :]:
                name = ctx.unique_name(f"{left}_x_{right}")
                values = _numeric_view(ctx, left) * _numeric_view(ctx, right)  # type: ignore[operator]
                if ctx.add(name, values, spec=_spec("numeric")):
                    made.append(name)
    if made:
        ctx.record(
            FeatureOp.POLYNOMIAL,
            f"degree-{degree} expansion of {usable} added {len(made)} column(s)",
            decision.rationale,
        )


def _op_log_transform(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    for column in columns:
        series = _numeric_view(ctx, column)
        if series is None:
            ctx.skip(FeatureOp.LOG_TRANSFORM, f"'{column}' is not numeric")
            continue
        minimum = float(series.min(skipna=True)) if series.notna().any() else 0.0
        if minimum < 0:
            # Sign-preserving log keeps negative values usable instead of
            # discarding them to NaN.
            values = np.sign(series) * np.log1p(series.abs())
            how = "sign-preserving log1p (the column contains negatives)"
        else:
            values = np.log1p(series)
            how = "log1p"
        name = ctx.unique_name(decision.output_name_hint or f"{column}_log")
        if ctx.add(name, values, spec=_spec("numeric")):
            ctx.consume(column, f"re-expressed as '{name}'")
            ctx.record(
                FeatureOp.LOG_TRANSFORM,
                f"{how} of '{column}' -> '{name}' (raw column dropped from the "
                f"feature set to avoid a collinear duplicate)",
                decision.rationale,
            )


def _op_sqrt_transform(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    for column in columns:
        series = _numeric_view(ctx, column)
        if series is None:
            ctx.skip(FeatureOp.SQRT_TRANSFORM, f"'{column}' is not numeric")
            continue
        values = np.sign(series) * np.sqrt(series.abs())
        name = ctx.unique_name(decision.output_name_hint or f"{column}_sqrt")
        if ctx.add(name, values, spec=_spec("numeric")):
            ctx.consume(column, f"re-expressed as '{name}'")
            ctx.record(
                FeatureOp.SQRT_TRANSFORM,
                f"sign-preserving square root of '{column}' -> '{name}'",
                decision.rationale,
            )


def _op_boxcox(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    for column in columns:
        series = _numeric_view(ctx, column)
        if series is None:
            ctx.skip(FeatureOp.BOXCOX, f"'{column}' is not numeric")
            continue
        strictly_positive = bool(series.dropna().gt(0).all())
        method = "box-cox" if strictly_positive else "yeo-johnson"
        # The lambda is a fitted statistic, so it belongs in the pipeline where it
        # is estimated from the training partition only.
        ctx.specs[column] = _spec("power", method=method)
        ctx.record(
            FeatureOp.BOXCOX,
            f"'{column}' will be {method} power-transformed inside the fitted "
            f"pipeline (lambda estimated on the training partition)",
            decision.rationale,
        )


def _op_binning(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    n_bins = max(2, min(50, int(params.get("n_bins", params.get("bins", 5)))))
    strategy = str(params.get("strategy", "quantile"))
    if strategy not in ("quantile", "uniform", "kmeans"):
        strategy = "quantile"
    for column in columns:
        if _numeric_view(ctx, column) is None:
            ctx.skip(FeatureOp.BINNING, f"'{column}' is not numeric")
            continue
        ctx.specs[column] = _spec("bin", n_bins=n_bins, strategy=strategy)
        ctx.record(
            FeatureOp.BINNING,
            f"'{column}' will be discretised into {n_bins} {strategy} bins inside the "
            f"fitted pipeline (edges learned from the training partition)",
            decision.rationale,
        )


# ---------------------------------------------------------------------------
# Operations: categorical encoding
# ---------------------------------------------------------------------------


def _coerce_min_frequency(value: Any, *, on_invalid: Callable[[str], None]) -> int | float | None:
    """Normalise ``min_frequency`` to what :class:`OneHotEncoder` actually accepts.

    sklearn overloads this parameter on type, and the distinction is load-bearing:
    an ``int`` >= 1 is an absolute row count, a ``float`` in (0.0, 1.0) is a
    proportion of rows, and nothing else validates. Coercing everything to
    ``float`` turns a perfectly reasonable "pool levels seen fewer than 20 times"
    into ``20.0``, which is neither a count nor a proportion, and sklearn rejects
    it — failing *every* candidate model, baseline included, because the encoder
    lives in the shared preprocessor.

    Args:
        value: The agent-supplied parameter, of unknown type.
        on_invalid: Called with an explanation when the value cannot be honoured.

    Returns:
        An ``int`` count, a ``float`` proportion, or ``None`` to leave the
        encoder's default in place.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number <= 0:  # NaN or non-positive
        on_invalid(f"min_frequency={value!r} is not positive; ignoring it")
        return None
    if 0.0 < number < 1.0:
        return number  # a proportion of rows
    if number.is_integer():
        return int(number)  # an absolute count
    on_invalid(
        f"min_frequency={value!r} is above 1 but fractional; sklearn reads values "
        f">= 1 as absolute counts, so it was rounded down to {int(number)}"
    )
    return int(number)


def _op_one_hot(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    max_categories = params.get("max_categories")
    max_categories = int(max_categories) if isinstance(max_categories, (int, float)) else None
    min_frequency = _coerce_min_frequency(
        params.get("min_frequency"),
        on_invalid=lambda message: ctx.state.add_warning(f"features: {message}"),
    )
    for column in columns:
        levels = int(ctx.frame[column].nunique(dropna=True))
        if levels > MAX_ONEHOT_CARDINALITY and max_categories is None:
            max_categories = MAX_ONEHOT_CARDINALITY
            ctx.state.add_warning(
                f"features: '{column}' has {levels} levels; one-hot was capped to the "
                f"{MAX_ONEHOT_CARDINALITY} most frequent with the rest pooled as "
                f"'infrequent'"
            )
        if column not in ctx.features:
            ctx.features.append(column)
            ctx.excluded.pop(column, None)
        ctx.specs[column] = _spec(
            "onehot", max_categories=max_categories, min_frequency=min_frequency
        )
        ctx.record(
            FeatureOp.ONE_HOT_ENCODE,
            f"'{column}' ({levels} level(s)) will be one-hot encoded in the fitted "
            f"pipeline, with unseen levels mapped to all-zeros",
            decision.rationale,
        )


def _op_ordinal(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    order = params.get("categories") or params.get("order") or params.get("levels")
    for column in columns:
        levels = [str(v) for v in order] if isinstance(order, list) else None
        if levels:
            present = set(_category_key(ctx.frame[column]).unique())
            missing = [v for v in present if v not in set(levels) and v != "__MISSING__"]
            if missing:
                ctx.state.add_warning(
                    f"features: ordinal order for '{column}' omits {missing[:6]}; those "
                    f"levels will encode as -1"
                )
            ctx.specs[column] = _spec("ordinal", per_column=True, categories=tuple(levels))
        else:
            ctx.specs[column] = _spec("ordinal")
        if column not in ctx.features:
            ctx.features.append(column)
            ctx.excluded.pop(column, None)
        ctx.record(
            FeatureOp.ORDINAL_ENCODE,
            f"'{column}' will be ordinal-encoded"
            + (f" in the stated order {levels}" if levels else " in sorted level order")
            + " inside the fitted pipeline",
            decision.rationale,
        )


def _op_target_encode(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    """Out-of-fold mean encoding with smoothing toward the prior."""
    y = _target_values(ctx)
    if y is None:
        ctx.skip(
            FeatureOp.TARGET_ENCODE,
            "target encoding needs a numeric or binary target; a multiclass or absent "
            "target would require one column per class",
        )
        return

    smoothing = float(params.get("smoothing", DEFAULT_TARGET_SMOOTHING))
    n_splits = int(params.get("n_splits", params.get("folds", DEFAULT_TARGET_ENCODE_FOLDS)))
    fit_index = _fit_index(ctx)
    fitted_on_train_only = len(fit_index) < len(ctx.frame)
    y_fit = y.loc[fit_index].dropna()
    if y_fit.empty:
        ctx.skip(FeatureOp.TARGET_ENCODE, "no non-null target values to encode against")
        return
    # Time order matters: a TimeSeriesSplit over rows in arbitrary order would
    # build folds that mix past and future.
    fit_index = _time_ordered(ctx, y_fit.index)
    y_fit = y_fit.loc[fit_index]
    prior = float(y_fit.mean())

    from .splitter import make_cv_splitter

    splitter, groups = make_cv_splitter(
        ctx.state, n_splits=min(n_splits, max(2, len(fit_index) // 2)), y=y_fit.to_numpy()
    )
    if groups is not None:
        if ctx.group and ctx.group in ctx.frame.columns:
            groups = _category_key(ctx.frame.loc[fit_index, ctx.group]).to_numpy()
        else:  # pragma: no cover - grouped strategy implies a group column
            groups = None

    for column in columns:
        keys = _category_key(ctx.frame[column])
        oof = pd.Series(np.nan, index=ctx.frame.index, dtype="float64")
        positions = np.arange(len(fit_index))
        try:
            folds = list(splitter.split(positions.reshape(-1, 1), y_fit.to_numpy(), groups))
        except (ValueError, TypeError) as exc:
            ctx.skip(FeatureOp.TARGET_ENCODE, f"could not build folds for '{column}' ({exc})")
            continue
        for train_pos, valid_pos in folds:
            train_labels = fit_index[train_pos]
            valid_labels = fit_index[valid_pos]
            mapping, fold_prior = _smoothed_means(
                keys.loc[train_labels], y_fit.loc[train_labels], smoothing
            )
            oof.loc[valid_labels] = (
                keys.loc[valid_labels].map(mapping).astype("float64").fillna(fold_prior)
            )

        full_mapping, _ = _smoothed_means(keys.loc[fit_index], y_fit, smoothing)
        # Rows outside the fitting partition (and any fold the splitter never used
        # for validation, e.g. the first block of a time-series split) take the
        # mapping fitted on the whole fitting partition.
        untouched = oof.isna()
        if bool(untouched.any()):
            oof.loc[untouched] = (
                keys.loc[untouched].map(full_mapping).astype("float64").fillna(prior)
            )

        name = ctx.unique_name(decision.output_name_hint or f"{column}_target_enc")
        if ctx.add(name, oof, spec=_spec("numeric")) is None:
            continue
        ctx.consume(column, f"replaced by the target encoding '{name}'")
        ctx.encodings.setdefault("target_encode", {})[column] = {
            "output": name,
            "prior": prior,
            "smoothing": smoothing,
            "mapping": {str(k): float(v) for k, v in full_mapping.items()},
        }
        ctx.record(
            FeatureOp.TARGET_ENCODE,
            f"'{column}' -> '{name}' as a {len(folds)}-fold out-of-fold mean smoothed "
            f"with {smoothing:g} pseudo-observations toward the prior {prior:.4g}"
            + (
                "; the mapping was fitted on the training partition only"
                if fitted_on_train_only
                else "; each row's value excludes its own fold, so no row sees its own label"
            ),
            decision.rationale,
        )


def _smoothed_means(
    keys: pd.Series, y: pd.Series, smoothing: float
) -> tuple[dict[Any, float], float]:
    """Per-level mean shrunk toward the global mean by ``smoothing`` counts."""
    prior = float(y.mean())
    grouped = y.groupby(keys.reindex(y.index), sort=False, observed=True)
    stats = grouped.agg(["mean", "count"])
    weight = stats["count"].astype("float64")
    shrunk = (weight * stats["mean"] + smoothing * prior) / (weight + smoothing)
    return {k: float(v) for k, v in shrunk.items()}, prior


def _op_frequency_encode(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    normalise = bool(params.get("normalize", params.get("normalise", True)))
    fit_index = _fit_index(ctx)
    for column in columns:
        keys = _category_key(ctx.frame[column])
        # Counting levels is target-independent, so it is safe on the fitting
        # partition without folding; only the *target* needs out-of-fold care.
        counts = keys.loc[fit_index].value_counts(normalize=normalise)
        values = keys.map(counts).astype("float64").fillna(0.0)
        name = ctx.unique_name(decision.output_name_hint or f"{column}_freq")
        if ctx.add(name, values, spec=_spec("numeric")) is None:
            continue
        ctx.consume(column, f"replaced by the frequency encoding '{name}'")
        ctx.encodings.setdefault("frequency_encode", {})[column] = {
            "output": name,
            "normalized": normalise,
            "mapping": {str(k): float(v) for k, v in counts.items()},
        }
        ctx.record(
            FeatureOp.FREQUENCY_ENCODE,
            f"'{column}' -> '{name}' as level "
            f"{'share' if normalise else 'count'} over {len(counts)} level(s); unseen "
            f"levels score 0",
            decision.rationale,
        )


def _op_hash_encode(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    n_buckets = max(2, min(256, int(params.get("n_features", params.get("n_buckets", 8)))))
    for column in columns:
        keys = _category_key(ctx.frame[column])
        # md5 rather than hash(): Python salts string hashing per process, so
        # hash() would give a different bucket on every run.
        buckets = keys.map(
            lambda v: int(hashlib.md5(str(v).encode("utf-8")).hexdigest()[:8], 16)
            % n_buckets
        )
        name = ctx.unique_name(decision.output_name_hint or f"{column}_hash")
        if ctx.add(name, buckets.astype("str"), spec=_spec("onehot")) is None:
            continue
        ctx.consume(column, f"replaced by the hashed buckets '{name}'")
        ctx.encodings.setdefault("hash_encode", {})[column] = {
            "output": name,
            "n_buckets": n_buckets,
            "algorithm": "md5-mod",
        }
        ctx.record(
            FeatureOp.HASH_ENCODE,
            f"'{column}' ({int(keys.nunique())} level(s)) hashed into {n_buckets} "
            f"stable buckets as '{name}', which caps width and absorbs unseen levels",
            decision.rationale,
        )


def _op_aggregate_by_group(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    by = params.get("group") or params.get("by") or ctx.group
    if by not in ctx.frame.columns:
        candidates = [c for c in columns if _numeric_view(ctx, c) is None]
        by = candidates[0] if candidates else None
    if by is None or by not in ctx.frame.columns:
        ctx.skip(FeatureOp.AGGREGATE_BY_GROUP, "no grouping column available")
        return
    stats = _str_list(params.get("stats") or params.get("agg") or ["mean", "std"])
    values = [c for c in columns if c != by and _numeric_view(ctx, c) is not None]
    if ctx.target and ctx.target in values:
        ctx.state.add_warning(
            f"features: refused to aggregate the target '{ctx.target}' by '{by}' — a "
            f"group mean of the target is target leakage; use target_encode instead"
        )
        values = [c for c in values if c != ctx.target]
    if not values:
        ctx.skip(FeatureOp.AGGREGATE_BY_GROUP, "no numeric value columns to aggregate")
        return

    fit_index = _fit_index(ctx)
    keys = _category_key(ctx.frame[by])
    for column in values:
        series = _numeric_view(ctx, column)
        assert series is not None
        for stat in stats:
            try:
                mapping = series.loc[fit_index].groupby(
                    keys.loc[fit_index], sort=False, observed=True
                ).agg(stat)
            except (AttributeError, ValueError, TypeError):
                continue
            mapped = keys.map(mapping).astype("float64")
            name = ctx.unique_name(f"{column}_{stat}_by_{by}")
            if ctx.add(name, mapped, spec=_spec("numeric")) is None:
                continue
            ctx.encodings.setdefault("group_aggregate", {})[f"{by}|{column}|{stat}"] = {
                "output": name,
                "mapping": {str(k): _finite(v) for k, v in mapping.items()},
            }
        ctx.record(
            FeatureOp.AGGREGATE_BY_GROUP,
            f"aggregated '{column}' by '{by}' as {stats}, giving each row its group's "
            f"profile as context",
            decision.rationale,
        )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Operations: text and geo
# ---------------------------------------------------------------------------


def _op_text_tfidf(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    max_features = int(params.get("max_features", 100))
    max_features = max(2, min(TFIDF_MAX_FEATURES, max_features))
    ngram_max = max(1, min(3, int(params.get("ngram_max", params.get("ngram_range", 1)))))
    min_df = params.get("min_df", 2)
    min_df = int(min_df) if isinstance(min_df, (int, float)) and min_df >= 1 else 1
    sublinear = bool(params.get("sublinear_tf", True))
    stop_words = params.get("stop_words", "english")
    stop_words = str(stop_words) if stop_words in ("english",) else None

    for column in columns:
        series = ctx.frame[column]
        if not _is_stringish(series):
            ctx.skip(FeatureOp.TEXT_TFIDF, f"'{column}' is not textual")
            continue
        # The vectoriser cannot see nulls; normalising them to empty strings here
        # is deterministic and identical at scoring time.
        ctx.frame[column] = _text_view(series).where(series.notna(), other="").astype("str")
        spec = _spec(
            "tfidf",
            per_column=True,
            max_features=max_features,
            ngram_max=ngram_max,
            min_df=min_df,
            sublinear_tf=sublinear,
            stop_words=stop_words,
        )
        # A vectoriser that cannot build a vocabulary — all-null, stop-words-only,
        # single-character tokens, or no term surviving min_df — raises inside the
        # trainer, past this per-op guard, and takes the whole run with it. Probe the
        # fit here (on the fitting partition only) and skip the op instead.
        reason = _tfidf_unfittable(ctx, column, spec)
        if reason:
            ctx.skip(
                FeatureOp.TEXT_TFIDF,
                f"'{column}' yields no usable vocabulary ({reason})",
            )
            continue
        if column not in ctx.features:
            ctx.features.append(column)
            ctx.excluded.pop(column, None)
        ctx.specs[column] = spec
        ctx.record(
            FeatureOp.TEXT_TFIDF,
            f"'{column}' will be TF-IDF vectorised inside the fitted pipeline, capped at "
            f"{max_features} terms (1-{ngram_max} grams), so the vocabulary comes from "
            f"the training partition only",
            decision.rationale,
        )


def _tfidf_unfittable(ctx: _Ctx, column: str, spec: _Spec) -> str | None:
    """Why a TF-IDF branch for ``column`` could not be fitted, or ``None`` if it can.

    Fitting a throwaway copy on the fitting partition is the only reliable test:
    whether a vocabulary survives ``stop_words``, the token pattern, and ``min_df``
    is not something we can predict from the raw strings.
    """
    documents = ctx.frame.loc[_fit_index(ctx), column].astype("str")
    try:
        _build_transformer(spec, ctx).fit(documents)
    except ValueError as exc:
        return " ".join(str(exc).split())[:160]
    return None


def _op_text_length(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    for column in columns:
        series = ctx.frame[column]
        if not _is_stringish(series):
            ctx.skip(FeatureOp.TEXT_LENGTH, f"'{column}' is not textual")
            continue
        text = _text_view(series).where(series.notna(), other="")
        char_name = ctx.unique_name(f"{column}_char_count")
        word_name = ctx.unique_name(f"{column}_word_count")
        ctx.add(char_name, text.str.len().astype("float64"), spec=_spec("numeric"))
        ctx.add(
            word_name,
            text.str.split().map(lambda parts: float(len(parts))),
            spec=_spec("numeric"),
        )
        ctx.record(
            FeatureOp.TEXT_LENGTH,
            f"measured '{column}' as {char_name} and {word_name}",
            decision.rationale,
        )


def _op_geo_distance(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    lat = params.get("lat") or params.get("latitude")
    lon = params.get("lon") or params.get("longitude")
    pairs = [c for c in (lat, lon) if c in ctx.frame.columns]
    if len(pairs) != 2:
        pairs = [c for c in columns if _numeric_view(ctx, c) is not None]
    if len(pairs) not in (2, 4):
        ctx.skip(
            FeatureOp.GEO_DISTANCE,
            "needs two numeric columns (lat, lon) or four (lat1, lon1, lat2, lon2)",
        )
        return

    if len(pairs) == 4:
        lat1, lon1, lat2, lon2 = (_numeric_view(ctx, c) for c in pairs)
        label = f"{pairs[0]}_{pairs[1]}_to_{pairs[2]}_{pairs[3]}"
        reference = "the second coordinate pair"
    else:
        lat1, lon1 = (_numeric_view(ctx, c) for c in pairs)
        ref_lat = params.get("ref_lat")
        ref_lon = params.get("ref_lon")
        fit_index = _fit_index(ctx)
        if not isinstance(ref_lat, (int, float)):
            ref_lat = float(lat1.loc[fit_index].mean())  # type: ignore[union-attr]
            ref_lon = float(lon1.loc[fit_index].mean())  # type: ignore[union-attr]
            reference = f"the dataset centroid ({ref_lat:.4f}, {ref_lon:.4f})"
        else:
            ref_lon = float(ref_lon) if isinstance(ref_lon, (int, float)) else 0.0
            reference = f"the reference point ({float(ref_lat):.4f}, {ref_lon:.4f})"
        lat2 = pd.Series(float(ref_lat), index=ctx.frame.index)
        lon2 = pd.Series(float(ref_lon), index=ctx.frame.index)
        label = f"{pairs[0]}_{pairs[1]}_distance_km"

    distance = _haversine_km(lat1, lon1, lat2, lon2)  # type: ignore[arg-type]
    name = ctx.unique_name(decision.output_name_hint or label)
    ctx.add(name, distance, spec=_spec("numeric"))
    ctx.record(
        FeatureOp.GEO_DISTANCE,
        f"'{name}' is the great-circle distance in km to {reference}",
        decision.rationale,
    )


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    phi1 = np.radians(lat1.astype("float64"))
    phi2 = np.radians(lat2.astype("float64"))
    dphi = phi2 - phi1
    dlambda = np.radians(lon2.astype("float64") - lon1.astype("float64"))
    inner = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(inner.clip(0.0, 1.0)))


# ---------------------------------------------------------------------------
# Operations: scaling, reduction, selection
# ---------------------------------------------------------------------------


_SCALE_KINDS = {
    FeatureOp.STANDARD_SCALE: "scale_standard",
    FeatureOp.MINMAX_SCALE: "scale_minmax",
    FeatureOp.ROBUST_SCALE: "scale_robust",
    FeatureOp.QUANTILE_TRANSFORM: "scale_quantile",
}


def _op_scale(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    kind = _SCALE_KINDS[decision.op]
    targets = [c for c in (columns or list(ctx.features)) if _numeric_view(ctx, c) is not None]
    if not targets:
        ctx.skip(decision.op, "no numeric columns to scale")
        return
    options: dict[str, Any] = {}
    if decision.op is FeatureOp.QUANTILE_TRANSFORM:
        options["output_distribution"] = str(
            params.get("output_distribution", "normal")
        )
    scaled: list[str] = []
    for column in targets:
        existing = ctx.specs.get(column)
        if existing is not None and existing.kind not in ("numeric", kind):
            # An earlier op already chose a distribution transform for this
            # column; two of them in sequence is never what was intended.
            ctx.state.add_warning(
                f"features: '{column}' is already assigned {existing.kind}; "
                f"{kind} was not applied on top of it"
            )
            continue
        ctx.specs[column] = _spec(kind, **options)
        scaled.append(column)
    if not scaled:
        ctx.skip(decision.op, "every candidate column already has a transform assigned")
        return
    ctx.record(
        decision.op,
        f"{len(scaled)} numeric column(s) will be {kind.replace('scale_', '')}-scaled "
        f"inside the fitted pipeline, so the statistics come from the training "
        f"partition and never from the test set",
        decision.rationale,
    )


def _op_reduce(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    kind = "svd" if decision.op is FeatureOp.SVD else "pca"
    n_components = int(params.get("n_components", params.get("k", 10)))
    n_components = max(1, n_components)
    numeric = [c for c in columns if _numeric_view(ctx, c) is not None]

    if numeric and len(numeric) < len(ctx.features):
        block = _spec(f"reduce_{kind}", n_components=n_components)
        for column in numeric:
            ctx.specs[column] = block
        ctx.record(
            decision.op,
            f"{kind.upper()} will compress {len(numeric)} column(s) to "
            f"{n_components} component(s) inside the fitted pipeline",
            decision.rationale,
        )
        return

    ctx.post_steps.append(
        (
            f"{kind}_reduce",
            SafeDimensionReduction(
                n_components=n_components,
                kind=kind,
                random_state=ctx.state.config.random_state,
            ),
        )
    )
    ctx.record(
        decision.op,
        f"{kind.upper()} to at most {n_components} component(s) appended to the fitted "
        f"pipeline, after encoding and scaling",
        decision.rationale,
    )


def _op_select_k_best(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    if not ctx.target or ctx.target not in ctx.frame.columns:
        ctx.skip(
            FeatureOp.SELECT_K_BEST,
            "univariate selection scores features against a target, and none is set",
        )
        return
    k = int(params.get("k", params.get("n_features", 20)))
    k = max(1, k)
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    score_func = f_classif if (ctx.task and ctx.task.is_classification) else f_regression
    ctx.post_steps.append(("select_k_best", SelectKBest(score_func=score_func, k=k)))
    ctx.record(
        FeatureOp.SELECT_K_BEST,
        f"the top {k} feature(s) by {score_func.__name__} will be selected inside the "
        f"fitted pipeline, scored on the training partition only",
        decision.rationale,
    )


def _op_variance_threshold(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    threshold = params.get("threshold", 0.0)
    threshold = float(threshold) if isinstance(threshold, (int, float)) else 0.0
    from sklearn.feature_selection import VarianceThreshold

    ctx.post_steps.append(("variance_threshold", VarianceThreshold(threshold=threshold)))
    ctx.record(
        FeatureOp.VARIANCE_THRESHOLD,
        f"features with variance <= {threshold:g} will be dropped inside the fitted "
        f"pipeline",
        decision.rationale,
    )


def _op_drop_correlated(
    ctx: _Ctx, decision: FeatureDecision, columns: list[str], params: dict[str, Any]
) -> None:
    profile = ctx.state.profile
    if profile is None or not profile.highly_correlated_pairs:
        ctx.skip(
            FeatureOp.DROP_CORRELATED,
            "the profile lists no highly correlated pairs to act on",
        )
        return
    threshold = params.get("threshold")
    threshold = float(threshold) if isinstance(threshold, (int, float)) else 0.0

    target_strength = {
        pair.left if pair.right == ctx.target else pair.right: abs(pair.coefficient)
        for pair in profile.target_correlations
    }
    dropped: list[str] = []
    for pair in profile.highly_correlated_pairs:
        if abs(pair.coefficient) < threshold:
            continue
        left, right = pair.left, pair.right
        if left not in ctx.features or right not in ctx.features:
            continue
        left_strength = target_strength.get(left, -1.0)
        right_strength = target_strength.get(right, -1.0)
        loser = right if left_strength >= right_strength else left
        keeper = left if loser == right else right
        ctx.consume(
            loser,
            f"correlated {pair.coefficient:+.2f} with '{keeper}', which tracks the "
            f"target more strongly",
        )
        dropped.append(loser)
    if dropped:
        ctx.record(
            FeatureOp.DROP_CORRELATED,
            f"dropped {dropped} as redundant, keeping the member of each pair with the "
            f"stronger target correlation",
            decision.rationale,
        )
    else:
        ctx.skip(
            FeatureOp.DROP_CORRELATED,
            "no correlated pair had both members still in the feature set",
        )


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------


_HANDLERS: dict[FeatureOp, Any] = {
    FeatureOp.DATE_DECOMPOSE: _op_date_decompose,
    FeatureOp.CYCLICAL_ENCODE: _op_cyclical_encode,
    FeatureOp.LAG: _op_lag,
    FeatureOp.ROLLING: _op_rolling,
    FeatureOp.DIFF: _op_diff,
    FeatureOp.EXPANDING: _op_expanding,
    FeatureOp.INTERACTION: _op_interaction,
    FeatureOp.RATIO: _op_ratio,
    FeatureOp.POLYNOMIAL: _op_polynomial,
    FeatureOp.LOG_TRANSFORM: _op_log_transform,
    FeatureOp.SQRT_TRANSFORM: _op_sqrt_transform,
    FeatureOp.BOXCOX: _op_boxcox,
    FeatureOp.BINNING: _op_binning,
    FeatureOp.ONE_HOT_ENCODE: _op_one_hot,
    FeatureOp.ORDINAL_ENCODE: _op_ordinal,
    FeatureOp.TARGET_ENCODE: _op_target_encode,
    FeatureOp.FREQUENCY_ENCODE: _op_frequency_encode,
    FeatureOp.HASH_ENCODE: _op_hash_encode,
    FeatureOp.TEXT_TFIDF: _op_text_tfidf,
    FeatureOp.TEXT_LENGTH: _op_text_length,
    FeatureOp.GEO_DISTANCE: _op_geo_distance,
    FeatureOp.AGGREGATE_BY_GROUP: _op_aggregate_by_group,
    FeatureOp.STANDARD_SCALE: _op_scale,
    FeatureOp.MINMAX_SCALE: _op_scale,
    FeatureOp.ROBUST_SCALE: _op_scale,
    FeatureOp.QUANTILE_TRANSFORM: _op_scale,
    FeatureOp.PCA: _op_reduce,
    FeatureOp.SVD: _op_reduce,
    FeatureOp.SELECT_K_BEST: _op_select_k_best,
    FeatureOp.DROP_CORRELATED: _op_drop_correlated,
    FeatureOp.VARIANCE_THRESHOLD: _op_variance_threshold,
}

#: Generation before transformation before selection: a scaling op that names no
#: columns must see every column an earlier op created, and a selection op must
#: run after both.
_OP_ORDER: dict[FeatureOp, int] = {
    FeatureOp.DATE_DECOMPOSE: 0,
    FeatureOp.CYCLICAL_ENCODE: 1,
    FeatureOp.LAG: 2,
    FeatureOp.DIFF: 3,
    FeatureOp.ROLLING: 4,
    FeatureOp.EXPANDING: 5,
    FeatureOp.AGGREGATE_BY_GROUP: 6,
    FeatureOp.TEXT_LENGTH: 7,
    FeatureOp.GEO_DISTANCE: 8,
    FeatureOp.RATIO: 9,
    FeatureOp.INTERACTION: 10,
    FeatureOp.POLYNOMIAL: 11,
    FeatureOp.LOG_TRANSFORM: 12,
    FeatureOp.SQRT_TRANSFORM: 13,
    FeatureOp.BOXCOX: 14,
    FeatureOp.BINNING: 15,
    FeatureOp.TARGET_ENCODE: 20,
    FeatureOp.FREQUENCY_ENCODE: 21,
    FeatureOp.HASH_ENCODE: 22,
    FeatureOp.ONE_HOT_ENCODE: 23,
    FeatureOp.ORDINAL_ENCODE: 24,
    FeatureOp.TEXT_TFIDF: 25,
    FeatureOp.DROP_CORRELATED: 30,
    FeatureOp.STANDARD_SCALE: 40,
    FeatureOp.MINMAX_SCALE: 41,
    FeatureOp.ROBUST_SCALE: 42,
    FeatureOp.QUANTILE_TRANSFORM: 43,
    FeatureOp.VARIANCE_THRESHOLD: 60,
    FeatureOp.SELECT_K_BEST: 61,
    FeatureOp.PCA: 70,
    FeatureOp.SVD: 71,
}


# ---------------------------------------------------------------------------
# Preprocessor assembly
# ---------------------------------------------------------------------------


class SafeDimensionReduction(BaseEstimator, TransformerMixin):
    """PCA/SVD that adapts its component count to the matrix it is given.

    ``n_components`` is chosen by an agent before the transformed width is known,
    and sklearn raises if it exceeds ``min(n_samples, n_features)``. Because this
    estimator is fitted deep inside the trainer, that exception would abort a run
    for a recoverable reason, so the request is clamped at fit time instead. A
    sparse input silently routes to :class:`~sklearn.decomposition.TruncatedSVD`,
    which is the only one of the two that accepts it.

    Args:
        n_components: Requested component count; an upper bound in practice.
        kind: ``"pca"`` or ``"svd"``.
        random_state: Seed for the randomised solvers.
    """

    def __init__(
        self, n_components: int = 10, kind: str = "pca", random_state: int = 42
    ) -> None:
        self.n_components = n_components
        self.kind = kind
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> SafeDimensionReduction:
        """Fit the underlying decomposition with a clamped component count."""
        from scipy import sparse as sp
        from sklearn.decomposition import PCA, TruncatedSVD

        n_samples, n_features = X.shape
        limit = max(1, min(int(n_features), max(1, int(n_samples) - 1)))
        components = max(1, min(int(self.n_components), limit))
        is_sparse = bool(sp.issparse(X))
        if is_sparse or self.kind == "svd":
            # TruncatedSVD needs n_components < n_features, so it cannot run at
            # all on a single column. A preceding select_k_best(k=1) or a PCA that
            # clamped to one component both produce exactly that, and raising here
            # would abort the run inside the trainer — the very thing this class
            # exists to prevent. One column is already one-dimensional, so there is
            # nothing to reduce: pass it through.
            if int(n_features) < 2:
                self.reducer_ = None
                self.kind_used_ = "passthrough"
                self.n_features_in_ = int(n_features)
                self.n_components_ = int(n_features)
                return self
            components = max(1, min(components, int(n_features) - 1))
            self.reducer_ = TruncatedSVD(
                n_components=components, random_state=self.random_state
            )
            self.kind_used_ = "svd"
        else:
            self.reducer_ = PCA(
                n_components=components, random_state=self.random_state
            )
            self.kind_used_ = "pca"
        self.reducer_.fit(X)
        self.n_features_in_ = int(n_features)
        self.n_components_ = int(components)
        return self

    def transform(self, X: Any) -> Any:
        """Project ``X`` onto the fitted components."""
        if self.reducer_ is None:  # too narrow to reduce; see fit()
            return X
        return self.reducer_.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the projected components, e.g. ``pca0 … pcaN``."""
        prefix = getattr(self, "kind_used_", self.kind)
        if prefix == "passthrough":
            prefix = self.kind
        return np.asarray(
            [f"{prefix}{i}" for i in range(getattr(self, "n_components_", 0))],
            dtype=object,
        )


def transformed_feature_names(preprocessor: Any) -> list[str]:
    """Best-effort names of the columns a *fitted* preprocessor emits.

    Returns an empty list when the pipeline cannot describe itself, which lets
    the explainer fall back to positional names instead of failing.
    """
    if preprocessor is None:
        return []
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - naming is a nicety, never a requirement
        logger.debug("preprocessor could not report feature names", exc_info=True)
        return []


def _build_transformer(spec: _Spec, ctx: _Ctx) -> Any:
    """Turn one :class:`_Spec` into an unfitted sklearn transformer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        KBinsDiscretizer,
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    options = spec.options
    numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    if spec.kind == "tfidf":
        return TfidfVectorizer(
            max_features=int(options.get("max_features", 100)),
            ngram_range=(1, int(options.get("ngram_max", 1))),
            min_df=int(options.get("min_df", 1)),
            sublinear_tf=bool(options.get("sublinear_tf", True)),
            stop_words=options.get("stop_words"),
            lowercase=True,
        )

    if spec.kind == "onehot":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                        max_categories=options.get("max_categories"),
                        min_frequency=options.get("min_frequency"),
                    ),
                ),
            ]
        )

    if spec.kind == "ordinal":
        categories = options.get("categories")
        return Pipeline(
            [
                (
                    "encode",
                    OrdinalEncoder(
                        categories=[list(categories)] if categories else "auto",
                        # Unseen and missing levels share code -1, which keeps the
                        # encoding total without inventing an ordering position.
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )

    steps: list[tuple[str, Any]] = [("impute", numeric_imputer)]
    if spec.kind == "scale_standard":
        steps.append(("scale", StandardScaler()))
    elif spec.kind == "scale_minmax":
        steps.append(("scale", MinMaxScaler()))
    elif spec.kind == "scale_robust":
        steps.append(("scale", RobustScaler()))
    elif spec.kind == "scale_quantile":
        steps.append(
            (
                "scale",
                QuantileTransformer(
                    output_distribution=str(options.get("output_distribution", "normal")),
                    random_state=ctx.state.config.random_state,
                ),
            )
        )
    elif spec.kind == "power":
        steps.append(
            ("scale", PowerTransformer(method=str(options.get("method", "yeo-johnson"))))
        )
    elif spec.kind == "bin":
        steps.append(
            (
                "bin",
                KBinsDiscretizer(
                    n_bins=int(options.get("n_bins", 5)),
                    encode="ordinal",
                    strategy=str(options.get("strategy", "quantile")),
                ),
            )
        )
    elif spec.kind.startswith("reduce_"):
        steps.append(("scale", StandardScaler()))
        steps.append(
            (
                "reduce",
                SafeDimensionReduction(
                    n_components=int(options.get("n_components", 10)),
                    kind=spec.kind.removeprefix("reduce_"),
                    random_state=ctx.state.config.random_state,
                ),
            )
        )
    elif spec.kind != "numeric":  # pragma: no cover - the registry is closed
        raise ExecutionError(f"unknown transformer spec kind {spec.kind!r}")
    return Pipeline(steps)


def _finalise(ctx: _Ctx) -> None:
    """Classify leftover columns, then assemble ``state.preprocessor``."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    features = [c for c in ctx.features if c in ctx.frame.columns]

    for column in list(features):
        spec = ctx.specs.get(column)
        series = ctx.frame[column]

        if pdt.is_datetime64_any_dtype(series):
            ctx.exclude(
                column,
                "raw datetime: decompose it into calendar parts to use it",
            )
            features.remove(column)
            continue

        if spec is not None:
            continue

        if pdt.is_bool_dtype(series) or (
            _is_numeric(series) and pdt.is_extension_array_dtype(series.dtype)
        ):
            # Nullable extension dtypes and booleans reach sklearn as object
            # arrays; float64 is the one representation every estimator accepts.
            ctx.frame[column] = series.astype("float64")
            ctx.specs[column] = _spec("numeric")
            continue

        if _is_numeric(series):
            ctx.specs[column] = _spec("numeric")
            continue

        if _is_stringish(series):
            levels = int(series.nunique(dropna=True))
            if levels <= 1:
                ctx.exclude(column, f"only {levels} distinct value(s): no variance")
                features.remove(column)
            elif levels <= MAX_ONEHOT_CARDINALITY:
                ctx.specs[column] = _spec("onehot")
            else:
                ctx.exclude(
                    column,
                    f"{levels} distinct text values exceeds the "
                    f"{MAX_ONEHOT_CARDINALITY}-level one-hot limit; a target, frequency, "
                    f"or hash encoding would be needed",
                )
                features.remove(column)
            continue

        ctx.exclude(column, f"unsupported dtype {series.dtype}")
        features.remove(column)

    if not features:
        # Better a numeric-only model than no model: fall back before giving up.
        features = [
            c
            for c in ctx.frame.columns
            if c != ctx.target and _numeric_view(ctx, c) is not None
        ]
        for column in features:
            ctx.specs[column] = _spec("numeric")
            ctx.excluded.pop(column, None)
        if features:
            ctx.state.add_warning(
                "features: the plan left no usable feature columns; fell back to the "
                f"{len(features)} raw numeric column(s)"
            )
        else:
            raise ExecutionError(
                "apply_feature_plan: no usable feature columns remain after the plan; "
                f"excluded {list(ctx.excluded)[:12]}"
            )
    ctx.features = features

    # sklearn's encoders recognise np.nan as missing but treat pandas' pd.NA as a
    # category, which would create a phantom level. Normalise once, here.
    for column in features:
        series = ctx.frame[column]
        if _is_stringish(series):
            as_object = _text_view(series).astype("object")
            ctx.frame[column] = as_object.where(series.notna(), other=np.nan)

    branches: list[tuple[str, Any, Any]] = []
    grouped: dict[_Spec, list[str]] = {}
    for column in features:
        spec = ctx.specs.get(column, _spec("numeric"))
        if spec.per_column:
            selector = column if spec.wants_1d else [column]
            branches.append(
                (f"{spec.kind}_{_slug(column)}", _build_transformer(spec, ctx), selector)
            )
        else:
            grouped.setdefault(spec, []).append(column)

    for index, (spec, columns) in enumerate(grouped.items()):
        branches.append((f"{spec.kind}_{index}", _build_transformer(spec, ctx), columns))

    has_sparse_branch = any(
        ctx.specs.get(c, _spec("numeric")).kind == "tfidf" for c in features
    )
    column_transformer = ColumnTransformer(
        branches,
        remainder="drop",
        # Dense output keeps every downstream estimator (including PCA and SHAP)
        # usable; only a TF-IDF branch justifies the memory of staying sparse.
        sparse_threshold=0.3 if has_sparse_branch else 0.0,
        # Prefixes are noise in a report, but TF-IDF tokens can collide with real
        # column names and duplicate output names break feature naming entirely.
        verbose_feature_names_out=has_sparse_branch,
    )

    steps: list[tuple[str, Any]] = [("columns", column_transformer)]
    for name, step in sorted(ctx.post_steps, key=lambda item: _POST_ORDER.get(item[0], 50)):
        if any(existing == name for existing, _ in steps):
            ctx.state.add_warning(
                f"features: pipeline step '{name}' was requested more than once; "
                f"only the first was kept"
            )
            continue
        steps.append((name, step))
    ctx.state.preprocessor = Pipeline(steps)
    ctx.state.extras["feature_preprocessor_steps"] = [
        f"{name}: {type(step).__name__}" for name, step in steps
    ]
    ctx.state.extras["feature_column_specs"] = {
        column: ctx.specs.get(column, _spec("numeric")).kind for column in features
    }


_POST_ORDER = {"variance_threshold": 0, "select_k_best": 1, "pca_reduce": 2, "svd_reduce": 3}


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower() or "column"


# ---------------------------------------------------------------------------
# Parameter coercion
# ---------------------------------------------------------------------------


def _int_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out or [1]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return [1]


def _str_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    return [text] if text else ["mean"]
