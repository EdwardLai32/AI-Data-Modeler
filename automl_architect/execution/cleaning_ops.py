"""Deterministic execution of a :class:`~automl_architect.core.schemas.CleaningPlan`.

The Cleaning Agent decides *what* to do and *why*; this module is the only place
that touches the dataframe. Two properties make the result auditable:

1.  **Order is imposed here, not by the agent.** An agent may list its decisions
    in any order — narrating imputation before it mentions dropping the useless
    column it would have imputed. Applying them in the listed order produces
    correct-looking but wrong statistics (a mode computed over rows that a later
    decision deletes). :data:`ACTION_ORDER` pins the safe sequence: target-row
    drops, column drops, duplicates, dtype/datetime normalisation, imputation,
    then outlier handling.
2.  **Fitted statistics are kept, not recomputed.** Every fill value lands in
    ``state.extras['cleaning_fill_values']``. Recomputing a median at scoring
    time is the classic silent train/serve skew: the code looks identical and the
    numbers are not.

Every decision is applied defensively. A decision that would empty the frame,
delete the target, or impute the target is refused with a warning rather than
executed, and any unexpected exception degrades to a skipped action so one bad
instruction cannot abort the run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..core.errors import ExecutionError
from ..core.schemas import (
    AgentName,
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    EventKind,
    MissingStrategy,
    params_to_dict,
)
from ..core.state import RunState

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_ORDER",
    "ITERATIVE_MAX_COLUMNS",
    "ITERATIVE_MAX_ROWS",
    "KNN_MAX_COLUMNS",
    "KNN_MAX_ROWS",
    "MIN_ROWS_AFTER_ACTION",
    "OUTLIER_REMOVAL_REFUSE_FRACTION",
    "OUTLIER_REMOVAL_WARN_FRACTION",
    "apply_cleaning_plan",
]

# --- guards ---------------------------------------------------------------
# Multivariate imputation is O(n_rows * n_cols^2) at best and reads the whole
# frame into a dense float matrix. Above these sizes the wall-clock cost dwarfs
# any accuracy gain over a median, so we degrade instead of stalling the run.
KNN_MAX_ROWS = 50_000
KNN_MAX_COLUMNS = 40
ITERATIVE_MAX_ROWS = 20_000
ITERATIVE_MAX_COLUMNS = 25

#: Removing more of the table than this as "outliers" is a modelling decision,
#: not cleaning; it proceeds but is flagged loudly.
OUTLIER_REMOVAL_WARN_FRACTION = 0.05
#: Beyond this the action is refused outright.
OUTLIER_REMOVAL_REFUSE_FRACTION = 0.50

#: A cleaning step must never leave fewer rows than this.
MIN_ROWS_AFTER_ACTION = 10

_NULL_TOKENS = {
    "",
    "-",
    "--",
    "?",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "nil",
    "unknown",
    "missing",
}

#: Safe application order, independent of the order the agent listed decisions.
ACTION_ORDER: dict[CleaningAction, int] = {
    CleaningAction.DROP_ROWS_MISSING_TARGET: 0,
    CleaningAction.DROP_LEAKAGE_COLUMN: 10,
    CleaningAction.DROP_CONSTANT_COLUMN: 11,
    CleaningAction.DROP_COLUMN: 12,
    CleaningAction.DROP_DUPLICATE_ROWS: 20,
    CleaningAction.STRIP_WHITESPACE: 30,
    CleaningAction.NORMALISE_CATEGORIES: 31,
    CleaningAction.PARSE_DATETIME: 32,
    CleaningAction.CAST_DTYPE: 33,
    CleaningAction.IMPUTE_MISSING: 40,
    CleaningAction.CLIP_OUTLIERS: 50,
    CleaningAction.REMOVE_OUTLIER_ROWS: 51,
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Mutable working state for one cleaning pass."""

    state: RunState
    frame: pd.DataFrame
    target: str | None
    fill_values: dict[str, Any] = field(default_factory=dict)
    clip_bounds: dict[str, list[float | None]] = field(default_factory=dict)
    imputers: list[dict[str, Any]] = field(default_factory=list)
    dtype_casts: dict[str, str] = field(default_factory=dict)
    datetime_columns: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    def refuse(self, action: CleaningAction, reason: str) -> None:
        self.state.add_warning(f"cleaning: refused {action.value} — {reason}")

    def record(self, action: CleaningAction, message: str, rationale: str = "") -> None:
        line = f"{action.value}: {message}"
        if rationale:
            line = f"{line} — {_trim(rationale)}"
        self.state.applied_cleaning.append(line)
        self.state.bus.emit(
            EventKind.LOG,
            line,
            agent=AgentName.CLEANING,
            payload={"action": action.value, "detail": message},
        )

    def note_dropped(self, columns: list[str]) -> None:
        for col in columns:
            if col not in self.state.dropped_columns:
                self.state.dropped_columns.append(col)


def _trim(text: str, limit: int = 180) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _is_stringish(series: pd.Series) -> bool:
    """True for object/str/categorical columns.

    pandas 3 gives CSV text columns the ``str`` dtype rather than ``object``, so
    an ``is_object_dtype`` check alone silently misses every text column.
    """
    if _is_categorical(series):
        return True
    return bool(pdt.is_string_dtype(series) or pdt.is_object_dtype(series))


def _is_numeric(series: pd.Series) -> bool:
    return bool(pdt.is_numeric_dtype(series)) and not pdt.is_bool_dtype(series)


def _numeric_columns(frame: pd.DataFrame, exclude: str | None = None) -> list[str]:
    return [
        c for c in frame.columns if c != exclude and _is_numeric(frame[c])
    ]


def _as_python(value: Any) -> Any:
    """Coerce a numpy/pandas scalar to a JSON-friendly Python scalar."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _mode_of(series: pd.Series) -> Any:
    modes = series.dropna()
    if modes.empty:
        return None
    counts = modes.value_counts()
    if counts.empty:
        return None
    return counts.index[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_cleaning_plan(state: RunState, plan: CleaningPlan) -> Any:
    """Apply every decision in ``plan`` to the run's working dataframe.

    Args:
        state: The run blackboard. ``state.working_df`` (falling back to
            ``state.raw_df``) supplies the input frame and receives the result.
        plan: The Cleaning Agent's decisions. Order is re-imposed by
            :data:`ACTION_ORDER`; unknown column references are filtered out.

    Returns:
        The cleaned :class:`pandas.DataFrame`, which is also assigned to
        ``state.working_df``.

    Raises:
        ExecutionError: If no dataframe has been loaded yet.
    """
    source = state.working_df if state.working_df is not None else state.raw_df
    if source is None:
        raise ExecutionError("apply_cleaning_plan: no dataframe on the run state")

    frame = source.copy()
    if frame.index.has_duplicates:
        # Later ops assign engineered columns by index alignment; duplicate
        # labels would fan those assignments out across unrelated rows.
        frame = frame.reset_index(drop=True)
        state.add_warning("cleaning: duplicate index labels found; index was reset")

    ctx = _Ctx(state=state, frame=frame, target=state.target)
    rows_before, cols_before = ctx.frame.shape

    decisions = [d for d in plan.decisions if isinstance(d, CleaningDecision)]
    decisions.sort(key=lambda d: ACTION_ORDER.get(d.action, 99))

    # The target is never imputed and never modelled from null rows, whether or
    # not the agent thought to say so.
    explicit_target_drop = next(
        (d for d in decisions if d.action is CleaningAction.DROP_ROWS_MISSING_TARGET),
        None,
    )
    _drop_rows_missing_target(
        ctx,
        rationale=explicit_target_drop.rationale if explicit_target_drop else "",
        implicit=explicit_target_drop is None,
    )

    for decision in decisions:
        if decision is explicit_target_drop:
            continue
        try:
            _dispatch(ctx, decision)
        except Exception as exc:  # noqa: BLE001 - one bad decision must not abort
            logger.exception("cleaning action %s failed", decision.action)
            state.add_warning(
                f"cleaning: {decision.action.value} on "
                f"{decision.columns or 'table'} failed ({exc}); skipped"
            )

    # Columns the plan flagged at the top level rather than as a decision.
    extra_drops = [c for c in plan.columns_to_drop if c in ctx.frame.columns]
    if extra_drops:
        _drop_columns(ctx, extra_drops, CleaningAction.DROP_COLUMN, "plan.columns_to_drop")

    state.working_df = ctx.frame
    state.extras["cleaning_fill_values"] = ctx.fill_values
    state.extras["cleaning_clip_bounds"] = ctx.clip_bounds
    state.extras["cleaning_dtype_casts"] = ctx.dtype_casts
    state.extras["cleaning_datetime_columns"] = ctx.datetime_columns
    if ctx.imputers:
        # Fitted multivariate imputers cannot be expressed as scalar fill values;
        # they are kept as live objects for the scoring path.
        state.extras["cleaning_imputers"] = ctx.imputers

    rows_after, cols_after = ctx.frame.shape
    state.bus.emit(
        EventKind.LOG,
        f"cleaning applied: {rows_before}x{cols_before} -> {rows_after}x{cols_after}, "
        f"{len(state.applied_cleaning)} action(s)",
        agent=AgentName.CLEANING,
        payload={
            "rows_before": rows_before,
            "rows_after": rows_after,
            "columns_before": cols_before,
            "columns_after": cols_after,
        },
    )
    return ctx.frame


def _dispatch(ctx: _Ctx, decision: CleaningDecision) -> None:
    action = decision.action
    params = params_to_dict(decision.parameters)
    columns = _resolve_columns(ctx, decision)

    if action in (
        CleaningAction.DROP_COLUMN,
        CleaningAction.DROP_CONSTANT_COLUMN,
        CleaningAction.DROP_LEAKAGE_COLUMN,
    ):
        _drop_columns(ctx, columns, action, decision.rationale)
    elif action is CleaningAction.DROP_DUPLICATE_ROWS:
        _drop_duplicates(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.STRIP_WHITESPACE:
        _strip_whitespace(ctx, columns, decision.rationale)
    elif action is CleaningAction.NORMALISE_CATEGORIES:
        _normalise_categories(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.PARSE_DATETIME:
        _parse_datetime(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.CAST_DTYPE:
        _cast_dtype(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.IMPUTE_MISSING:
        _impute(ctx, columns, decision, params)
    elif action is CleaningAction.CLIP_OUTLIERS:
        _clip_outliers(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.REMOVE_OUTLIER_ROWS:
        _remove_outlier_rows(ctx, columns, params, decision.rationale)
    elif action is CleaningAction.DROP_ROWS_MISSING_TARGET:
        _drop_rows_missing_target(ctx, decision.rationale, implicit=False)
    else:  # pragma: no cover - CleaningAction is exhaustively handled above
        ctx.state.add_warning(f"cleaning: unhandled action {action}")


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _resolve_columns(ctx: _Ctx, decision: CleaningDecision) -> list[str]:
    """Filter agent-named columns against the frame, defaulting sensibly.

    An empty ``columns`` list means "the whole table"; what that expands to is
    action-specific, and for anything that writes values the target column is
    always excluded.
    """
    known = set(ctx.frame.columns)
    named = [c for c in decision.columns if c in known]
    unknown = [c for c in decision.columns if c not in known]
    if unknown:
        ctx.state.add_warning(
            f"cleaning: {decision.action.value} referenced unknown column(s) "
            f"{unknown[:8]}{'...' if len(unknown) > 8 else ''}; ignoring them"
        )
    if named:
        return _protect_target(ctx, named, decision.action)
    return _default_columns(ctx, decision.action)


def _protect_target(ctx: _Ctx, columns: list[str], action: CleaningAction) -> list[str]:
    """Strip the target from any action that would rewrite or delete it."""
    if not ctx.target or ctx.target not in columns:
        return columns
    mutating = action in {
        CleaningAction.IMPUTE_MISSING,
        CleaningAction.DROP_COLUMN,
        CleaningAction.DROP_CONSTANT_COLUMN,
        CleaningAction.DROP_LEAKAGE_COLUMN,
        CleaningAction.CLIP_OUTLIERS,
    }
    if not mutating:
        return columns
    ctx.state.add_warning(
        f"cleaning: refused {action.value} on the target column "
        f"'{ctx.target}' — the target must not be imputed, clipped, or dropped"
    )
    return [c for c in columns if c != ctx.target]


def _default_columns(ctx: _Ctx, action: CleaningAction) -> list[str]:
    frame = ctx.frame
    target = ctx.target
    if action is CleaningAction.IMPUTE_MISSING:
        return [
            c
            for c in frame.columns
            if c != target and bool(frame[c].isna().any())
        ]
    if action is CleaningAction.STRIP_WHITESPACE:
        return [c for c in frame.columns if _is_stringish(frame[c])]
    if action is CleaningAction.NORMALISE_CATEGORIES:
        return [
            c
            for c in frame.columns
            if c != target and _is_stringish(frame[c])
        ]
    if action in (CleaningAction.CLIP_OUTLIERS, CleaningAction.REMOVE_OUTLIER_ROWS):
        return _numeric_columns(frame, exclude=target)
    if action is CleaningAction.DROP_CONSTANT_COLUMN:
        constant = list(ctx.state.profile.constant_columns) if ctx.state.profile else []
        return [
            c
            for c in (constant or frame.columns)
            if c in frame.columns and c != target and frame[c].nunique(dropna=False) <= 1
        ]
    return []


# ---------------------------------------------------------------------------
# Row / column removal
# ---------------------------------------------------------------------------


def _drop_rows_missing_target(
    ctx: _Ctx, rationale: str = "", *, implicit: bool = False
) -> None:
    target = ctx.target
    if not target:
        if not implicit:
            ctx.state.add_warning(
                "cleaning: drop_rows_missing_target requested but no target is set"
            )
        return
    if target not in ctx.frame.columns:
        if not implicit:
            ctx.state.add_warning(
                f"cleaning: target '{target}' is not in the frame; cannot drop null-target rows"
            )
        return

    mask = ctx.frame[target].isna()
    n_missing = int(mask.sum())
    if n_missing == 0:
        return
    remaining = ctx.n_rows - n_missing
    if remaining < MIN_ROWS_AFTER_ACTION:
        ctx.refuse(
            CleaningAction.DROP_ROWS_MISSING_TARGET,
            f"only {remaining} row(s) have a non-null target; the frame would be unusable",
        )
        return

    ctx.frame = ctx.frame.loc[~mask]
    reason = rationale or (
        "a row with no label cannot be trained on or scored, and imputing a "
        "target invents ground truth"
    )
    ctx.record(
        CleaningAction.DROP_ROWS_MISSING_TARGET,
        f"dropped {n_missing} row(s) with a missing '{target}' "
        f"({n_missing / max(1, n_missing + remaining):.1%} of the table)"
        + (" [applied automatically]" if implicit else ""),
        reason,
    )


def _drop_columns(
    ctx: _Ctx, columns: list[str], action: CleaningAction, rationale: str
) -> None:
    present = [c for c in dict.fromkeys(columns) if c in ctx.frame.columns]
    if ctx.target and ctx.target in present:
        ctx.refuse(action, f"'{ctx.target}' is the target column")
        present = [c for c in present if c != ctx.target]
    if not present:
        return
    if len(present) >= len(ctx.frame.columns):
        ctx.refuse(action, "that would drop every column")
        return

    ctx.frame = ctx.frame.drop(columns=present)
    ctx.note_dropped(present)
    ctx.record(action, f"dropped column(s) {present}", rationale)


def _drop_duplicates(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    subset = columns or None
    keep = params.get("keep", "first")
    if keep not in ("first", "last", False):
        keep = "first"
    mask = ctx.frame.duplicated(subset=subset, keep=keep)
    n_dupes = int(mask.sum())
    if n_dupes == 0:
        return
    if ctx.n_rows - n_dupes < MIN_ROWS_AFTER_ACTION:
        ctx.refuse(
            CleaningAction.DROP_DUPLICATE_ROWS,
            f"removing {n_dupes} duplicate row(s) would leave "
            f"{ctx.n_rows - n_dupes} row(s)",
        )
        return
    ctx.frame = ctx.frame.loc[~mask]
    ctx.record(
        CleaningAction.DROP_DUPLICATE_ROWS,
        f"removed {n_dupes} duplicate row(s)"
        + (f" keyed on {subset}" if subset else " (full-row match)"),
        rationale,
    )


# ---------------------------------------------------------------------------
# Text / dtype normalisation
# ---------------------------------------------------------------------------


def _string_view(series: pd.Series) -> pd.Series:
    """A string-dtype view of a text or categorical column."""
    if _is_categorical(series):
        return series.astype("object").astype("str")
    return series.astype("str")


def _strip_whitespace(ctx: _Ctx, columns: list[str], rationale: str) -> None:
    touched: list[str] = []
    for col in columns:
        series = ctx.frame[col]
        if not _is_stringish(series):
            continue
        was_categorical = _is_categorical(series)
        stripped = _string_view(series).str.strip()
        stripped = stripped.where(series.notna(), other=pd.NA)
        if was_categorical:
            stripped = stripped.astype("category")
        if not stripped.equals(series):
            ctx.frame[col] = stripped
            touched.append(col)
    if touched:
        ctx.record(
            CleaningAction.STRIP_WHITESPACE,
            f"trimmed leading/trailing whitespace in {touched}",
            rationale,
        )


def _normalise_categories(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    lower = bool(params.get("lowercase", params.get("lower", True)))
    collapse = bool(params.get("collapse_whitespace", True))
    nullify = bool(params.get("nullify_placeholders", True))
    mapping = params.get("mapping") or params.get("replacements")
    mapping = mapping if isinstance(mapping, dict) else {}

    touched: list[str] = []
    for col in columns:
        series = ctx.frame[col]
        if not _is_stringish(series):
            continue
        was_categorical = _is_categorical(series)
        text = _string_view(series).str.strip()
        if lower:
            text = text.str.lower()
        if collapse:
            text = text.map(
                lambda v: re.sub(r"\s+", " ", v) if isinstance(v, str) else v
            )
        if mapping:
            text = text.replace({str(k): str(v) for k, v in mapping.items()})
        if nullify:
            text = text.where(~text.str.lower().isin(_NULL_TOKENS), other=pd.NA)
        text = text.where(series.notna(), other=pd.NA)
        if was_categorical:
            text = text.astype("category")
        before = int(series.nunique(dropna=True))
        after = int(text.nunique(dropna=True))
        ctx.frame[col] = text
        touched.append(f"{col} ({before}->{after} levels)")
    if touched:
        ctx.record(
            CleaningAction.NORMALISE_CATEGORIES,
            "normalised category spelling for " + ", ".join(touched),
            rationale,
        )


def _parse_datetime(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    fmt = params.get("format") or params.get("fmt")
    dayfirst = bool(params.get("dayfirst", False))
    unit = params.get("unit")
    for col in columns:
        series = ctx.frame[col]
        if pdt.is_datetime64_any_dtype(series):
            if col not in ctx.datetime_columns:
                ctx.datetime_columns.append(col)
            continue
        parsed = _to_datetime(series, fmt=fmt, dayfirst=dayfirst, unit=unit)
        if parsed is None:
            ctx.state.add_warning(
                f"cleaning: could not parse '{col}' as a datetime; left unchanged"
            )
            continue
        n_failed = int(parsed.isna().sum() - series.isna().sum())
        ctx.frame[col] = parsed
        ctx.datetime_columns.append(col)
        detail = f"parsed '{col}' to datetime64"
        if n_failed > 0:
            detail += f" ({n_failed} unparseable value(s) became NaT)"
        ctx.record(CleaningAction.PARSE_DATETIME, detail, rationale)


def _to_datetime(
    series: pd.Series, *, fmt: str | None, dayfirst: bool, unit: str | None
) -> pd.Series | None:
    attempts: list[dict[str, Any]] = []
    if unit:
        attempts.append({"unit": str(unit)})
    if fmt:
        attempts.append({"format": str(fmt)})
    attempts.append({"format": "mixed", "dayfirst": dayfirst})
    attempts.append({"dayfirst": dayfirst})
    non_null = int(series.notna().sum())
    for kwargs in attempts:
        try:
            parsed = pd.to_datetime(series, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if non_null == 0 or int(parsed.notna().sum()) > 0:
            return parsed
    return None


_DTYPE_ALIASES = {
    "int": "int64",
    "integer": "int64",
    "float": "float64",
    "double": "float64",
    "str": "str",
    "string": "str",
    "text": "str",
    "bool": "boolean",
    "boolean": "boolean",
    "cat": "category",
    "categorical": "category",
}


def _cast_dtype(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    requested = params.get("dtype") or params.get("to") or params.get("type")
    for col in columns:
        target_dtype = str(params.get(col, requested) or "").strip().lower()
        if not target_dtype:
            ctx.state.add_warning(
                f"cleaning: cast_dtype for '{col}' has no 'dtype' parameter; skipped"
            )
            continue
        resolved = _DTYPE_ALIASES.get(target_dtype, target_dtype)
        if resolved.startswith("datetime"):
            _parse_datetime(ctx, [col], params, rationale)
            continue
        series = ctx.frame[col]
        try:
            if resolved in ("int64", "Int64", "float64", "float32", "int32"):
                numeric = pd.to_numeric(series, errors="coerce")
                n_failed = int(numeric.isna().sum() - series.isna().sum())
                if resolved.lower().startswith("int"):
                    # A nullable integer is the only honest landing place when
                    # coercion produced NaNs.
                    cast = (
                        numeric.astype("Int64")
                        if numeric.isna().any()
                        else numeric.astype("int64")
                    )
                else:
                    cast = numeric.astype(resolved)
                if n_failed:
                    ctx.state.add_warning(
                        f"cleaning: cast '{col}' to {resolved} coerced "
                        f"{n_failed} unparseable value(s) to null"
                    )
            else:
                cast = series.astype(resolved)
        except (ValueError, TypeError) as exc:
            ctx.state.add_warning(
                f"cleaning: could not cast '{col}' to {resolved} ({exc}); left as "
                f"{series.dtype}"
            )
            continue
        ctx.frame[col] = cast
        ctx.dtype_casts[col] = resolved
        ctx.record(
            CleaningAction.CAST_DTYPE,
            f"cast '{col}' from {series.dtype} to {cast.dtype}",
            rationale,
        )


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def _default_strategy(series: pd.Series) -> MissingStrategy:
    if _is_numeric(series):
        return MissingStrategy.MEDIAN
    if pdt.is_datetime64_any_dtype(series):
        return MissingStrategy.FORWARD_FILL
    return MissingStrategy.MODE


def _impute(
    ctx: _Ctx,
    columns: list[str],
    decision: CleaningDecision,
    params: dict[str, Any],
) -> None:
    strategy = decision.strategy
    columns = [c for c in columns if c in ctx.frame.columns]
    if not columns:
        return

    if strategy is MissingStrategy.LEAVE_AS_IS:
        ctx.record(
            CleaningAction.IMPUTE_MISSING,
            f"left missing values in {columns} untouched",
            decision.rationale,
        )
        return
    if strategy is MissingStrategy.DROP_COLUMN:
        _drop_columns(ctx, columns, CleaningAction.DROP_COLUMN, decision.rationale)
        return
    if strategy is MissingStrategy.DROP_ROWS:
        _drop_rows_with_missing(ctx, columns, decision.rationale)
        return
    if strategy in (MissingStrategy.KNN, MissingStrategy.ITERATIVE):
        _multivariate_impute(ctx, columns, strategy, params, decision.rationale)
        return

    for col in columns:
        series = ctx.frame[col]
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            continue
        chosen = strategy or _default_strategy(series)
        _impute_one(ctx, col, chosen, params, decision.rationale, n_missing)


def _impute_one(
    ctx: _Ctx,
    col: str,
    strategy: MissingStrategy,
    params: dict[str, Any],
    rationale: str,
    n_missing: int,
) -> None:
    series = ctx.frame[col]
    numeric = _is_numeric(series)

    if strategy in (MissingStrategy.MEAN, MissingStrategy.MEDIAN) and not numeric:
        ctx.state.add_warning(
            f"cleaning: {strategy.value} imputation is undefined for non-numeric "
            f"'{col}' ({series.dtype}); used the mode instead"
        )
        strategy = MissingStrategy.MODE

    if strategy in (
        MissingStrategy.FORWARD_FILL,
        MissingStrategy.BACKWARD_FILL,
        MissingStrategy.INTERPOLATE,
    ):
        _sequential_impute(ctx, col, strategy, params, rationale, n_missing)
        return

    if strategy is MissingStrategy.MEAN:
        fill: Any = _as_python(series.mean())
        label = f"mean {fill:.6g}" if isinstance(fill, (int, float)) else "mean"
    elif strategy is MissingStrategy.MEDIAN:
        fill = _as_python(series.median())
        label = f"median {fill:.6g}" if isinstance(fill, (int, float)) else "median"
    elif strategy is MissingStrategy.MODE:
        fill = _as_python(_mode_of(series))
        label = f"mode {fill!r}"
    elif strategy is MissingStrategy.MISSING_CATEGORY:
        fill = str(params.get("fill_value", params.get("value", "MISSING")))
        label = f"explicit level {fill!r}"
    elif strategy is MissingStrategy.CONSTANT:
        fill = params.get("fill_value", params.get("value", 0 if numeric else "MISSING"))
        label = f"constant {fill!r}"
    else:  # pragma: no cover - every MissingStrategy is handled
        ctx.state.add_warning(f"cleaning: unhandled missing strategy {strategy}")
        return

    if fill is None:
        ctx.state.add_warning(
            f"cleaning: '{col}' has no usable {strategy.value} (all values missing?); "
            f"left as-is"
        )
        return

    if _is_categorical(series) and fill not in set(series.cat.categories):
        ctx.frame[col] = series.cat.add_categories([fill]).fillna(fill)
    else:
        ctx.frame[col] = series.fillna(fill)
    ctx.fill_values[col] = _as_python(fill)
    ctx.record(
        CleaningAction.IMPUTE_MISSING,
        f"filled {n_missing} missing value(s) in '{col}' with the {label}",
        rationale,
    )


def _sequential_impute(
    ctx: _Ctx,
    col: str,
    strategy: MissingStrategy,
    params: dict[str, Any],
    rationale: str,
    n_missing: int,
) -> None:
    """Fill along the time axis, per group when the data has entities.

    Forward-filling a frame in row order is only meaningful if row order *is*
    time order, which is why the temporal column is applied first and the group
    column bounds the fill: without the grouping, one entity's last observation
    leaks into the next entity's first.
    """
    order_col = _temporal_column(ctx)
    group_col = _group_column(ctx)
    frame = ctx.frame
    ordered_index = (
        frame.sort_values(order_col, kind="stable").index
        if order_col
        else frame.index
    )
    series = frame.loc[ordered_index, col]

    limit = params.get("limit")
    limit = int(limit) if isinstance(limit, (int, float)) and limit else None

    def _fill(chunk: pd.Series) -> pd.Series:
        if strategy is MissingStrategy.FORWARD_FILL:
            return chunk.ffill(limit=limit)
        if strategy is MissingStrategy.BACKWARD_FILL:
            return chunk.bfill(limit=limit)
        method = str(params.get("method", "linear"))
        try:
            return chunk.interpolate(method=method, limit=limit, limit_direction="both")
        except (ValueError, TypeError):
            return chunk.ffill(limit=limit).bfill(limit=limit)

    if strategy is MissingStrategy.INTERPOLATE and not _is_numeric(series):
        ctx.state.add_warning(
            f"cleaning: interpolation needs a numeric column; '{col}' is "
            f"{series.dtype}, forward-filling instead"
        )
        strategy = MissingStrategy.FORWARD_FILL

    if group_col and group_col in frame.columns and group_col != col:
        groups = frame.loc[ordered_index, group_col]
        filled = series.groupby(groups, sort=False, observed=True).transform(_fill)
    else:
        filled = _fill(series)

    ctx.frame[col] = filled.reindex(frame.index)
    remaining = int(ctx.frame[col].isna().sum())
    detail = (
        f"{strategy.value} on '{col}' filled {n_missing - remaining} of {n_missing} "
        f"missing value(s)"
    )
    if order_col:
        detail += f", ordered by '{order_col}'"
    if group_col and group_col in frame.columns and group_col != col:
        detail += f" within each '{group_col}'"
    ctx.record(CleaningAction.IMPUTE_MISSING, detail, rationale)

    if remaining:
        # Edge rows have nothing to carry forward from; close the gap with a
        # static statistic so the column leaves cleaning complete.
        fallback = (
            MissingStrategy.MEDIAN if _is_numeric(ctx.frame[col]) else MissingStrategy.MODE
        )
        _impute_one(ctx, col, fallback, params, "residual gaps after sequential fill", remaining)


def _drop_rows_with_missing(ctx: _Ctx, columns: list[str], rationale: str) -> None:
    mask = ctx.frame[columns].isna().any(axis=1)
    n_drop = int(mask.sum())
    if n_drop == 0:
        return
    remaining = ctx.n_rows - n_drop
    if remaining < MIN_ROWS_AFTER_ACTION:
        ctx.refuse(
            CleaningAction.IMPUTE_MISSING,
            f"dropping rows missing {columns} would leave {remaining} row(s)",
        )
        return
    fraction = n_drop / max(1, ctx.n_rows)
    if fraction > 0.25:
        ctx.state.add_warning(
            f"cleaning: dropping rows missing {columns} removes {fraction:.1%} of the "
            f"table; imputation would preserve more signal"
        )
    ctx.frame = ctx.frame.loc[~mask]
    ctx.record(
        CleaningAction.IMPUTE_MISSING,
        f"dropped {n_drop} row(s) ({fraction:.1%}) missing {columns}",
        rationale,
    )


def _multivariate_impute(
    ctx: _Ctx,
    columns: list[str],
    strategy: MissingStrategy,
    params: dict[str, Any],
    rationale: str,
) -> None:
    """KNN / iterative imputation, guarded by dataset size."""
    targets = [c for c in columns if _is_numeric(ctx.frame[c])]
    non_numeric = [c for c in columns if c not in targets]
    if non_numeric:
        ctx.state.add_warning(
            f"cleaning: {strategy.value} imputation only applies to numeric columns; "
            f"{non_numeric} fell back to the mode"
        )
        for col in non_numeric:
            n_missing = int(ctx.frame[col].isna().sum())
            if n_missing:
                _impute_one(ctx, col, MissingStrategy.MODE, params, rationale, n_missing)
    if not targets:
        return

    donors = _numeric_columns(ctx.frame, exclude=ctx.target)
    max_rows, max_cols = (
        (KNN_MAX_ROWS, KNN_MAX_COLUMNS)
        if strategy is MissingStrategy.KNN
        else (ITERATIVE_MAX_ROWS, ITERATIVE_MAX_COLUMNS)
    )
    if ctx.n_rows > max_rows or len(donors) > max_cols:
        ctx.state.add_warning(
            f"cleaning: {strategy.value} imputation skipped — {ctx.n_rows} rows x "
            f"{len(donors)} numeric columns exceeds the guard "
            f"({max_rows} x {max_cols}); used the median/mode instead"
        )
        for col in targets:
            n_missing = int(ctx.frame[col].isna().sum())
            if n_missing:
                _impute_one(ctx, col, MissingStrategy.MEDIAN, params, rationale, n_missing)
        return

    try:
        if strategy is MissingStrategy.KNN:
            from sklearn.impute import KNNImputer

            n_neighbors = int(params.get("n_neighbors", params.get("k", 5)))
            n_neighbors = max(1, min(n_neighbors, max(1, ctx.n_rows - 1)))
            imputer: Any = KNNImputer(
                n_neighbors=n_neighbors, weights=str(params.get("weights", "uniform"))
            )
            label = f"KNN (k={n_neighbors})"
        else:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(
                max_iter=int(params.get("max_iter", 10)),
                random_state=ctx.state.config.random_state,
            )
            label = "iterative (MICE-style) regression"
    except ImportError as exc:  # pragma: no cover - sklearn is a hard dependency
        ctx.state.add_warning(
            f"cleaning: {strategy.value} imputation unavailable ({exc}); using the median"
        )
        for col in targets:
            n_missing = int(ctx.frame[col].isna().sum())
            if n_missing:
                _impute_one(ctx, col, MissingStrategy.MEDIAN, params, rationale, n_missing)
        return

    block = ctx.frame[donors].astype("float64")
    n_missing_before = {c: int(ctx.frame[c].isna().sum()) for c in targets}
    filled = pd.DataFrame(
        imputer.fit_transform(block), columns=donors, index=ctx.frame.index
    )
    for col in targets:
        original = ctx.frame[col]
        ctx.frame[col] = filled[col].astype(original.dtype, errors="ignore")
        # A scalar fallback keeps the scoring path working even if the fitted
        # imputer object cannot be shipped with the model.
        ctx.fill_values.setdefault(col, _as_python(original.median()))
    ctx.imputers.append(
        {"kind": strategy.value, "columns": list(donors), "imputer": imputer}
    )
    ctx.record(
        CleaningAction.IMPUTE_MISSING,
        f"imputed {sum(n_missing_before.values())} missing value(s) across {targets} "
        f"with {label} over {len(donors)} numeric donor column(s)",
        rationale,
    )


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def _iqr_bounds(
    ctx: _Ctx, col: str, params: dict[str, Any]
) -> tuple[float | None, float | None, str]:
    """Bounds for one column: explicit params, then the profile, then the frame."""
    lower = params.get("lower", params.get("lower_bound"))
    upper = params.get("upper", params.get("upper_bound"))
    if isinstance(lower, (int, float)) or isinstance(upper, (int, float)):
        return (
            float(lower) if isinstance(lower, (int, float)) else None,
            float(upper) if isinstance(upper, (int, float)) else None,
            "explicit bounds",
        )

    profile = ctx.state.profile
    if profile is not None:
        column_profile = profile.column(col)
        if column_profile is not None and column_profile.outliers is not None:
            summary = column_profile.outliers
            if summary.lower_bound is not None or summary.upper_bound is not None:
                return (
                    summary.lower_bound,
                    summary.upper_bound,
                    f"profile {summary.method.upper()} bounds",
                )

    series = ctx.frame[col].astype("float64")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    if not np.isfinite(q1) or not np.isfinite(q3):
        return None, None, "unavailable"
    multiplier = float(params.get("iqr_multiplier", params.get("k", 1.5)))
    iqr = float(q3 - q1)
    return (
        float(q1) - multiplier * iqr,
        float(q3) + multiplier * iqr,
        f"recomputed IQR x{multiplier:g}",
    )


def _clip_outliers(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    for col in columns:
        series = ctx.frame[col]
        if not _is_numeric(series):
            ctx.state.add_warning(
                f"cleaning: clip_outliers skipped non-numeric column '{col}'"
            )
            continue
        lower, upper, source = _iqr_bounds(ctx, col, params)
        if lower is None and upper is None:
            ctx.state.add_warning(
                f"cleaning: no outlier bounds available for '{col}'; not clipped"
            )
            continue
        n_low = int((series < lower).sum()) if lower is not None else 0
        n_high = int((series > upper).sum()) if upper is not None else 0
        ctx.clip_bounds[col] = [lower, upper]
        if n_low + n_high == 0:
            continue
        ctx.frame[col] = series.clip(lower=lower, upper=upper)
        ctx.record(
            CleaningAction.CLIP_OUTLIERS,
            f"winsorised '{col}' to [{lower:.6g}, {upper:.6g}] using {source} "
            f"({n_low} below, {n_high} above)",
            rationale,
        )


def _remove_outlier_rows(
    ctx: _Ctx, columns: list[str], params: dict[str, Any], rationale: str
) -> None:
    numeric = [c for c in columns if _is_numeric(ctx.frame[c])]
    if not numeric:
        ctx.state.add_warning(
            "cleaning: remove_outlier_rows found no numeric columns to test"
        )
        return

    mask = pd.Series(False, index=ctx.frame.index)
    used: list[str] = []
    for col in numeric:
        lower, upper, _ = _iqr_bounds(ctx, col, params)
        if lower is None and upper is None:
            continue
        series = ctx.frame[col]
        col_mask = pd.Series(False, index=ctx.frame.index)
        if lower is not None:
            col_mask |= series < lower
        if upper is not None:
            col_mask |= series > upper
        mask |= col_mask.fillna(False)
        used.append(col)

    n_drop = int(mask.sum())
    if n_drop == 0:
        return
    fraction = n_drop / max(1, ctx.n_rows)
    if fraction > OUTLIER_REMOVAL_REFUSE_FRACTION:
        ctx.refuse(
            CleaningAction.REMOVE_OUTLIER_ROWS,
            f"{fraction:.1%} of rows lie outside the bounds for {used}; that is a "
            f"distribution, not a set of outliers",
        )
        return
    if ctx.n_rows - n_drop < MIN_ROWS_AFTER_ACTION:
        ctx.refuse(
            CleaningAction.REMOVE_OUTLIER_ROWS,
            f"only {ctx.n_rows - n_drop} row(s) would remain",
        )
        return
    if fraction > OUTLIER_REMOVAL_WARN_FRACTION:
        ctx.state.add_warning(
            f"cleaning: remove_outlier_rows deletes {n_drop} row(s) ({fraction:.1%}) "
            f"— above the {OUTLIER_REMOVAL_WARN_FRACTION:.0%} guideline; extreme values "
            f"may be signal rather than noise"
        )
    ctx.frame = ctx.frame.loc[~mask]
    ctx.record(
        CleaningAction.REMOVE_OUTLIER_ROWS,
        f"removed {n_drop} row(s) ({fraction:.1%}) with values outside the IQR bounds "
        f"of {used}",
        rationale,
    )


# ---------------------------------------------------------------------------
# Shared lookups
# ---------------------------------------------------------------------------


def _temporal_column(ctx: _Ctx) -> str | None:
    problem = ctx.state.problem
    if problem and problem.temporal_column in ctx.frame.columns:
        return problem.temporal_column
    profile = ctx.state.profile
    if profile:
        for name in profile.temporal_columns:
            if name in ctx.frame.columns:
                return name
    for name in ctx.frame.columns:
        if pdt.is_datetime64_any_dtype(ctx.frame[name]):
            return str(name)
    return None


def _group_column(ctx: _Ctx) -> str | None:
    problem = ctx.state.problem
    if problem and problem.group_column and problem.group_column in ctx.frame.columns:
        return problem.group_column
    return None
