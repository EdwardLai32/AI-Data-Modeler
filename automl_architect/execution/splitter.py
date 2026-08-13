"""Train / validation / test partitioning.

The split *is* the experiment. Every score the rest of the pipeline reports is
only as trustworthy as the boundary drawn here, and the three ways to draw it
wrong are all invisible afterwards:

* random-splitting time-ordered data lets the model train on the future;
* random-splitting entity data puts the same customer on both sides, so the
  model memorises the entity instead of learning the pattern;
* random-splitting a rare-class problem can leave the test set with no positives
  at all, which makes accuracy look excellent and recall undefined.

:func:`resolve_split_strategy` therefore inspects the problem definition and the
profile and returns both the choice and the reason for it, and
:func:`make_splits` records that reason on :class:`DataSplits` so the Evaluation
Agent can judge whether a number deserves belief.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..core.errors import ExecutionError
from ..core.schemas import AgentName, EventKind, TaskType
from ..core.state import DataSplits, RunState

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_ROWS_FOR_VALIDATION",
    "MIN_VALIDATION_ROWS",
    "SPLIT_STRATEGIES",
    "make_cv_splitter",
    "make_splits",
    "resolve_split_strategy",
]

SPLIT_STRATEGIES = ("temporal", "grouped", "stratified", "random")

#: Below this row count a three-way split spends more precision than it buys:
#: the validation estimate becomes noise and the training set pays for it.
MIN_ROWS_FOR_VALIDATION = 150
#: A validation partition smaller than this cannot resolve a metric difference.
MIN_VALIDATION_ROWS = 20
#: Never emit a training partition smaller than this.
MIN_TRAIN_ROWS = 5


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _frame(state: RunState) -> pd.DataFrame | None:
    frame = state.df
    return frame if frame is not None else None


def _columns_of(state: RunState) -> set[str]:
    frame = _frame(state)
    if frame is not None:
        return set(map(str, frame.columns))
    if state.profile:
        return {c.name for c in state.profile.columns}
    return set()


def _temporal_column(state: RunState) -> str | None:
    """The column that defines time order, if one has been *declared*.

    Deliberately narrower than ``feature_ops``, which will fall back to any
    datetime column it finds. Ordering lags by the wrong timestamp costs one
    useless feature; ordering the *split* by ``date_of_birth`` silently invents a
    train/test boundary that means nothing, so the time axis has to come from the
    Problem Agent or the profiler rather than from a dtype scan.
    """
    known = _columns_of(state)
    problem = state.problem
    if problem and problem.temporal_column and problem.temporal_column in known:
        return problem.temporal_column
    if state.profile:
        for name in state.profile.temporal_columns:
            if name in known:
                return name
    return None


def _group_column(state: RunState) -> str | None:
    known = _columns_of(state)
    problem = state.problem
    if problem and problem.group_column and problem.group_column in known:
        return problem.group_column
    return None


def _training_groups(state: RunState) -> Any:
    """Group labels aligned to the training partition, or ``None``.

    ``X_train`` carries only ``feature_names``, and the group key is deliberately
    excluded from those (its raw levels never generalise across an entity-wise
    boundary). So the labels almost always have to be recovered from the source
    frame and aligned to ``X_train``'s index rather than read off ``X_train``.
    """
    group_col = _group_column(state)
    if not group_col:
        return None

    train = state.splits.X_train
    if train is not None and group_col in getattr(train, "columns", []):
        return train[group_col].to_numpy()

    frame = _frame(state)
    if frame is None or group_col not in frame.columns:
        return None
    if train is None:
        return frame[group_col].to_numpy()
    if frame.index.has_duplicates or not train.index.isin(frame.index).all():
        return None
    return frame.loc[train.index, group_col].to_numpy()


def _target_series(state: RunState, frame: pd.DataFrame) -> pd.Series | None:
    target = state.target
    if not target or target not in frame.columns:
        return None
    return frame[target]


def _class_counts(state: RunState) -> pd.Series | None:
    frame = _frame(state)
    if frame is None:
        return None
    y = _target_series(state, frame)
    if y is None:
        return None
    return y.dropna().value_counts()


def _is_classification(state: RunState) -> bool:
    task = state.task_type
    return bool(task and task.is_classification)


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


def resolve_split_strategy(state: RunState) -> tuple[str, str]:
    """Choose how to partition this dataset.

    Precedence is deliberate. Forecasting is decided by time before anything
    else, because a model that has seen the future cannot be salvaged by a good
    grouping. A group key comes next: entity overlap inflates scores in a way no
    later diagnostic can detect. Stratification is last, because it only
    protects the *composition* of the partitions, not their independence.

    Args:
        state: The run blackboard; reads ``problem``, ``profile``, and the most
            processed dataframe available.

    Returns:
        ``(strategy, rationale)`` where ``strategy`` is one of
        :data:`SPLIT_STRATEGIES` and ``rationale`` is the evidence-based reason,
        suitable for a report.
    """
    task = state.task_type
    temporal = _temporal_column(state)
    group = _group_column(state)

    if task is TaskType.TIME_SERIES_FORECASTING:
        if temporal:
            return (
                "temporal",
                f"Forecasting task: the split is a single cut in '{temporal}' so that "
                f"training data strictly precedes evaluation data. A random split would "
                f"let the model learn from periods it is meant to predict.",
            )
        return (
            "temporal",
            "Forecasting task with no parseable timestamp column: the existing row "
            "order is treated as time order and the split is taken from the tail. "
            "A random split would train on the future.",
        )

    if group:
        return (
            "grouped",
            f"'{group}' identifies a repeating entity, so rows are not independent. "
            f"Splitting by entity keeps every row of a given '{group}' on one side of "
            f"the boundary; a random split would let the model memorise entities it is "
            f"then scored on.",
        )

    if temporal:
        return (
            "temporal",
            f"'{temporal}' gives the rows a real time order, so the split is a cut in "
            f"time rather than a shuffle. This measures the only thing that matters in "
            f"deployment — performance on data recorded after training.",
        )

    if _is_classification(state):
        counts = _class_counts(state)
        detail = ""
        if counts is not None and not counts.empty:
            n_classes = int(counts.size)
            ratio = float(counts.max() / max(1, counts.min()))
            if n_classes > 2:
                detail = f" There are {n_classes} classes"
                if ratio > 3:
                    detail += f" and the majority:minority ratio is {ratio:.1f}:1"
                detail += "."
            elif ratio > 3:
                detail = f" The majority:minority ratio is {ratio:.1f}:1."
        return (
            "stratified",
            "Classification task: each partition preserves the class proportions of "
            "the full dataset, so the test metric is comparable to the training "
            "metric and rare classes cannot vanish from evaluation." + detail,
        )

    return (
        "random",
        "No temporal ordering, entity key, or class structure constrains the split, "
        "so a shuffled partition at the configured sizes is the unbiased choice.",
    )


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------


def make_cv_splitter(
    state: RunState,
    *,
    n_splits: int | None = None,
    y: Any = None,
    groups: Any = None,
) -> tuple[Any, Any]:
    """Build the cross-validation splitter that matches the split strategy.

    Sharing one definition between the holdout split, hyperparameter tuning, and
    out-of-fold target encoding is what keeps their scores comparable: folds that
    respect a different boundary than the test split produce a validation number
    that does not predict the test number.

    Args:
        state: The run blackboard.
        n_splits: Fold count. Defaults to ``config.cv_folds``, reduced when a
            class or group is too small to appear in every fold.
        y: Optional label array used to cap folds for stratification.
        groups: Optional group labels; defaults to the group column of the
            training partition when one exists.

    Returns:
        ``(splitter, groups)`` — the second element is ``None`` unless the
        splitter needs group labels passed to ``split()``.
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    strategy = state.splits.strategy or resolve_split_strategy(state)[0]
    folds = int(n_splits or state.config.cv_folds or 5)
    folds = max(2, folds)
    random_state = state.config.random_state

    if groups is None and strategy == "grouped":
        groups = _training_groups(state)

    if strategy == "temporal":
        return TimeSeriesSplit(n_splits=folds), None

    if strategy == "grouped":
        if groups is not None:
            n_groups = int(pd.Series(groups).nunique())
            folds = max(2, min(folds, n_groups))
            return GroupKFold(n_splits=folds), groups
        # Falling through to KFold here would silently undo the entity boundary the
        # grouped strategy exists to enforce, so say so rather than scoring folds
        # that share entities.
        state.add_warning(
            "split: the grouped strategy is active but no group labels could be "
            "recovered for cross-validation; folds may share entities and CV scores "
            "will be optimistic"
        )

    if strategy == "stratified" and y is not None:
        counts = pd.Series(np.asarray(y).ravel()).value_counts()
        if not counts.empty:
            folds = max(2, min(folds, int(counts.min())))
        return (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
            None,
        )

    return KFold(n_splits=folds, shuffle=True, random_state=random_state), None


# ---------------------------------------------------------------------------
# Main split
# ---------------------------------------------------------------------------


def make_splits(state: RunState) -> DataSplits:
    """Partition the working data into train / validation / test.

    Honours ``config.test_size``, ``config.validation_size`` and
    ``config.random_state``. Degrades rather than failing: an unstratifiable
    class distribution falls back to a shuffled split, too few groups falls back
    likewise, and a dataset too small for three partitions gets two.

    For classification the target is label-encoded when it is not already
    integer-coded, with the fitted encoder stored on ``state.label_encoder`` and
    the class order mirrored into ``state.extras['class_names']``.

    Args:
        state: The run blackboard. Reads ``feature_frame`` (falling back to
            ``working_df`` then ``raw_df``) plus ``feature_names``, and writes
            ``state.splits``.

    Returns:
        The populated :class:`~automl_architect.core.state.DataSplits`, also
        assigned to ``state.splits``.

    Raises:
        ExecutionError: If no dataframe is available, or the frame has too few
            rows to split at all.
    """
    frame = _frame(state)
    if frame is None:
        raise ExecutionError("make_splits: no dataframe on the run state")
    if frame.index.has_duplicates:
        frame = frame.reset_index(drop=True)
        state.add_warning("split: duplicate index labels found; index was reset")

    target = state.target
    task = state.task_type
    supervised = target is not None and target in frame.columns

    if target and target not in frame.columns:
        state.add_warning(
            f"split: target '{target}' is not present in the frame; producing an "
            f"unsupervised split with no labels"
        )

    if supervised:
        null_target = frame[target].isna()
        if bool(null_target.any()):
            state.add_warning(
                f"split: dropped {int(null_target.sum())} row(s) with a missing "
                f"'{target}' that survived cleaning"
            )
            frame = frame.loc[~null_target]

    if len(frame) < MIN_TRAIN_ROWS * 2:
        raise ExecutionError(
            f"make_splits: {len(frame)} row(s) is too few to partition; "
            f"at least {MIN_TRAIN_ROWS * 2} are required"
        )

    features = _resolve_features(state, frame)
    strategy, rationale = resolve_split_strategy(state)

    y_full: pd.Series | None = None
    if supervised:
        y_full = frame[target]
        if task and task.is_classification:
            y_full = _encode_labels(state, y_full)

    notes: list[str] = []
    test_size = _clamp(state.config.test_size, 0.0, 0.5, "test_size", state)
    validation_size = _clamp(
        state.config.validation_size, 0.0, 0.5, "validation_size", state
    )
    if test_size + validation_size >= 0.8:
        state.add_warning(
            f"split: test_size + validation_size = {test_size + validation_size:.2f} "
            f"leaves too little training data; validation was reduced"
        )
        validation_size = max(0.0, 0.8 - test_size)

    n_rows = int(len(frame))
    if validation_size > 0 and (
        n_rows < MIN_ROWS_FOR_VALIDATION
        or round(n_rows * validation_size) < MIN_VALIDATION_ROWS
    ):
        notes.append(
            f"Validation partition skipped: {n_rows} rows cannot support three "
            f"partitions (a {validation_size:.0%} slice would be "
            f"{round(n_rows * validation_size)} rows). Model selection uses "
            f"cross-validation on the training partition instead."
        )
        state.add_warning(notes[-1])
        validation_size = 0.0

    # Stratification needs enough members per class to fill every fold; check
    # before committing to the strategy so the fallback is recorded, not crashed.
    if strategy == "stratified":
        strategy, extra = _check_stratifiable(state, y_full)
        if extra:
            notes.append(extra)
    if strategy == "grouped":
        strategy, extra = _check_groupable(state, frame)
        if extra:
            notes.append(extra)

    if strategy == "temporal":
        train_idx, valid_idx, test_idx, extra = _temporal_partition(
            state, frame, test_size, validation_size
        )
    elif strategy == "grouped":
        train_idx, valid_idx, test_idx, extra = _grouped_partition(
            state, frame, test_size, validation_size
        )
    elif strategy == "stratified":
        train_idx, valid_idx, test_idx, extra = _random_partition(
            state, frame, y_full, test_size, validation_size, stratify=True
        )
    else:
        train_idx, valid_idx, test_idx, extra = _random_partition(
            state, frame, y_full, test_size, validation_size, stratify=False
        )
    if extra:
        notes.append(extra)

    if len(train_idx) < MIN_TRAIN_ROWS:
        raise ExecutionError(
            f"make_splits: the {strategy} split left only {len(train_idx)} training "
            f"row(s); reduce test_size or supply more data"
        )

    def _x(index: pd.Index) -> pd.DataFrame | None:
        return frame.loc[index, features] if len(index) else None

    def _y(index: pd.Index) -> pd.Series | None:
        if y_full is None or not len(index):
            return None
        return y_full.loc[index]

    # A grouped or temporal cut lands on whole entities or whole periods, so the
    # realised sizes can miss the requested ones badly. Saying so is the
    # difference between an auditable split and a surprising one.
    realised_test = len(test_idx) / max(1, n_rows)
    if test_size > 0 and abs(realised_test - test_size) > 0.10:
        notes.append(
            f"The {strategy} split had to respect whole "
            f"{'entities' if strategy == 'grouped' else 'periods'}, so the test "
            f"partition came out at {realised_test:.0%} rather than the requested "
            f"{test_size:.0%}."
        )
        state.add_warning(notes[-1])

    splits = DataSplits(
        X_train=_x(train_idx),
        X_valid=_x(valid_idx),
        X_test=_x(test_idx),
        y_train=_y(train_idx),
        y_valid=_y(valid_idx),
        y_test=_y(test_idx),
        strategy=strategy,
        rationale=" ".join([rationale, *notes]).strip(),
    )
    state.splits = splits
    sizes = splits.sizes()
    state.extras["split_strategy"] = strategy
    state.extras["split_sizes"] = sizes
    state.extras["split_feature_columns"] = list(features)
    state.bus.emit(
        EventKind.LOG,
        f"split ({strategy}): train={sizes['train']} validation={sizes['validation']} "
        f"test={sizes['test']} over {len(features)} feature column(s)",
        agent=AgentName.PLANNER,
        payload={"strategy": strategy, **{f"n_{k}": v for k, v in sizes.items()}},
    )
    return splits


# ---------------------------------------------------------------------------
# Feature / label preparation
# ---------------------------------------------------------------------------


def _resolve_features(state: RunState, frame: pd.DataFrame) -> list[str]:
    """The X columns: what feature engineering declared, filtered to reality."""
    declared = [c for c in state.feature_names if c in frame.columns]
    if declared:
        return declared

    target = state.target
    excluded = {target} if target else set()
    if state.profile:
        excluded.update(state.profile.identifier_columns)
        excluded.update(state.profile.constant_columns)
    group = _group_column(state)
    if group:
        excluded.add(group)
    derived = [
        c
        for c in frame.columns
        if c not in excluded and not pdt.is_datetime64_any_dtype(frame[c])
    ]
    if not derived:
        raise ExecutionError("make_splits: no usable feature columns remain")
    state.add_warning(
        "split: state.feature_names was empty; derived "
        f"{len(derived)} feature column(s) directly from the frame"
    )
    return derived


def _encode_labels(state: RunState, y: pd.Series) -> pd.Series:
    """Integer-code a classification target, keeping the encoder for inversion.

    Skipped when the labels are already ``0..k-1`` integers, so that a dataset
    arriving pre-encoded is not gratuitously relabelled.
    """
    values = y.dropna()
    if pdt.is_integer_dtype(values) or pdt.is_bool_dtype(values):
        codes = np.sort(pd.unique(values.astype("int64")))
        if codes.size and codes[0] == 0 and codes[-1] == codes.size - 1:
            state.extras.setdefault("class_names", [str(c) for c in codes])
            return y.astype("int64")

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    as_text = y.astype("object").astype("str")
    encoded = encoder.fit_transform(as_text.to_numpy())
    state.label_encoder = encoder
    class_names = [str(c) for c in encoder.classes_]
    state.extras["class_names"] = class_names

    positive = getattr(state.problem, "positive_class", None)
    if positive is not None and str(positive) in class_names:
        index = class_names.index(str(positive))
        state.extras["positive_label"] = index
        if len(class_names) == 2 and index != 1:
            # LabelEncoder orders classes alphabetically, which does not always
            # put the class of interest at 1; downstream metrics must read the
            # index rather than assume it.
            state.add_warning(
                f"split: positive class '{positive}' encodes to {index}, not 1; "
                f"metrics must use extras['positive_label']"
            )
    elif len(class_names) == 2:
        state.extras.setdefault("positive_label", 1)

    shown = ", ".join(f"{name}->{i}" for i, name in enumerate(class_names[:8]))
    state.bus.emit(
        EventKind.LOG,
        f"label encoding for '{y.name}': {shown}"
        + (" …" if len(class_names) > 8 else ""),
        agent=AgentName.PLANNER,
        payload={"n_classes": len(class_names)},
    )
    return pd.Series(encoded, index=y.index, name=y.name)


def _clamp(
    value: float, low: float, high: float, label: str, state: RunState
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    if number < low or number > high:
        clamped = min(max(number, low), high)
        state.add_warning(
            f"split: {label}={value} is out of range; clamped to {clamped}"
        )
        return clamped
    return number


# ---------------------------------------------------------------------------
# Strategy feasibility
# ---------------------------------------------------------------------------


def _check_stratifiable(state: RunState, y: pd.Series | None) -> tuple[str, str]:
    if y is None:
        return "random", (
            "Stratification requires labels; none are available, so the split is "
            "shuffled."
        )
    counts = y.value_counts()
    if counts.empty:
        return "random", "Stratification requires labels; the target is empty."
    minimum = int(counts.min())
    folds = max(2, int(state.config.cv_folds or 5))
    if minimum < 2:
        note = (
            f"Stratification abandoned: class {counts.idxmin()!r} has {minimum} "
            f"member(s), so it cannot appear in both partitions. Fell back to a "
            f"shuffled split; metrics for that class will be unreliable."
        )
        state.add_warning(note)
        return "random", note
    if minimum < folds:
        note = (
            f"Stratification abandoned: class {counts.idxmin()!r} has {minimum} "
            f"member(s), fewer than the {folds} cross-validation folds, so stratified "
            f"folds would come out empty. Fell back to a shuffled split."
        )
        state.add_warning(note)
        return "random", note
    return "stratified", ""


def _check_groupable(state: RunState, frame: pd.DataFrame) -> tuple[str, str]:
    group = _group_column(state)
    if not group or group not in frame.columns:
        note = "Grouped split abandoned: the group column is not in the frame."
        state.add_warning(note)
        return "random", note
    n_groups = int(frame[group].nunique(dropna=False))
    if n_groups < 3:
        note = (
            f"Grouped split abandoned: '{group}' has only {n_groups} distinct "
            f"value(s), too few to hold entities out. Fell back to a shuffled split."
        )
        state.add_warning(note)
        return "random", note
    if n_groups >= len(frame) * 0.95:
        note = (
            f"'{group}' is nearly unique per row ({n_groups} groups over "
            f"{len(frame)} rows), so a grouped split is equivalent to a random one."
        )
        state.add_warning(note)
        return "random", note
    return "grouped", ""


# ---------------------------------------------------------------------------
# Partitioners — each returns (train, valid, test, note)
# ---------------------------------------------------------------------------


def _empty(frame: pd.DataFrame) -> pd.Index:
    return frame.index[:0]


def _temporal_partition(
    state: RunState,
    frame: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> tuple[pd.Index, pd.Index, pd.Index, str]:
    temporal = _temporal_column(state)
    if temporal:
        column = frame[temporal]
        if not pdt.is_datetime64_any_dtype(column) and not pdt.is_numeric_dtype(column):
            parsed = pd.to_datetime(column, errors="coerce")
            column = parsed if int(parsed.notna().sum()) > 0 else column
        order = column.sort_values(kind="stable").index
        note = f"Rows ordered by '{temporal}'; the most recent rows form the test set."
    else:
        order = frame.index
        note = "No timestamp column; the existing row order is treated as time order."

    n = len(order)
    n_test = int(round(n * test_size))
    n_valid = int(round(n * validation_size))
    if test_size > 0:
        n_test = max(1, n_test)
    if validation_size > 0:
        n_valid = max(1, n_valid)
    if n - n_test - n_valid < MIN_TRAIN_ROWS:
        n_valid = 0
        n_test = max(0, min(n_test, n - MIN_TRAIN_ROWS))
        note += " Partition sizes were reduced to preserve a usable training set."

    test_idx = order[n - n_test :] if n_test else _empty(frame)
    valid_start = n - n_test - n_valid
    valid_idx = order[valid_start : n - n_test] if n_valid else _empty(frame)
    train_idx = order[:valid_start] if valid_start > 0 else order[: max(0, n - n_test)]

    if temporal and n_test:
        boundary = frame.loc[test_idx[0], temporal]
        note += f" The train/test boundary sits at {boundary}."
    return train_idx, valid_idx, test_idx, note


def _grouped_partition(
    state: RunState,
    frame: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> tuple[pd.Index, pd.Index, pd.Index, str]:
    from sklearn.model_selection import GroupShuffleSplit

    group = _group_column(state)
    if group is None:  # pragma: no cover - _check_groupable ran first
        raise ExecutionError("_grouped_partition: no group column resolved")
    groups = frame[group].astype("object").astype("str")
    random_state = state.config.random_state

    if test_size > 0:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_pos, test_pos = next(splitter.split(frame, groups=groups))
        train_idx = frame.index[train_pos]
        test_idx = frame.index[test_pos]
    else:
        train_idx, test_idx = frame.index, _empty(frame)

    valid_idx = _empty(frame)
    if validation_size > 0 and len(train_idx) > MIN_TRAIN_ROWS:
        relative = min(0.5, validation_size / max(1e-9, 1.0 - test_size))
        sub_groups = groups.loc[train_idx]
        if int(sub_groups.nunique()) >= 3:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=relative, random_state=random_state
            )
            inner_train_pos, valid_pos = next(
                splitter.split(train_idx.to_frame(), groups=sub_groups)
            )
            valid_idx = train_idx[valid_pos]
            train_idx = train_idx[inner_train_pos]

    n_train_groups = int(groups.loc[train_idx].nunique())
    n_test_groups = int(groups.loc[test_idx].nunique()) if len(test_idx) else 0
    note = (
        f"Entities are disjoint across partitions: {n_train_groups} distinct "
        f"'{group}' value(s) in train and {n_test_groups} in test, with no overlap."
    )
    return train_idx.sort_values(), valid_idx.sort_values(), test_idx.sort_values(), note


def _random_partition(
    state: RunState,
    frame: pd.DataFrame,
    y: pd.Series | None,
    test_size: float,
    validation_size: float,
    *,
    stratify: bool,
) -> tuple[pd.Index, pd.Index, pd.Index, str]:
    random_state = state.config.random_state
    index = frame.index
    labels = y.loc[index] if (stratify and y is not None) else None

    if test_size > 0:
        train_idx, test_idx = _safe_train_test_split(
            state, index, labels, test_size, random_state, "test"
        )
    else:
        train_idx, test_idx = index, _empty(frame)

    valid_idx = _empty(frame)
    if validation_size > 0 and len(train_idx) > MIN_TRAIN_ROWS:
        relative = min(0.5, validation_size / max(1e-9, 1.0 - test_size))
        sub_labels = y.loc[train_idx] if (stratify and y is not None) else None
        train_idx, valid_idx = _safe_train_test_split(
            state, train_idx, sub_labels, relative, random_state, "validation"
        )

    note = (
        "Class proportions are preserved in every partition."
        if stratify
        else "Rows were shuffled with a fixed seed, so the partition is reproducible."
    )
    return train_idx.sort_values(), valid_idx.sort_values(), test_idx.sort_values(), note


def _safe_train_test_split(
    state: RunState,
    index: pd.Index,
    labels: pd.Series | None,
    size: float,
    random_state: int,
    which: str,
) -> tuple[pd.Index, pd.Index]:
    """``train_test_split`` that degrades to unstratified instead of raising."""
    from sklearn.model_selection import train_test_split

    if labels is not None:
        try:
            left, right = train_test_split(
                index,
                test_size=size,
                random_state=random_state,
                shuffle=True,
                stratify=labels.to_numpy(),
            )
            return pd.Index(left), pd.Index(right)
        except ValueError as exc:
            state.add_warning(
                f"split: stratified {which} split failed ({exc}); retried unstratified"
            )
    left, right = train_test_split(
        index, test_size=size, random_state=random_state, shuffle=True
    )
    return pd.Index(left), pd.Index(right)
