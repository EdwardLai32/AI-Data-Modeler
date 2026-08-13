"""The training loop: where every number on the leaderboard is produced.

The design constraints, in priority order:

1.  **Nothing leaks.** The preprocessor is cloned into the Pipeline so it is
    re-fitted inside every cross-validation fold. Fitting a scaler or a target
    encoder on the whole training set before splitting is the single most common
    way an AutoML tool reports a score it cannot reproduce.
2.  **One broken family cannot end the run.** Every candidate is trained inside
    its own guard; a failure becomes ``ExperimentResult(failed=True, error=...)``
    and the loop moves on. :class:`NoViableModelError` is raised only when the
    entire slate failed.
3.  **Budgets are respected out loud.** When the time budget or
    ``max_experiments`` cuts the slate short, that is a recorded warning, not a
    silent truncation — a leaderboard of two models must not look like a
    leaderboard of eight.

Scores are always in natural units (``rmse`` positive, ``log_loss`` positive);
sklearn's ``neg_*`` convention is unwound at the boundary by
:class:`~automl_architect.execution.metrics.ScorerSpec`.
"""

from __future__ import annotations

import io
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..core.errors import NoViableModelError, TrainingError
from ..core.schemas import (
    ExperimentLog,
    ExperimentResult,
    MetricValue,
    ModelCandidate,
    ModelFamily,
    TaskType,
    dict_to_params,
    params_to_dict,
)
from ..core.state import RunState
from .metrics import (
    ScorerSpec,
    canonical_metric,
    higher_is_better,
    is_better,
    metric_names_for,
    primary_metric_for,
    score_predictions,
    scorer_for,
)
from .model_zoo import (
    available_families,
    build_estimator,
    is_available,
    supports_proba,
    supports_task,
    unsupported_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingContext",
    "build_training_context",
    "cluster_labels",
    "cross_validate_pipeline",
    "fit_final_model",
    "fit_measured",
    "format_leaderboard",
    "record_experiment",
    "run_experiments",
    "score_fitted_model",
    "wrap_with_preprocessor",
]

# Below this much remaining budget there is no point starting another fit; the
# run needs time to explain, evaluate, and report what it already has.
_MIN_CANDIDATE_SECONDS = 6.0
# Cross-validation multiplies fit cost by n_splits, so it is the first thing
# dropped when the clock is tight.
_MIN_CV_SECONDS = 25.0

_MODEL_STEP = "model"
_PREPROCESSOR_STEP = "preprocessor"

# Rows per season for the frequency strings pandas infers.
_SEASON_BY_FREQ: dict[str, int] = {
    "H": 24,
    "h": 24,
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "Y": 1,
    "A": 1,
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class TrainingContext:
    """Everything the loop needs, resolved once so every candidate is judged identically.

    Building this per candidate would risk two models being scored against
    different label encodings or different folds, which is the subtlest way a
    leaderboard can lie.
    """

    task: TaskType
    metric: str
    higher_better: bool
    X_train: Any = None
    y_train: Any = None
    X_eval: Any = None
    y_eval: Any = None
    eval_partition: str = "none"
    labels: list[Any] | None = None
    pos_label: Any = None
    class_names: list[str] = field(default_factory=list)
    cv: Any = None
    cv_groups: Any = None
    n_splits: int = 0
    scorer: ScorerSpec | None = None
    random_state: int = 42
    zoo_ctx: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def supervised(self) -> bool:
        """Whether a target is available to score against."""
        return self.y_train is not None


def build_training_context(state: RunState) -> TrainingContext:
    """Resolve task, metric, partitions, folds, and label encoding for a run.

    Shared by :func:`run_experiments` and the tuner so that a tuned score and an
    untuned score are always measured the same way on the same partition.

    Args:
        state: The run blackboard. ``state.splits`` must be populated; if it is
            not, the splitter is invoked lazily as a degradation path.

    Returns:
        A fully resolved :class:`TrainingContext`.

    Raises:
        TrainingError: If no feature matrix can be found at all.
    """
    # Splits first: the task inference below reads the target out of them, so a
    # lazily built split has to exist before it runs.
    _ensure_splits(state)
    task = _resolve_task(state)
    splits = state.splits

    X_train = splits.X_train
    if X_train is None or _n_rows(X_train) == 0:
        raise TrainingError(
            "no training partition available; run the splitter before training"
        )

    y_train = splits.y_train
    X_eval, y_eval, partition = _pick_eval_partition(state)

    context = TrainingContext(
        task=task,
        metric="",
        higher_better=True,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        eval_partition=partition,
        random_state=int(state.config.random_state),
    )

    _apply_row_cap(state, context)
    if task.is_classification and context.y_train is not None:
        _prepare_class_labels(state, context)
    _resolve_metric(state, context)
    _resolve_folds(state, context)
    context.zoo_ctx = _zoo_context(state, context)
    return context


def _resolve_task(state: RunState) -> TaskType:
    task = state.task_type
    if task is not None:
        return task
    # A missing problem definition should not stop training: infer the coarse
    # family from the target itself and say so.
    y = state.splits.y_train if state.splits else None
    if y is None:
        state.add_warning("no task type on the run; assuming clustering (no target)")
        return TaskType.CLUSTERING
    values = np.asarray(y).ravel()
    if values.dtype.kind in "OUSb" or len(np.unique(values)) <= 2:
        inferred = TaskType.BINARY_CLASSIFICATION
    elif values.dtype.kind in "iu" and len(np.unique(values)) <= 20:
        inferred = TaskType.MULTICLASS_CLASSIFICATION
    else:
        inferred = TaskType.REGRESSION
    state.add_warning(
        f"no task type on the run; inferred '{inferred.value}' from the target dtype"
    )
    return inferred


def _ensure_splits(state: RunState) -> None:
    """Populate ``state.splits`` from the splitter if a previous step skipped it."""
    if state.splits is not None and state.splits.X_train is not None:
        return
    try:
        from .splitter import make_splits  # local import: optional sibling module

        state.splits = make_splits(state)
        state.add_warning("training had to build the data splits itself")
    except Exception as exc:  # noqa: BLE001 - degrade to the caller's error
        logger.debug("could not lazily build splits: %s", exc)


def _pick_eval_partition(state: RunState) -> tuple[Any, Any, str]:
    """Choose the partition every candidate is scored on.

    Validation is preferred: the test set exists to measure the *winner* once,
    and selecting on it would turn the final number into a training score.
    """
    splits = state.splits
    if splits.has_validation:
        return splits.X_valid, splits.y_valid, "validation"
    if splits.has_test:
        return splits.X_test, splits.y_test, "test"
    return None, None, "none"


def _apply_row_cap(state: RunState, ctx: TrainingContext) -> None:
    cap = int(getattr(state.settings, "max_train_rows", 0) or 0)
    n = _n_rows(ctx.X_train)
    if cap <= 0 or n <= cap:
        return
    rng = np.random.default_rng(ctx.random_state)
    keep = np.sort(rng.choice(n, size=cap, replace=False))
    ctx.X_train = _take_rows(ctx.X_train, keep)
    if ctx.y_train is not None:
        ctx.y_train = _take_rows(ctx.y_train, keep)
    message = (
        f"training subsampled from {n:,} to {cap:,} rows "
        f"(settings.max_train_rows); scores reflect the sample"
    )
    # The context is rebuilt by the tuner and the final refit, so the warning is
    # deduplicated rather than repeated three times in one run summary.
    if message not in state.warnings:
        state.add_warning(message)
    ctx.notes.append(message)


def _prepare_class_labels(state: RunState, ctx: TrainingContext) -> None:
    """Normalise classification targets to contiguous integer codes.

    XGBoost and LightGBM require ``0..k-1`` labels, and every metric here is
    invariant to label naming, so encoding once up front removes a whole class
    of per-family failure. ``state.label_encoder`` is reused when it already
    covers the data and populated when it is empty, so the original class names
    survive for the report.
    """
    present: set[Any] = set()
    for part in (ctx.y_train, ctx.y_eval):
        if part is None:
            continue
        array = np.asarray(part).ravel()
        present.update(array[~_null_mask(array)].tolist())
    if not present:
        return

    encoder = state.label_encoder
    if encoder is not None and hasattr(encoder, "classes_"):
        known = set(np.asarray(encoder.classes_).tolist())
        if present.issubset(known):
            as_str = np.asarray(encoder.classes_).dtype.kind in "OUS"
            ctx.y_train = _encode_labels(encoder, ctx.y_train, as_str)
            ctx.y_eval = _encode_labels(encoder, ctx.y_eval, as_str)
            _finish_labels(state, ctx, list(encoder.classes_))
            return

    numeric = all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in present
    )
    if numeric and present == set(range(len(present))):
        # Already contiguous codes: leave the values alone so downstream
        # artefacts keep matching whatever produced them, but recover the
        # original class names. Without them, a positive_class of "churned"
        # cannot be matched to its code and binary precision/recall/f1 would
        # silently be reported for the other class.
        _finish_labels(state, ctx, _decode_names(state, len(present)))
        return

    values, as_str = _encoder_domain(present)
    new_encoder = LabelEncoder()
    new_encoder.fit(values)
    ctx.y_train = _encode_labels(new_encoder, ctx.y_train, as_str)
    ctx.y_eval = _encode_labels(new_encoder, ctx.y_eval, as_str)
    _finish_labels(state, ctx, list(new_encoder.classes_))
    if state.label_encoder is None:
        # Preserve the original class names for the report; nothing else has
        # claimed this slot, and the codes here would be meaningless without it.
        state.label_encoder = new_encoder


def _finish_labels(state: RunState, ctx: TrainingContext, classes: list[Any]) -> None:
    """Record the encoded label space and the positive class within it."""
    ctx.class_names = [str(c) for c in classes]
    ctx.labels = list(range(len(classes)))
    ctx.pos_label = _positive_index(state, classes)


def _decode_names(state: RunState, n_classes: int) -> list[Any]:
    """Original class names for an already-encoded target, in code order.

    The splitter encodes the target and leaves the mapping on
    ``state.label_encoder`` (and mirrors it into ``extras['class_names']``).
    Either channel is usable; the integer codes are the last resort.
    """
    encoder = state.label_encoder
    if encoder is not None and hasattr(encoder, "classes_"):
        names = list(np.asarray(encoder.classes_).tolist())
        if len(names) == n_classes:
            return names
    extras = state.extras if isinstance(state.extras, dict) else {}
    names = list(extras.get("class_names") or [])
    if len(names) == n_classes:
        return names
    return list(range(n_classes))


def _encoder_domain(present: set[Any]) -> tuple[np.ndarray, bool]:
    """The array to fit a LabelEncoder on, and whether values need stringifying.

    Mixed-type targets cannot be sorted by numpy, so they are compared as text.
    Uniform int, float, or str targets keep their own type.
    """
    ordered = sorted(present, key=str)
    uniform_numeric = all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, (bool, np.bool_))
        for v in ordered
    )
    uniform_text = all(isinstance(v, str) for v in ordered)
    if uniform_numeric or uniform_text:
        return np.asarray(ordered), False
    return np.asarray([str(v) for v in ordered]), True


def _positive_index(state: RunState, classes: list[Any]) -> Any:
    """Map ``problem.positive_class`` onto the encoded label space.

    Returns the integer code, because the target is always encoded to
    ``0..k-1`` by the time metrics see it.
    """
    if len(classes) != 2:
        return None
    wanted = getattr(state.problem, "positive_class", None) if state.problem else None
    if wanted is None:
        return None
    for index, value in enumerate(classes):
        if str(value) == str(wanted):
            return index
    return None


def _resolve_metric(state: RunState, ctx: TrainingContext) -> None:
    requested = canonical_metric(state.primary_metric)
    default = primary_metric_for(ctx.task)
    metric = requested or default
    if metric not in metric_names_for(ctx.task):
        message = (
            f"primary metric '{state.primary_metric}' is not computable for "
            f"{ctx.task.value}; using '{default}' instead"
        )
        if message not in state.warnings:
            state.add_warning(message)
        ctx.notes.append(message)
        metric = default
    ctx.metric = metric
    ctx.higher_better = higher_is_better(metric)
    ctx.scorer = scorer_for(
        metric, ctx.task, pos_label=ctx.pos_label, labels=ctx.labels
    )


def _resolve_folds(state: RunState, ctx: TrainingContext) -> None:
    """Pick the fold generator implied by how the data was split."""
    if not ctx.supervised:
        return  # clustering and anomaly detection have no fold semantics here

    strategy = (state.splits.strategy or "").lower()
    n = _n_rows(ctx.X_train)
    n_splits = max(2, int(state.config.cv_folds or 5))
    n_splits = min(n_splits, max(2, n // 2))
    if n < 2 * n_splits:
        ctx.notes.append(f"only {n} training rows; cross-validation skipped")
        return

    # The splitter owns the split boundary, so it also owns the fold boundary.
    # Folds that respect a different boundary than the holdout split produce a
    # validation number that does not predict the test number.
    if _folds_from_splitter(state, ctx, n_splits):
        return

    groups = _resolve_groups(state, ctx)
    if "group" in strategy:
        if groups is None:
            state.add_warning(
                "split strategy is grouped but the group column is unavailable; "
                "cross-validating with plain K-fold instead"
            )
        else:
            n_groups = len(np.unique(groups))
            if n_groups < 2:
                state.add_warning("only one group present; using plain K-fold")
            else:
                ctx.cv = GroupKFold(n_splits=min(n_splits, n_groups))
                ctx.cv_groups = groups
                ctx.n_splits = ctx.cv.get_n_splits()
                return

    if any(token in strategy for token in ("time", "temporal", "expand", "chrono")):
        ctx.cv = TimeSeriesSplit(n_splits=n_splits)
        ctx.n_splits = n_splits
        return

    if ctx.task.is_classification:
        counts = np.unique(np.asarray(ctx.y_train).ravel(), return_counts=True)[1]
        smallest = int(counts.min()) if counts.size else 0
        if smallest < 2:
            state.add_warning(
                "a class has fewer than 2 training rows; stratified folds are "
                "impossible, using plain K-fold"
            )
            ctx.cv = KFold(
                n_splits=n_splits, shuffle=True, random_state=ctx.random_state
            )
        else:
            folds = min(n_splits, smallest)
            if folds < n_splits:
                ctx.notes.append(
                    f"reduced to {folds} folds: the rarest class has only "
                    f"{smallest} training rows"
                )
            ctx.cv = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=ctx.random_state
            )
        ctx.n_splits = ctx.cv.get_n_splits()
        return

    if ctx.task is TaskType.TIME_SERIES_FORECASTING:
        ctx.cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        ctx.cv = KFold(n_splits=n_splits, shuffle=True, random_state=ctx.random_state)
    ctx.n_splits = ctx.cv.get_n_splits()


def _folds_from_splitter(
    state: RunState, ctx: TrainingContext, n_splits: int
) -> bool:
    """Adopt ``splitter.make_cv_splitter`` when that module is available.

    Returns False if the splitter is absent, declines, or hands back group
    labels that do not line up with the training rows — in which case the local
    fold resolution below takes over.
    """
    try:
        from .splitter import make_cv_splitter
    except Exception as exc:  # noqa: BLE001 - sibling module is optional
        logger.debug("splitter.make_cv_splitter unavailable: %s", exc)
        return False
    try:
        cv, groups = make_cv_splitter(
            state, n_splits=n_splits, y=ctx.y_train, groups=None
        )
    except Exception as exc:  # noqa: BLE001 - fall back to local resolution
        logger.debug("make_cv_splitter failed: %s", exc)
        return False
    if cv is None:
        return False
    if groups is not None and _n_rows(groups) != _n_rows(ctx.X_train):
        # Row-capped training data, most likely. Local resolution re-derives the
        # groups against the rows actually being fitted.
        logger.debug("splitter groups do not match the training rows; resolving locally")
        return False

    # The splitter matches its own canonical strategy vocabulary exactly. If the
    # strategy string came from somewhere else and says "grouped" or "temporal"
    # while the returned folds are plain K-fold, adopting it would quietly throw
    # away the leakage protection the strategy was chosen for.
    strategy = (state.splits.strategy or "").lower()
    name = type(cv).__name__
    if "group" in strategy and "Group" not in name:
        return False
    if any(t in strategy for t in ("time", "temporal", "expand", "chrono")) and (
        "TimeSeries" not in name
    ):
        return False
    if "Stratified" in name and not ctx.task.is_classification:
        # The splitter keys stratification off the strategy string, not the task.
        # A stale "stratified" strategy left on the state by an earlier
        # classification reading of the problem would hand back a StratifiedKFold
        # that raises "Supported target types are: (binary, multiclass)" on a
        # continuous target -- losing cross-validation here and, worse, the whole
        # tuning step, which cannot recover from a raising search. The local
        # resolution below already keys off the task, so defer to it.
        logger.debug("declining a stratified splitter for a %s task", ctx.task.value)
        return False
    ctx.cv = cv
    ctx.cv_groups = groups
    try:
        ctx.n_splits = int(cv.get_n_splits())
    except Exception:  # noqa: BLE001 - purely informational
        ctx.n_splits = n_splits
    return True


def _resolve_groups(state: RunState, ctx: TrainingContext) -> Any:
    """Recover the grouping key for ``X_train``, or None."""
    column = getattr(state.problem, "group_column", None) if state.problem else None
    if not column:
        return None
    frame = ctx.X_train
    try:
        if hasattr(frame, "columns") and column in frame.columns:
            return np.asarray(frame[column])
        source = state.working_df if state.working_df is not None else state.raw_df
        if source is None or not hasattr(source, "columns"):
            return None
        if column not in source.columns:
            return None
        if hasattr(frame, "index"):
            return np.asarray(source.loc[frame.index, column])
    except Exception as exc:  # noqa: BLE001 - grouping is a nice-to-have
        logger.debug("could not resolve groups for %s: %s", column, exc)
    return None


def _zoo_context(state: RunState, ctx: TrainingContext) -> dict[str, Any]:
    extras = state.extras if isinstance(state.extras, dict) else {}
    n_classes = len(ctx.labels) if ctx.labels else None
    out: dict[str, Any] = {
        "n_jobs": state.settings.n_jobs,
        "n_classes": n_classes,
        "season_length": extras.get("season_length") or _season_length(state),
    }
    for key in ("n_clusters", "contamination", "class_weight"):
        if extras.get(key) is not None:
            out[key] = extras[key]
    return out


def _season_length(state: RunState) -> int:
    """Guess rows-per-season from the profiled frequency of the time column."""
    profile = state.profile
    column = getattr(state.problem, "temporal_column", None) if state.problem else None
    if profile is None or not column:
        return 1
    info = profile.column(column)
    frequency = getattr(info, "inferred_frequency", None) if info else None
    if not frequency:
        return 1
    token = str(frequency).split("-")[0].strip()
    return _SEASON_BY_FREQ.get(token, _SEASON_BY_FREQ.get(token[:1], 1))


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


#: Families whose objective is distance- or penalty-based, so an unscaled input
#: does not merely train slower — it trains something different. Feature scaling
#: is the Feature Agent's *choice*; for these it is the model's *requirement*,
#: which is why it is enforced here rather than left to the plan.
_SCALE_SENSITIVE: frozenset[ModelFamily] = frozenset(
    {
        ModelFamily.LINEAR,
        ModelFamily.LOGISTIC,
        ModelFamily.RIDGE,
        ModelFamily.LASSO,
        ModelFamily.ELASTIC_NET,
        ModelFamily.SVM,
        ModelFamily.KNN,
        ModelFamily.NEURAL_NETWORK,
        ModelFamily.ONE_CLASS_SVM,
        ModelFamily.KMEANS,
        ModelFamily.DBSCAN,
        ModelFamily.LOCAL_OUTLIER_FACTOR,
    }
)

_SCALE_STEP = "requires_scale"


def _needs_scaling(estimator: Any, family: ModelFamily | None) -> bool:
    """Whether this family needs a scaler the plan may not have supplied."""
    if family is None or family not in _SCALE_SENSITIVE:
        return False
    # An estimator that already normalises internally does not need a second pass.
    params = getattr(estimator, "get_params", dict)()
    return not bool(params.get("normalize", False))


def wrap_with_preprocessor(
    state: RunState, estimator: Any, family: ModelFamily | None = None
) -> Any:
    """Put ``estimator`` behind the run's preprocessor in a Pipeline.

    The preprocessor is *cloned*, not reused: an already-fitted transformer
    dropped into cross-validation would have seen every fold's validation rows,
    and the resulting score would be unreproducible.

    For a scale-sensitive ``family`` a ``StandardScaler`` is inserted between the
    preprocessor and the model. That is deliberately not left to the feature
    plan: whether to engineer a scaled feature is a modelling judgement, but
    logistic regression, SVM, and KNN on raw-magnitude inputs are simply wrong —
    the penalty term and the distance metric are both dominated by whichever
    column happens to be measured in the largest units. A live run surfaced this
    as ``lbfgs failed to converge after 2000 iterations``. ``with_mean=False``
    keeps the step valid when upstream one-hot encoding produced a sparse matrix.

    Args:
        state: The run blackboard, read for ``state.preprocessor``.
        estimator: The unfitted model.
        family: The model family, used to decide whether scaling is mandatory.

    Returns:
        A ``Pipeline``, or the bare estimator when nothing needs to wrap it.
    """
    steps: list[tuple[str, Any]] = []

    preprocessor = state.preprocessor
    if preprocessor is not None:
        try:
            fresh = clone(preprocessor)
        except Exception as exc:  # noqa: BLE001 - non-estimator preprocessor
            state.add_warning(
                f"preprocessor could not be cloned ({exc}); fitting it inside "
                "cross-validation is not possible, so folds may be optimistic"
            )
            fresh = preprocessor
        steps.append((_PREPROCESSOR_STEP, fresh))

    if _needs_scaling(estimator, family):
        steps.append((_SCALE_STEP, StandardScaler(with_mean=False)))

    if not steps:
        return estimator
    steps.append((_MODEL_STEP, estimator))
    return Pipeline(steps)


def _final_estimator(fitted: Any) -> Any:
    if isinstance(fitted, Pipeline):
        return fitted.named_steps.get(_MODEL_STEP, fitted[-1])
    return fitted


def cluster_labels(fitted: Any, X: Any) -> Any:
    """Cluster assignments from a fitted estimator, inductive or not.

    KMeans and GaussianMixture can label unseen rows; DBSCAN cannot, and exposes
    only the ``labels_`` it produced during fitting. Both are handled here so the
    trainer does not have to special-case families.

    Args:
        fitted: A fitted clustering estimator or Pipeline.
        X: Rows to label.

    Returns:
        Array of cluster labels, or None when neither route is available.
    """
    model = _final_estimator(fitted)
    if hasattr(model, "predict"):
        try:
            return np.asarray(fitted.predict(X))
        except Exception as exc:  # noqa: BLE001 - fall through to labels_
            logger.debug("clustering predict failed: %s", exc)
    labels = getattr(model, "labels_", None)
    if labels is not None:
        return np.asarray(labels)
    return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_experiments(state: RunState) -> ExperimentLog:
    """Train and score every selected candidate, best-first.

    Each candidate is cross-validated on the training partition and then scored
    once on the held-out evaluation partition. Both numbers are recorded: the CV
    mean with its spread says whether the single held-out score is trustworthy.

    Args:
        state: The run blackboard. Reads ``model_selection``, ``splits``,
            ``preprocessor``, and the time budget; writes ``experiments``,
            ``best_model``, and ``best_pipeline``.

    Returns:
        The populated :class:`ExperimentLog`, also assigned to
        ``state.experiments``.

    Raises:
        NoViableModelError: When every candidate failed to train.
    """
    ctx = build_training_context(state)
    candidates = _resolve_candidates(state, ctx)
    log = ExperimentLog(
        primary_metric=ctx.metric, higher_is_better=ctx.higher_better
    )
    state.experiments = log

    limit = max(1, int(state.config.max_experiments or 1))
    if len(candidates) > limit:
        state.add_warning(
            f"{len(candidates)} candidates proposed but max_experiments={limit}; "
            f"training the top {limit} by rank"
        )
        candidates = candidates[:limit]

    best_fitted: Any = None
    state.bus.log(
        f"training {len(candidates)} candidate(s), optimising {ctx.metric} "
        f"on the {ctx.eval_partition} partition"
    )
    for note in ctx.notes:
        state.bus.log(f"training setup: {note}")

    for position, candidate in enumerate(candidates, start=1):
        remaining = state.time_remaining
        if remaining <= _MIN_CANDIDATE_SECONDS and position > 1:
            skipped = [c.family.value for c in candidates[position - 1 :]]
            state.add_warning(
                f"time budget nearly exhausted ({remaining:.0f}s left); skipped "
                f"{len(skipped)} candidate(s): {', '.join(skipped)}"
            )
            break

        result, fitted = _train_candidate(state, ctx, candidate)
        log.results.append(result)

        if result.failed:
            state.bus.warn(
                f"{candidate.family.value} failed: {result.error}",
            )
            continue

        if result.primary_score is not None:
            state.bus.metric(
                f"{candidate.family.value}.{ctx.metric}", result.primary_score
            )
        if is_better(result.primary_score, _best_score(log), ctx.metric):
            log.best_experiment_id = result.experiment_id
            best_fitted = fitted

    successes = [r for r in log.results if not r.failed]
    if not successes:
        errors = "; ".join(f"{r.family.value}: {r.error}" for r in log.results) or "none attempted"
        raise NoViableModelError(f"every candidate failed to train ({errors})")

    if log.best_experiment_id is None:
        # Everything trained but nothing could be scored on the primary metric.
        fallback = successes[0]
        log.best_experiment_id = fallback.experiment_id
        state.add_warning(
            f"no candidate produced a usable {ctx.metric}; falling back to the "
            f"first model that trained ({fallback.family.value})"
        )

    if best_fitted is not None:
        state.best_pipeline = best_fitted
        state.best_model = _final_estimator(best_fitted)

    log.leaderboard_notes = format_leaderboard(log, ctx)
    best = log.best()
    if best is not None:
        state.bus.log(
            f"winner: {best.family.value} with {ctx.metric}="
            f"{_fmt(best.primary_score)} on {ctx.eval_partition}"
        )
    return log


def _resolve_candidates(state: RunState, ctx: TrainingContext) -> list[ModelCandidate]:
    """Rank-ordered candidates, filtered to what can actually be built."""
    selection = state.model_selection
    proposed = list(selection.candidates) if selection else []
    proposed.sort(key=lambda c: (c.rank, c.family.value))

    usable: list[ModelCandidate] = []
    for candidate in proposed:
        if not is_available(candidate.family):
            state.add_warning(
                f"candidate '{candidate.family.value}' skipped: its package is "
                "not installed in this environment"
            )
            continue
        if not supports_task(candidate.family, ctx.task):
            # It cannot train, so letting it consume an experiment slot and fail
            # would shorten the leaderboard for no information gain.
            state.add_warning(
                f"candidate '{candidate.family.value}' skipped: it does not apply "
                f"to a {ctx.task.value} task"
            )
            continue
        usable.append(candidate)

    if usable:
        return usable

    fallback = available_families(ctx.task)[:3]
    if not fallback:
        raise TrainingError(
            f"no model family in the zoo supports task '{ctx.task.value}'"
        )
    state.add_warning(
        "no usable model candidates were selected; falling back to "
        + ", ".join(f.value for f in fallback)
    )
    return [
        ModelCandidate(
            family=family,
            rank=index + 1,
            suitability="fair",
            rationale="Default fallback: the Model Selection Agent produced no "
            "usable candidate for this task.",
            is_baseline=family is ModelFamily.BASELINE_DUMMY,
        )
        for index, family in enumerate(fallback)
    ]


def _best_score(log: ExperimentLog) -> float | None:
    current = log.best()
    return current.primary_score if current else None


def _train_candidate(
    state: RunState, ctx: TrainingContext, candidate: ModelCandidate
) -> tuple[ExperimentResult, Any]:
    """Train, score, and persist one candidate. Never raises."""
    family = candidate.family
    params = params_to_dict(candidate.initial_params)
    try:
        rejected = unsupported_params(family, ctx.task, params, **ctx.zoo_ctx)
        if rejected:
            state.add_warning(
                f"{family.value}: ignored unsupported hyperparameter(s) "
                f"{', '.join(rejected)}"
            )
        estimator = build_estimator(
            family,
            ctx.task,
            params,
            random_state=ctx.random_state,
            **ctx.zoo_ctx,
        )
        pipeline = wrap_with_preprocessor(state, estimator, family)

        cv_scores = cross_validate_pipeline(state, ctx, pipeline, family)
        fitted, train_seconds, peak_mb = fit_measured(pipeline, ctx)

        result = record_experiment(
            state,
            ctx,
            family,
            params,
            fitted,
            cv_scores=cv_scores,
            train_seconds=train_seconds,
            peak_memory_mb=peak_mb,
            is_baseline=candidate.is_baseline,
        )
    except Exception as exc:  # noqa: BLE001 - a family failing is expected
        failure = ExperimentResult(
            family=family,
            label=family.value,
            params=dict_to_params(params),
            primary_metric=ctx.metric,
            is_baseline=candidate.is_baseline,
            failed=True,
            error=f"{type(exc).__name__}: {_brief(exc, limit=600)}",
        )
        logger.warning("candidate %s failed: %s", family.value, _brief(exc))
        logger.debug("candidate %s traceback", family.value, exc_info=True)
        return failure, None

    return result, fitted


def record_experiment(
    state: RunState,
    ctx: TrainingContext,
    family: ModelFamily,
    params: dict,
    fitted: Any,
    *,
    label: str = "",
    tuned: bool = False,
    cv_scores: list[float] | None = None,
    train_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
    is_baseline: bool = False,
) -> ExperimentResult:
    """Score, size, persist, and describe one already-fitted model.

    Shared by the training loop and the tuner so a tuned model and an untuned
    one are measured by identical code on the identical partition — the only way
    ``improvement`` means anything.

    Args:
        state: The run blackboard.
        ctx: The resolved training context.
        family: Which family produced ``fitted``.
        params: The hyperparameters actually used.
        fitted: A fitted estimator or Pipeline.
        label: Display label; defaults to the family name.
        tuned: Whether these params came out of hyperparameter search.
        cv_scores: Cross-validated scores in natural units, if any.
        train_seconds: Measured fit time.
        peak_memory_mb: Measured peak Python allocation during fit.
        is_baseline: Whether this candidate is the reference baseline.

    Returns:
        A populated :class:`ExperimentResult`.
    """
    folds = list(cv_scores or [])
    result = ExperimentResult(
        family=family,
        label=label or (f"{family.value} (tuned)" if tuned else family.value),
        params=dict_to_params(params),
        primary_metric=ctx.metric,
        is_baseline=is_baseline,
        tuned=tuned,
        cv_scores=folds,
        train_seconds=train_seconds,
        peak_memory_mb=peak_memory_mb,
    )

    metrics, predict_seconds = score_fitted_model(state, ctx, fitted, family)
    result.predict_seconds = predict_seconds
    result.metrics = [
        MetricValue(name=name, value=value) for name, value in sorted(metrics.items())
    ]
    result.primary_score = metrics.get(ctx.metric)

    if result.primary_score is None and folds:
        # The held-out partition could not produce the primary metric but the
        # folds could; the record says explicitly which number is on the board.
        result.primary_score = float(np.mean(folds))
        result.metrics.append(
            MetricValue(
                name=f"{ctx.metric}_cv_mean",
                value=result.primary_score,
                std=float(np.std(folds)),
            )
        )
    elif result.primary_score is None:
        # It trained but cannot be ranked. Saying so is the point: an unscored
        # model silently sorted to the bottom would look like a bad model.
        state.add_warning(
            f"{family.value}: trained successfully but '{ctx.metric}' was not "
            "computable from its predictions, so it cannot be ranked"
        )
    if folds:
        result.metrics.append(
            MetricValue(
                name="cv_mean",
                value=float(np.mean(folds)),
                std=float(np.std(folds)),
            )
        )

    result.n_features_in = _n_features(fitted, ctx.X_train)
    size, path = _persist(state, fitted, result.experiment_id, family)
    result.model_size_bytes = size
    result.artifact_path = path
    return result


def cross_validate_pipeline(
    state: RunState, ctx: TrainingContext, pipeline: Any, family: ModelFamily
) -> list[float]:
    """Cross-validate on the training partition, in the metric's natural units.

    Args:
        state: The run blackboard, for warnings and the time budget.
        ctx: The resolved training context, which owns the fold generator.
        pipeline: An unfitted estimator or Pipeline. It is cloned per fold.
        family: The family, used only in warning messages.

    Returns:
        One score per fold that produced one. Empty when the folds could not be
        run at all — cross-validation is informative here, not required.
    """
    if ctx.cv is None or ctx.scorer is None or not ctx.supervised:
        return []
    if state.time_remaining < _MIN_CV_SECONDS:
        state.add_warning(
            f"{family.value}: skipped cross-validation to stay inside the time budget"
        )
        return []
    try:
        # n_jobs=1: the estimators already parallelise internally, and nesting
        # loky pools on Windows costs more than it saves.
        outcome = cross_validate(
            clone(pipeline),
            ctx.X_train,
            ctx.y_train,
            groups=ctx.cv_groups,
            scoring=ctx.scorer.scorer,
            cv=ctx.cv,
            n_jobs=1,
            error_score=np.nan,
        )
    except Exception as exc:  # noqa: BLE001 - CV is informative, not required
        state.add_warning(
            f"{family.value}: cross-validation failed ({_brief(exc)})"
        )
        return []
    raw = np.asarray(outcome.get("test_score", []), dtype=float)
    scores = [ctx.scorer.to_natural(v) for v in raw if np.isfinite(v)]
    if len(scores) < raw.size:
        state.add_warning(
            f"{family.value}: {raw.size - len(scores)} of {raw.size} folds "
            "produced no score and were excluded from the CV mean"
        )
    return scores


def fit_measured(pipeline: Any, ctx: TrainingContext) -> tuple[Any, float, float]:
    """Fit under tracemalloc, then report seconds and peak Python memory.

    Only the fit runs traced. tracemalloc roughly doubles allocation cost, so
    prediction is timed afterwards with tracing off — that number feeds the
    deployment latency estimate and must not be inflated. It also cannot see
    memory allocated in worker processes, so this is a floor, not a ceiling.

    Args:
        pipeline: An unfitted estimator or Pipeline.
        ctx: The resolved training context supplying the training partition.

    Returns:
        ``(fitted, train_seconds, peak_memory_mb)``.
    """
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()

    start = time.perf_counter()
    try:
        if ctx.supervised:
            pipeline.fit(ctx.X_train, ctx.y_train)
        else:
            pipeline.fit(ctx.X_train)
        train_seconds = time.perf_counter() - start
        peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        if not already_tracing:
            tracemalloc.stop()
    return pipeline, train_seconds, peak_bytes / 1e6


def score_fitted_model(
    state: RunState, ctx: TrainingContext, fitted: Any, family: ModelFamily
) -> tuple[dict[str, float], float]:
    """Predict on the evaluation partition and score, timing the prediction.

    Args:
        state: The run blackboard, for warnings.
        ctx: The resolved training context, which owns the partition choice.
        fitted: A fitted estimator or Pipeline.
        family: The family, used to decide whether probabilities are available.

    Returns:
        ``(metrics, predict_seconds)``. Metrics absent from the dict were not
        computable for these inputs.
    """
    if ctx.task is TaskType.CLUSTERING:
        return _score_clustering_model(state, ctx, fitted)
    if ctx.task is TaskType.ANOMALY_DETECTION:
        return _score_anomaly_model(state, ctx, fitted)

    X_eval, y_eval, partition = ctx.X_eval, ctx.y_eval, ctx.eval_partition
    if X_eval is None or y_eval is None:
        # No held-out data at all: scoring on train is the only option, and it
        # is labelled as such rather than reported as a generalisation estimate.
        X_eval, y_eval, partition = ctx.X_train, ctx.y_train, "train"
        state.add_warning(
            f"{family.value}: no validation or test partition; the reported "
            "score is an in-sample training score, not a generalisation estimate"
        )

    start = time.perf_counter()
    y_pred = fitted.predict(X_eval)
    proba = _probabilities(fitted, X_eval, ctx, family)
    predict_seconds = time.perf_counter() - start

    metrics = score_predictions(
        ctx.task,
        y_eval,
        y_pred,
        proba,
        ctx.labels,
        pos_label=ctx.pos_label,
    )
    if partition == "train":
        # Flagged in the record itself so no downstream reader mistakes an
        # in-sample number for a held-out one.
        metrics["evaluated_on_train"] = 1.0
    return metrics, predict_seconds


def _score_clustering_model(
    state: RunState, ctx: TrainingContext, fitted: Any
) -> tuple[dict[str, float], float]:
    model = _final_estimator(fitted)
    inductive = hasattr(model, "predict")
    X_metric_source = ctx.X_eval if (inductive and ctx.X_eval is not None) else ctx.X_train
    start = time.perf_counter()
    labels = cluster_labels(fitted, X_metric_source)
    predict_seconds = time.perf_counter() - start
    if labels is None:
        raise TrainingError("clustering model produced no labels")
    if not inductive:
        state.add_warning(
            f"{type(model).__name__} cannot label unseen rows; cluster quality "
            "is measured on the training partition"
        )
    # Internal indices only mean something in the space the model clustered in.
    return (
        score_predictions(
            ctx.task, None, labels, X=_transformed(fitted, X_metric_source)
        ),
        predict_seconds,
    )


def _score_anomaly_model(
    state: RunState, ctx: TrainingContext, fitted: Any
) -> tuple[dict[str, float], float]:
    X_eval = ctx.X_eval if ctx.X_eval is not None else ctx.X_train
    start = time.perf_counter()
    flags = np.asarray(fitted.predict(X_eval))
    scores = None
    model = _final_estimator(fitted)
    for method in ("decision_function", "score_samples"):
        if hasattr(model, method):
            try:
                scores = np.asarray(getattr(fitted, method)(X_eval), dtype=float)
                break
            except Exception as exc:  # noqa: BLE001 - scores are optional
                logger.debug("%s failed: %s", method, exc)
    predict_seconds = time.perf_counter() - start
    return (
        score_predictions(
            ctx.task,
            ctx.y_eval,
            flags,
            scores,
            X=_transformed(fitted, X_eval),
        ),
        predict_seconds,
    )


def _probabilities(
    fitted: Any, X: Any, ctx: TrainingContext, family: ModelFamily
) -> Any:
    """Class scores for ranking metrics: probabilities, else decision margins."""
    if not ctx.task.is_classification:
        return None
    if supports_proba(family, ctx.task) and hasattr(fitted, "predict_proba"):
        try:
            return np.asarray(fitted.predict_proba(X), dtype=float)
        except Exception as exc:  # noqa: BLE001 - fall back to margins
            logger.debug("predict_proba failed for %s: %s", family.value, exc)
    if hasattr(fitted, "decision_function"):
        try:
            return np.asarray(fitted.decision_function(X), dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.debug("decision_function failed for %s: %s", family.value, exc)
    return None


def _transformed(fitted: Any, X: Any) -> Any:
    """``X`` in the space the final estimator actually saw."""
    if not isinstance(fitted, Pipeline):
        return X
    try:
        return fitted[:-1].transform(X)
    except Exception as exc:  # noqa: BLE001 - raw space is a usable fallback
        logger.debug("could not transform X for metrics: %s", exc)
        return X


def _persist(
    state: RunState, fitted: Any, experiment_id: str, family: ModelFamily
) -> tuple[int, str | None]:
    """Serialise once: the buffer gives both the size and the artifact bytes."""
    buffer = io.BytesIO()
    try:
        joblib.dump(fitted, buffer)
    except Exception as exc:  # noqa: BLE001 - unpicklable model is not fatal
        state.add_warning(f"{family.value}: model could not be serialised ({exc})")
        return 0, None
    payload = buffer.getvalue()
    try:
        path = state.artifact_path("models", f"{experiment_id}_{family.value}.joblib")
        path.write_bytes(payload)
        state.bus.artifact(str(path), kind="model")
        return len(payload), str(path)
    except Exception as exc:  # noqa: BLE001 - disk failure must not lose the run
        state.add_warning(f"{family.value}: model could not be written to disk ({exc})")
        return len(payload), None


def _n_features(fitted: Any, X: Any) -> int:
    for candidate in (fitted, _final_estimator(fitted)):
        value = getattr(candidate, "n_features_in_", None)
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
    shape = getattr(X, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        return int(shape[1])
    return 0


# ---------------------------------------------------------------------------
# Final refit
# ---------------------------------------------------------------------------


def fit_final_model(state: RunState, family: ModelFamily, params: dict) -> Any:
    """Refit one family on train + validation with fixed hyperparameters.

    Used once the winner is known: the validation rows were spent on model
    selection, so folding them back in gives the deployed model more data
    without touching the test partition that still has to measure it.

    Args:
        state: The run blackboard.
        family: Which family to refit.
        params: Concrete hyperparameters (already a plain dict).

    Returns:
        The fitted Pipeline, or the fitted estimator when the run has no
        preprocessor.

    Raises:
        TrainingError: If the estimator cannot be built or fitted.
    """
    ctx = build_training_context(state)
    # Only fold in validation when its *encoded* target is available; adding
    # rows to X without their labels would corrupt the fit silently.
    use_validation = ctx.eval_partition == "validation" and (
        not ctx.supervised or ctx.y_eval is not None
    )
    X = _concat_rows(ctx.X_train, state.splits.X_valid if use_validation else None)
    y = None
    if ctx.supervised:
        y = _concat_rows(ctx.y_train, ctx.y_eval if use_validation else None)

    estimator = build_estimator(
        family, ctx.task, params or {}, random_state=ctx.random_state, **ctx.zoo_ctx
    )
    pipeline = wrap_with_preprocessor(state, estimator, family)
    try:
        if y is None:
            pipeline.fit(X)
        else:
            pipeline.fit(X, y)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed failure
        raise TrainingError(f"final refit of {family.value} failed: {exc}") from exc
    state.bus.log(
        f"refitted {family.value} on {_n_rows(X):,} train+validation rows"
    )
    return pipeline


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def format_leaderboard(log: ExperimentLog, ctx: TrainingContext | None = None) -> str:
    """Render the experiment log as a plain-text leaderboard.

    Args:
        log: The populated experiment log.
        ctx: Optional context, used to name the evaluation partition.

    Returns:
        A multi-line string, best model first, failures listed last.
    """
    metric = log.primary_metric or "score"
    partition = ctx.eval_partition if ctx else "held-out"
    ranked = [r for r in log.results if not r.failed and r.primary_score is not None]
    ranked.sort(key=lambda r: r.primary_score, reverse=log.higher_is_better)

    lines = [
        f"Leaderboard by {metric} on the {partition} partition "
        f"({'higher' if log.higher_is_better else 'lower'} is better):",
        f"{'#':>2}  {'family':<24}{metric:>14}{'cv mean':>12}{'cv sd':>9}"
        f"{'fit s':>9}{'size KB':>10}",
    ]
    for index, result in enumerate(ranked, start=1):
        cv_mean = float(np.mean(result.cv_scores)) if result.cv_scores else float("nan")
        cv_sd = float(np.std(result.cv_scores)) if result.cv_scores else float("nan")
        marker = "*" if result.experiment_id == log.best_experiment_id else " "
        lines.append(
            f"{index:>2}{marker} {result.family.value:<24}"
            f"{_fmt(result.primary_score):>14}{_fmt(cv_mean):>12}{_fmt(cv_sd):>9}"
            f"{result.train_seconds:>9.2f}{result.model_size_bytes / 1024:>10.1f}"
        )
    unscored = [r for r in log.results if not r.failed and r.primary_score is None]
    for result in unscored:
        lines.append(f"    {result.family.value:<24}{'trained, unscored':>14}")
    for result in (r for r in log.results if r.failed):
        lines.append(f"    {result.family.value:<24}FAILED: {result.error}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.5g}"


def _brief(exc: BaseException, limit: int = 240) -> str:
    """One-line, length-capped exception text.

    sklearn's aggregate cross-validation error embeds full tracebacks for every
    failed fold. Verbatim, that turns one warning into two kilobytes of the run
    summary; the detail is already in the log at WARNING level with exc_info.
    """
    text = " ".join(str(exc).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Frame helpers (pandas 3.x: no chained assignment, copy-on-write default)
# ---------------------------------------------------------------------------


def _n_rows(frame: Any) -> int:
    if frame is None:
        return 0
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0])
    try:
        return int(len(frame))
    except TypeError:
        return 0


def _take_rows(frame: Any, positions: np.ndarray) -> Any:
    if frame is None:
        return None
    if hasattr(frame, "iloc"):
        return frame.iloc[positions]
    return np.asarray(frame)[positions]


def _concat_rows(first: Any, second: Any) -> Any:
    if second is None or _n_rows(second) == 0:
        return first
    if first is None:
        return second
    if hasattr(first, "iloc") and hasattr(second, "iloc"):
        import pandas as pd

        return pd.concat([first, second], axis=0)
    return np.concatenate([np.asarray(first), np.asarray(second)], axis=0)


def _null_mask(array: np.ndarray) -> np.ndarray:
    """Boolean mask of missing entries, safe for object and extension dtypes."""
    import pandas as pd

    return np.asarray(pd.isna(array), dtype=bool)


def _encode_labels(encoder: Any, y: Any, as_str: bool) -> Any:
    if y is None:
        return None
    values = np.asarray(y).ravel()
    if as_str:
        values = values.astype(str)
    return np.asarray(encoder.transform(values))
