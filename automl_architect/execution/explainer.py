"""Explainability executor: measured attributions, not plausible ones.

The Explainability Agent narrates whatever this module measures, so everything
here is computed from the real fitted pipeline against held-out rows. Three
choices are load-bearing:

*   **Permutation importance is always computed, on held-out data.** Importance
    measured on the training partition reflects what the model memorised, not
    what it uses to generalise, so the validation partition is preferred and the
    test partition is the fallback. It is also the only method that works for
    every estimator, which is why it is the floor the report can never drop
    below.
*   **SHAP is best-effort and strictly additive.** The package may be missing,
    the model may be unsupported, and ``KernelExplainer`` is slow enough to be a
    liability under a wall-clock budget. Each of those degrades to a warning and
    leaves the permutation numbers untouched.
*   **Names are translated back to the user's vocabulary.** A model trained on a
    ``ColumnTransformer`` sees ``onehot__city_Paris``; the person reading the
    report has a column called ``city``. Expanded columns are folded back into
    their source column so the narrative speaks about the dataset rather than
    about the encoder.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..core.schemas import (
    Counterfactual,
    ExplainabilityReport,
    FeatureAttribution,
    Param,
    TaskType,
)
from ..core.state import RunState
from ._metric_bridge import metrics_source, resolve_scorer

logger = logging.getLogger(__name__)

# --- runtime budgets -------------------------------------------------------
# Every cap here exists because the orchestrator runs under a wall-clock budget;
# an unbounded explanation pass on a wide frame can outlast the training it is
# explaining.
MAX_PERMUTATION_ROWS = 5_000
PERMUTATION_REPEATS = 5
MAX_SHAP_EXPLAIN_ROWS = 500
MAX_SHAP_BACKGROUND_ROWS = 100
KERNEL_SHAP_MAX_ROWS = 50
KERNEL_SHAP_MAX_BACKGROUND = 25
KERNEL_SHAP_MAX_FEATURES = 30
KERNEL_SHAP_NSAMPLES = 100
KERNEL_SHAP_MIN_SECONDS_LEFT = 60.0
MAX_PDP_FEATURES = 4
MAX_PDP_ROWS = 1_000
PDP_GRID_RESOLUTION = 30
MAX_COUNTERFACTUAL_FEATURES = 3
MAX_ATTRIBUTIONS = 25
SHAP_PLOT_MAX_FEATURES = 20

_TREE_MODULE_PREFIXES = (
    "sklearn.ensemble",
    "sklearn.tree",
    "xgboost",
    "lightgbm",
    "catboost",
)


# ---------------------------------------------------------------------------
# Frame helpers (tolerate DataFrame or ndarray inputs)
# ---------------------------------------------------------------------------


def row_count(X: Any) -> int:
    """Number of rows in a DataFrame or 2-D array, 0 for ``None``."""
    if X is None:
        return 0
    try:
        return int(len(X))
    except TypeError:
        return 0


def is_frame(X: Any) -> bool:
    """Whether ``X`` looks like a pandas DataFrame."""
    return hasattr(X, "columns") and hasattr(X, "iloc")


def column_names(X: Any) -> list[str]:
    """Column labels for a DataFrame, positional names for an array."""
    if X is None:
        return []
    if is_frame(X):
        return [str(c) for c in X.columns]
    arr = np.asarray(X)
    width = arr.shape[1] if arr.ndim > 1 else 1
    return [f"feature_{i}" for i in range(width)]


def take_rows(X: Any, positions: Sequence[int] | np.ndarray) -> Any:
    """Positional row selection that works for frames, Series, and arrays."""
    idx = np.asarray(positions, dtype=int)
    if X is None:
        return None
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def head_rows(X: Any, n: int) -> Any:
    """First ``n`` rows."""
    total = row_count(X)
    if total == 0:
        return X
    return take_rows(X, np.arange(min(n, total)))


def to_dense_array(X: Any) -> np.ndarray:
    """Materialise a dense float array from a frame, array, or sparse matrix."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    return np.asarray(X)


def _subsample_positions(n: int, cap: int, random_state: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=cap, replace=False))


_NO_MATPLOTLIB_NOTE = (
    "matplotlib is not installed, so charts were rendered with plotly/kaleido "
    "instead (bar chart in place of a SHAP beeswarm)."
)
_pyplot_cache: list[Any] = []


def _load_pyplot() -> Any | None:
    """Import ``pyplot`` on the Agg backend, or ``None`` if matplotlib is absent.

    matplotlib is an optional dependency here: it is the nicest way to draw a
    SHAP beeswarm and sklearn's PDP display, but the run must not lose its
    explanations because a headless worker lacks it.
    """
    if _pyplot_cache:
        return _pyplot_cache[0]
    plt: Any | None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)  # headless: no display on a worker
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:
        logger.debug("matplotlib unavailable: %s", exc)
        plt = None
    _pyplot_cache.append(plt)
    return plt


def _note_once(notes: list[str], message: str) -> None:
    if message not in notes:
        notes.append(message)


# ---------------------------------------------------------------------------
# Prediction context
# ---------------------------------------------------------------------------


@dataclass
class PredictionContext:
    """A fitted model plus the partitions it can actually be called on.

    Resolving this once is what lets both explainer and diagnostics stay honest
    about *which* matrices the estimator accepts: a full pipeline takes the raw
    feature frame, a bare estimator needs ``state.preprocessor`` applied first.
    """

    estimator: Any
    final_model: Any
    task: TaskType
    primary_metric: str = ""
    X_train: Any = None
    y_train: Any = None
    X_valid: Any = None
    y_valid: Any = None
    X_test: Any = None
    y_test: Any = None
    eval_partition: str = "test"
    inner_preprocessor: Any = None
    input_names: list[str] = field(default_factory=list)
    transformed_names: list[str] = field(default_factory=list)
    original_columns: list[str] = field(default_factory=list)
    labels: list[Any] | None = None
    positive_index: int = 1
    pre_transformed: bool = False
    random_state: int = 42
    notes: list[str] = field(default_factory=list)

    # -- partitions -------------------------------------------------------

    @property
    def is_classification(self) -> bool:
        return bool(self.task and self.task.is_classification)

    def partition(self, name: str) -> tuple[Any, Any]:
        """``(X, y)`` for ``'train'``, ``'validation'``, or ``'test'``."""
        return {
            "train": (self.X_train, self.y_train),
            "validation": (self.X_valid, self.y_valid),
            "test": (self.X_test, self.y_test),
        }.get(name, (None, None))

    def available_partitions(self) -> list[str]:
        return [
            name
            for name in ("train", "validation", "test")
            if row_count(self.partition(name)[0]) > 0
        ]

    @property
    def X_eval(self) -> Any:
        return self.partition(self.eval_partition)[0]

    @property
    def y_eval(self) -> Any:
        return self.partition(self.eval_partition)[1]

    # -- inference --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray | None:
        """Class probabilities, or ``None`` when the model cannot produce them."""
        method = getattr(self.estimator, "predict_proba", None)
        if method is None or not self.is_classification:
            return None
        try:
            return np.asarray(method(X))
        except Exception as exc:
            logger.debug("predict_proba failed: %s", exc)
            return None

    def transform(self, X: Any) -> np.ndarray:
        """Model-space matrix for ``X`` — what SHAP must be handed."""
        if self.inner_preprocessor is not None and not self.pre_transformed:
            return to_dense_array(self.inner_preprocessor.transform(X))
        return to_dense_array(X)

    def model_space_names(self) -> list[str]:
        return self.transformed_names or self.input_names

    @property
    def positive_label(self) -> Any:
        """The class probabilities are reported for, on binary problems."""
        if self.labels and 0 <= self.positive_index < len(self.labels):
            return self.labels[self.positive_index]
        return 1

    @property
    def scorer_pos_label(self) -> Any:
        """``pos_label`` for sklearn's binary threshold scorers, else ``None``.

        Only binary classification has a positive class to bind; passing one for
        a regression or multiclass metric would be meaningless.
        """
        if self.task is not TaskType.BINARY_CLASSIFICATION:
            return None
        if self.labels and len(self.labels) != 2:
            return None
        return self.positive_label


def _unwrap_pipeline(estimator: Any) -> tuple[Any, Any]:
    """Split a sklearn ``Pipeline`` into ``(preprocessing, final estimator)``."""
    steps = getattr(estimator, "steps", None)
    if not steps:
        return None, estimator
    final = steps[-1][1]
    if len(steps) == 1:
        return None, final
    try:
        return estimator[:-1], final
    except Exception:  # pragma: no cover - non-standard pipeline subclass
        return None, final


def _probe(estimator: Any, X: Any) -> bool:
    if estimator is None or row_count(X) == 0:
        return False
    try:
        estimator.predict(head_rows(X, min(3, row_count(X))))
        return True
    except Exception as exc:
        logger.debug("predict probe failed: %s", exc)
        return False


def build_prediction_context(state: RunState) -> PredictionContext | None:
    """Resolve the fitted model and the matrices it accepts.

    Returns ``None`` (after recording a warning) when there is no fitted model or
    no usable partition, which is the signal for callers to emit an empty report
    rather than raise.

    Args:
        state: The run blackboard, read for ``best_pipeline``/``best_model``,
            ``preprocessor``, and ``splits``.

    Returns:
        A :class:`PredictionContext`, or ``None`` if predictions are impossible.
    """
    estimator = state.best_pipeline if state.best_pipeline is not None else state.best_model
    if estimator is None:
        state.add_warning(
            "explainability: no fitted model on the run state; skipping attributions."
        )
        return None

    splits = state.splits
    parts: dict[str, tuple[Any, Any]] = {
        "train": (splits.X_train, splits.y_train),
        "validation": (splits.X_valid, splits.y_valid),
        "test": (splits.X_test, splits.y_test),
    }
    if all(row_count(X) == 0 for X, _ in parts.values()):
        state.add_warning(
            "explainability: no materialised splits on the run state; skipping attributions."
        )
        return None

    first_X = next(X for X, _ in parts.values() if row_count(X) > 0)
    notes: list[str] = []
    pre_transformed = False

    if not _probe(estimator, first_X):
        # A bare estimator was stored: apply the run's preprocessor ourselves.
        preprocessor = state.preprocessor
        candidate = state.best_model if state.best_model is not None else estimator
        transformed: dict[str, tuple[Any, Any]] | None = None
        if preprocessor is not None:
            try:
                transformed = {
                    name: (
                        (to_dense_array(preprocessor.transform(X)) if row_count(X) else X),
                        y,
                    )
                    for name, (X, y) in parts.items()
                }
            except Exception as exc:
                transformed = None
                notes.append(f"preprocessor.transform failed ({exc}).")
        if transformed is not None and _probe(
            candidate, next(X for X, _ in transformed.values() if row_count(X) > 0)
        ):
            parts = transformed
            estimator = candidate
            pre_transformed = True
            notes.append(
                "model requires pre-transformed input; state.preprocessor was applied "
                "before scoring, so feature names come from the encoder."
            )
        else:
            state.add_warning(
                "explainability: the fitted model could not predict on the stored "
                "splits; skipping attributions."
            )
            return None

    inner_pre, final_model = _unwrap_pipeline(estimator)

    if row_count(parts["validation"][0]) > 0:
        eval_partition = "validation"
    elif row_count(parts["test"][0]) > 0:
        eval_partition = "test"
    else:
        eval_partition = "train"
        notes.append(
            "no held-out partition available; importance was measured on the "
            "training rows and therefore reflects memorisation as well as signal."
        )

    input_names = column_names(parts[eval_partition][0])
    transformed_names: list[str] = []
    if inner_pre is not None:
        try:
            transformed_names = [str(n) for n in inner_pre.get_feature_names_out()]
        except Exception as exc:
            logger.debug("get_feature_names_out failed: %s", exc)
    if not transformed_names:
        n_out = getattr(final_model, "n_features_in_", None)
        if n_out and int(n_out) == len(input_names):
            transformed_names = list(input_names)

    labels: list[Any] | None = None
    if state.task_type is not None and state.task_type.is_classification:
        raw_labels = getattr(final_model, "classes_", None)
        if raw_labels is None:
            y_ref = parts["train"][1] if parts["train"][1] is not None else parts[eval_partition][1]
            try:
                raw_labels = np.unique(np.asarray(y_ref).ravel())
            except Exception:
                raw_labels = None
        if raw_labels is not None:
            labels = list(np.asarray(raw_labels).tolist())

    # The operator's positive class decides which probability column every
    # downstream number is quoted against, so it is resolved once here.
    positive_index = 1 if labels and len(labels) > 1 else 0
    wanted = state.problem.positive_class if state.problem else None
    if wanted is not None and labels:
        for index, label in enumerate(labels):
            if str(label) == str(wanted):
                positive_index = index
                break

    original = _original_columns(state)

    return PredictionContext(
        estimator=estimator,
        final_model=final_model,
        task=state.task_type or TaskType.REGRESSION,
        primary_metric=state.primary_metric,
        X_train=parts["train"][0],
        y_train=parts["train"][1],
        X_valid=parts["validation"][0],
        y_valid=parts["validation"][1],
        X_test=parts["test"][0],
        y_test=parts["test"][1],
        eval_partition=eval_partition,
        inner_preprocessor=inner_pre,
        input_names=input_names,
        transformed_names=transformed_names,
        original_columns=original,
        labels=labels,
        positive_index=positive_index,
        pre_transformed=pre_transformed,
        random_state=state.config.random_state,
        notes=notes,
    )


def _original_columns(state: RunState) -> list[str]:
    """The user's own column vocabulary, longest names first for prefix matching."""
    names: list[str] = []
    for frame in (state.working_df, state.raw_df):
        if frame is not None and hasattr(frame, "columns"):
            names.extend(str(c) for c in frame.columns)
    if state.profile:
        names.extend(c.name for c in state.profile.columns)
    target = state.target
    unique = {n for n in names if n and n != target}
    return sorted(unique, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Name mapping and normalisation
# ---------------------------------------------------------------------------


def display_name(name: str, *, model_features: set[str], originals: Sequence[str]) -> str:
    """Translate an encoder-mangled feature name back to the user's column.

    ``onehot__city_Paris`` becomes ``city``; ``num__age`` becomes ``age``; a
    genuinely engineered feature such as ``amount_per_item`` is left alone
    because no original column explains it better.
    """
    label = str(name)
    if label in model_features:
        base = label
    else:
        base = label.split("__", 1)[1] if "__" in label else label
    if base in model_features or base in originals:
        return base
    candidates = [c for c in originals if base.startswith(f"{c}_")]
    if candidates:
        return max(candidates, key=len)
    return base


def _merge_direction(current: str | None, incoming: str) -> str:
    if current is None or current == incoming:
        return incoming
    if current == "unknown":
        return incoming
    if incoming == "unknown":
        return current
    return "mixed"


def _display_directions(
    directions: dict[str, str], ctx: PredictionContext
) -> dict[str, str]:
    """Collapse per-encoded-column directions onto display names."""
    model_features = set(ctx.input_names)
    merged: dict[str, str] = {}
    for raw_name, direction in directions.items():
        label = display_name(
            raw_name, model_features=model_features, originals=ctx.original_columns
        )
        merged[label] = _merge_direction(merged.get(label), direction)
    return merged


def _finalise_attributions(
    values: dict[str, float],
    directions: dict[str, str],
    method: str,
    ctx: PredictionContext,
    *,
    top_n: int = MAX_ATTRIBUTIONS,
    fallback_directions: dict[str, str] | None = None,
) -> list[FeatureAttribution]:
    """Fold expanded columns together, clip, normalise to 1.0, and rank."""
    if not values:
        return []
    model_features = set(ctx.input_names)
    originals = ctx.original_columns

    merged: dict[str, float] = {}
    merged_dir: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        # Negative permutation importance means "the model was better without a
        # faithful copy of this column" — noise, not evidence of harm, so it is
        # floored at zero before normalisation.
        value = max(value, 0.0)
        label = display_name(raw_name, model_features=model_features, originals=originals)
        merged[label] = merged.get(label, 0.0) + value
        merged_dir[label] = _merge_direction(
            merged_dir.get(label), directions.get(raw_name, "unknown")
        )

    total = sum(merged.values())
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    out: list[FeatureAttribution] = []
    for label, value in ranked:
        direction = merged_dir.get(label, "unknown")
        if direction == "unknown" and fallback_directions:
            direction = fallback_directions.get(label, "unknown")
        out.append(
            FeatureAttribution(
                feature=label,
                importance=float(value / total) if total > 0 else 0.0,
                direction=direction,  # type: ignore[arg-type]
                method=method,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def _permutation_importance(
    state: RunState, ctx: PredictionContext, notes: list[str]
) -> tuple[dict[str, float], list[str]]:
    """Permutation importance on the held-out partition.

    Returns ``(values by model-input feature, features ranked best first)``.
    """
    from sklearn.inspection import permutation_importance

    X, y = ctx.partition(ctx.eval_partition)
    if row_count(X) == 0 or y is None:
        notes.append("permutation importance skipped: evaluation partition is empty.")
        return {}, []

    positions = _subsample_positions(row_count(X), MAX_PERMUTATION_ROWS, ctx.random_state)
    X_use, y_use = take_rows(X, positions), take_rows(y, positions)
    if len(positions) < row_count(X):
        notes.append(
            f"permutation importance sampled {len(positions):,} of {row_count(X):,} "
            f"{ctx.eval_partition} rows for runtime."
        )

    scorer, scorer_name, _ = resolve_scorer(
        ctx.primary_metric, ctx.task, pos_label=ctx.scorer_pos_label
    )
    try:
        result = permutation_importance(
            ctx.estimator,
            X_use,
            y_use,
            scoring=scorer if scorer is not None else None,
            n_repeats=PERMUTATION_REPEATS,
            random_state=ctx.random_state,
            n_jobs=None,
        )
    except Exception as exc:
        state.add_warning(f"permutation importance failed: {exc}")
        notes.append(f"permutation importance failed: {exc}")
        return {}, []

    names = column_names(X_use)
    means = np.asarray(result.importances_mean, dtype=float)
    values = {
        name: float(means[i]) for i, name in enumerate(names) if i < means.shape[0]
    }
    ranked = [name for name, _ in sorted(values.items(), key=lambda kv: kv[1], reverse=True)]
    notes.append(
        f"permutation importance: {PERMUTATION_REPEATS} repeats on the "
        f"{ctx.eval_partition} partition, scoring="
        f"{scorer_name or 'estimator.score'}."
    )
    return values, ranked


# ---------------------------------------------------------------------------
# Native model attributions
# ---------------------------------------------------------------------------


def _native_attributions(
    ctx: PredictionContext, notes: list[str]
) -> tuple[dict[str, float], dict[str, str], str]:
    """``feature_importances_`` or ``coef_`` from the final estimator."""
    model = ctx.final_model
    names = ctx.model_space_names()

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        arr = np.asarray(importances, dtype=float).ravel()
        if names and len(names) == arr.shape[0]:
            return (
                {names[i]: float(arr[i]) for i in range(arr.shape[0])},
                {},
                "feature_importances_",
            )
        notes.append(
            "native feature_importances_ ignored: length "
            f"{arr.shape[0]} does not match {len(names)} known feature names."
        )

    coef = getattr(model, "coef_", None)
    if coef is not None:
        arr = np.asarray(coef, dtype=float)
        magnitude = np.abs(arr).mean(axis=0) if arr.ndim > 1 else np.abs(arr)
        signed = arr.mean(axis=0) if arr.ndim > 1 else arr
        magnitude = magnitude.ravel()
        signed = signed.ravel()
        if names and len(names) == magnitude.shape[0]:
            values = {names[i]: float(magnitude[i]) for i in range(magnitude.shape[0])}
            directions = {
                names[i]: (
                    "increases"
                    if signed[i] > 0
                    else "decreases"
                    if signed[i] < 0
                    else "mixed"
                )
                for i in range(signed.shape[0])
            }
            if arr.ndim > 1 and arr.shape[0] > 1:
                # Averaged over one-vs-rest coefficients, so the sign is not a
                # single class's direction.
                directions = {name: "mixed" for name in values}
            notes.append(
                "native attributions from coef_ (absolute value; scale depends on "
                "whether features were standardised)."
            )
            return values, directions, "coef_"
        notes.append(
            f"coef_ ignored: length {magnitude.shape[0]} does not match "
            f"{len(names)} known feature names."
        )

    return {}, {}, ""


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def _looks_like_tree(model: Any) -> bool:
    module = type(model).__module__ or ""
    if module.startswith(_TREE_MODULE_PREFIXES):
        return True
    return hasattr(model, "feature_importances_") and hasattr(model, "n_features_in_")


def _reduce_shap_values(
    raw: Any, positive_index: int = 1
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Normalise SHAP output to ``(signed 2-D or None, magnitude 2-D)``.

    Modern SHAP returns ``(n_rows, n_features, n_outputs)`` for classifiers and
    older versions return a list per class. Binary problems collapse to the
    *resolved positive class* so that a reported direction refers to the same
    outcome the rest of the report quotes probabilities for; multiclass collapses
    to the mean absolute contribution across classes, where a single sign is
    meaningless.
    """
    if raw is None:
        return None, None
    if isinstance(raw, list):
        try:
            raw = np.stack([np.asarray(v) for v in raw], axis=-1)
        except Exception:
            raw = np.asarray(raw[0])
    values = np.asarray(raw, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim == 2:
        return values, np.abs(values)
    if values.ndim == 3:
        if values.shape[2] == 2:
            signed = values[:, :, min(max(positive_index, 0), 1)]
            return signed, np.abs(signed)
        return None, np.abs(values).mean(axis=2)
    return None, None


def _shap_attributions(
    state: RunState, ctx: PredictionContext, notes: list[str]
) -> tuple[dict[str, float], dict[str, str], str | None, bool]:
    """Best-effort SHAP values, a summary chart, and the availability flag."""
    try:
        import shap  # noqa: F401  (presence check; used below)
    except Exception as exc:
        notes.append(f"SHAP unavailable ({exc.__class__.__name__}); skipped.")
        return {}, {}, None, False

    X, _ = ctx.partition(ctx.eval_partition)
    if row_count(X) == 0:
        notes.append("SHAP skipped: evaluation partition is empty.")
        return {}, {}, None, False

    positions = _subsample_positions(row_count(X), MAX_SHAP_EXPLAIN_ROWS, ctx.random_state)
    X_explain = take_rows(X, positions)
    try:
        Xt = ctx.transform(X_explain)
    except Exception as exc:
        state.add_warning(f"SHAP skipped: could not transform features ({exc}).")
        notes.append(f"SHAP skipped: transform failed ({exc}).")
        return {}, {}, None, False
    if Xt.ndim != 2 or Xt.shape[0] == 0:
        notes.append("SHAP skipped: transformed matrix is not 2-D.")
        return {}, {}, None, False

    names = ctx.model_space_names()
    if len(names) != Xt.shape[1]:
        names = [f"feature_{i}" for i in range(Xt.shape[1])]

    # ``explained`` is the matrix the values actually line up with: KernelExplainer
    # scores a subsample of Xt, and correlating a 50-row contribution column
    # against a 500-row feature column silently yields "unknown" for every
    # direction, so the rows must be carried back with the values.
    signed, magnitude, method, explained = _run_shap_explainer(state, ctx, Xt, notes)
    if magnitude is None:
        return {}, {}, None, False
    if explained is None or explained.shape[0] != magnitude.shape[0]:
        explained = Xt

    values = {
        names[i]: float(magnitude[:, i].mean())
        for i in range(min(len(names), magnitude.shape[1]))
    }
    directions: dict[str, str] = {}
    if signed is not None and signed.shape[0] == explained.shape[0]:
        for i in range(min(len(names), signed.shape[1], explained.shape[1])):
            directions[names[i]] = _direction_from_correlation(
                explained[:, i], signed[:, i]
            )

    png = _save_shap_summary(state, signed, magnitude, explained, names, notes)
    notes.append(
        f"SHAP method: {method} on {magnitude.shape[0]:,} {ctx.eval_partition} rows."
    )
    return values, directions, png, True


def _run_shap_explainer(
    state: RunState, ctx: PredictionContext, Xt: np.ndarray, notes: list[str]
) -> tuple[np.ndarray | None, np.ndarray | None, str, np.ndarray | None]:
    """``(signed, magnitude, method, matrix the values correspond to)``."""
    import shap

    model = ctx.final_model
    background_positions = _subsample_positions(
        Xt.shape[0], MAX_SHAP_BACKGROUND_ROWS, ctx.random_state
    )
    background = Xt[background_positions]

    if _looks_like_tree(model):
        try:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(Xt, check_additivity=False)
            signed, magnitude = _reduce_shap_values(raw, ctx.positive_index)
            if magnitude is not None:
                return signed, magnitude, "TreeExplainer", Xt
        except Exception as exc:
            notes.append(f"TreeExplainer failed ({exc}); trying a fallback explainer.")

    if hasattr(model, "coef_"):
        try:
            explainer = shap.LinearExplainer(model, background)
            signed, magnitude = _reduce_shap_values(
                explainer.shap_values(Xt), ctx.positive_index
            )
            if magnitude is not None:
                return signed, magnitude, "LinearExplainer", Xt
        except Exception as exc:
            notes.append(f"LinearExplainer failed ({exc}); trying a fallback explainer.")

    # KernelExplainer is model-agnostic but costs O(rows x nsamples) model
    # calls, so it only runs on a small sample and only with budget to spare.
    if Xt.shape[1] > KERNEL_SHAP_MAX_FEATURES:
        notes.append(
            f"KernelExplainer skipped: {Xt.shape[1]} features exceeds the "
            f"{KERNEL_SHAP_MAX_FEATURES}-feature budget."
        )
        return None, None, "none", None
    if state.time_remaining and state.time_remaining < KERNEL_SHAP_MIN_SECONDS_LEFT:
        notes.append(
            "KernelExplainer skipped: less than "
            f"{KERNEL_SHAP_MIN_SECONDS_LEFT:.0f}s of run budget remaining."
        )
        return None, None, "none", None

    try:
        predict_fn = _model_space_predict(ctx)
        bg = Xt[_subsample_positions(Xt.shape[0], KERNEL_SHAP_MAX_BACKGROUND, ctx.random_state)]
        sample = Xt[_subsample_positions(Xt.shape[0], KERNEL_SHAP_MAX_ROWS, ctx.random_state)]
        started = time.perf_counter()
        explainer = shap.KernelExplainer(predict_fn, bg)
        raw = explainer.shap_values(sample, nsamples=KERNEL_SHAP_NSAMPLES, silent=True)
        signed, magnitude = _reduce_shap_values(raw, ctx.positive_index)
        notes.append(
            f"KernelExplainer ran on {sample.shape[0]} rows in "
            f"{time.perf_counter() - started:.1f}s (approximation)."
        )
        return signed, magnitude, "KernelExplainer", sample
    except Exception as exc:
        state.add_warning(f"SHAP unavailable for this model: {exc}")
        notes.append(f"KernelExplainer failed ({exc}).")
        return None, None, "none", None


def _model_space_predict(ctx: PredictionContext) -> Callable[[np.ndarray], np.ndarray]:
    """A scalar prediction function over the model-space matrix, for SHAP.

    Three things have to hold for KernelExplainer's output to mean anything:
    the binary column selected must be the *resolved positive class* (so the
    sign convention matches every other number in the report), a classifier
    without ``predict_proba`` must still expose a continuous score
    (``decision_function``), and a classifier that only predicts labels must be
    mapped onto numbers — ``float("yes")`` raises and would lose SHAP entirely.
    """
    model = ctx.final_model
    label_positions = {
        str(label): index for index, label in enumerate(ctx.labels or [])
    }

    def predict(matrix: np.ndarray) -> np.ndarray:
        if ctx.is_classification and hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(matrix))
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, min(max(ctx.positive_index, 0), 1)]
            return proba.max(axis=1) if proba.ndim == 2 else proba
        if ctx.is_classification and hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(matrix), dtype=float)
            if scores.ndim == 2 and scores.shape[1] > 1:
                index = min(max(ctx.positive_index, 0), scores.shape[1] - 1)
                return scores[:, index]
            scores = scores.ravel()
            # A 1-D decision function is oriented towards classes_[1]; flip it
            # when the operator's positive class is the other one.
            return -scores if ctx.positive_index == 0 else scores
        raw = np.asarray(model.predict(matrix))
        try:
            return raw.astype(float)
        except (TypeError, ValueError):
            if label_positions:
                return np.asarray(
                    [float(label_positions.get(str(v), -1)) for v in raw.ravel()],
                    dtype=float,
                )
            codes = {value: i for i, value in enumerate(sorted(set(raw.ravel().tolist())))}
            return np.asarray([float(codes[v]) for v in raw.ravel()], dtype=float)

    return predict


def _direction_from_correlation(feature: np.ndarray, contribution: np.ndarray) -> str:
    try:
        x = np.asarray(feature, dtype=float)
        s = np.asarray(contribution, dtype=float)
        if x.std() == 0 or s.std() == 0:
            return "unknown"
        corr = float(np.corrcoef(x, s)[0, 1])
    except Exception:
        return "unknown"
    if not np.isfinite(corr):
        return "unknown"
    if corr > 0.1:
        return "increases"
    if corr < -0.1:
        return "decreases"
    return "mixed"


def _save_shap_summary(
    state: RunState,
    signed: np.ndarray | None,
    magnitude: np.ndarray,
    Xt: np.ndarray,
    names: list[str],
    notes: list[str],
) -> str | None:
    """Write a SHAP summary PNG, preferring matplotlib and falling back to plotly."""
    path = state.artifact_path("charts", "shap_summary.png")
    plt = _load_pyplot()
    if plt is None:
        _note_once(notes, _NO_MATPLOTLIB_NOTE)
    else:
        try:
            import shap

            plot_values = signed if signed is not None else magnitude
            plt.figure()
            shap.summary_plot(
                plot_values,
                Xt,
                feature_names=names,
                max_display=SHAP_PLOT_MAX_FEATURES,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close("all")
            state.bus.artifact(str(path), kind="chart")
            return str(path)
        except Exception as exc:
            notes.append(f"matplotlib SHAP summary failed ({exc}); trying plotly.")

    try:
        import plotly.graph_objects as go

        mean_abs = np.abs(magnitude).mean(axis=0)
        order = np.argsort(mean_abs)[-SHAP_PLOT_MAX_FEATURES:]
        fig = go.Figure(
            go.Bar(
                x=[float(mean_abs[i]) for i in order],
                y=[names[i] for i in order],
                orientation="h",
            )
        )
        fig.update_layout(
            title="Mean absolute SHAP contribution",
            xaxis_title="mean |SHAP value|",
            template="plotly_white",
            height=max(320, 26 * len(order)),
        )
        fig.write_image(str(path), width=900, height=max(320, 26 * len(order)))
        state.bus.artifact(str(path), kind="chart")
        return str(path)
    except Exception as exc:
        state.add_warning(f"SHAP summary chart could not be written: {exc}")
        notes.append(f"SHAP summary chart failed ({exc}).")
        return None


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------


def _partial_dependence_charts(
    state: RunState, ctx: PredictionContext, ranked: list[str], notes: list[str]
) -> list[str]:
    """One PNG per top feature. Each feature is guarded independently."""
    X, _ = ctx.partition(ctx.eval_partition)
    if row_count(X) == 0 or not ranked:
        return []
    positions = _subsample_positions(row_count(X), MAX_PDP_ROWS, ctx.random_state)
    X_use = take_rows(X, positions)
    known = set(column_names(X_use))
    chosen = [f for f in ranked if f in known][:MAX_PDP_FEATURES]
    if not chosen:
        notes.append("partial dependence skipped: no ranked feature is a model input.")
        return []

    target = None
    if ctx.task is TaskType.MULTICLASS_CLASSIFICATION and ctx.labels:
        # sklearn requires an explicit class for multiclass partial dependence;
        # reusing the resolved positive class keeps every chart in this report
        # about the same outcome.
        target = ctx.positive_label

    paths: list[str] = []
    all_names = column_names(X_use)
    for feature in chosen:
        # sklearn addresses columns by label for frames and by position for
        # arrays, so the key and the human-readable name can differ.
        key: Any = feature if is_frame(X_use) else all_names.index(feature)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in feature)
        path = state.artifact_path("charts", f"partial_dependence_{safe}.png")
        rendered = _pdp_matplotlib(ctx, X_use, feature, key, path, target, notes)
        if not rendered:
            rendered = _pdp_plotly(ctx, X_use, feature, key, path, target, notes)
        if rendered:
            paths.append(str(path))
            state.bus.artifact(str(path), kind="chart")
    if not paths:
        state.add_warning("partial dependence charts could not be rendered.")
    return paths


def _is_numeric_column(X: Any, feature: str) -> bool:
    if not is_frame(X):
        return True
    try:
        import pandas as pd

        return bool(pd.api.types.is_numeric_dtype(X[feature]))
    except Exception:
        return False


def _pdp_matplotlib(
    ctx: PredictionContext,
    X: Any,
    feature: str,
    key: Any,
    path: Any,
    target: Any,
    notes: list[str],
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        _note_once(notes, _NO_MATPLOTLIB_NOTE)
        return False
    try:
        from sklearn.inspection import PartialDependenceDisplay

        categorical = [] if _is_numeric_column(X, feature) else [key]
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            ctx.estimator,
            X,
            features=[key],
            categorical_features=categorical or None,
            grid_resolution=PDP_GRID_RESOLUTION,
            target=target,
            ax=ax,
        )
        ax.set_title(f"Partial dependence: {feature}")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return True
    except Exception as exc:
        notes.append(f"matplotlib partial dependence for '{feature}' failed ({exc}).")
        return False


def _pdp_plotly(
    ctx: PredictionContext,
    X: Any,
    feature: str,
    key: Any,
    path: Any,
    target: Any,
    notes: list[str],
) -> bool:
    try:
        import plotly.graph_objects as go
        from sklearn.inspection import partial_dependence

        categorical = [] if _is_numeric_column(X, feature) else [key]
        result = partial_dependence(
            ctx.estimator,
            X,
            features=[key],
            categorical_features=categorical or None,
            grid_resolution=PDP_GRID_RESOLUTION,
            kind="average",
        )
        grid = np.asarray(result.get("grid_values", result.get("values"))[0])
        average = np.asarray(result["average"])
        row = 0
        if average.shape[0] > 1 and target is not None and ctx.labels:
            try:
                row = list(ctx.labels).index(target)
            except ValueError:
                row = 0
        y = average[row]
        fig = go.Figure(
            go.Scatter(x=[str(g) for g in grid] if categorical else grid, y=y, mode="lines+markers")
        )
        fig.update_layout(
            title=f"Partial dependence: {feature}",
            xaxis_title=feature,
            yaxis_title="average prediction",
            template="plotly_white",
        )
        fig.write_image(str(path), width=760, height=460)
        return True
    except Exception as exc:
        notes.append(f"plotly partial dependence for '{feature}' failed ({exc}).")
        return False


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------


def _render_prediction(ctx: PredictionContext, row: Any) -> tuple[str, float | None]:
    """Human-readable prediction for a single row, plus a comparable number."""
    proba = ctx.predict_proba(row)
    if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        if proba.shape[1] == 2:
            index = min(ctx.positive_index, proba.shape[1] - 1)
            positive = float(proba[0, index])
            return f"P({ctx.positive_label})={positive:.3f}", positive
        index = int(np.argmax(proba[0]))
        label = ctx.labels[index] if ctx.labels and index < len(ctx.labels) else index
        return f"{label} (p={float(proba[0, index]):.3f})", float(proba[0, index])
    value = ctx.predict(row)
    scalar = np.asarray(value).ravel()[0]
    if ctx.is_classification:
        # A hard label has no comparable magnitude, so no delta is reported.
        return f"class {scalar}", None
    try:
        numeric = float(scalar)
        return f"{numeric:.4g}", numeric
    except (TypeError, ValueError):
        return str(scalar), None


def _with_value(row: Any, feature: str, index: int, value: float) -> Any:
    """A copy of a one-row container with one feature overwritten.

    Explicit assignment on a copy, never chained assignment: under pandas 3's
    copy-on-write a chained write is silently discarded.
    """
    if is_frame(row):
        probe = row.copy()
        probe.loc[:, feature] = value
        return probe
    probe = to_dense_array(row).astype(float).copy()
    probe[0, index] = value
    return probe


def _counterfactuals(
    state: RunState, ctx: PredictionContext, ranked: list[str], notes: list[str]
) -> list[Counterfactual]:
    """Perturb one representative row's top features to their p10 and p90."""
    X, _ = ctx.partition(ctx.eval_partition)
    if row_count(X) == 0 or not ranked:
        return []

    try:
        all_names = column_names(X)
        numeric_ranked = [
            f for f in ranked if f in set(all_names) and _is_numeric_column(X, f)
        ]
        if not numeric_ranked:
            notes.append("counterfactuals skipped: no numeric feature among the top drivers.")
            return []
        features = numeric_ranked[:MAX_COUNTERFACTUAL_FEATURES]
        columns = {name: all_names.index(name) for name in features}
        block = np.column_stack(
            [
                np.asarray(
                    X[name].to_numpy() if is_frame(X) else to_dense_array(X)[:, index],
                    dtype=float,
                )
                for name, index in columns.items()
            ]
        )

        # A row near the median is a fairer baseline than row 0: the reported
        # delta then reflects a typical case rather than an outlier.
        medians = np.nanmedian(block, axis=0)
        spread = np.nanstd(block, axis=0)
        spread[spread == 0] = 1.0
        distance = np.nansum(np.abs(block - medians) / spread, axis=1)
        position = int(np.nanargmin(distance))
        base_row = take_rows(X, [position])
        base_text, base_value = _render_prediction(ctx, base_row)

        out: list[Counterfactual] = []
        for offset, feature in enumerate(features):
            original = float(block[position, offset])
            for quantile, label in ((0.10, "10th"), (0.90, "90th")):
                target_value = float(np.nanquantile(block[:, offset], quantile))
                if not np.isfinite(target_value) or np.isclose(target_value, original):
                    continue
                probe = _with_value(base_row, feature, columns[feature], target_value)
                new_text, new_value = _render_prediction(ctx, probe)
                delta = ""
                if base_value is not None and new_value is not None:
                    delta = f" (change {new_value - base_value:+.4g})"
                out.append(
                    Counterfactual(
                        description=(
                            f"Moving `{feature}` from {original:.4g} to its {label} "
                            f"percentile ({target_value:.4g}) shifts the prediction from "
                            f"{base_text} to {new_text}{delta}."
                        ),
                        changed_features=[
                            Param(key=feature, value=f"{original:.6g} -> {target_value:.6g}")
                        ],
                        original_prediction=base_text,
                        new_prediction=new_text,
                    )
                )
        if not out:
            notes.append("counterfactuals produced no usable perturbation.")
        return out
    except Exception as exc:
        state.add_warning(f"counterfactual generation failed: {exc}")
        notes.append(f"counterfactuals skipped: {exc}.")
        return []


# ---------------------------------------------------------------------------
# Plain-language rendering
# ---------------------------------------------------------------------------


def _plain_language(
    attributions: list[FeatureAttribution],
    counterfactuals: list[Counterfactual],
    ctx: PredictionContext,
) -> list[str]:
    reference = (
        f"the probability of '{ctx.positive_label}'"
        if ctx.is_classification
        else "the predicted target value"
    )
    lines: list[str] = [
        f"Directions below describe the effect on {reference}."
    ]
    verb = {
        "increases": "pushes the prediction up",
        "decreases": "pushes the prediction down",
        "mixed": "acts in both directions depending on the row",
        "unknown": "matters in magnitude, though the direction was not measured",
    }
    for attribution in attributions[:5]:
        share = attribution.importance * 100
        lines.append(
            f"`{attribution.feature}` contributes approximately {share:.1f}% of the "
            f"model's total feature importance and {verb[attribution.direction]} "
            f"(method: {attribution.method})."
        )
    if len(attributions) >= 3:
        head = sum(a.importance for a in attributions[:3]) * 100
        lines.append(
            f"The top three drivers together account for about {head:.1f}% of the "
            "measured importance, so the model is "
            + ("concentrated on a few signals." if head >= 60 else "drawing on many signals.")
        )
    if counterfactuals:
        lines.append(counterfactuals[0].description)
    if ctx.eval_partition == "train":
        lines.append(
            "Caveat: importance was measured on training rows, so it may overstate "
            "how much the model relies on memorised detail."
        )
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_explanations(state: RunState) -> ExplainabilityReport:
    """Measure global and local explanations for the run's best model.

    Runs permutation importance (always), native attributions, SHAP (when
    importable and affordable), partial dependence, and counterfactual probes.
    Each stage is independently guarded: a failure downgrades that stage to a
    warning in ``state.warnings`` plus a line in ``method_notes`` and leaves the
    rest of the report intact.

    Args:
        state: The run blackboard. Must carry a fitted ``best_pipeline`` or
            ``best_model`` and materialised ``splits`` for anything to be
            measured.

    Returns:
        An :class:`ExplainabilityReport` with normalised attributions, chart
        paths, and counterfactuals. Also assigned to ``state.explainability`` so
        the narrating agent can read it back.
    """
    started = time.perf_counter()
    notes: list[str] = [f"metric functions: {metrics_source()}"]
    report = ExplainabilityReport()

    ctx = build_prediction_context(state)
    if ctx is None:
        report.method_notes = (
            "No explanations were computed: no fitted model or usable data splits "
            "were available on the run state."
        )
        state.explainability = report
        return report

    notes.extend(ctx.notes)
    notes.append(f"attributions measured on the {ctx.eval_partition} partition.")
    notes.append(
        "attribution direction refers to "
        + (
            f"the probability of the positive class '{ctx.positive_label}'."
            if ctx.is_classification
            else "the predicted target value."
        )
    )

    perm_values, perm_ranked = _permutation_importance(state, ctx, notes)
    native_values, native_directions, native_method = _native_attributions(ctx, notes)

    shap_values, shap_directions, shap_png, shap_ok = _shap_attributions(state, ctx, notes)
    report.shap_available = shap_ok
    report.shap_summary_path = shap_png

    # Permutation importance is a magnitude, never a sign, so any direction it
    # carries has to be borrowed from a signed method.
    report.permutation_importance = _finalise_attributions(
        perm_values,
        {},
        "permutation",
        ctx,
        fallback_directions=_display_directions({**native_directions, **shap_directions}, ctx),
    )

    # Preference order for the headline numbers: the model's own coefficients or
    # split gains describe it exactly; SHAP is the next most faithful; the
    # permutation values are the guaranteed floor.
    shap_display_directions = _display_directions(shap_directions, ctx)
    if native_values:
        report.global_attributions = _finalise_attributions(
            native_values,
            native_directions,
            native_method,
            ctx,
            fallback_directions=shap_display_directions,
        )
    elif shap_values:
        report.global_attributions = _finalise_attributions(
            shap_values, shap_directions, "shap", ctx
        )
    else:
        report.global_attributions = list(report.permutation_importance)
        notes.append(
            "global attributions fall back to permutation importance (no native "
            "coefficients or SHAP values were available)."
        )

    if shap_values and native_values:
        notes.append(
            "SHAP values were computed as well; the headline attributions use the "
            "model's native importances, with SHAP supplying the direction the "
            "native method has no sign for."
        )

    ranking = perm_ranked or [
        name for name, _ in sorted(native_values.items(), key=lambda kv: kv[1], reverse=True)
    ]
    # Chart/probe ordering must use model-input names, which native attributions
    # may not be in (they live in encoder space) — fall back to input columns.
    input_set = set(ctx.input_names)
    ranking_inputs = [name for name in ranking if name in input_set]
    if not ranking_inputs:
        ranking_inputs = [
            name
            for name in (
                display_name(n, model_features=input_set, originals=ctx.original_columns)
                for n in ranking
            )
            if name in input_set
        ]

    report.partial_dependence_paths = _partial_dependence_charts(
        state, ctx, ranking_inputs, notes
    )
    report.counterfactuals = _counterfactuals(state, ctx, ranking_inputs, notes)
    report.plain_language_explanations = _plain_language(
        report.global_attributions, report.counterfactuals, ctx
    )

    notes.append(f"explainability pass took {time.perf_counter() - started:.1f}s.")
    report.method_notes = " ".join(notes)
    state.explainability = report
    return report


__all__ = [
    "PredictionContext",
    "build_prediction_context",
    "column_names",
    "compute_explanations",
    "display_name",
    "head_rows",
    "is_frame",
    "row_count",
    "take_rows",
    "to_dense_array",
]
