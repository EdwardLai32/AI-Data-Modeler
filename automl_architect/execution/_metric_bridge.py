"""Late-bound access to ``execution.metrics`` with a self-contained fallback.

The explainability and diagnostics executors need four things from the metrics
module: the default metric for a task, its orientation, a scorer name for
sklearn's ``scoring=`` argument, and a way to score raw predictions. Importing
``execution.metrics`` at module scope would make *this* module unimportable
whenever that one is absent, half-written, or itself broken — which would take
the whole diagnostics stage down with it and violate the degrade-never-crash
rule. So every lookup goes through :func:`_resolve`, which prefers the real
implementation, per-function, and falls back to a small local one.

The fallback is deliberately conservative: it covers the metric names the
supported task types actually use and returns a partial dict rather than
raising when one metric cannot be computed on the given inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from ..core.schemas import TaskType

logger = logging.getLogger(__name__)

_REQUIRED = (
    "primary_metric_for",
    "higher_is_better",
    "score_predictions",
    "sklearn_scorer_name",
)

_cache: dict[str, Callable[..., Any]] = {}
_source: dict[str, str] = {}


def _resolve(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``execution.metrics.<name>`` if it exists, else ``fallback``."""
    if name in _cache:
        return _cache[name]
    func: Callable[..., Any] = fallback
    origin = "fallback"
    try:
        from . import metrics as _metrics  # local import: may not exist yet

        candidate = getattr(_metrics, name, None)
        if callable(candidate):
            func = candidate
            origin = "execution.metrics"
    except Exception as exc:  # pragma: no cover - depends on sibling module
        logger.debug("execution.metrics unavailable (%s); using fallback %s", exc, name)
    _cache[name] = func
    _source[name] = origin
    return func


_CANONICAL_ALIASES = {
    "auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "brier_score",
    "logloss": "log_loss",
    "medae": "median_absolute_error",
    "mad": "mae",
}


def _fb_canonical_metric(metric: str) -> str:
    name = (metric or "").strip().lower().removeprefix("neg_")
    return _CANONICAL_ALIASES.get(name, name)


def canonical_metric(metric: str) -> str:
    """Normalise a metric name to the spelling the metric panel uses."""
    try:
        return str(_resolve("canonical_metric", _fb_canonical_metric)(metric))
    except Exception:
        return _fb_canonical_metric(metric)


def metrics_source() -> str:
    """Where the metric functions currently in use came from."""
    for name in _REQUIRED:
        _resolve(name, _FALLBACKS[name])
    if all(_source.get(n) == "execution.metrics" for n in _REQUIRED):
        return "execution.metrics"
    if all(_source.get(n) == "fallback" for n in _REQUIRED):
        return "internal fallback (execution.metrics unavailable)"
    return "mixed (execution.metrics partially available)"


# ---------------------------------------------------------------------------
# Fallback implementations
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = {
    "rmse",
    "mse",
    "mae",
    "mad",
    "mape",
    "smape",
    "msle",
    "rmsle",
    "medae",
    "median_absolute_error",
    "max_error",
    "log_loss",
    "logloss",
    "cross_entropy",
    "brier",
    "brier_score",
    "hamming_loss",
    "davies_bouldin",
}

_SCORER_ALIASES = {
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "roc_auc_ovr": "roc_auc_ovr_weighted",
    "average_precision": "average_precision",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision": "precision",
    "recall": "recall",
    "log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
    "brier_score": "neg_brier_score",
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "medae": "neg_median_absolute_error",
    "explained_variance": "explained_variance",
    "silhouette": "",
}


def _fb_primary_metric_for(task: TaskType) -> str:
    if task is TaskType.BINARY_CLASSIFICATION:
        return "roc_auc"
    if task in (
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
    ):
        return "f1_macro"
    if task in (TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING):
        return "rmse"
    if task is TaskType.CLUSTERING:
        return "silhouette"
    if task is TaskType.ANOMALY_DETECTION:
        return "average_precision"
    return "accuracy"


def _fb_higher_is_better(metric: str) -> bool:
    name = (metric or "").strip().lower()
    if name.startswith("neg_"):
        # sklearn's negated scorers are already oriented higher-is-better.
        return True
    return name not in _LOWER_IS_BETTER


def _fb_sklearn_scorer_name(metric: str, task: TaskType) -> str:
    name = (metric or "").strip().lower()
    if name.startswith("neg_"):
        return name
    mapped = _SCORER_ALIASES.get(name)
    if mapped is not None:
        if mapped == "roc_auc" and task is TaskType.MULTICLASS_CLASSIFICATION:
            return "roc_auc_ovr_weighted"
        if mapped in ("f1", "precision", "recall") and task in (
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ):
            return f"{mapped}_macro"
        return mapped
    return name


def _positive_column(labels: Any, proba: np.ndarray) -> int:
    """Index of the column holding the positive class for binary problems."""
    if proba.ndim == 2 and proba.shape[1] == 2:
        return 1
    return 0


def _fb_score_predictions(
    task: TaskType,
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    labels: Any = None,
) -> dict[str, float]:
    """Compute a small metric panel, skipping anything that cannot be computed."""
    from sklearn import metrics as skm

    out: dict[str, float] = {}
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    proba = None if y_proba is None else np.asarray(y_proba)

    def attempt(name: str, fn: Callable[[], float]) -> None:
        try:
            value = float(fn())
        except Exception:  # one unusable metric must not void the panel
            return
        if np.isfinite(value):
            out[name] = value

    if task.is_classification:
        class_labels = (
            list(labels)
            if labels is not None
            else sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
        )
        multiclass = len(class_labels) > 2
        average = "macro" if multiclass else "binary"
        pos_label = class_labels[-1] if len(class_labels) == 2 else None

        attempt("accuracy", lambda: skm.accuracy_score(y_true_arr, y_pred_arr))
        attempt(
            "balanced_accuracy",
            lambda: skm.balanced_accuracy_score(y_true_arr, y_pred_arr),
        )
        kwargs: dict[str, Any] = {"average": average, "zero_division": 0}
        if pos_label is not None:
            kwargs["pos_label"] = pos_label
        attempt("precision", lambda: skm.precision_score(y_true_arr, y_pred_arr, **kwargs))
        attempt("recall", lambda: skm.recall_score(y_true_arr, y_pred_arr, **kwargs))
        attempt("f1", lambda: skm.f1_score(y_true_arr, y_pred_arr, **kwargs))
        attempt(
            "f1_macro",
            lambda: skm.f1_score(
                y_true_arr, y_pred_arr, average="macro", zero_division=0
            ),
        )
        if proba is not None and proba.size:
            if multiclass:
                attempt(
                    "roc_auc",
                    lambda: skm.roc_auc_score(
                        y_true_arr,
                        proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=class_labels,
                    ),
                )
            else:
                col = _positive_column(class_labels, proba)
                scores = proba[:, col] if proba.ndim == 2 else proba
                binary_true = (y_true_arr == class_labels[-1]).astype(int)
                attempt("roc_auc", lambda: skm.roc_auc_score(binary_true, scores))
                attempt(
                    "average_precision",
                    lambda: skm.average_precision_score(binary_true, scores),
                )
                attempt("brier", lambda: skm.brier_score_loss(binary_true, scores))
            attempt(
                "log_loss",
                lambda: skm.log_loss(y_true_arr, proba, labels=class_labels),
            )
        return out

    # regression / forecasting
    attempt(
        "rmse",
        lambda: skm.root_mean_squared_error(y_true_arr, y_pred_arr),
    )
    attempt("mse", lambda: skm.mean_squared_error(y_true_arr, y_pred_arr))
    attempt("mae", lambda: skm.mean_absolute_error(y_true_arr, y_pred_arr))
    attempt("medae", lambda: skm.median_absolute_error(y_true_arr, y_pred_arr))
    attempt("r2", lambda: skm.r2_score(y_true_arr, y_pred_arr))
    attempt(
        "explained_variance",
        lambda: skm.explained_variance_score(y_true_arr, y_pred_arr),
    )
    if np.all(y_true_arr != 0):
        attempt(
            "mape",
            lambda: skm.mean_absolute_percentage_error(y_true_arr, y_pred_arr),
        )
    return out


class _FallbackScorerSpec:
    """The subset of ``metrics.ScorerSpec`` :func:`resolve_scorer` reads."""

    __slots__ = ("metric", "name", "scorer", "sign")

    def __init__(self, metric: str, name: str, scorer: Any, sign: int = 1) -> None:
        self.metric = metric
        self.name = name
        self.scorer = scorer
        self.sign = sign


def _fb_scorer_for(
    metric: str, task: TaskType | str | None, *, pos_label: Any = None, labels: Any = None
) -> Any:
    """Rebuild the binary threshold scorers around ``pos_label``.

    Only the case the registry gets wrong is handled here; returning None for
    everything else lets :func:`resolve_scorer` fall through to ``get_scorer``.
    """
    name = _fb_canonical_metric(metric)
    if pos_label is None or name not in ("precision", "recall", "f1"):
        return None
    registry_name = _fb_sklearn_scorer_name(name, task)
    if registry_name not in ("precision", "recall", "f1"):
        return None  # macro/micro variants already ignore pos_label
    try:
        from sklearn import metrics as skm

        func = {
            "precision": skm.precision_score,
            "recall": skm.recall_score,
            "f1": skm.f1_score,
        }[registry_name]
        scorer = skm.make_scorer(
            func,
            greater_is_better=True,
            pos_label=pos_label,
            average="binary",
            zero_division=0,
        )
    except Exception as exc:  # pragma: no cover - make_scorer is stable
        logger.debug("fallback scorer_for failed for %s: %s", metric, exc)
        return None
    return _FallbackScorerSpec(metric=name, name=registry_name, scorer=scorer, sign=1)


_FALLBACKS: dict[str, Callable[..., Any]] = {
    "primary_metric_for": _fb_primary_metric_for,
    "higher_is_better": _fb_higher_is_better,
    "score_predictions": _fb_score_predictions,
    "sklearn_scorer_name": _fb_sklearn_scorer_name,
    "scorer_for": _fb_scorer_for,
}


# ---------------------------------------------------------------------------
# Public delegating API
# ---------------------------------------------------------------------------


def primary_metric_for(task: TaskType) -> str:
    """Default primary metric for ``task``."""
    return str(_resolve("primary_metric_for", _fb_primary_metric_for)(task))


def higher_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` means a better model."""
    return bool(_resolve("higher_is_better", _fb_higher_is_better)(metric))


def sklearn_scorer_name(metric: str, task: TaskType) -> str:
    """The ``scoring=`` string sklearn understands for ``metric``."""
    return str(_resolve("sklearn_scorer_name", _fb_sklearn_scorer_name)(metric, task))


def score_predictions(
    task: TaskType,
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    labels: Any = None,
) -> dict[str, float]:
    """Metric panel for one set of predictions. Never raises."""
    func = _resolve("score_predictions", _fb_score_predictions)
    try:
        result = func(task, y_true, y_pred, y_proba=y_proba, labels=labels)
    except TypeError:
        # A stricter signature in the real module: retry positionally.
        try:
            result = func(task, y_true, y_pred, y_proba, labels)
        except Exception as exc:
            logger.debug("score_predictions failed: %s", exc)
            return {}
    except Exception as exc:
        logger.debug("score_predictions failed: %s", exc)
        return {}
    if not isinstance(result, dict):
        return {}
    clean: dict[str, float] = {}
    for key, value in result.items():
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fvalue):
            clean[str(key)] = fvalue
    return clean


def resolve_scorer(
    metric: str, task: TaskType, *, pos_label: Any = None
) -> tuple[Any, str, bool]:
    """Return ``(scorer, name, negated)`` for sklearn's ``scoring=`` argument.

    ``scorer`` is ``None`` when no sklearn scorer matches, which callers treat as
    "use the estimator's own ``score`` method". ``negated`` says whether scores
    from the scorer need their sign flipped to read as the metric itself.

    ``pos_label`` matters for the binary ``precision``/``recall``/``f1`` family:
    ``get_scorer("f1")`` hardcodes ``pos_label=1``, which raises on a target
    labelled ``"churn"``/``"stay"`` and would take permutation importance and the
    learning curve down with it. ``execution.metrics.scorer_for`` already rebuilds
    those scorers around the intended class, so it is preferred when available.
    """
    spec_func = _resolve("scorer_for", _fb_scorer_for)
    try:
        spec = spec_func(metric, task, pos_label=pos_label)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("scorer_for failed for %s: %s", metric, exc)
        spec = None
    if spec is not None:
        scorer = getattr(spec, "scorer", None)
        if scorer is not None:
            name = str(getattr(spec, "name", "") or "")
            sign = int(getattr(spec, "sign", 1) or 1)
            return scorer, name or f"{canonical_metric(metric)} (local)", sign < 0

    name = sklearn_scorer_name(metric, task)
    if not name:
        return None, "", False
    try:
        from sklearn.metrics import get_scorer

        scorer = get_scorer(name)
    except Exception:
        return None, "", False
    return scorer, name, name.startswith("neg_")


__all__ = [
    "canonical_metric",
    "higher_is_better",
    "metrics_source",
    "primary_metric_for",
    "resolve_scorer",
    "score_predictions",
    "sklearn_scorer_name",
]
