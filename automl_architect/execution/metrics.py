"""The measurement layer.

Every score the system reports passes through this module, so two properties
matter more than breadth:

1.  **Direction correctness.** :func:`higher_is_better` drives model selection.
    Getting ``rmse`` or ``log_loss`` backwards would silently crown the worst
    candidate, and nothing downstream would notice. The direction table is
    therefore explicit and exhaustive rather than inferred from a name.
2.  **No fabrication.** :func:`score_predictions` never raises and never
    substitutes. A metric that cannot be computed for the given inputs (single
    class present, no probabilities, all-zero actuals) is *omitted* from the
    result dict. A missing key means "not measurable here"; it never means zero.

Metric values are always reported in their **natural units** — ``rmse`` is a
positive error, ``log_loss`` is a positive loss. sklearn's ``neg_*`` scorers are
an artefact of its "greater is better" convention, so :class:`ScorerSpec`
carries the sign needed to convert cross-validation output back.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn import metrics as skm

from ..core.schemas import TaskType

logger = logging.getLogger(__name__)

__all__ = [
    "ANOMALY_METRICS",
    "CLASSIFICATION_METRICS",
    "CLUSTERING_METRICS",
    "REGRESSION_METRICS",
    "ScorerSpec",
    "canonical_metric",
    "higher_is_better",
    "is_better",
    "metric_names_for",
    "primary_metric_for",
    "score_predictions",
    "scorer_for",
    "sklearn_scorer_name",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

# Agents, config overrides, and humans all spell metrics differently. Everything
# is folded onto one canonical name before any lookup happens.
_ALIASES: dict[str, str] = {
    "acc": "accuracy",
    "accuracy_score": "accuracy",
    "auc": "roc_auc",
    "auroc": "roc_auc",
    "roc": "roc_auc",
    "roc_auc_score": "roc_auc",
    "auc_roc": "roc_auc",
    "roc_auc_ovr_macro": "roc_auc_ovr",
    "pr_auc": "average_precision",
    "auprc": "average_precision",
    "auc_pr": "average_precision",
    "average_precision_score": "average_precision",
    "f1_score": "f1",
    "f_1": "f1",
    "fbeta": "f1",
    "precision_score": "precision",
    "recall_score": "recall",
    "sensitivity": "recall",
    "tpr": "recall",
    "logloss": "log_loss",
    "cross_entropy": "log_loss",
    "binary_crossentropy": "log_loss",
    "mcc": "matthews_corrcoef",
    "matthews": "matthews_corrcoef",
    "matthews_correlation": "matthews_corrcoef",
    "kappa": "cohen_kappa",
    "cohens_kappa": "cohen_kappa",
    "cohen_kappa_score": "cohen_kappa",
    "balanced_acc": "balanced_accuracy",
    "brier": "brier_score",
    "brier_score_loss": "brier_score",
    "root_mean_squared_error": "rmse",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "mean_absolute_percentage_error": "mape",
    "median_ae": "median_absolute_error",
    "medae": "median_absolute_error",
    "mdae": "median_absolute_error",
    "r_squared": "r2",
    "r2_score": "r2",
    "rsquared": "r2",
    "coefficient_of_determination": "r2",
    "explained_variance_score": "explained_variance",
    "rmsle": "root_mean_squared_log_error",
    "msle": "mean_squared_log_error",
    "silhouette_score": "silhouette",
    "calinski_harabasz_score": "calinski_harabasz",
    "calinski": "calinski_harabasz",
    "davies_bouldin_score": "davies_bouldin",
    "davies_bouldin_index": "davies_bouldin",
}

# The single source of truth for optimisation direction.
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "rmse",
        "mse",
        "mae",
        "mape",
        "smape",
        "median_absolute_error",
        "max_error",
        "mean_squared_log_error",
        "root_mean_squared_log_error",
        "mean_poisson_deviance",
        "mean_gamma_deviance",
        "log_loss",
        "brier_score",
        "hamming_loss",
        "zero_one_loss",
        "expected_calibration_error",
        "davies_bouldin",
        "noise_fraction",
        "outlier_fraction",
    }
)

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "f1_micro",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovr_weighted",
    "average_precision",
    "log_loss",
    "matthews_corrcoef",
    "cohen_kappa",
)

REGRESSION_METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "mape",
    "smape",
    "r2",
    "explained_variance",
    "median_absolute_error",
    "mse",
    "max_error",
)

CLUSTERING_METRICS: tuple[str, ...] = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "noise_fraction",
    "n_clusters",
)

ANOMALY_METRICS: tuple[str, ...] = (
    "score_separation",
    "outlier_fraction",
    "mean_anomaly_score",
)

_PRIMARY_BY_TASK: dict[TaskType, str] = {
    TaskType.BINARY_CLASSIFICATION: "roc_auc",
    TaskType.MULTICLASS_CLASSIFICATION: "f1_macro",
    TaskType.MULTILABEL_CLASSIFICATION: "f1_micro",
    TaskType.REGRESSION: "rmse",
    TaskType.TIME_SERIES_FORECASTING: "rmse",
    TaskType.RANKING: "roc_auc",
    TaskType.RECOMMENDATION: "roc_auc",
    TaskType.CLUSTERING: "silhouette",
    TaskType.ANOMALY_DETECTION: "score_separation",
    TaskType.SURVIVAL_ANALYSIS: "r2",
    TaskType.CAUSAL_INFERENCE: "r2",
    TaskType.NLP: "f1_macro",
    TaskType.COMPUTER_VISION: "accuracy",
    TaskType.GRAPH_LEARNING: "silhouette",
}

_EPS = 1e-12


def canonical_metric(name: str | None) -> str:
    """Fold a metric spelling onto its canonical name.

    ``"AUC"``, ``"roc_auc_score"`` and ``"neg_root_mean_squared_error"`` all
    resolve to the names used as keys by :func:`score_predictions`.

    Args:
        name: Any metric spelling, or None.

    Returns:
        The canonical metric name, or ``""`` when ``name`` is empty.
    """
    if not name:
        return ""
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text.startswith("neg_"):
        text = text[4:]
    return _ALIASES.get(text, text)


def primary_metric_for(task: TaskType | str | None) -> str:
    """The default optimisation metric for a task type.

    Used when no agent or config override supplies one, and as the fallback when
    an override names a metric that cannot be computed for the task.

    Args:
        task: The task type, its string value, or None.

    Returns:
        A canonical metric name; ``"accuracy"`` for unknown tasks.
    """
    resolved = _as_task(task)
    if resolved is None:
        return "accuracy"
    return _PRIMARY_BY_TASK.get(resolved, "accuracy")


def higher_is_better(metric: str) -> bool:
    """Whether larger values of ``metric`` mean a better model.

    This is the function model selection turns on. Unknown metrics default to
    True, which is the correct guess for the score-like names agents invent, but
    every loss and error metric this package can emit is listed explicitly.

    Args:
        metric: Any metric spelling. sklearn ``neg_*`` scorer names are
            recognised and always report True.

    Returns:
        True when the metric should be maximised.
    """
    if not metric:
        return True
    raw = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    if raw.startswith("neg_"):
        # sklearn's sign-flipped scorers are maximised by construction.
        return True
    name = canonical_metric(raw)
    if name in _LOWER_IS_BETTER:
        return False
    # Averaging suffixes never change direction: f1_macro is still maximised.
    for suffix in ("_macro", "_micro", "_weighted", "_samples", "_ovr", "_ovo"):
        if name.endswith(suffix) and name[: -len(suffix)] in _LOWER_IS_BETTER:
            return False
    if name.endswith("_loss") or name.endswith("_error"):
        return False
    return True


def is_better(candidate: float | None, incumbent: float | None, metric: str) -> bool:
    """Whether ``candidate`` beats ``incumbent`` on ``metric``.

    Args:
        candidate: The challenger score, or None.
        incumbent: The current best score, or None.
        metric: Metric name, used only for its direction.

    Returns:
        True when candidate is a strict improvement. A None or non-finite
        candidate never wins; any finite candidate beats a None incumbent.
    """
    if candidate is None or not math.isfinite(float(candidate)):
        return False
    if incumbent is None or not math.isfinite(float(incumbent)):
        return True
    if higher_is_better(metric):
        return float(candidate) > float(incumbent)
    return float(candidate) < float(incumbent)


def metric_names_for(task: TaskType | str | None) -> tuple[str, ...]:
    """The metrics :func:`score_predictions` may emit for a task type."""
    resolved = _as_task(task)
    if resolved is None:
        return CLASSIFICATION_METRICS + REGRESSION_METRICS
    if resolved.is_classification:
        return CLASSIFICATION_METRICS
    if resolved is TaskType.CLUSTERING:
        return CLUSTERING_METRICS
    if resolved is TaskType.ANOMALY_DETECTION:
        return ANOMALY_METRICS + CLASSIFICATION_METRICS
    return REGRESSION_METRICS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_predictions(
    task: TaskType | str | None,
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    labels: Any = None,
    *,
    pos_label: Any = None,
    X: Any = None,
) -> dict[str, float]:
    """Score predictions, omitting anything that cannot be honestly computed.

    Never raises. Each metric is computed in isolation, and one that fails —
    because only a single class is present, because probabilities were not
    supplied, because the actuals are all zero — is left out of the result.
    Non-finite results are dropped for the same reason.

    Args:
        task: Task type driving which metric family applies. When None, the
            family is inferred from the dtype of ``y_true``/``y_pred``.
        y_true: Ground truth. May be None for unsupervised tasks.
        y_pred: Predicted labels or values.
        y_proba: Probabilities (classification), or ``decision_function``
            scores (anomaly detection). Optional.
        labels: Full label set, needed when a partition is missing a class.
        pos_label: The positive class for binary metrics. Defaults to ``1`` when
            present in the labels, else the last label in sorted order.
        X: Feature matrix, required only for clustering's internal indices.

    Returns:
        Mapping of canonical metric name to finite float value. For multiclass
        problems ``precision``/``recall``/``f1`` are the *macro* averages;
        the ``_macro``/``_weighted``/``_micro`` variants are also reported
        explicitly.
    """
    out: dict[str, float] = {}
    resolved = _as_task(task) or _infer_task(y_true, y_pred)

    if resolved is TaskType.CLUSTERING:
        _score_clustering(out, y_pred, X)
    elif resolved is TaskType.ANOMALY_DETECTION:
        _score_anomaly(out, y_true, y_pred, y_proba, X)
    elif resolved is not None and resolved.is_classification:
        _score_classification(out, y_true, y_pred, y_proba, labels, pos_label)
    else:
        _score_regression(out, y_true, y_pred)

    return {k: v for k, v in out.items() if math.isfinite(v)}


def _score_classification(
    out: dict[str, float],
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    labels: Any,
    pos_label: Any,
) -> None:
    if y_true is None or y_pred is None:
        return
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if truth.size == 0 or truth.size != pred.size:
        return

    label_list = _resolve_labels(truth, pred, labels)
    binary = len(label_list) <= 2
    positive = pos_label if pos_label is not None else _default_pos_label(label_list)

    _safe(out, "accuracy", lambda: skm.accuracy_score(truth, pred))
    _safe(out, "balanced_accuracy", lambda: skm.balanced_accuracy_score(truth, pred))

    for average in ("macro", "weighted", "micro"):
        _safe(
            out,
            f"precision_{average}",
            lambda a=average: skm.precision_score(
                truth, pred, average=a, labels=label_list, zero_division=0
            ),
        )
        _safe(
            out,
            f"recall_{average}",
            lambda a=average: skm.recall_score(
                truth, pred, average=a, labels=label_list, zero_division=0
            ),
        )
        _safe(
            out,
            f"f1_{average}",
            lambda a=average: skm.f1_score(
                truth, pred, average=a, labels=label_list, zero_division=0
            ),
        )

    if binary:
        _safe(
            out,
            "precision",
            lambda: skm.precision_score(
                truth, pred, pos_label=positive, average="binary", zero_division=0
            ),
        )
        _safe(
            out,
            "recall",
            lambda: skm.recall_score(
                truth, pred, pos_label=positive, average="binary", zero_division=0
            ),
        )
        _safe(
            out,
            "f1",
            lambda: skm.f1_score(
                truth, pred, pos_label=positive, average="binary", zero_division=0
            ),
        )
    else:
        # "f1" without qualification on a multiclass problem means the macro
        # average here: it weights every class equally, so a model that ignores
        # a rare class cannot hide behind the majority class.
        for base in ("precision", "recall", "f1"):
            if f"{base}_macro" in out:
                out[base] = out[f"{base}_macro"]

    _safe(out, "matthews_corrcoef", lambda: skm.matthews_corrcoef(truth, pred))
    _safe(out, "cohen_kappa", lambda: skm.cohen_kappa_score(truth, pred))

    if y_proba is None:
        return
    proba = np.asarray(y_proba, dtype=float)
    if proba.size == 0:
        return

    if binary:
        scores = _positive_column(proba, label_list, positive)
        if scores is None or scores.size != truth.size:
            return
        truth_bin = (truth == positive).astype(int)
        # roc_auc and average_precision are rank statistics, so a
        # decision_function margin is a legitimate input for them.
        _safe(out, "roc_auc", lambda: skm.roc_auc_score(truth_bin, scores))
        _safe(
            out,
            "average_precision",
            lambda: skm.average_precision_score(truth_bin, scores),
        )
        # log_loss and Brier are not. Feeding them unbounded margins would
        # produce a plausible-looking number that means nothing, so they are
        # computed only when the input really is a calibrated probability.
        if not _looks_like_probabilities(proba):
            return
        if proba.ndim == 2 and proba.shape[1] == 2:
            _safe(
                out,
                "log_loss",
                lambda: skm.log_loss(truth, proba, labels=label_list),
            )
        else:
            _safe(
                out,
                "log_loss",
                lambda: skm.log_loss(truth_bin, scores, labels=[0, 1]),
            )
        _safe(out, "brier_score", lambda: skm.brier_score_loss(truth_bin, scores))
        return

    if proba.ndim != 2 or proba.shape[1] != len(label_list):
        return
    _safe(
        out,
        "roc_auc_ovr",
        lambda: skm.roc_auc_score(
            truth, proba, multi_class="ovr", average="macro", labels=label_list
        ),
    )
    _safe(
        out,
        "roc_auc_ovr_weighted",
        lambda: skm.roc_auc_score(
            truth, proba, multi_class="ovr", average="weighted", labels=label_list
        ),
    )
    if "roc_auc_ovr" in out:
        # An agent that asked for "roc_auc" on a multiclass target gets the
        # one-vs-rest form, which is the only thing that is defined.
        out["roc_auc"] = out["roc_auc_ovr"]
    if _looks_like_probabilities(proba):
        _safe(out, "log_loss", lambda: skm.log_loss(truth, proba, labels=label_list))

    def _multiclass_ap() -> float:
        from sklearn.preprocessing import label_binarize

        binarised = label_binarize(truth, classes=list(label_list))
        return float(
            skm.average_precision_score(binarised, proba, average="macro")
        )

    _safe(out, "average_precision", _multiclass_ap)


def _score_regression(out: dict[str, float], y_true: Any, y_pred: Any) -> None:
    if y_true is None or y_pred is None:
        return
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if truth.size == 0 or truth.size != pred.size:
        return
    finite = np.isfinite(truth) & np.isfinite(pred)
    if not finite.any():
        return
    truth = truth[finite]
    pred = pred[finite]

    _safe(out, "rmse", lambda: skm.root_mean_squared_error(truth, pred))
    _safe(out, "mse", lambda: skm.mean_squared_error(truth, pred))
    _safe(out, "mae", lambda: skm.mean_absolute_error(truth, pred))
    _safe(out, "r2", lambda: skm.r2_score(truth, pred))
    _safe(out, "explained_variance", lambda: skm.explained_variance_score(truth, pred))
    _safe(
        out,
        "median_absolute_error",
        lambda: skm.median_absolute_error(truth, pred),
    )
    _safe(out, "max_error", lambda: skm.max_error(truth, pred))

    # MAPE explodes on near-zero actuals. Rather than report a number driven by
    # a handful of denominators, drop those rows -- and drop MAPE entirely if
    # that would silently change what is being measured on most of the data.
    scale = float(np.mean(np.abs(truth))) if truth.size else 0.0
    threshold = max(_EPS, 1e-6 * scale)
    usable = np.abs(truth) > threshold
    if usable.sum() >= max(1, int(0.5 * truth.size)):
        _safe(
            out,
            "mape",
            lambda: skm.mean_absolute_percentage_error(truth[usable], pred[usable]),
        )

    def _smape() -> float:
        denom = (np.abs(truth) + np.abs(pred)) / 2.0
        mask = denom > _EPS
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs(truth[mask] - pred[mask]) / denom[mask]))

    _safe(out, "smape", _smape)


def _score_clustering(out: dict[str, float], y_pred: Any, X: Any) -> None:
    if y_pred is None:
        return
    labels = np.asarray(y_pred).ravel()
    if labels.size == 0:
        return

    noise = labels == -1
    out["noise_fraction"] = float(noise.mean())
    out["n_clusters"] = float(len({int(v) for v in labels[~noise]}))

    if X is None:
        return
    matrix = _as_matrix(X)
    if matrix is None or matrix.shape[0] != labels.size:
        return
    # Noise points are not a cluster; including them distorts every index.
    keep = ~noise
    subset = matrix[keep]
    sub_labels = labels[keep]
    n_labels = len({int(v) for v in sub_labels})
    if n_labels < 2 or n_labels >= subset.shape[0]:
        return

    sample_size = 10_000 if subset.shape[0] > 10_000 else None
    _safe(
        out,
        "silhouette",
        lambda: skm.silhouette_score(
            subset, sub_labels, sample_size=sample_size, random_state=0
        ),
    )
    _safe(
        out,
        "calinski_harabasz",
        lambda: skm.calinski_harabasz_score(subset, sub_labels),
    )
    _safe(out, "davies_bouldin", lambda: skm.davies_bouldin_score(subset, sub_labels))


def _score_anomaly(
    out: dict[str, float], y_true: Any, y_pred: Any, y_proba: Any, X: Any
) -> None:
    if y_pred is None:
        return
    pred = np.asarray(y_pred).ravel()
    if pred.size == 0:
        return
    # sklearn outlier detectors emit -1 for outliers, +1 for inliers.
    is_outlier = pred == -1
    if not is_outlier.any() and set(np.unique(pred)).issubset({0, 1}):
        is_outlier = pred == 1
    out["outlier_fraction"] = float(is_outlier.mean())

    if y_proba is not None:
        scores = np.asarray(y_proba, dtype=float).ravel()
        if scores.size == pred.size and np.isfinite(scores).any():
            out["mean_anomaly_score"] = float(np.nanmean(scores))
            if is_outlier.any() and (~is_outlier).any():
                # decision_function is higher for inliers, so a wide positive
                # gap means the detector actually separated the two groups.
                gap = float(
                    np.nanmean(scores[~is_outlier]) - np.nanmean(scores[is_outlier])
                )
                if math.isfinite(gap):
                    out["score_separation"] = gap

    if X is not None and is_outlier.any() and (~is_outlier).any():
        matrix = _as_matrix(X)
        if matrix is not None and matrix.shape[0] == pred.size:
            _safe(
                out,
                "silhouette",
                lambda: skm.silhouette_score(
                    matrix,
                    is_outlier.astype(int),
                    sample_size=10_000 if matrix.shape[0] > 10_000 else None,
                    random_state=0,
                ),
            )

    if y_true is None:
        return
    truth = np.asarray(y_true).ravel()
    if truth.size != pred.size:
        return
    # A labelled anomaly benchmark is rare but when present it is the only
    # measurement that means anything, so it is reported alongside.
    truth_bin = _to_outlier_flags(truth)
    if truth_bin is None:
        return
    _safe(
        out,
        "f1",
        lambda: skm.f1_score(truth_bin, is_outlier.astype(int), zero_division=0),
    )
    _safe(
        out,
        "precision",
        lambda: skm.precision_score(
            truth_bin, is_outlier.astype(int), zero_division=0
        ),
    )
    _safe(
        out,
        "recall",
        lambda: skm.recall_score(truth_bin, is_outlier.astype(int), zero_division=0),
    )
    if y_proba is not None:
        scores = np.asarray(y_proba, dtype=float).ravel()
        if scores.size == truth_bin.size:
            _safe(out, "roc_auc", lambda: skm.roc_auc_score(truth_bin, -scores))
            _safe(
                out,
                "average_precision",
                lambda: skm.average_precision_score(truth_bin, -scores),
            )


# ---------------------------------------------------------------------------
# sklearn scorer bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorerSpec:
    """A cross-validation scorer plus what is needed to read its output.

    Attributes:
        metric: Canonical metric name being measured.
        name: The sklearn registry name, or ``""`` for a locally built scorer.
        scorer: A callable ``scorer(estimator, X, y)``.
        sign: Multiply raw scorer output by this to get natural units. ``-1``
            for the ``neg_*`` family and for any lower-is-better metric wrapped
            with ``make_scorer``.
    """

    metric: str
    name: str
    scorer: Any
    sign: int = 1

    def to_natural(self, value: float) -> float:
        """Convert one raw scorer value into the metric's natural units."""
        return self.sign * float(value)


_SCORER_NAMES: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_weighted": "f1_weighted",
    "precision_macro": "precision_macro",
    "precision_micro": "precision_micro",
    "precision_weighted": "precision_weighted",
    "recall_macro": "recall_macro",
    "recall_micro": "recall_micro",
    "recall_weighted": "recall_weighted",
    "matthews_corrcoef": "matthews_corrcoef",
    "log_loss": "neg_log_loss",
    "brier_score": "neg_brier_score",
    "average_precision": "average_precision",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "median_absolute_error": "neg_median_absolute_error",
    "max_error": "neg_max_error",
    "mean_squared_log_error": "neg_mean_squared_log_error",
    "root_mean_squared_log_error": "neg_root_mean_squared_log_error",
    "r2": "r2",
    "explained_variance": "explained_variance",
}


def sklearn_scorer_name(metric: str, task: TaskType | str | None) -> str:
    """Map a metric onto an sklearn scorer registry name.

    Args:
        metric: Any metric spelling.
        task: Task type, which disambiguates the binary vs multiclass forms of
            ``roc_auc``, ``precision``, ``recall`` and ``f1``.

    Returns:
        A name accepted by ``sklearn.metrics.get_scorer``, or ``""`` when no
        built-in scorer measures this metric (``cohen_kappa``, ``smape``, and
        every clustering/anomaly index). Callers should use :func:`scorer_for`,
        which fills those gaps locally.
    """
    name = canonical_metric(metric)
    resolved = _as_task(task)
    multiclass = resolved is TaskType.MULTICLASS_CLASSIFICATION
    multilabel = resolved is TaskType.MULTILABEL_CLASSIFICATION

    if name in ("precision", "recall", "f1"):
        if multiclass:
            candidate = f"{name}_macro"
        elif multilabel:
            candidate = f"{name}_micro"
        else:
            candidate = name
    elif name == "roc_auc":
        candidate = "roc_auc_ovr_weighted" if multiclass else "roc_auc"
    else:
        candidate = _SCORER_NAMES.get(name, "")

    if not candidate:
        return ""
    try:
        if candidate not in skm.get_scorer_names():
            return ""
    except Exception:  # pragma: no cover - registry always available
        return ""
    return candidate


def scorer_for(
    metric: str,
    task: TaskType | str | None,
    *,
    pos_label: Any = None,
    labels: Any = None,
) -> ScorerSpec | None:
    """Build a cross-validation scorer for ``metric``.

    Prefers sklearn's registry, and falls back to wrapping this module's own
    implementation with ``make_scorer`` for metrics sklearn does not ship
    (``cohen_kappa``, ``smape``). Clustering and anomaly metrics have no
    supervised scorer form and return None.

    Args:
        metric: Any metric spelling.
        task: Task type, used to pick binary vs multiclass scorer variants.
        pos_label: Positive class for binary metrics.
        labels: Full label set, for metrics that need it.

    Returns:
        A :class:`ScorerSpec`, or None when the metric cannot be cross-validated.
    """
    name = canonical_metric(metric)
    if not name:
        return None
    resolved = _as_task(task)
    if resolved in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
        return None

    registry_name = sklearn_scorer_name(name, task)
    if registry_name:
        try:
            scorer = skm.get_scorer(registry_name)
        except Exception:  # pragma: no cover - guarded by name check above
            scorer = None
        if scorer is not None:
            if registry_name in ("precision", "recall", "f1") and pos_label is not None:
                # get_scorer's binary variants hardcode pos_label=1; rebuild so
                # the reported precision refers to the intended class.
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
            sign = -1 if registry_name.startswith("neg_") else 1
            return ScorerSpec(metric=name, name=registry_name, scorer=scorer, sign=sign)

    local = _LOCAL_SCORE_FUNCS.get(name)
    if local is None:
        return None
    greater = higher_is_better(name)
    try:
        scorer = skm.make_scorer(local, greater_is_better=greater)
    except Exception:  # pragma: no cover - make_scorer is stable
        return None
    return ScorerSpec(metric=name, name="", scorer=scorer, sign=1 if greater else -1)


def _cohen_kappa(y_true: Any, y_pred: Any) -> float:
    return float(skm.cohen_kappa_score(y_true, y_pred))


def _smape_func(y_true: Any, y_pred: Any) -> float:
    out: dict[str, float] = {}
    _score_regression(out, y_true, y_pred)
    return float(out.get("smape", float("nan")))


_LOCAL_SCORE_FUNCS: dict[str, Any] = {
    "cohen_kappa": _cohen_kappa,
    "smape": _smape_func,
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe(out: dict[str, float], name: str, compute: Any) -> None:
    """Record ``compute()`` under ``name``, or record nothing at all."""
    try:
        # sklearn warns loudly for undefined metrics (single class present, no
        # common labels). Those metrics get dropped below, so the warning would
        # only add noise to a run log that already reports their absence.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = compute()
    except Exception as exc:  # a metric that does not apply is simply absent
        logger.debug("metric %s not computable: %s", name, exc)
        return
    if value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(numeric):
        out[name] = numeric


def _as_task(task: TaskType | str | None) -> TaskType | None:
    if task is None:
        return None
    if isinstance(task, TaskType):
        return task
    try:
        return TaskType(str(task))
    except ValueError:
        return None


def _infer_task(y_true: Any, y_pred: Any) -> TaskType | None:
    """Guess classification vs regression from the values themselves."""
    for candidate in (y_true, y_pred):
        if candidate is None:
            continue
        array = np.asarray(candidate).ravel()
        if array.size == 0:
            continue
        if array.dtype.kind in "OUSb":
            return TaskType.BINARY_CLASSIFICATION
        if array.dtype.kind in "iu" or (
            array.dtype.kind == "f" and np.allclose(array, np.round(array), equal_nan=True)
        ):
            n_unique = len(np.unique(array))
            if n_unique <= 2:
                return TaskType.BINARY_CLASSIFICATION
            if n_unique <= 20 and array.dtype.kind in "iu":
                return TaskType.MULTICLASS_CLASSIFICATION
        return TaskType.REGRESSION
    return None


def _resolve_labels(truth: np.ndarray, pred: np.ndarray, labels: Any) -> list[Any]:
    if labels is not None:
        listed = [v for v in np.asarray(labels).ravel().tolist()]
        if listed:
            return listed
    union = set(np.unique(truth).tolist()) | set(np.unique(pred).tolist())
    try:
        return sorted(union)
    except TypeError:  # mixed types: fall back to string ordering
        return sorted(union, key=str)


def _default_pos_label(labels: list[Any]) -> Any:
    if not labels:
        return 1
    for candidate in (1, "1", True, "yes", "true"):
        if candidate in labels:
            return candidate
    return labels[-1]


def _positive_column(
    proba: np.ndarray, labels: list[Any], positive: Any
) -> np.ndarray | None:
    """Extract the score for the positive class from a probability array."""
    if proba.ndim == 1:
        return proba
    if proba.ndim != 2:
        return None
    if proba.shape[1] == 1:
        return proba[:, 0]
    index = labels.index(positive) if positive in labels else proba.shape[1] - 1
    if index >= proba.shape[1]:
        index = proba.shape[1] - 1
    return proba[:, index]


def _looks_like_probabilities(proba: np.ndarray) -> bool:
    """Whether an array can honestly be read as class probabilities."""
    if proba.size == 0 or not np.isfinite(proba).all():
        return False
    if proba.min() < -1e-9 or proba.max() > 1 + 1e-9:
        return False
    if proba.ndim == 2 and proba.shape[1] > 1:
        return bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-3))
    return True


def _as_matrix(X: Any) -> np.ndarray | None:
    try:
        if hasattr(X, "to_numpy"):
            matrix = X.to_numpy(dtype=float, copy=False)
        else:
            matrix = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        return None
    return matrix


def _to_outlier_flags(truth: np.ndarray) -> np.ndarray | None:
    """Normalise ground-truth anomaly labels to 1=outlier, 0=inlier."""
    unique = set(np.unique(truth).tolist())
    if unique.issubset({-1, 1}):
        return (truth == -1).astype(int)
    if unique.issubset({0, 1}):
        return truth.astype(int)
    if unique.issubset({True, False}):
        return truth.astype(int)
    return None
