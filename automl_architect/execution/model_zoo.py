"""Every ``ModelFamily`` mapped to a real, fitted-by-someone-else estimator.

This module is the only place in the system that knows how a family name turns
into an estimator object. Three rules shape it:

*   **Optional dependencies are probed once, at import.** XGBoost, LightGBM and
    CatBoost may be absent; :func:`is_available` reports the truth so the Model
    Selection Agent's postprocess can drop families that cannot run instead of
    the trainer discovering it mid-loop.
*   **Agent-supplied parameters are filtered, never trusted.** A hyperparameter
    the estimator does not accept is dropped with a log line rather than
    crashing the fit. Values are still validated by sklearn at ``fit`` time,
    where the trainer catches the failure and records it.
*   **No silent substitution across task types.** ``linear`` on a classification
    target legitimately means logistic regression, and that mapping is
    documented. ``naive_bayes`` on a regression target has no honest
    equivalent, so it raises and the trainer records a failed experiment.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.mixture import GaussianMixture
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..config import get_settings
from ..core.errors import MissingDependencyError, TrainingError
from ..core.schemas import ModelFamily, TaskType

logger = logging.getLogger(__name__)

__all__ = [
    "FAMILY_TASKS",
    "OPTIONAL_PACKAGES",
    "SeasonalNaiveRegressor",
    "StatsmodelsForecaster",
    "available_families",
    "build_estimator",
    "default_search_space",
    "is_available",
    "supports_proba",
    "supports_task",
    "unsupported_params",
]


def _installed(module: str) -> bool:
    """Whether ``module`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # broken or shadowed distribution
        return False


def _require(module: str, feature: str) -> Any:
    """Import an optional backend, or raise the documented typed error.

    ``find_spec`` only proves the distribution is *present*, not that it loads.
    A half-installed xgboost/lightgbm/catboost — the classic Windows "DLL load
    failed" — passes the probe and then raises ``ImportError`` here. Letting that
    escape would break :func:`build_estimator`'s contract, which promises
    :class:`MissingDependencyError`, and would cost the caller the install hint
    that error carries.
    """
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - ImportError, OSError, anything
        logger.warning("optional package %r is present but failed to import: %s", module, exc)
        # The top-level distribution, not the submodule: "pip install
        # statsmodels.tsa.statespace.sarimax" is not an installable name.
        raise MissingDependencyError(module.split(".")[0], feature) from exc


# Probed once at import so availability is a cheap, stable fact for the whole
# run rather than a per-candidate import attempt.
_HAS_XGBOOST = _installed("xgboost")
_HAS_LIGHTGBM = _installed("lightgbm")
_HAS_CATBOOST = _installed("catboost")
_HAS_STATSMODELS = _installed("statsmodels")

OPTIONAL_PACKAGES: dict[ModelFamily, str] = {
    ModelFamily.XGBOOST: "xgboost",
    ModelFamily.LIGHTGBM: "lightgbm",
    ModelFamily.CATBOOST: "catboost",
    ModelFamily.THETA: "statsmodels",
    ModelFamily.EXPONENTIAL_SMOOTHING: "statsmodels",
    ModelFamily.SARIMAX: "statsmodels",
}

_AVAILABILITY: dict[ModelFamily, bool] = {
    ModelFamily.XGBOOST: _HAS_XGBOOST,
    ModelFamily.LIGHTGBM: _HAS_LIGHTGBM,
    ModelFamily.CATBOOST: _HAS_CATBOOST,
    ModelFamily.THETA: _HAS_STATSMODELS,
    ModelFamily.EXPONENTIAL_SMOOTHING: _HAS_STATSMODELS,
    ModelFamily.SARIMAX: _HAS_STATSMODELS,
}

_CLASSIFICATION = frozenset(
    {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
    }
)
_REGRESSION_LIKE = frozenset({TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING})
_TABULAR_SUPERVISED = _CLASSIFICATION | _REGRESSION_LIKE

FAMILY_TASKS: dict[ModelFamily, frozenset[TaskType]] = {
    ModelFamily.LINEAR: _TABULAR_SUPERVISED,
    ModelFamily.LOGISTIC: _CLASSIFICATION,
    ModelFamily.RIDGE: _TABULAR_SUPERVISED,
    ModelFamily.LASSO: _TABULAR_SUPERVISED,
    ModelFamily.ELASTIC_NET: _TABULAR_SUPERVISED,
    ModelFamily.DECISION_TREE: _TABULAR_SUPERVISED,
    ModelFamily.RANDOM_FOREST: _TABULAR_SUPERVISED,
    ModelFamily.EXTRA_TREES: _TABULAR_SUPERVISED,
    ModelFamily.GRADIENT_BOOSTING: _TABULAR_SUPERVISED,
    ModelFamily.HIST_GRADIENT_BOOSTING: _TABULAR_SUPERVISED,
    ModelFamily.XGBOOST: _TABULAR_SUPERVISED,
    ModelFamily.LIGHTGBM: _TABULAR_SUPERVISED,
    ModelFamily.CATBOOST: _TABULAR_SUPERVISED,
    ModelFamily.SVM: _TABULAR_SUPERVISED,
    ModelFamily.KNN: _TABULAR_SUPERVISED,
    ModelFamily.NAIVE_BAYES: _CLASSIFICATION,
    ModelFamily.NEURAL_NETWORK: _TABULAR_SUPERVISED,
    ModelFamily.BASELINE_DUMMY: _TABULAR_SUPERVISED,
    ModelFamily.KMEANS: frozenset({TaskType.CLUSTERING}),
    ModelFamily.DBSCAN: frozenset({TaskType.CLUSTERING}),
    ModelFamily.GAUSSIAN_MIXTURE: frozenset({TaskType.CLUSTERING}),
    ModelFamily.ISOLATION_FOREST: frozenset({TaskType.ANOMALY_DETECTION}),
    ModelFamily.LOCAL_OUTLIER_FACTOR: frozenset({TaskType.ANOMALY_DETECTION}),
    ModelFamily.ONE_CLASS_SVM: frozenset({TaskType.ANOMALY_DETECTION}),
    ModelFamily.SEASONAL_NAIVE: frozenset({TaskType.TIME_SERIES_FORECASTING}),
    ModelFamily.THETA: frozenset({TaskType.TIME_SERIES_FORECASTING}),
    ModelFamily.EXPONENTIAL_SMOOTHING: frozenset({TaskType.TIME_SERIES_FORECASTING}),
    ModelFamily.SARIMAX: frozenset({TaskType.TIME_SERIES_FORECASTING}),
}

# Rough desirability order for tabular work, used only to order the fallback
# candidate list when no agent has ranked anything.
_PREFERENCE: tuple[ModelFamily, ...] = (
    ModelFamily.BASELINE_DUMMY,
    ModelFamily.SEASONAL_NAIVE,
    ModelFamily.LIGHTGBM,
    ModelFamily.XGBOOST,
    ModelFamily.CATBOOST,
    ModelFamily.HIST_GRADIENT_BOOSTING,
    ModelFamily.RANDOM_FOREST,
    ModelFamily.EXTRA_TREES,
    ModelFamily.GRADIENT_BOOSTING,
    ModelFamily.LOGISTIC,
    ModelFamily.RIDGE,
    ModelFamily.LINEAR,
    ModelFamily.ELASTIC_NET,
    ModelFamily.LASSO,
    ModelFamily.SVM,
    ModelFamily.KNN,
    ModelFamily.NEURAL_NETWORK,
    ModelFamily.NAIVE_BAYES,
    ModelFamily.DECISION_TREE,
    ModelFamily.KMEANS,
    ModelFamily.GAUSSIAN_MIXTURE,
    ModelFamily.DBSCAN,
    ModelFamily.ISOLATION_FOREST,
    ModelFamily.LOCAL_OUTLIER_FACTOR,
    ModelFamily.ONE_CLASS_SVM,
    ModelFamily.EXPONENTIAL_SMOOTHING,
    ModelFamily.THETA,
    ModelFamily.SARIMAX,
)

# Families whose sklearn estimator handles an indicator matrix directly; the
# rest get wrapped in OneVsRestClassifier for multilabel targets.
_NATIVE_MULTILABEL = frozenset(
    {
        ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.EXTRA_TREES,
        ModelFamily.KNN,
        ModelFamily.NEURAL_NETWORK,
    }
)

_NO_PROBA = frozenset({ModelFamily.RIDGE})


# ---------------------------------------------------------------------------
# Time-series estimators
# ---------------------------------------------------------------------------


class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    """Forecast each horizon step with the value one season earlier.

    The honest baseline for any seasonal series: if a learned model cannot beat
    "same as last week", the model has found nothing. Implemented directly
    against the sklearn estimator API so it drops into a Pipeline and
    ``TimeSeriesSplit`` like anything else. ``X`` is used only for its length —
    predictions come from the tail of the training target.

    Args:
        season_length: Period length in rows. 1 reduces to a last-value naive
            forecast.
    """

    def __init__(self, season_length: int = 1) -> None:
        self.season_length = season_length

    def fit(self, X: Any, y: Any = None) -> SeasonalNaiveRegressor:
        """Memorise the final season of the training target.

        Args:
            X: Feature matrix; only its column count is recorded.
            y: The target series, in chronological order.

        Returns:
            self.

        Raises:
            TrainingError: If ``y`` is empty or None.
        """
        if y is None:
            raise TrainingError("SeasonalNaiveRegressor requires a target series")
        values = np.asarray(y, dtype=float).ravel()
        if values.size == 0:
            raise TrainingError("SeasonalNaiveRegressor received an empty target")
        period = max(1, int(self.season_length))
        period = min(period, values.size)
        self.season_ = values[-period:]
        finite = values[np.isfinite(values)]
        self.fallback_ = float(finite.mean()) if finite.size else 0.0
        self.n_features_in_ = int(getattr(X, "shape", (0, 0))[1]) if _has_2d(X) else 0
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Repeat the memorised season across ``len(X)`` steps.

        Args:
            X: Rows to forecast; only the row count matters.

        Returns:
            Array of forecasts, one per row of ``X``.
        """
        if not hasattr(self, "season_"):
            raise TrainingError("SeasonalNaiveRegressor is not fitted")
        n = _row_count(X)
        if n == 0:
            return np.empty(0, dtype=float)
        out = self.season_[np.arange(n) % self.season_.size].astype(float, copy=True)
        return np.where(np.isfinite(out), out, self.fallback_)


class StatsmodelsForecaster(BaseEstimator, RegressorMixin):
    """sklearn-shaped wrapper over statsmodels univariate forecasters.

    Covers Theta, Holt-Winters exponential smoothing, and SARIMAX behind one
    estimator so the trainer treats them like any other candidate. Fitting uses
    only ``y``; ``predict`` forecasts ``len(X)`` steps beyond the training
    window, which is exactly what a temporal split asks for.

    Args:
        kind: One of ``theta``, ``exponential_smoothing``, ``sarimax``.
        season_length: Seasonal period. Seasonality is disabled automatically
            when the training window is shorter than two full periods, because
            statsmodels cannot estimate it and would raise.
        trend: Trend component for exponential smoothing (``"add"``, ``"mul"``
            or None).
        damped_trend: Whether to damp the exponential-smoothing trend.
        order: SARIMAX non-seasonal ``(p, d, q)``.
        seasonal_order: SARIMAX seasonal ``(P, D, Q, s)``. When ``s`` is 0 it is
            replaced by ``season_length`` if that is usable.
    """

    def __init__(
        self,
        kind: str = "exponential_smoothing",
        season_length: int = 1,
        trend: str | None = "add",
        damped_trend: bool = False,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        self.kind = kind
        self.season_length = season_length
        self.trend = trend
        self.damped_trend = damped_trend
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X: Any, y: Any = None) -> StatsmodelsForecaster:
        """Fit the underlying statsmodels model on the target series.

        Args:
            X: Feature matrix; unused except for ``n_features_in_``.
            y: The target series, in chronological order.

        Returns:
            self.

        Raises:
            MissingDependencyError: If statsmodels is not installed.
            TrainingError: If the series is unusable or the model does not
                converge to a fitted result.
        """
        if not _HAS_STATSMODELS:
            raise MissingDependencyError("statsmodels", f"{self.kind} forecasting")
        if y is None:
            raise TrainingError("StatsmodelsForecaster requires a target series")
        series = np.asarray(y, dtype=float).ravel()
        series = series[np.isfinite(series)]
        if series.size < 4:
            raise TrainingError(
                f"{self.kind} needs at least 4 observations, got {series.size}"
            )

        period = max(1, int(self.season_length))
        seasonal_ok = period > 1 and series.size >= 2 * period
        self.n_features_in_ = int(getattr(X, "shape", (0, 0))[1]) if _has_2d(X) else 0
        self.fallback_ = float(series[-1])

        with warnings.catch_warnings():
            # statsmodels is chatty about missing frequency information on a
            # plain array and about convergence on short series; neither
            # changes the forecast, and a failure surfaces as an exception.
            warnings.simplefilter("ignore")
            try:
                self.result_ = self._fit_backend(series, period, seasonal_ok)
            except MissingDependencyError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported as a train failure
                raise TrainingError(f"{self.kind} failed to fit: {exc}") from exc
        return self

    def _fit_backend(self, series: np.ndarray, period: int, seasonal_ok: bool) -> Any:
        kind = str(self.kind).lower()
        if kind == "theta":
            module = _require("statsmodels.tsa.forecasting.theta", "Theta forecasting")
            model = module.ThetaModel(
                series, period=period if seasonal_ok else None, deseasonalize=seasonal_ok
            )
            return model.fit()
        if kind in ("exponential_smoothing", "holt_winters"):
            module = _require("statsmodels.tsa.holtwinters", "exponential smoothing")
            model = module.ExponentialSmoothing(
                series,
                trend=self.trend,
                damped_trend=bool(self.damped_trend) and self.trend is not None,
                seasonal="add" if seasonal_ok else None,
                seasonal_periods=period if seasonal_ok else None,
                initialization_method="estimated",
            )
            return model.fit(optimized=True)
        if kind == "sarimax":
            module = _require("statsmodels.tsa.statespace.sarimax", "SARIMAX forecasting")
            seasonal_order = tuple(int(v) for v in self.seasonal_order)
            if seasonal_order[3] == 0 and seasonal_ok:
                seasonal_order = (1, 0, 0, period)
            if not seasonal_ok:
                seasonal_order = (0, 0, 0, 0)
            model = module.SARIMAX(
                series,
                order=tuple(int(v) for v in self.order),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False)
        raise TrainingError(f"unknown statsmodels forecaster kind: {self.kind!r}")

    def predict(self, X: Any) -> np.ndarray:
        """Forecast ``len(X)`` steps past the end of the training series.

        Args:
            X: Rows to forecast; only the row count matters.

        Returns:
            Array of forecasts, one per row of ``X``.
        """
        if not hasattr(self, "result_"):
            raise TrainingError("StatsmodelsForecaster is not fitted")
        n = _row_count(X)
        if n == 0:
            return np.empty(0, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = np.asarray(self.result_.forecast(n), dtype=float).ravel()
        if raw.size < n:
            raw = np.concatenate([raw, np.full(n - raw.size, self.fallback_)])
        out = raw[:n]
        return np.where(np.isfinite(out), out, self.fallback_)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_available(family: ModelFamily) -> bool:
    """Whether this family can actually be built in this environment.

    sklearn-backed families are always available. Boosted-tree and statsmodels
    families reflect the import probe performed when this module loaded.

    Args:
        family: The model family to check.

    Returns:
        True when :func:`build_estimator` can construct the family.
    """
    return _AVAILABILITY.get(family, True)


def supports_task(family: ModelFamily, task: TaskType) -> bool:
    """Whether ``family`` is a defensible choice for ``task``."""
    return task in FAMILY_TASKS.get(family, frozenset())


def available_families(task: TaskType) -> list[ModelFamily]:
    """Families that suit ``task`` and are installed, most promising first.

    Args:
        task: The task type being solved.

    Returns:
        Ordered list of usable families. Empty for task types the execution
        layer cannot train (NLP, computer vision, graph learning).
    """
    return [
        family
        for family in _PREFERENCE
        if supports_task(family, task) and is_available(family)
    ]


def supports_proba(family: ModelFamily, task: TaskType) -> bool:
    """Whether the built estimator exposes ``predict_proba``.

    Drives whether the trainer bothers asking for probabilities, and therefore
    whether ``roc_auc``/``log_loss`` can be reported at all.

    Args:
        family: The model family.
        task: The task type it will be built for.

    Returns:
        True only for classification estimators with a probability output.
        ``ridge`` is False: ``RidgeClassifier`` offers ``decision_function``
        only.
    """
    if task not in _CLASSIFICATION:
        return False
    if not supports_task(family, task):
        return False
    return family not in _NO_PROBA


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_estimator(
    family: ModelFamily,
    task: TaskType,
    params: dict,
    random_state: int = 42,
    **ctx: Any,
) -> Any:
    """Construct a configured, unfitted estimator.

    Defaults are chosen to be reasonable without tuning: ``n_jobs`` from
    settings, a fixed ``random_state`` wherever the estimator accepts one, and
    verbosity off for the boosted-tree libraries so training does not flood the
    run log. Parameters the estimator does not accept are dropped with a log
    line — an agent naming ``max_leaves`` on a random forest should not end the
    run.

    Args:
        family: Which model family to build.
        task: The task it will be trained for. Determines classifier vs
            regressor and multilabel wrapping.
        params: Agent-supplied hyperparameters, already converted from
            ``list[Param]`` via ``params_to_dict``.
        random_state: Seed for every stochastic component.
        **ctx: Optional context. Recognised keys: ``n_jobs``, ``n_clusters``,
            ``n_classes``, ``season_length``, ``contamination``,
            ``class_weight``. Unknown keys are ignored.

    Returns:
        An unfitted estimator implementing the sklearn API.

    Raises:
        MissingDependencyError: The family's optional package is not installed.
        TrainingError: The family has no honest estimator for this task type.
    """
    if not supports_task(family, task):
        raise TrainingError(
            f"model family '{family.value}' does not apply to task '{task.value}'"
        )
    if not is_available(family):
        raise MissingDependencyError(
            OPTIONAL_PACKAGES.get(family, family.value), f"{family.value} models"
        )

    context = _Context(random_state=random_state, ctx=ctx)
    builder = _BUILDERS.get(family)
    if builder is None:  # pragma: no cover - table covers the whole enum
        raise TrainingError(f"no estimator registered for family '{family.value}'")

    estimator = builder(task, context)
    # Parameters are set on the base estimator before any multilabel wrapper
    # goes on, so an agent's `n_estimators` lands on the forest rather than
    # being rejected by OneVsRestClassifier.
    rejected = _apply_params(estimator, params or {})
    if rejected:
        logger.warning(
            "dropped unsupported params for %s/%s: %s",
            family.value,
            task.value,
            ", ".join(sorted(rejected)),
        )
    if task is TaskType.MULTILABEL_CLASSIFICATION and family not in _NATIVE_MULTILABEL:
        estimator = OneVsRestClassifier(estimator, n_jobs=context.n_jobs)
    return estimator


def unsupported_params(
    family: ModelFamily, task: TaskType, params: dict, **ctx: Any
) -> list[str]:
    """Which of ``params`` the estimator for this family/task would ignore.

    Lets the trainer surface a dropped hyperparameter as a run warning rather
    than only a log line, without duplicating the acceptance rules.

    Args:
        family: The model family.
        task: The task type the estimator would be built for.
        params: Candidate hyperparameters.
        **ctx: Same optional context keys as :func:`build_estimator`.

    Returns:
        Sorted names the estimator does not accept. Empty when the family
        cannot be built at all, since then nothing was dropped.
    """
    if not params or not supports_task(family, task) or not is_available(family):
        return []
    builder = _BUILDERS.get(family)
    if builder is None:  # pragma: no cover - table covers the whole enum
        return []
    try:
        probe = builder(task, _Context(random_state=0, ctx=ctx))
    except Exception as exc:  # noqa: BLE001 - construction problems surface later
        logger.debug("could not probe params for %s: %s", family.value, exc)
        return []
    return sorted(_apply_params(probe, params))


class _Context:
    """Normalised construction context, with settings-derived defaults."""

    def __init__(self, random_state: int, ctx: dict[str, Any]) -> None:
        settings_n_jobs = -1
        try:
            settings_n_jobs = int(get_settings().n_jobs)
        except Exception:  # pragma: no cover - settings always constructible
            pass
        self.random_state = int(random_state)
        self.n_jobs = int(ctx.get("n_jobs") or settings_n_jobs)
        self.n_clusters = max(2, int(ctx.get("n_clusters") or 3))
        self.n_classes = ctx.get("n_classes")
        self.season_length = max(1, int(ctx.get("season_length") or 1))
        self.contamination = ctx.get("contamination") or "auto"
        self.class_weight = ctx.get("class_weight")


def _modernise_params(estimator: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Rewrite hyperparameters sklearn has deprecated but not yet removed.

    Agents propose hyperparameters from the vocabulary they learned, which lags
    the installed library. ``penalty`` on :class:`LogisticRegression` is the live
    example: deprecated in sklearn 1.8, removed in 1.10, and still the name most
    references use. Dropping it would silently discard the agent's intent, and
    passing it through emits a deprecation warning per fold and breaks outright
    on 1.10 — so translate it to the replacement knob and keep the meaning.

    Only applied when the estimator exposes the modern parameter, so this is a
    no-op on older sklearn where ``penalty`` is still the correct spelling.
    """
    if "penalty" not in params:
        return params
    available = estimator.get_params(deep=False)
    if "l1_ratio" not in available or "C" not in available:
        return params

    penalty = params["penalty"]
    updated = {k: v for k, v in params.items() if k != "penalty"}
    if penalty is None or str(penalty).lower() == "none":
        updated.setdefault("C", float("inf"))
    elif penalty == "l2":
        updated.setdefault("l1_ratio", 0.0)
    elif penalty == "l1":
        updated.setdefault("l1_ratio", 1.0)
        updated.setdefault("solver", "saga")
    elif penalty == "elasticnet":
        updated.setdefault("l1_ratio", 0.5)
        updated.setdefault("solver", "saga")
    else:
        return params  # unrecognised value; let sklearn own the error message

    logger.debug("translated deprecated penalty=%r to %s", penalty, updated)
    return updated


def _apply_params(estimator: Any, params: dict[str, Any]) -> list[str]:
    """Set what the estimator accepts; return the names it rejected."""
    params = _modernise_params(estimator, params)
    rejected: list[str] = []
    for key, value in params.items():
        if not isinstance(key, str) or key.startswith("_"):
            rejected.append(str(key))
            continue
        try:
            # set_params raises for unknown names on sklearn estimators; the
            # boosted-tree wrappers accept extras through **kwargs, which is
            # their documented behaviour and safe to honour.
            estimator.set_params(**{key: value})
        except (ValueError, TypeError, KeyError):
            rejected.append(key)
    return rejected


# --- per-family builders --------------------------------------------------


def _linear(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        # "Linear model" for a categorical target means logistic regression.
        return _logistic(task, ctx)
    return LinearRegression()


def _logistic(task: TaskType, ctx: _Context) -> Any:
    # No n_jobs: it has been a no-op on LogisticRegression since sklearn 1.8 and
    # passing it only emits a FutureWarning on every fit.
    return LogisticRegression(
        max_iter=1000,
        random_state=ctx.random_state,
        class_weight=ctx.class_weight,
    )


def _ridge(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return RidgeClassifier(
            alpha=1.0, random_state=ctx.random_state, class_weight=ctx.class_weight
        )
    return Ridge(alpha=1.0, random_state=ctx.random_state)


def _lasso(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        # sklearn 1.8 deprecated `penalty`; l1_ratio=1.0 is the pure-L1 form.
        return LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            max_iter=2000,
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
        )
    return Lasso(alpha=0.001, max_iter=5000, random_state=ctx.random_state)


def _elastic_net(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            max_iter=2000,
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
        )
    return ElasticNet(
        alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=ctx.random_state
    )


def _decision_tree(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
        )
    return DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=5, random_state=ctx.random_state
    )


def _random_forest(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            n_jobs=ctx.n_jobs,
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
        )
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        n_jobs=ctx.n_jobs,
        random_state=ctx.random_state,
    )


def _extra_trees(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return ExtraTreesClassifier(
            n_estimators=300,
            n_jobs=ctx.n_jobs,
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
        )
    return ExtraTreesRegressor(
        n_estimators=300, n_jobs=ctx.n_jobs, random_state=ctx.random_state
    )


def _gradient_boosting(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return GradientBoostingClassifier(random_state=ctx.random_state)
    return GradientBoostingRegressor(random_state=ctx.random_state)


def _hist_gradient_boosting(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return HistGradientBoostingClassifier(
            random_state=ctx.random_state,
            class_weight=ctx.class_weight,
            categorical_features=None,
        )
    return HistGradientBoostingRegressor(
        random_state=ctx.random_state, categorical_features=None
    )


def _xgboost(task: TaskType, ctx: _Context) -> Any:
    xgb = _require("xgboost", "XGBoost models")
    common = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": ctx.n_jobs,
        "random_state": ctx.random_state,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if task in _CLASSIFICATION:
        n_classes = ctx.n_classes
        objective = (
            "multi:softprob"
            if isinstance(n_classes, int) and n_classes > 2
            else "binary:logistic"
        )
        return xgb.XGBClassifier(objective=objective, eval_metric="logloss", **common)
    return xgb.XGBRegressor(objective="reg:squarederror", **common)


def _lightgbm(task: TaskType, ctx: _Context) -> Any:
    lgb = _require("lightgbm", "LightGBM models")
    common = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": ctx.n_jobs,
        "random_state": ctx.random_state,
        "verbose": -1,
    }
    if task in _CLASSIFICATION:
        return lgb.LGBMClassifier(class_weight=ctx.class_weight, **common)
    return lgb.LGBMRegressor(**common)


def _catboost(task: TaskType, ctx: _Context) -> Any:
    catboost = _require("catboost", "CatBoost models")
    common = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": ctx.random_state,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": ctx.n_jobs if ctx.n_jobs and ctx.n_jobs > 0 else -1,
    }
    if task in _CLASSIFICATION:
        return catboost.CatBoostClassifier(**common)
    return catboost.CatBoostRegressor(**common)


def _svm(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        # probability=True costs an internal 5-fold Platt calibration, and is
        # worth it: without it roc_auc and log_loss cannot be reported at all.
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight=ctx.class_weight,
            random_state=ctx.random_state,
        )
    return SVR(C=1.0, kernel="rbf", gamma="scale")


def _knn(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return KNeighborsClassifier(n_neighbors=15, n_jobs=ctx.n_jobs)
    return KNeighborsRegressor(n_neighbors=15, n_jobs=ctx.n_jobs)


def _naive_bayes(task: TaskType, ctx: _Context) -> Any:
    # GaussianNB rather than Multinomial/Bernoulli: the feature frame reaching
    # the zoo is already scaled and may contain negatives, which the count-based
    # variants reject outright.
    return GaussianNB()


def _neural_network(task: TaskType, ctx: _Context) -> Any:
    common = {
        "hidden_layer_sizes": (128, 64),
        "alpha": 1e-4,
        "max_iter": 500,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "random_state": ctx.random_state,
    }
    if task in _CLASSIFICATION:
        return MLPClassifier(**common)
    return MLPRegressor(**common)


def _baseline_dummy(task: TaskType, ctx: _Context) -> Any:
    if task in _CLASSIFICATION:
        return DummyClassifier(strategy="prior", random_state=ctx.random_state)
    return DummyRegressor(strategy="mean")


def _kmeans(task: TaskType, ctx: _Context) -> Any:
    return KMeans(
        n_clusters=ctx.n_clusters, n_init="auto", random_state=ctx.random_state
    )


def _dbscan(task: TaskType, ctx: _Context) -> Any:
    return DBSCAN(eps=0.5, min_samples=5, n_jobs=ctx.n_jobs)


def _gaussian_mixture(task: TaskType, ctx: _Context) -> Any:
    return GaussianMixture(
        n_components=ctx.n_clusters,
        covariance_type="full",
        random_state=ctx.random_state,
    )


def _isolation_forest(task: TaskType, ctx: _Context) -> Any:
    return IsolationForest(
        n_estimators=200,
        contamination=ctx.contamination,
        n_jobs=ctx.n_jobs,
        random_state=ctx.random_state,
    )


def _local_outlier_factor(task: TaskType, ctx: _Context) -> Any:
    # novelty=True so the detector can score held-out rows; with the default
    # novelty=False, LOF exposes only fit_predict on the training data.
    return LocalOutlierFactor(
        n_neighbors=20,
        contamination=ctx.contamination,
        novelty=True,
        n_jobs=ctx.n_jobs,
    )


def _one_class_svm(task: TaskType, ctx: _Context) -> Any:
    return OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")


def _seasonal_naive(task: TaskType, ctx: _Context) -> Any:
    return SeasonalNaiveRegressor(season_length=ctx.season_length)


def _theta(task: TaskType, ctx: _Context) -> Any:
    return StatsmodelsForecaster(kind="theta", season_length=ctx.season_length)


def _exponential_smoothing(task: TaskType, ctx: _Context) -> Any:
    return StatsmodelsForecaster(
        kind="exponential_smoothing", season_length=ctx.season_length
    )


def _sarimax(task: TaskType, ctx: _Context) -> Any:
    return StatsmodelsForecaster(
        kind="sarimax",
        season_length=ctx.season_length,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
    )


_BUILDERS: dict[ModelFamily, Any] = {
    ModelFamily.LINEAR: _linear,
    ModelFamily.LOGISTIC: _logistic,
    ModelFamily.RIDGE: _ridge,
    ModelFamily.LASSO: _lasso,
    ModelFamily.ELASTIC_NET: _elastic_net,
    ModelFamily.DECISION_TREE: _decision_tree,
    ModelFamily.RANDOM_FOREST: _random_forest,
    ModelFamily.EXTRA_TREES: _extra_trees,
    ModelFamily.GRADIENT_BOOSTING: _gradient_boosting,
    ModelFamily.HIST_GRADIENT_BOOSTING: _hist_gradient_boosting,
    ModelFamily.XGBOOST: _xgboost,
    ModelFamily.LIGHTGBM: _lightgbm,
    ModelFamily.CATBOOST: _catboost,
    ModelFamily.SVM: _svm,
    ModelFamily.KNN: _knn,
    ModelFamily.NAIVE_BAYES: _naive_bayes,
    ModelFamily.NEURAL_NETWORK: _neural_network,
    ModelFamily.BASELINE_DUMMY: _baseline_dummy,
    ModelFamily.KMEANS: _kmeans,
    ModelFamily.DBSCAN: _dbscan,
    ModelFamily.GAUSSIAN_MIXTURE: _gaussian_mixture,
    ModelFamily.ISOLATION_FOREST: _isolation_forest,
    ModelFamily.LOCAL_OUTLIER_FACTOR: _local_outlier_factor,
    ModelFamily.ONE_CLASS_SVM: _one_class_svm,
    ModelFamily.SEASONAL_NAIVE: _seasonal_naive,
    ModelFamily.THETA: _theta,
    ModelFamily.EXPONENTIAL_SMOOTHING: _exponential_smoothing,
    ModelFamily.SARIMAX: _sarimax,
}


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

_SPACES: dict[ModelFamily, dict[str, list[Any]]] = {
    ModelFamily.LOGISTIC: {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ModelFamily.DECISION_TREE: {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    ModelFamily.RANDOM_FOREST: {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 8, 16, 24],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
    },
    ModelFamily.EXTRA_TREES: {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 8, 16, 24],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
    },
    ModelFamily.GRADIENT_BOOSTING: {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.85, 1.0],
    },
    ModelFamily.HIST_GRADIENT_BOOSTING: {
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 400],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 20, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    ModelFamily.XGBOOST: {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 6, 8],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 7],
        "reg_lambda": [0.5, 1.0, 5.0],
    },
    ModelFamily.LIGHTGBM: {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 20, 50],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    ModelFamily.CATBOOST: {
        "iterations": [300, 600, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 9.0],
    },
    ModelFamily.KNN: {
        "n_neighbors": [3, 5, 11, 21, 35],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    ModelFamily.NAIVE_BAYES: {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    ModelFamily.NEURAL_NETWORK: {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
    ModelFamily.KMEANS: {
        "n_clusters": [2, 3, 4, 5, 6, 8, 10],
        "init": ["k-means++", "random"],
    },
    ModelFamily.DBSCAN: {
        "eps": [0.2, 0.5, 1.0, 1.5, 2.0],
        "min_samples": [3, 5, 10, 20],
    },
    ModelFamily.GAUSSIAN_MIXTURE: {
        "n_components": [2, 3, 4, 5, 6, 8],
        "covariance_type": ["full", "tied", "diag", "spherical"],
    },
    ModelFamily.ISOLATION_FOREST: {
        "n_estimators": [100, 200, 400],
        "max_samples": ["auto", 0.5, 0.8],
        "max_features": [0.5, 0.8, 1.0],
    },
    ModelFamily.LOCAL_OUTLIER_FACTOR: {
        "n_neighbors": [5, 10, 20, 35, 50],
        "p": [1, 2],
    },
    ModelFamily.ONE_CLASS_SVM: {
        "nu": [0.01, 0.05, 0.1, 0.2, 0.35],
        "gamma": ["scale", "auto", 0.01, 0.1],
    },
    ModelFamily.SEASONAL_NAIVE: {"season_length": [1, 7, 12, 24, 52]},
    ModelFamily.EXPONENTIAL_SMOOTHING: {
        "trend": ["add", None],
        "damped_trend": [True, False],
    },
    ModelFamily.THETA: {"season_length": [1, 4, 7, 12, 52]},
    ModelFamily.SARIMAX: {
        "order": [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    },
}

_REGRESSION_ONLY_SPACES: dict[ModelFamily, dict[str, list[Any]]] = {
    ModelFamily.LINEAR: {"fit_intercept": [True, False]},
    ModelFamily.RIDGE: {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ModelFamily.LASSO: {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0]},
    ModelFamily.ELASTIC_NET: {
        "alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    ModelFamily.SVM: {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "epsilon": [0.01, 0.1, 0.5],
    },
}

_CLASSIFICATION_ONLY_SPACES: dict[ModelFamily, dict[str, list[Any]]] = {
    ModelFamily.LINEAR: {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ModelFamily.RIDGE: {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ModelFamily.LASSO: {"C": [0.01, 0.1, 1.0, 10.0]},
    ModelFamily.ELASTIC_NET: {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    ModelFamily.SVM: {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "linear"],
    },
}


def default_search_space(family: ModelFamily, task: TaskType) -> dict:
    """A sane hyperparameter grid for this family and task.

    Used as the fallback when the Tuning Agent's ``search_space`` is empty or
    every entry names a parameter the estimator rejects. Values are discrete
    lists so one space works unchanged for grid search, random search, halving,
    and Optuna's categorical suggestion.

    Args:
        family: The model family being tuned.
        task: The task type, which switches the linear-model spaces between
            their ``alpha`` (regression) and ``C`` (classification) forms.

    Returns:
        Mapping of parameter name to candidate values. Empty for families with
        nothing meaningful to tune, such as ``baseline_dummy``.
    """
    if task in _CLASSIFICATION and family in _CLASSIFICATION_ONLY_SPACES:
        space = _CLASSIFICATION_ONLY_SPACES[family]
    elif task in _REGRESSION_LIKE and family in _REGRESSION_ONLY_SPACES:
        space = _REGRESSION_ONLY_SPACES[family]
    else:
        space = _SPACES.get(family, {})
    return {key: list(values) for key, values in space.items()}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _has_2d(X: Any) -> bool:
    shape = getattr(X, "shape", None)
    return isinstance(shape, tuple) and len(shape) == 2


def _row_count(X: Any) -> int:
    if X is None:
        return 0
    shape = getattr(X, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0])
    try:
        return int(len(X))
    except TypeError:
        return 0
