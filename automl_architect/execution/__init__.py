"""Deterministic executors: where an agent's decision becomes real computation.

Nothing in this package reasons. Each module takes a typed decision plus the run
state and applies it with pandas, scikit-learn, XGBoost, LightGBM, Optuna, or
SHAP, recording what it did on the state so the result is reproducible and
auditable.

Imports are resolved lazily through :func:`__getattr__` for two reasons. The
first is startup cost — pulling in sklearn, xgboost, and shap to read a schema is
wasteful. The second matters more: several executors depend on optional packages,
and a lazy lookup means an absent XGBoost breaks exactly one attribute rather
than the whole package. A module that cannot be imported degrades to a missing
attribute (logged), which callers can detect with :func:`hasattr` and route
around, instead of taking the run down at import time.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Public symbol -> module that defines it. The executor interface contract.
_EXPORTS: dict[str, str] = {
    # splitter
    "make_splits": "splitter",
    "resolve_split_strategy": "splitter",
    "make_cv_splitter": "splitter",
    # cleaning
    "apply_cleaning_plan": "cleaning_ops",
    # features
    "apply_feature_plan": "feature_ops",
    "transformed_feature_names": "feature_ops",
    "SafeDimensionReduction": "feature_ops",
    # model zoo
    "available_families": "model_zoo",
    "is_available": "model_zoo",
    "build_estimator": "model_zoo",
    "default_search_space": "model_zoo",
    "supports_proba": "model_zoo",
    # metrics
    "primary_metric_for": "metrics",
    "higher_is_better": "metrics",
    "score_predictions": "metrics",
    "sklearn_scorer_name": "metrics",
    # training
    "run_experiments": "trainer",
    "fit_final_model": "trainer",
    # tuning
    "run_tuning": "tuner",
    # explainability
    "compute_explanations": "explainer",
    # diagnostics
    "compute_diagnostics": "diagnostics",
    "DiagnosticsBundle": "diagnostics",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public executor symbol on first use."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except Exception as exc:  # noqa: BLE001 - one unavailable executor, not a dead package
        logger.warning(
            "execution.%s is unavailable (%s); %r cannot be used in this environment",
            module_name,
            exc,
            name,
        )
        raise AttributeError(
            f"{name!r} is unavailable because automl_architect.execution."
            f"{module_name} failed to import: {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache, so the next lookup skips this path entirely
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
