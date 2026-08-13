"""AI Data Modeler — an autonomous multi-agent AI data scientist.

The shape of the system in one paragraph: a deterministic orchestrator walks a
plan that a reasoning agent wrote, invoking specialist agents that *decide* and
executor modules that *do*. Claude never computes a statistic or trains a model;
it reads measured facts, chooses among options, and explains itself in a typed
schema. Everything numeric comes from pandas, scikit-learn, XGBoost, LightGBM,
Optuna, and SHAP. That split is what makes the output both trustworthy and
auditable — every number is reproducible, and every choice has a recorded reason.

Typical use::

    from automl_architect import analyse

    summary = analyse("data/churn.csv", target="churned")
    print(summary.report.executive_summary)
"""

from __future__ import annotations

from .config import Settings, get_settings
from .core.schemas import (
    DataSource,
    RunConfig,
    RunStatus,
    RunSummary,
    SourceKind,
    TaskType,
)

__version__ = "0.1.0"

__all__ = [
    "DataSource",
    "RunConfig",
    "RunStatus",
    "RunSummary",
    "Settings",
    "SourceKind",
    "TaskType",
    "__version__",
    "analyse",
    "analyze",
    "get_settings",
]


def __getattr__(name: str):  # pragma: no cover - lazy import shim
    """Defer the heavy imports (pandas, sklearn) until an entry point is used.

    Keeps ``import automl_architect`` cheap for the CLI's ``--help`` path and for
    consumers that only want the schemas.
    """
    if name in ("analyse", "analyze"):
        from .runner import analyse as _analyse

        return _analyse
    if name == "AutoMLArchitect":
        from .runner import AutoMLArchitect as _cls

        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
