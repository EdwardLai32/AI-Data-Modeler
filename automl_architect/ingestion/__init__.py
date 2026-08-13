"""Data ingestion: thirteen source kinds behind one router.

Everything enters the system through :func:`load_source`, which dispatches on
:class:`~automl_architect.core.schemas.SourceKind` via a registry and returns the
frame together with a fully-populated
:class:`~automl_architect.core.schemas.IngestionResult`::

    from automl_architect.ingestion import load_source
    from automl_architect.core.schemas import DataSource, SourceKind

    frame, result = load_source(
        DataSource(kind=SourceKind.CSV, uri="data/churn.csv"), max_rows=100_000
    )
    print(result.n_rows, result.validation_warnings)

Attribute access is lazy so that ``import automl_architect.ingestion`` stays
cheap for callers that only want the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .base import Connector, LoadOutcome, register, registered_kinds
    from .dataframe import make_dataframe_source, register_frame
    from .router import available_kinds, infer_source, load_source

__all__ = [
    "Connector",
    "LoadOutcome",
    "available_kinds",
    "infer_source",
    "load_source",
    "make_dataframe_source",
    "register",
    "register_frame",
    "registered_kinds",
]

_EXPORTS = {
    "Connector": ("base", "Connector"),
    "LoadOutcome": ("base", "LoadOutcome"),
    "register": ("base", "register"),
    "registered_kinds": ("base", "registered_kinds"),
    "make_dataframe_source": ("dataframe", "make_dataframe_source"),
    "register_frame": ("dataframe", "register_frame"),
    "available_kinds": ("router", "available_kinds"),
    "infer_source": ("router", "infer_source"),
    "load_source": ("router", "load_source"),
}


def __getattr__(name: str) -> Any:
    """Resolve the public API on first use."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
