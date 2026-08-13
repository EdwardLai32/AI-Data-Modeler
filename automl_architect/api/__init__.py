"""HTTP surface: FastAPI app, endpoints, and run supervision.

Imports are lazy so that ``from automl_architect.api.schemas import ...`` (which
the CLI does) never pulls FastAPI into the process.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "get_run_manager", "router"]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import shim
    if name == "create_app":
        from .app import create_app

        return create_app
    if name == "router":
        from .routes import router

        return router
    if name == "get_run_manager":
        from .service import get_run_manager

        return get_run_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
