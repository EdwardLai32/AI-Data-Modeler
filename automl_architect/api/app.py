"""FastAPI application factory.

The factory shape matters for two reasons: tests need an app with patched
settings, and ``uvicorn --reload`` needs an import string. Both are served —
:func:`create_app` for the former, a lazily-constructed module attribute ``app``
for the latter (``uvicorn automl_architect.api.app:app``), built on first access
so importing this module for its factory does not open a database.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from ..core.errors import (
    ConfigurationError,
    IngestionError,
    MissingDependencyError,
    ValidationError,
)
from ..storage.artifacts import ArtifactAccessError
from .routes import get_repository, router
from .schemas import ErrorResponse
from .service import get_run_manager, reset_run_manager

logger = logging.getLogger(__name__)

DESCRIPTION = """\
An autonomous multi-agent AI data scientist.

Start a run with `POST /api/runs` (JSON with a `uri`/`source`, or a multipart
file upload). The response returns immediately; the run executes in the
background. Follow it with `GET /api/runs/{run_id}/stream` for Server-Sent
Events, or poll `GET /api/runs/{run_id}/events?after=<sequence>`.

The event stream is resumable: pass the last sequence you saw and nothing that
happened in the gap is lost.
"""

#: Exception type -> HTTP status. Anything unlisted becomes a 500.
_STATUS_BY_ERROR: tuple[tuple[type[Exception], int], ...] = (
    (MissingDependencyError, 503),
    (ConfigurationError, 503),
    (ArtifactAccessError, 404),
    (ValidationError, 422),
    (IngestionError, 400),
)


def _configure_logging(settings: Settings) -> None:
    """Set up root logging once, honouring the configured level."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Settings override; the cached singleton by default.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    resolved = settings or get_settings()
    _configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        resolved.ensure_dirs()
        # Touching the repository here surfaces an unusable database at startup
        # rather than on the first request.
        try:
            get_repository().ping()
        except Exception:
            logger.exception("database is not usable at startup")
        yield
        reset_run_manager()

    app = FastAPI(
        title="AI Data Modeler",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The SSE cursor header must survive the browser's CORS filter, or a
        # reconnecting EventSource silently loses its place.
        expose_headers=["Last-Event-ID", "Content-Disposition"],
    )

    app.include_router(router)
    _register_error_handlers(app)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        """Point a curious browser at the docs and the health check."""
        return {
            "name": "AI Data Modeler",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
            "runs": "/api/runs",
            "active_runs": get_run_manager(resolved).active_ids(),
        }

    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Map the package's exception hierarchy onto HTTP status codes."""

    async def handle_known(request: Request, exc: Exception) -> JSONResponse:
        code = next(
            (status for kind, status in _STATUS_BY_ERROR if isinstance(exc, kind)), 500
        )
        if code >= 500:
            logger.exception("unhandled error on %s", request.url.path)
        payload = ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
            run_id=request.path_params.get("run_id"),
        )
        return JSONResponse(status_code=code, content=payload.model_dump(mode="json"))

    for error_type, _ in _STATUS_BY_ERROR:
        app.add_exception_handler(error_type, handle_known)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=type(exc).__name__,
                detail=str(exc),
                run_id=request.path_params.get("run_id"),
            ).model_dump(mode="json"),
        )


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Build ``app`` on first access, for ``uvicorn automl_architect.api.app:app``."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app"]
