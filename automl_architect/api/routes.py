"""HTTP endpoints.

Three things here are load-bearing beyond the obvious CRUD.

*   **The SSE stream is lossless across reconnects.** A client hands back the
    last sequence it saw (as a query parameter, or the ``Last-Event-ID`` header
    that browsers send automatically). The handler subscribes to the live bus
    *first*, then replays from storage, then tails the subscription while
    discarding anything already sent. Subscribing after replaying would drop
    every event emitted during the gap, which is exactly the failure the cursor
    exists to prevent.
*   **The event loop is never blocked.** The repository is synchronous
    SQLAlchemy, so every read from it inside an async handler goes through
    :func:`asyncio.to_thread`. The bus is synchronous and fires on the run's own
    thread, so its callback only does ``call_soon_threadsafe`` onto a queue.
*   **Artifacts are served from inside the run directory or not at all.** Every
    client-supplied path goes through :func:`~automl_architect.storage.artifacts.resolve_artifact`,
    which resolves symlinks and rejects anything that escapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .. import __version__
from ..config import Settings, get_settings
from ..core.schemas import (
    ApprovalRequest,
    QuestionAnswer,
    RunEvent,
    RunSummary,
)
from ..storage.artifacts import (
    ArtifactAccessError,
    list_artifacts,
    resolve_artifact,
    store_upload,
)
from ..storage.repository import RunRepository
from .schemas import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalListResponse,
    ArtifactEntry,
    ArtifactListResponse,
    AskRequest,
    CancelResponse,
    EventPage,
    HealthResponse,
    RunListItem,
    RunListResponse,
    StartRunRequest,
    StartRunResponse,
    UploadResponse,
    data_source_from_uri,
)
from .service import (
    REPORT_FORMAT_ATTRS,
    TERMINAL_STATUSES,
    RunManager,
    answer_question,
    available_report_formats,
    detect_features,
    get_run_manager,
    locate_report,
    module_available,
    python_version,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["runs"])

#: Fields on StartRunRequest that arrive as comma-separated text in a form post.
_LIST_FIELDS = ("report_formats", "fairness_attributes")

#: Which report to serve when the caller does not name a format: most
#: presentable first.
_REPORT_PREFERENCE = ("html", "pdf", "pptx", "markdown", "json")

#: Media types for what this platform writes. ``mimetypes`` does not know
#: ``.md`` or ``.joblib`` on every OS, and guessing octet-stream for a report
#: turns "view it" into "download it".
_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".joblib": "application/octet-stream",
    ".parquet": "application/vnd.apache.parquet",
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
}

#: 422. Starlette renamed the constant; read whichever this version exposes so
#: the module neither emits a deprecation warning nor breaks on an older install.
HTTP_422 = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", None
) or status.HTTP_422_UNPROCESSABLE_ENTITY

#: 413, read the same way and for the same reason.
HTTP_413 = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", None
) or status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

#: Modules Starlette will use for multipart parsing if either is installed.
_MULTIPART_MODULES = ("python_multipart", "multipart")

#: Cap on a body buffered in memory by the fallback multipart parser (256 MiB).
#: Starlette's parser spools to disk and is not subject to this.
MAX_INLINE_UPLOAD_BYTES = 256 * 1024 * 1024

_MULTIPART_HINT = (
    "this form body could not be parsed. Install the optional 'python-multipart' "
    "package for full multipart support (pip install python-multipart), or POST "
    "application/json with a 'uri' or 'source' field instead."
)


@lru_cache(maxsize=4)
def _repository_for(database_url: str) -> RunRepository:
    return RunRepository(database_url)


def get_repository() -> RunRepository:
    """The repository every handler shares.

    Cached per database URL rather than per process: a settings change (a test
    redirecting the workspace, an operator repointing the database) then yields a
    fresh repository instead of silently serving reads from the previous one.
    """
    return _repository_for(get_settings().resolved_database_url)


def reset_repository_cache() -> None:
    """Drop cached repositories. For tests and for a settings reload."""
    _repository_for.cache_clear()


def get_manager() -> RunManager:
    """The process-wide run manager."""
    return get_run_manager()


def _settings() -> Settings:
    return get_settings()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report whether this deployment can actually execute a run."""
    settings = _settings()
    warnings: list[str] = []

    workspace_writable = await asyncio.to_thread(_probe_workspace, settings)
    if not workspace_writable:
        warnings.append(f"workspace {settings.workspace} is not writable")

    # Reading the default database URL creates the workspace directory, so this
    # whole block can fail on a misconfigured deployment. Health must still answer.
    database_url = "unresolved"
    database_reachable = False
    stored = 0
    try:
        repository = get_repository()
        database_url = _redact(repository.database_url)
        database_reachable = await asyncio.to_thread(repository.ping)
        if database_reachable:
            stored = await asyncio.to_thread(repository.count_runs)
        else:
            warnings.append("database is not reachable")
    except Exception as exc:
        logger.exception("health check could not open the database")
        warnings.append(f"database is unusable: {type(exc).__name__}: {exc}")

    credentials = settings.has_api_key()
    if settings.offline:
        # Not a warning: offline mode is a deliberate configuration in which no
        # credential is consulted, so flagging its absence would be noise.
        credentials = True
    elif not credentials:
        warnings.append(
            "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment; "
            "agent steps will fail unless the SDK finds credentials elsewhere. "
            "Set AUTOML_OFFLINE=1 to run on the deterministic rule engine instead"
        )

    try:
        active_runs = len(get_manager().active_ids())
    except Exception as exc:  # the manager also needs a resolvable workspace
        logger.exception("health check could not reach the run manager")
        active_runs = 0
        warnings.append(f"the run manager is unavailable: {type(exc).__name__}: {exc}")

    features = detect_features()

    return HealthResponse(
        status="ok" if not warnings else "degraded",
        version=__version__,
        model=settings.model,
        python_version=python_version(),
        credentials_configured=credentials,
        workspace=str(settings.workspace),
        workspace_writable=workspace_writable,
        database_url=database_url,
        database_reachable=database_reachable,
        active_runs=active_runs,
        stored_runs=stored,
        features=features,
        warnings=warnings,
    )


def _probe_workspace(settings: Settings) -> bool:
    try:
        settings.ensure_dirs()
        probe = settings.workspace / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _redact(url: str) -> str:
    """Strip credentials out of a database URL before returning it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse, tags=["data"])
async def upload_dataset(
    request: Request,
    filename: str | None = Query(
        default=None, description="Required when posting a raw body instead of a form."
    ),
) -> UploadResponse:
    """Store a dataset and return the :class:`DataSource` that reads it.

    Accepts either ``multipart/form-data`` with a ``file`` part, or a raw binary
    body plus ``?filename=``. The raw path exists so the endpoint still works
    when ``python-multipart`` is not installed.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        _fields, upload = await _read_form(request)
        if upload is None:
            raise HTTPException(
                HTTP_422, "no file part found in the form"
            )
        name, payload = upload
    else:
        payload = await request.body()
        if not payload:
            raise HTTPException(HTTP_422, "empty request body")
        if not filename:
            raise HTTPException(
                HTTP_422,
                "?filename= is required when uploading a raw body",
            )
        name = filename

    path = await asyncio.to_thread(store_upload, payload, name, settings=_settings())
    return UploadResponse(
        filename=path.name,
        path=str(path),
        size_bytes=len(payload),
        source=data_source_from_uri(str(path)),
    )


def _starlette_can_parse_multipart() -> bool:
    """Whether Starlette's own multipart parser is usable in this install."""
    return any(module_available(name) for name in _MULTIPART_MODULES)


async def _read_form(request: Request) -> tuple[dict[str, str], tuple[str, bytes] | None]:
    """Parse a form body into scalar fields plus at most one uploaded file.

    Prefers Starlette's parser, which spools large parts to disk and handles the
    full grammar. Falls back to a local parser when the optional
    ``python-multipart`` package is absent, because "upload a dataset" is the
    primary way a browser starts a run and it should not depend on an extra.
    Starlette gates *every* form body on that package — it reads the content type
    with ``parse_options_header`` before it dispatches — so the urlencoded shape
    needs the same fallback as the multipart one, not just multipart.

    Raises:
        HTTPException: 413 if the body exceeds the inline cap, 422 on a malformed
            body, 503 if form parsing is unavailable for an unexpected reason.
    """
    content_type = request.headers.get("content-type", "")
    if not _starlette_can_parse_multipart():
        if content_type.startswith("multipart/form-data"):
            boundary = _multipart_boundary(content_type)
            if boundary is None:
                raise HTTPException(HTTP_422, "multipart body declares no boundary")
            return _parse_multipart(await _read_capped_body(request), boundary)
        if content_type.startswith("application/x-www-form-urlencoded"):
            return _parse_urlencoded(await _read_capped_body(request)), None

    try:
        form = await request.form()
    except Exception as exc:  # starlette asserts rather than raising ImportError
        logger.warning("form parsing unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _MULTIPART_HINT) from exc

    fields: dict[str, str] = {}
    upload: tuple[str, bytes] | None = None
    try:
        for key, value in form.multi_items():
            candidate_name = getattr(value, "filename", None)
            if candidate_name:
                upload = (candidate_name, await value.read())  # type: ignore[union-attr]
            else:
                fields[key] = str(value)
    finally:
        await form.close()
    return fields, upload


async def _read_capped_body(
    request: Request, cap: int = MAX_INLINE_UPLOAD_BYTES
) -> bytes:
    """Buffer the request body, refusing anything over ``cap``.

    Streamed rather than ``await request.body()`` so an oversized upload is
    rejected as it arrives instead of after the whole thing is in memory.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise HTTPException(
                HTTP_413,
                f"upload exceeds the {cap // (1024 * 1024)} MiB inline limit; "
                "write the file to the server and pass its path as 'uri' instead",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_urlencoded(body: bytes) -> dict[str, str]:
    """Decode an ``application/x-www-form-urlencoded`` body into scalar fields.

    Blank values are kept so that an explicitly-cleared form field stays visible
    to the caller, which drops it rather than validating an empty string into a
    typed model. A repeated key takes its last value, matching Starlette.
    """
    pairs = parse_qsl(
        body.decode("utf-8", errors="replace"),
        keep_blank_values=True,
        errors="replace",
    )
    return {key: value for key, value in pairs}


def _multipart_boundary(content_type: str) -> bytes | None:
    """Extract the boundary token from a ``multipart/form-data`` content type."""
    for parameter in content_type.split(";")[1:]:
        key, _, value = parameter.strip().partition("=")
        if key.strip().lower() != "boundary":
            continue
        token = value.strip().strip('"')
        return token.encode("latin-1") if token else None
    return None


def _parse_multipart(
    body: bytes, boundary: bytes
) -> tuple[dict[str, str], tuple[str, bytes] | None]:
    """Minimal RFC 7578 parser: scalar fields plus the last file part.

    Deliberately narrow — one file, no nested multipart, no transfer encodings —
    which is the whole shape this API accepts. Anything richer should install
    ``python-multipart`` and get Starlette's parser instead.
    """
    fields: dict[str, str] = {}
    upload: tuple[str, bytes] | None = None
    delimiter = b"--" + boundary

    for segment in body.split(delimiter):
        block = segment[2:] if segment.startswith(b"\r\n") else segment
        if not block or block.startswith(b"--"):  # preamble or closing marker
            continue
        head, separator, content = block.partition(b"\r\n\r\n")
        if not separator:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]

        name, filename = _part_names(head)
        if name is None:
            continue
        if filename:
            upload = (filename, content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, upload


def _part_names(head: bytes) -> tuple[str | None, str | None]:
    """Read ``name`` and ``filename`` out of one part's header block."""
    name: str | None = None
    filename: str | None = None
    for line in head.decode("utf-8", errors="replace").splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() != "content-disposition":
            continue
        for parameter in value.split(";")[1:]:
            attribute, _, raw = parameter.strip().partition("=")
            token = raw.strip().strip('"')
            if attribute.strip().lower() == "name":
                name = token
            elif attribute.strip().lower() == "filename":
                filename = token
    return name, filename


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.post(
    "/runs",
    response_model=StartRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/StartRunRequest"}
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "target_column": {"type": "string"},
                            "project": {"type": "string"},
                        },
                    }
                },
            }
        }
    },
)
async def start_run(request: Request) -> StartRunResponse:
    """Start a run and return its id immediately.

    Accepts JSON (with ``source``, ``uri``, or ``upload_path``) or a multipart
    form carrying the dataset itself alongside the same field names. Execution
    happens on a background thread; follow it with ``/events`` or ``/stream``.
    """
    payload = await _parse_start_request(request)
    try:
        config = payload.to_run_config()
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc

    manager = get_manager()
    try:
        handle = await asyncio.to_thread(manager.start, config)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return StartRunResponse(
        run_id=handle.run_id,
        project=config.project,
        status=handle.status,
        stream_url=f"/api/runs/{handle.run_id}/stream",
        events_url=f"/api/runs/{handle.run_id}/events",
        run_url=f"/api/runs/{handle.run_id}",
    )


async def _parse_start_request(request: Request) -> StartRunRequest:
    """Build a :class:`StartRunRequest` from a JSON or multipart body."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        fields, upload = await _read_form(request)
        data: dict[str, Any] = {
            key: value for key, value in fields.items() if value != ""
        }
        for key in _LIST_FIELDS:
            if isinstance(data.get(key), str):
                data[key] = [item.strip() for item in data[key].split(",") if item.strip()]
        if upload is not None:
            name, blob = upload
            path = await asyncio.to_thread(store_upload, blob, name, settings=_settings())
            data["upload_path"] = str(path)
    else:
        raw = await request.body()
        if not raw:
            raise HTTPException(HTTP_422, "empty request body")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise HTTPException(
                HTTP_422, f"invalid JSON body: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                HTTP_422, "request body must be a JSON object"
            )

    try:
        return StartRunRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            HTTP_422, json.loads(exc.json())
        ) from exc


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    project: str | None = Query(default=None, description="Filter to one project."),
    limit: int = Query(default=50, ge=1, le=500),
) -> RunListResponse:
    """List runs, most recent first."""
    repository = get_repository()
    rows = await asyncio.to_thread(repository.list_run_index, project, limit)
    active = set(get_manager().active_ids())
    items = [
        RunListItem(**row, is_active=row["run_id"] in active) for row in rows
    ]
    return RunListResponse(project=project, count=len(items), runs=items)


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str) -> RunSummary:
    """The full run summary, live-snapshotted if the run is still executing."""
    summary = await asyncio.to_thread(get_manager().summary, run_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    return summary


@router.get("/runs/{run_id}/events", response_model=EventPage)
async def get_events(
    run_id: str,
    after: int = Query(default=0, ge=0, description="Return events after this sequence."),
    limit: int = Query(default=500, ge=1, le=5000),
) -> EventPage:
    """Poll the event log with a sequence cursor."""
    manager = get_manager()
    repository = get_repository()
    page = await asyncio.to_thread(
        _collect_events, manager, repository, run_id, after, limit
    )
    if not page and await asyncio.to_thread(manager.summary, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")

    run_status = await asyncio.to_thread(manager.status, run_id)
    return EventPage(
        run_id=run_id,
        after=after,
        last_sequence=page[-1].sequence if page else after,
        count=len(page),
        status=run_status,
        finished=bool(run_status and run_status in TERMINAL_STATUSES),
        events=page,
    )


def _collect_events(
    manager: RunManager,
    repository: RunRepository,
    run_id: str,
    after: int,
    limit: int | None = None,
) -> list[RunEvent]:
    """Merge persisted and in-memory events, deduplicated by sequence.

    The in-memory bus runs ahead of storage by up to one writer batch, so reading
    either source alone would lag or lose history. Both are ordered ascending
    from the cursor, so trimming the merge to ``limit`` yields the same window a
    single source would have.
    """
    merged: dict[int, RunEvent] = {
        event.sequence: event
        for event in repository.get_events(run_id, after, limit=limit)
    }
    handle = manager.get(run_id)
    if handle is not None:
        for event in handle.bus.since(after):
            merged.setdefault(event.sequence, event)
    ordered = [merged[key] for key in sorted(merged)]
    return ordered[:limit] if limit is not None else ordered


@router.get("/runs/{run_id}/stream", tags=["events"])
async def stream_run(
    request: Request,
    run_id: str,
    last_sequence: int = Query(default=0, ge=0),
    after: int | None = Query(
        default=None, ge=0, description="Alias for last_sequence."
    ),
) -> Response:
    """Server-Sent Events for a run, resumable without loss.

    The cursor is taken from ``after``, then ``last_sequence``, then the
    ``Last-Event-ID`` header — so a browser ``EventSource`` that reconnects
    resumes exactly where it stopped with no client-side bookkeeping.
    """
    from sse_starlette.sse import EventSourceResponse

    cursor = after if after is not None else last_sequence
    header_cursor = request.headers.get("last-event-id")
    if header_cursor and header_cursor.isdigit():
        cursor = max(cursor, int(header_cursor))

    manager = get_manager()
    repository = get_repository()
    if manager.get(run_id) is None:
        if await asyncio.to_thread(repository.get_run, run_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")

    generator = _event_stream(request, run_id, cursor, manager, repository)
    return EventSourceResponse(generator, ping=15)


async def _event_stream(
    request: Request,
    run_id: str,
    cursor: int,
    manager: RunManager,
    repository: RunRepository,
) -> Any:
    """Yield SSE payloads: replay from the cursor, then tail live events."""
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue[RunEvent] = asyncio.Queue()
    handle = manager.get(run_id)
    unsubscribe = None

    if handle is not None:
        def _push(event: RunEvent) -> None:
            # Runs on the orchestrator's thread: hand off and return immediately.
            loop.call_soon_threadsafe(inbox.put_nowait, event)

        unsubscribe = handle.bus.subscribe(_push)

    sent = cursor
    try:
        for event in await asyncio.to_thread(
            _collect_events, manager, repository, run_id, sent
        ):
            if event.sequence > sent:
                yield _sse(event)
                sent = event.sequence

        while True:
            if await request.is_disconnected():
                break
            run_status = handle.status if handle else await asyncio.to_thread(
                repository.run_status, run_id
            )
            terminal = bool(run_status and run_status in TERMINAL_STATUSES)
            try:
                event = await asyncio.wait_for(inbox.get(), timeout=1.0)
            except TimeoutError:
                if handle is None:
                    # The run belongs to another process; storage is the only feed.
                    for stored in await asyncio.to_thread(
                        repository.get_events, run_id, sent
                    ):
                        yield _sse(stored)
                        sent = stored.sequence
                if terminal and inbox.empty():
                    break
                continue
            if event.sequence <= sent:
                continue
            yield _sse(event)
            sent = event.sequence
    except asyncio.CancelledError:  # client vanished mid-write
        raise
    finally:
        if unsubscribe is not None:
            unsubscribe()

    final_status = await asyncio.to_thread(manager.status, run_id)
    yield {
        "event": "stream_closed",
        "id": str(sent),
        "data": json.dumps(
            {
                "run_id": run_id,
                "last_sequence": sent,
                "status": final_status.value if final_status else None,
            }
        ),
    }


def _sse(event: RunEvent) -> dict[str, str]:
    """One SSE frame. The id is the sequence, which is the reconnect cursor."""
    return {
        "event": event.kind.value,
        "id": str(event.sequence),
        "data": event.model_dump_json(),
    }


@router.post("/runs/{run_id}/cancel", response_model=CancelResponse)
async def cancel_run(run_id: str) -> CancelResponse:
    """Ask a run to stop at its next checkpoint."""
    manager = get_manager()
    if await asyncio.to_thread(manager.summary, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    requested, detail = await asyncio.to_thread(manager.cancel, run_id)
    return CancelResponse(
        run_id=run_id,
        status=await asyncio.to_thread(manager.status, run_id),
        cancellation_requested=requested,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/approvals", response_model=ApprovalListResponse)
async def get_approvals(run_id: str) -> ApprovalListResponse:
    """Approval requests for a run, split into pending and resolved."""
    manager = get_manager()
    if await asyncio.to_thread(manager.summary, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    approvals: list[ApprovalRequest] = await asyncio.to_thread(manager.approvals, run_id)
    return ApprovalListResponse(
        run_id=run_id,
        approvals=approvals,
        pending=[item for item in approvals if item.decision == "pending"],
        resolved=[item for item in approvals if item.decision != "pending"],
    )


@router.post(
    "/runs/{run_id}/approvals/{request_id}", response_model=ApprovalDecisionResponse
)
async def decide_approval(
    run_id: str, request_id: str, body: ApprovalDecisionRequest
) -> ApprovalDecisionResponse:
    """Approve or reject a pending request, releasing the paused run."""
    manager = get_manager()
    approval, resumed = await asyncio.to_thread(
        manager.resolve_approval,
        run_id,
        request_id,
        body.decision,
        note=body.note,
        decided_by=body.decided_by,
    )
    if approval is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no approval request {request_id} on run {run_id}",
        )
    detail = (
        "the run was waiting on this decision and has been released"
        if resumed
        else "decision recorded; no run thread was blocked on it"
    )
    return ApprovalDecisionResponse(
        run_id=run_id,
        request_id=request_id,
        approval=approval,
        resumed=resumed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@router.post("/runs/{run_id}/ask", response_model=QuestionAnswer, tags=["insight"])
async def ask_run(run_id: str, body: AskRequest) -> QuestionAnswer:
    """Answer a natural-language question about a run, grounded in its record."""
    summary = await asyncio.to_thread(get_manager().summary, run_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    try:
        return await asyncio.to_thread(answer_question, summary, body.question)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("question answering failed for run %s", run_id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"could not answer the question: {type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Reports and artifacts
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/report", tags=["artifacts"])
async def get_report(
    run_id: str,
    format: str | None = Query(
        default=None,
        description="markdown|html|pdf|pptx|json. Omit it to get the most "
        "presentable format the run actually rendered.",
    ),
    download: bool = Query(default=False),
) -> FileResponse:
    """Serve a rendered report for a run.

    With no ``format`` the best available one is served. A run configured for
    markdown only should not 404 a caller who simply asked for "the report".
    """
    summary = await asyncio.to_thread(get_manager().summary, run_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")

    if format is None:
        for key in _REPORT_PREFERENCE:
            path = await asyncio.to_thread(locate_report, run_id, summary, key)
            if path is not None:
                return _file_response(path, download=download)
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no report was rendered for run {run_id}"
        )

    key = format.lower().strip()
    if key not in REPORT_FORMAT_ATTRS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unsupported format {format!r}; choose from {sorted(set(REPORT_FORMAT_ATTRS))}",
        )
    path = await asyncio.to_thread(locate_report, run_id, summary, key)
    if path is None:
        available = await asyncio.to_thread(available_report_formats, run_id, summary)
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no {key} report for run {run_id}"
            + (f"; available: {available}" if available else ""),
        )
    return _file_response(path, download=download)




@router.get(
    "/runs/{run_id}/artifacts", response_model=ArtifactListResponse, tags=["artifacts"]
)
async def get_artifacts(run_id: str) -> ArtifactListResponse:
    """List every file this run wrote, with a URL to fetch each one."""
    manager = get_manager()
    if await asyncio.to_thread(manager.summary, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    entries = await asyncio.to_thread(list_artifacts, run_id)
    return ArtifactListResponse(
        run_id=run_id,
        count=len(entries),
        total_bytes=sum(entry.size_bytes for entry in entries),
        artifacts=[
            ArtifactEntry(
                relative_path=entry.relative_path,
                kind=entry.kind,
                size_bytes=entry.size_bytes,
                modified_at=entry.modified_at,
                url=f"/api/runs/{run_id}/artifacts/{entry.relative_path}",
            )
            for entry in entries
        ],
    )


@router.get("/runs/{run_id}/artifacts/{artifact_path:path}", tags=["artifacts"])
async def get_artifact(
    run_id: str, artifact_path: str, download: bool = Query(default=False)
) -> FileResponse:
    """Serve one artifact from inside the run directory."""
    try:
        path = await asyncio.to_thread(resolve_artifact, run_id, artifact_path)
    except ArtifactAccessError as exc:
        # Traversal attempts and missing files are both 404: distinguishing them
        # would confirm what exists outside the run directory.
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return _file_response(path, download=download)


def _file_response(path: Path, *, download: bool) -> FileResponse:
    media_type = (
        _MEDIA_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )
    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name if download else None,
        headers={
            "Content-Disposition": f'{disposition}; filename="{path.name}"',
            # Generated HTML and SVG are served from the API origin; stop the
            # browser from re-sniffing a mislabelled file into something active.
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/runs/{run_id}/experiments", tags=["runs"])
async def get_experiments(run_id: str) -> JSONResponse:
    """The stored leaderboard rows for a run, as plain JSON."""
    repository = get_repository()
    results = await asyncio.to_thread(repository.get_experiments, run_id)
    if not results and await asyncio.to_thread(get_manager().summary, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown run {run_id}")
    return JSONResponse(
        {
            "run_id": run_id,
            "count": len(results),
            "experiments": [result.model_dump(mode="json") for result in results],
        }
    )


@router.get("/runs/{run_id}/similar", tags=["insight"])
async def get_similar_runs(
    run_id: str, limit: int = Query(default=5, ge=1, le=25)
) -> JSONResponse:
    """Past runs on structurally similar datasets, from the dataset memory."""
    repository = get_repository()
    fingerprint = await asyncio.to_thread(repository.get_fingerprint_for_run, run_id)
    if fingerprint is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no dataset fingerprint recorded for run {run_id}",
        )
    matches = await asyncio.to_thread(repository.find_similar, fingerprint, limit)
    return JSONResponse(
        {
            "run_id": run_id,
            "count": len(matches),
            "similar_runs": [match.model_dump(mode="json") for match in matches],
        }
    )


@router.get("/projects", tags=["system"])
async def list_projects() -> JSONResponse:
    """Distinct project names that have at least one stored run."""
    repository = get_repository()
    projects = await asyncio.to_thread(repository.list_projects)
    return JSONResponse({"count": len(projects), "projects": projects})


__all__ = ["get_manager", "get_repository", "reset_repository_cache", "router"]
