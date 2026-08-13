"""Run supervision: background execution, cancellation, approvals, Q&A.

This module is the bridge between HTTP (or the CLI's progress display) and the
orchestrator, and it exists because a run is a long-lived, stateful thing that
outlives any single request. It owns three jobs:

*   **Background execution.** A run executes on a daemon thread. The manager
    keeps a :class:`RunHandle` per run so a later request can read status,
    cancel, resolve an approval, or subscribe to the live event bus.
*   **Durable events.** A dedicated writer thread drains events into the
    repository in batches. Persisting synchronously inside ``bus.emit`` would put
    a database round-trip on the orchestrator's critical path; losing events
    would break the reconnect guarantee the SSE endpoint makes. A queue is the
    only way to have both.
*   **Cooperative control.** :class:`RunControl` carries a cancellation flag and
    an approval gate. It is registered per run *and* exposed on a thread-local,
    so an orchestrator can find its own control without the manager having to
    thread an extra argument through every call site.

Nothing here imports FastAPI. The CLI uses the same manager for ``amla run``, so
the supervision logic must not depend on a web framework being installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Settings, get_settings
from ..core.errors import ConfigurationError, RunCancelled
from ..core.events import EventBus
from ..core.schemas import (
    AgentName,
    ApprovalRequest,
    EventKind,
    QuestionAnswer,
    RunConfig,
    RunEvent,
    RunStatus,
    RunSummary,
    Severity,
    params_to_dict,
)
from ..core.state import RunState
from .schemas import FeatureStatus

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)

#: How often the writer thread may snapshot a live run's summary into storage.
SNAPSHOT_INTERVAL_SECONDS = 3.0

#: Events that make a mid-run snapshot worth taking.
_SNAPSHOT_TRIGGERS = frozenset(
    {
        EventKind.STEP_COMPLETED,
        EventKind.STEP_FAILED,
        EventKind.APPROVAL_REQUESTED,
        EventKind.APPROVAL_RESOLVED,
        EventKind.REPLAN_TRIGGERED,
        EventKind.RUN_COMPLETED,
        EventKind.RUN_FAILED,
        EventKind.RUN_CANCELLED,
    }
)

_thread_local = threading.local()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Cooperative control
# ---------------------------------------------------------------------------


@dataclass
class RunControl:
    """The handle an executing run uses to cooperate with its supervisor.

    An orchestrator should do three things with it: poll :attr:`cancelled` (or
    call :meth:`raise_if_cancelled`) between steps, call :meth:`attach_state` as
    soon as it builds a :class:`RunState`, and route destructive-step gating
    through :meth:`request_approval` / :meth:`wait_for_approval`.

    Every method is safe to ignore. A runner that never touches its control still
    executes correctly; it simply cannot be cancelled or approved mid-flight.
    """

    run_id: str
    bus: EventBus
    cancel_event: threading.Event = field(default_factory=threading.Event)
    state: RunState | None = field(default=None, repr=False)
    _approvals: dict[str, ApprovalRequest] = field(default_factory=dict, repr=False)
    _gates: dict[str, threading.Event] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- cancellation ------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        """Whether a caller has asked this run to stop."""
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        """Request cancellation and release anything blocked on an approval."""
        self.cancel_event.set()
        with self._lock:
            for gate in self._gates.values():
                gate.set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`RunCancelled` if cancellation has been requested."""
        if self.cancelled:
            raise RunCancelled(f"run {self.run_id} was cancelled by a caller")

    # -- state -------------------------------------------------------------

    def attach_state(self, state: RunState) -> None:
        """Publish the live :class:`RunState` so the API can read progress."""
        self.state = state

    # -- approvals ---------------------------------------------------------

    def request_approval(self, request: ApprovalRequest) -> ApprovalRequest:
        """Register a pending approval and announce it on the event bus."""
        with self._lock:
            self._approvals[request.request_id] = request
            self._gates.setdefault(request.request_id, threading.Event())
        self.bus.emit(
            EventKind.APPROVAL_REQUESTED,
            request.action_summary,
            agent=request.agent,
            step_id=request.step_id,
            payload={
                "request_id": request.request_id,
                "step_id": request.step_id,
                "severity": request.severity.value,
                "affected_columns": request.affected_columns,
                "affected_row_estimate": request.affected_row_estimate,
            },
        )
        return request

    def wait_for_approval(
        self, request_id: str, timeout: float | None = None
    ) -> ApprovalRequest | None:
        """Block until a human decides, cancellation arrives, or ``timeout``.

        A decision that arrived *before* this call is honoured immediately. The
        orchestrator announces an approval and only then reaches its wait, so a
        fast reviewer can legitimately answer in between; treating that as "still
        pending" would hang the run until the timeout.

        Args:
            request_id: The request to wait on.
            timeout: Seconds to wait; ``None`` waits indefinitely.

        Returns:
            The decided request, or ``None`` if it timed out or is unknown.
        """
        request = self.find_approval(request_id)
        if request is None:
            return None
        with self._lock:
            self._approvals.setdefault(request_id, request)
            gate = self._gates.setdefault(request_id, threading.Event())
        if request.decision != "pending":
            return request
        if not gate.wait(timeout):
            return None
        self.raise_if_cancelled()
        return request

    def resolve_approval(
        self,
        request_id: str,
        decision: str,
        *,
        note: str | None = None,
        decided_by: str | None = None,
    ) -> tuple[ApprovalRequest | None, bool]:
        """Record a decision and open the gate the run waits on.

        Looks the request up through :meth:`find_approval`, so a decision that
        arrives before the executor has registered its gate still lands on the
        object the orchestrator holds — and creates the gate, so the executor's
        later wait returns immediately instead of blocking on an answered request.

        Returns:
            ``(request, released)`` where ``released`` is True if this call is
            what opened the gate.
        """
        request = self.find_approval(request_id)
        if request is None:
            return None, False
        with self._lock:
            self._approvals.setdefault(request_id, request)
            gate = self._gates.setdefault(request_id, threading.Event())
        request.decision = "approved" if decision == "approved" else "rejected"
        request.decided_at = _utcnow()
        request.decided_by = decided_by
        request.note = note
        released = not gate.is_set()
        gate.set()
        self.bus.emit(
            EventKind.APPROVAL_RESOLVED,
            f"{request.action_summary} -> {request.decision}",
            agent=request.agent,
            step_id=request.step_id,
            payload={"request_id": request_id, "decision": request.decision},
        )
        return request, released

    def track(self, request: ApprovalRequest) -> None:
        """Register an approval the orchestrator created without our help."""
        with self._lock:
            self._approvals.setdefault(request.request_id, request)
            self._gates.setdefault(request.request_id, threading.Event())

    def approvals(self) -> list[ApprovalRequest]:
        """Every approval this control knows about, oldest first."""
        source: list[ApprovalRequest]
        if self.state is not None:
            source = list(self.state.approvals)
            known = {item.request_id for item in source}
            with self._lock:
                source.extend(
                    item
                    for request_id, item in self._approvals.items()
                    if request_id not in known
                )
        else:
            with self._lock:
                source = list(self._approvals.values())
        return sorted(source, key=lambda item: item.created_at)

    def find_approval(self, request_id: str) -> ApprovalRequest | None:
        """Look up one approval by id, preferring the live run state's copy."""
        if self.state is not None:
            for item in self.state.approvals:
                if item.request_id == request_id:
                    return item
        with self._lock:
            return self._approvals.get(request_id)


def set_active_control(control: RunControl | None) -> None:
    """Bind ``control`` to the calling thread. Called by the manager."""
    _thread_local.control = control


def active_control() -> RunControl | None:
    """The :class:`RunControl` for the run executing on this thread, if any.

    Lets an orchestrator honour cancellation and approvals without the manager
    injecting an argument into every function signature.
    """
    return getattr(_thread_local, "control", None)


# ---------------------------------------------------------------------------
# Executor resolution
# ---------------------------------------------------------------------------

RunExecutor = Callable[[RunConfig, EventBus, RunControl], RunSummary]

_executor_override: RunExecutor | None = None

#: Where to look for the orchestrator class, in order.
_ORCHESTRATOR_SOURCES = (
    ("automl_architect.orchestrator", "Orchestrator"),
    ("automl_architect.orchestrator.engine", "Orchestrator"),
    ("automl_architect.runner", "Orchestrator"),
)

#: Config-shaped module-level entry points, tried if no orchestrator class exists.
#: Deliberately excludes ``analyse``/``analyze``, which take a *source* rather
#: than a :class:`RunConfig` and would silently mis-bind.
_EXECUTOR_FUNCTIONS = ("run_from_api", "execute_run", "run_config")
_EXECUTOR_CLASSES = ("Runner", "AutoMLArchitect")
_EXECUTOR_METHODS = ("run", "execute", "start", "__call__")

#: How long a run thread will hold itself open waiting for a human to approve a
#: destructive step before giving up and leaving the run suspended.
APPROVAL_WAIT_SECONDS = 3600.0

#: Guard against an approval loop that never converges.
MAX_APPROVAL_ROUNDS = 32

#: Finished runs kept in memory. Beyond this the oldest are dropped and served
#: from storage instead, so a server running for weeks does not accumulate every
#: event log it has ever emitted.
MAX_RETAINED_HANDLES = 32

_CONFIG_NAMES = ("config", "run_config", "cfg")
_BUS_NAMES = ("bus", "event_bus", "events", "eventbus")
_CONTROL_NAMES = ("control", "run_control", "controller", "supervisor")
_CANCEL_NAMES = ("cancel_event", "cancel_token", "cancel")


def register_run_executor(executor: RunExecutor | None) -> None:
    """Override how runs are executed.

    The orchestrator module can call this at import time to declare itself
    explicitly instead of relying on :func:`resolve_run_executor`'s introspection.
    Tests use it to inject a fake.

    Args:
        executor: Callable taking ``(config, bus, control)`` and returning a
            :class:`RunSummary`. ``None`` restores the default resolution.
    """
    global _executor_override
    _executor_override = executor


def resolve_run_executor() -> RunExecutor:
    """Return the callable that executes a run.

    Prefers an explicitly registered executor, then introspects
    ``automl_architect.runner`` for a function or class that accepts a
    :class:`RunConfig`.
    """
    if _executor_override is not None:
        return _executor_override
    return _default_executor


def _find_orchestrator_class() -> type | None:
    """Locate the orchestrator class, or ``None`` if this install lacks one."""
    for module_name, attribute in _ORCHESTRATOR_SOURCES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        candidate = getattr(module, attribute, None)
        if inspect.isclass(candidate):
            return candidate
    return None


def _default_executor(
    config: RunConfig, bus: EventBus, control: RunControl
) -> RunSummary:
    """Execute a run through whatever orchestrator this install provides."""
    orchestrator_class = _find_orchestrator_class()
    if orchestrator_class is not None:
        return _drive_orchestrator(orchestrator_class, config, bus, control)

    try:
        module = importlib.import_module("automl_architect.runner")
    except ImportError as exc:  # pragma: no cover - depends on sibling module
        raise ConfigurationError(
            "no orchestrator is importable (tried "
            f"{[name for name, _ in _ORCHESTRATOR_SOURCES]}), so runs cannot be "
            f"executed: {exc}"
        ) from exc

    for name in _EXECUTOR_FUNCTIONS:
        candidate = getattr(module, name, None)
        if callable(candidate) and not inspect.isclass(candidate):
            result = _invoke(candidate, config, bus, control)
            return _coerce_summary(result, config, control)

    for name in _EXECUTOR_CLASSES:
        cls = getattr(module, name, None)
        if not inspect.isclass(cls):
            continue
        instance = _invoke(cls, config, bus, control)
        for method_name in _EXECUTOR_METHODS:
            method = getattr(instance, method_name, None)
            if callable(method):
                result = _invoke(method, config, bus, control, allow_positional=False)
                return _coerce_summary(result, config, control)
        raise ConfigurationError(
            f"'{name}' in automl_architect.runner exposes no run/execute method"
        )

    raise ConfigurationError(
        "automl_architect.runner exposes no recognised entry point; expected an "
        f"Orchestrator class, one of {_EXECUTOR_FUNCTIONS}, or a class in "
        f"{_EXECUTOR_CLASSES}. Call "
        "automl_architect.api.service.register_run_executor() to declare one."
    )


def _drive_orchestrator(
    orchestrator_class: type,
    config: RunConfig,
    bus: EventBus,
    control: RunControl,
) -> RunSummary:
    """Run an :class:`Orchestrator` and service its approval suspensions.

    The orchestrator's approval model is suspend-and-return: it stops, sets
    ``AWAITING_APPROVAL``, and hands back a summary for a caller to resume. An
    HTTP client cannot be expected to hold a request open across that, so the run
    thread stays alive here, blocks on the approval gate, and calls ``resume``
    when a decision lands. From the API's point of view the run simply keeps
    running while a human thinks.
    """
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(orchestrator_class).parameters
    except (TypeError, ValueError):  # pragma: no cover
        parameters = {}
    if "settings" in parameters:
        kwargs["settings"] = get_settings()
    if "bus" in parameters:
        kwargs["bus"] = bus
    if "repository" in parameters:
        kwargs["repository"] = _executor_repository()

    orchestrator = orchestrator_class(config, **kwargs)

    state = getattr(orchestrator, "state", None)
    if isinstance(state, RunState):
        control.attach_state(state)
    # Share one flag rather than mirroring two: a cancel posted to the API and a
    # cancel called on the orchestrator must be the same event.
    if isinstance(getattr(orchestrator, "cancel_event", None), threading.Event):
        if control.cancel_event.is_set():
            orchestrator.cancel_event.set()
        orchestrator.cancel_event = control.cancel_event

    summary = _coerce_summary(orchestrator.run(), config, control)
    resume = getattr(orchestrator, "resume", None)
    if not callable(resume):
        return summary

    for _ in range(MAX_APPROVAL_ROUNDS):
        if summary.status is not RunStatus.AWAITING_APPROVAL or control.cancelled:
            break
        pending = next(
            (item for item in summary.approvals if item.decision == "pending"), None
        )
        if pending is None:
            break
        control.track(pending)
        decided = control.wait_for_approval(pending.request_id, APPROVAL_WAIT_SECONDS)
        if control.cancelled:
            break
        if decided is None:
            bus.warn(
                f"no decision on approval {pending.request_id} within "
                f"{APPROVAL_WAIT_SECONDS:.0f}s; leaving the run suspended. "
                "POST the decision and start a new run to continue."
            )
            break
        summary = _coerce_summary(
            resume(pending.request_id, decided.decision == "approved", decided.note),
            config,
            control,
        )
    return summary


def _executor_repository() -> Any | None:
    """A repository for the orchestrator's own persistence, or ``None``."""
    try:
        from ..storage.repository import RunRepository

        return RunRepository(settings=get_settings())
    except Exception:  # persistence is optional; the run still executes
        logger.exception("could not open the repository for the orchestrator")
        return None


def _invoke(
    target: Callable[..., Any],
    config: RunConfig,
    bus: EventBus,
    control: RunControl,
    *,
    allow_positional: bool = True,
) -> Any:
    """Call ``target`` with whichever of config/bus/control it declares."""
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # builtins and C callables
        return target(config)

    parameters = signature.parameters
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    args: list[Any] = []

    config_name = next((name for name in _CONFIG_NAMES if name in parameters), None)
    if config_name:
        kwargs[config_name] = config
    elif allow_positional:
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != "self"
        ]
        if positional:
            args.append(config)

    for names, value in (
        (_BUS_NAMES, bus),
        (_CONTROL_NAMES, control),
        (_CANCEL_NAMES, control.cancel_event),
    ):
        name = next((candidate for candidate in names if candidate in parameters), None)
        if name:
            kwargs[name] = value
        elif accepts_var_kw and names is _BUS_NAMES:
            kwargs["bus"] = value
        elif accepts_var_kw and names is _CONTROL_NAMES:
            kwargs["control"] = value

    return target(*args, **kwargs)


def _coerce_summary(
    result: Any, config: RunConfig, control: RunControl
) -> RunSummary:
    """Normalise whatever the runner returned into a :class:`RunSummary`."""
    if isinstance(result, RunSummary):
        return result
    if isinstance(result, RunState):
        return result.to_summary()
    to_summary = getattr(result, "to_summary", None)
    if callable(to_summary):
        candidate = to_summary()
        if isinstance(candidate, RunSummary):
            return candidate
    if control.state is not None:
        return control.state.to_summary()
    return RunSummary(
        run_id=config.run_id,
        project=config.project,
        status=RunStatus.COMPLETED,
        config=config,
        warnings=[
            "the orchestrator returned no RunSummary; this record is a stub built "
            "from the run configuration"
        ],
    )


# ---------------------------------------------------------------------------
# Handles and the manager
# ---------------------------------------------------------------------------


@dataclass
class RunHandle:
    """Live bookkeeping for one in-process run."""

    run_id: str
    config: RunConfig
    bus: EventBus
    control: RunControl
    thread: threading.Thread | None = field(default=None, repr=False)
    summary: RunSummary | None = field(default=None, repr=False)
    error: str | None = None
    accepted_at: datetime = field(default_factory=_utcnow)
    _status: RunStatus = RunStatus.PENDING

    @property
    def status(self) -> RunStatus:
        """Best available status: the live state's, else our own bookkeeping."""
        if self.control.state is not None and self.is_running:
            return self.control.state.status
        if self.summary is not None:
            return self.summary.status
        if self.is_running and any(
            item.decision == "pending" for item in self.control.approvals()
        ):
            return RunStatus.AWAITING_APPROVAL
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is still alive."""
        return self.thread is not None and self.thread.is_alive()

    @property
    def finished(self) -> bool:
        """Whether the run has reached a terminal state."""
        return not self.is_running and (
            self.summary is not None or self._status in TERMINAL_STATUSES
        )

    def current_summary(self) -> RunSummary | None:
        """The finished summary, or a snapshot of the live state."""
        if self.summary is not None:
            return self.summary
        if self.control.state is not None:
            try:
                return self.control.state.to_summary()
            except Exception:  # a half-built state is not worth an error response
                logger.debug("could not snapshot live state for %s", self.run_id)
        return None


class RunManager:
    """Starts runs on background threads and supervises them.

    Args:
        repository: Storage for summaries and events. Built from settings if omitted.
        settings: Settings override.
    """

    def __init__(
        self,
        repository: Any | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if repository is None:
            from ..storage.repository import RunRepository

            repository = RunRepository(settings=self.settings)
        self.repository = repository
        self._handles: dict[str, RunHandle] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[str, RunEvent] | None] = queue.Queue()
        self._last_snapshot: dict[str, float] = {}
        self._writer = threading.Thread(
            target=self._drain_events, name="amla-event-writer", daemon=True
        )
        self._writer.start()

    # -- lifecycle ---------------------------------------------------------

    def start(self, config: RunConfig) -> RunHandle:
        """Queue a run and return immediately.

        Args:
            config: Fully resolved run configuration.

        Returns:
            The handle for the newly started run.

        Raises:
            ValueError: If a run with this id is already active.
        """
        with self._lock:
            existing = self._handles.get(config.run_id)
            if existing is not None and existing.is_running:
                raise ValueError(f"run {config.run_id} is already running")

            bus = EventBus(config.run_id)
            control = RunControl(run_id=config.run_id, bus=bus)
            handle = RunHandle(
                run_id=config.run_id, config=config, bus=bus, control=control
            )
            self._handles[config.run_id] = handle

        bus.subscribe(lambda event: self._queue.put((config.run_id, event)))
        self._save(
            RunSummary(
                run_id=config.run_id,
                project=config.project,
                status=RunStatus.PENDING,
                config=config,
                artifact_dir=str(self.settings.run_dir(config.run_id)),
            )
        )

        thread = threading.Thread(
            target=self._execute,
            args=(handle,),
            name=f"amla-run-{config.run_id}",
            daemon=True,
        )
        handle.thread = thread
        thread.start()
        return handle

    def _execute(self, handle: RunHandle) -> None:
        """Worker body: run the orchestrator, then persist whatever came back."""
        set_active_control(handle.control)
        handle._status = RunStatus.RUNNING
        summary: RunSummary | None = None
        try:
            executor = resolve_run_executor()
            summary = executor(handle.config, handle.bus, handle.control)
            handle._status = summary.status
        except RunCancelled as exc:
            handle._status = RunStatus.CANCELLED
            handle.error = str(exc)
            summary = self._fallback_summary(handle, RunStatus.CANCELLED, str(exc))
            handle.bus.emit(EventKind.RUN_CANCELLED, str(exc))
        except Exception as exc:  # every failure must land as a stored FAILED run
            logger.exception("run %s failed", handle.run_id)
            handle._status = RunStatus.FAILED
            handle.error = str(exc)
            summary = self._fallback_summary(handle, RunStatus.FAILED, str(exc))
            handle.bus.emit(EventKind.RUN_FAILED, f"{type(exc).__name__}: {exc}")
        finally:
            set_active_control(None)
            if summary is not None:
                handle.summary = summary
                self._save(summary)
                self._remember(summary)
            self._flush()
            _release_dataframes(handle)
            self._evict_finished()

    def _evict_finished(self) -> None:
        """Forget the oldest finished runs so a long-lived server stays bounded.

        An evicted run is not lost — every summary, event, and artifact is in
        storage, which is precisely why the handle can be dropped. What goes is
        the in-memory event history and the live-state reference.
        """
        with self._lock:
            finished = [
                (handle.accepted_at, run_id)
                for run_id, handle in self._handles.items()
                if handle.finished
            ]
            if len(finished) <= MAX_RETAINED_HANDLES:
                return
            finished.sort()
            for _, run_id in finished[: len(finished) - MAX_RETAINED_HANDLES]:
                self._handles.pop(run_id, None)
                self._last_snapshot.pop(run_id, None)

    def _fallback_summary(
        self, handle: RunHandle, status: RunStatus, error: str
    ) -> RunSummary:
        """Build a summary when the run died before returning one."""
        state = handle.control.state
        if state is not None:
            try:
                state.mark_finished(status, error)
                return state.to_summary()
            except Exception:
                logger.debug("could not derive summary from state for %s", handle.run_id)
        return RunSummary(
            run_id=handle.run_id,
            project=handle.config.project,
            status=status,
            config=handle.config,
            started_at=handle.accepted_at,
            finished_at=_utcnow(),
            error=error,
            artifact_dir=str(self.settings.run_dir(handle.run_id)),
        )

    # -- reads -------------------------------------------------------------

    def get(self, run_id: str) -> RunHandle | None:
        """The live handle for a run, or ``None`` if it is not in this process."""
        with self._lock:
            return self._handles.get(run_id)

    def active_ids(self) -> list[str]:
        """Ids of runs whose worker thread is still alive."""
        with self._lock:
            return [
                run_id for run_id, handle in self._handles.items() if handle.is_running
            ]

    def summary(self, run_id: str) -> RunSummary | None:
        """Freshest summary available: live snapshot first, then storage."""
        handle = self.get(run_id)
        if handle is not None:
            snapshot = handle.current_summary()
            if snapshot is not None:
                return snapshot
        return self.repository.get_run(run_id)

    def status(self, run_id: str) -> RunStatus | None:
        """Status of a run from either source."""
        handle = self.get(run_id)
        if handle is not None:
            return handle.status
        return self.repository.run_status(run_id)

    def wait(self, run_id: str, timeout: float | None = None) -> RunSummary | None:
        """Block until a run finishes. Used by the CLI's foreground mode."""
        handle = self.get(run_id)
        if handle is None:
            return self.repository.get_run(run_id)
        if handle.thread is not None:
            handle.thread.join(timeout)
        self._flush()
        return handle.summary or self.repository.get_run(run_id)

    # -- control -----------------------------------------------------------

    def cancel(self, run_id: str) -> tuple[bool, str]:
        """Request cancellation.

        Returns:
            ``(requested, detail)``. ``requested`` is False when the run is
            already finished or unknown to this process.
        """
        handle = self.get(run_id)
        if handle is None:
            stored = self.repository.run_status(run_id)
            if stored is None:
                return False, "unknown run"
            if stored in TERMINAL_STATUSES:
                return False, f"run already {stored.value}"
            return False, (
                "the run is not executing in this process; cancellation must be "
                "requested from the process that started it"
            )
        if handle.finished:
            return False, f"run already {handle.status.value}"

        handle.control.cancel()
        handle.bus.emit(
            EventKind.LOG, "cancellation requested; the run will stop at the next checkpoint"
        )
        if not handle.is_running:
            handle._status = RunStatus.CANCELLED
            summary = self._fallback_summary(handle, RunStatus.CANCELLED, "cancelled")
            handle.summary = summary
            self._save(summary)
        return True, "cancellation requested"

    def pending_approvals(self, run_id: str) -> list[ApprovalRequest]:
        """Approvals still awaiting a human, from the live run or from storage."""
        return [item for item in self.approvals(run_id) if item.decision == "pending"]

    def approvals(self, run_id: str) -> list[ApprovalRequest]:
        """Every approval for a run, merging live and stored copies."""
        handle = self.get(run_id)
        live = handle.control.approvals() if handle else []
        known = {item.request_id for item in live}
        stored = [
            item
            for item in self.repository.get_approvals(run_id)
            if item.request_id not in known
        ]
        return sorted(live + stored, key=lambda item: item.created_at)

    def resolve_approval(
        self,
        run_id: str,
        request_id: str,
        decision: str,
        *,
        note: str | None = None,
        decided_by: str | None = None,
    ) -> tuple[ApprovalRequest | None, bool]:
        """Approve or reject a pending request and release the run.

        Args:
            run_id: Owning run.
            request_id: The approval to decide.
            decision: ``"approved"`` or ``"rejected"``.
            note: Free-text reason, stored with the decision.
            decided_by: Who decided.

        Returns:
            ``(request, resumed)``; ``resumed`` is True when a blocked run thread
            was actually released by this decision.
        """
        handle = self.get(run_id)
        if handle is not None:
            request, released = handle.control.resolve_approval(
                request_id, decision, note=note, decided_by=decided_by
            )
            if request is not None:
                self.repository.save_approval(run_id, request)
                snapshot = handle.current_summary()
                if snapshot is not None:
                    self._save(snapshot)
                return request, released

        stored = next(
            (
                item
                for item in self.repository.get_approvals(run_id)
                if item.request_id == request_id
            ),
            None,
        )
        if stored is None:
            return None, False
        stored.decision = "approved" if decision == "approved" else "rejected"
        stored.decided_at = _utcnow()
        stored.decided_by = decided_by
        stored.note = note
        self.repository.save_approval(run_id, stored)
        return stored, False

    def request_approval(
        self,
        run_id: str,
        *,
        step_id: str,
        agent: AgentName,
        action_summary: str,
        details: list[str] | None = None,
        affected_columns: list[str] | None = None,
        affected_row_estimate: int = 0,
        severity: Severity = Severity.MEDIUM,
    ) -> ApprovalRequest | None:
        """Create an approval request for a live run. Returns ``None`` if unknown."""
        handle = self.get(run_id)
        if handle is None:
            return None
        request = ApprovalRequest(
            step_id=step_id,
            agent=agent,
            action_summary=action_summary,
            details=details or [],
            affected_columns=affected_columns or [],
            affected_row_estimate=affected_row_estimate,
            severity=severity,
        )
        handle.control.request_approval(request)
        self.repository.save_approval(run_id, request)
        return request

    # -- event persistence -------------------------------------------------

    def _drain_events(self) -> None:
        """Writer loop: batch events into storage and snapshot live summaries."""
        batch: dict[str, list[RunEvent]] = {}
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                self._write_batch(batch)
                batch = {}
                continue
            if item is None:
                self._write_batch(batch)
                return
            run_id, event = item
            batch.setdefault(run_id, []).append(event)
            self._observe(run_id, event)
            if event.kind in _SNAPSHOT_TRIGGERS or sum(len(v) for v in batch.values()) >= 100:
                self._write_batch(batch)
                batch = {}
                if event.kind in _SNAPSHOT_TRIGGERS:
                    self._snapshot(run_id)

    def _write_batch(self, batch: dict[str, list[RunEvent]]) -> None:
        for run_id, events in batch.items():
            if not events:
                continue
            try:
                self.repository.append_events(run_id, events)
            except Exception:  # storage must never take the run down with it
                logger.exception("could not persist %d events for %s", len(events), run_id)

    def _observe(self, run_id: str, event: RunEvent) -> None:
        """Learn what we can from an event the orchestrator emitted itself.

        An orchestrator that announces an approval without going through
        :meth:`RunControl.request_approval` would otherwise leave the API with
        nothing to list, so the request is reconstructed from the payload.
        """
        if event.kind is not EventKind.APPROVAL_REQUESTED:
            return
        handle = self.get(run_id)
        if handle is None:
            return
        payload = params_to_dict(event.payload)
        request_id = payload.get("request_id") or payload.get("approval_id")
        if not isinstance(request_id, str) or handle.control.find_approval(request_id):
            return
        columns = payload.get("affected_columns")
        try:
            severity = Severity(str(payload.get("severity", "medium")))
        except ValueError:
            severity = Severity.MEDIUM
        request = ApprovalRequest(
            request_id=request_id,
            step_id=str(payload.get("step_id") or event.step_id or ""),
            agent=event.agent or AgentName.PLANNER,
            action_summary=event.message or "approval required",
            affected_columns=[str(c) for c in columns] if isinstance(columns, list) else [],
            affected_row_estimate=int(payload.get("affected_row_estimate") or 0),
            severity=severity,
        )
        handle.control.track(request)
        try:
            self.repository.save_approval(run_id, request)
        except Exception:
            logger.debug("could not persist reconstructed approval %s", request_id)

    def _snapshot(self, run_id: str) -> None:
        """Persist a mid-run summary, rate-limited per run."""
        now = time.monotonic()
        if now - self._last_snapshot.get(run_id, 0.0) < SNAPSHOT_INTERVAL_SECONDS:
            return
        handle = self.get(run_id)
        if handle is None or handle.summary is not None:
            return
        snapshot = handle.current_summary()
        if snapshot is None:
            return
        self._last_snapshot[run_id] = now
        self._save(snapshot)

    def _flush(self, timeout: float = 5.0) -> None:
        """Wait for the writer thread to drain, so a caller can read events back."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        # One more beat for the batch the writer is holding.
        time.sleep(0.3)

    def _save(self, summary: RunSummary) -> None:
        try:
            self.repository.save_run(summary)
        except Exception:
            logger.exception("could not persist run %s", summary.run_id)

    def _remember(self, summary: RunSummary) -> None:
        """Record a dataset fingerprint so future runs have precedent.

        The orchestrator fingerprints its own completed runs, so this is a
        backstop for executors that do not. Recording twice would put two rows
        for one run into the similarity search and double-count its vote.
        """
        if summary.status is not RunStatus.COMPLETED or summary.profile is None:
            return
        try:
            if self.repository.get_fingerprint_for_run(summary.run_id) is not None:
                return
            from ..storage.memory import DatasetMemory

            DatasetMemory(self.repository, settings=self.settings).remember(summary)
        except Exception:
            logger.exception("could not record dataset fingerprint for %s", summary.run_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the writer thread after flushing. Safe to call twice."""
        self._flush(timeout)
        self._queue.put(None)
        self._writer.join(timeout)


def _release_dataframes(handle: RunHandle) -> None:
    """Drop the run's in-memory dataframes once it has finished.

    A completed run's frames can be gigabytes, and nothing downstream reads them
    — the summary, the artifacts, and the fitted model on disk carry everything
    the API serves. The fitted estimator is deliberately kept, because an
    in-process caller may still want to predict with it.
    """
    state = handle.control.state
    if state is None:
        return
    for attribute in ("raw_df", "working_df", "feature_frame"):
        try:
            setattr(state, attribute, None)
        except Exception:  # a frozen or exotic state is not worth failing over
            logger.debug("could not release %s on run %s", attribute, handle.run_id)
    try:
        state.splits = type(state.splits)()
    except Exception:
        logger.debug("could not reset splits on run %s", handle.run_id)


_manager: RunManager | None = None
_manager_lock = threading.Lock()


def get_run_manager(settings: Settings | None = None) -> RunManager:
    """Process-wide :class:`RunManager`, created on first use."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = RunManager(settings=settings)
        return _manager


def reset_run_manager() -> None:
    """Drop the shared manager and the handlers' repository cache.

    One call is enough to fully detach from the current settings: a test that
    reset only the manager would leave the route handlers reading from the
    previous database.
    """
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
        _manager = None
    try:
        from .routes import reset_repository_cache  # deferred: routes imports us
    except ImportError:  # FastAPI absent; nothing was cached
        return
    reset_repository_cache()


# ---------------------------------------------------------------------------
# Natural-language questions
# ---------------------------------------------------------------------------

_QA_INSTRUCTIONS = """\
You are the natural-language interface to a completed AI Data Modeler run. A \
user asks a question; you answer it strictly from the run record supplied below.

Rules:
1. Ground every claim in the record. Quote the metric, decision, or event that \
supports it in `evidence`. If the record does not contain the answer, say so \
plainly rather than inferring.
2. Never invent a number. If a figure is absent, name what is missing.
3. Match the register of the question: a business question gets a business \
answer, a methodological question gets specifics.
4. Set `confidence` honestly. "low" is correct when the record is thin.
"""


def render_run_digest(summary: RunSummary, *, max_chars: int = 20_000) -> str:
    """Render a run into the text an answering agent reasons over.

    Deliberately dense and factual: the point is to make every number the
    question could need present, so the answer never has to guess.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# RUN {summary.run_id} ({summary.status.value})")
    add(f"project: {summary.project}")
    if summary.started_at:
        add(f"started: {summary.started_at.isoformat()}")
    add(f"duration: {summary.duration_seconds:.1f}s")
    add(
        f"cost: ${summary.usage.cost_usd:.4f} across {summary.usage.llm_calls} model calls"
    )
    if summary.error:
        add(f"error: {summary.error}")
    add("")

    source = summary.config.source
    add(f"## SOURCE\n{source.kind.value}: {source.uri or '(inline)'}")
    if summary.ingestion:
        add(
            f"loaded {summary.ingestion.n_rows:,} rows x {summary.ingestion.n_columns} "
            f"columns in {summary.ingestion.load_seconds:.2f}s"
        )
    add("")

    if summary.profile:
        profile = summary.profile
        add("## DATA PROFILE")
        add(
            f"{profile.n_rows:,} rows, {profile.n_columns} columns, "
            f"{profile.missing_cell_fraction:.2%} cells missing, "
            f"{profile.n_duplicate_rows:,} duplicate rows"
        )
        if profile.target:
            target = profile.target
            add(
                f"target `{target.name}` kind={target.kind.value} "
                f"classes={target.n_classes} imbalance={target.imbalance_ratio}"
            )
        if profile.leakage_findings:
            add(
                "leakage candidates: "
                + ", ".join(
                    f"{f.column} ({f.score:.3f}, {f.severity.value})"
                    for f in profile.leakage_findings[:8]
                )
            )
        if profile.quality_issues:
            add(
                "quality issues: "
                + "; ".join(
                    f"[{i.severity.value}] {i.code}: {i.detail}"
                    for i in profile.quality_issues[:8]
                )
            )
        add("")

    if summary.understanding:
        add("## DATASET UNDERSTANDING")
        add(f"headline: {summary.understanding.headline}")
        add(f"domain: {summary.understanding.likely_domain}; grain: {summary.understanding.grain}")
        for finding in summary.understanding.key_findings[:8]:
            add(f"- {finding}")
        add("")

    if summary.problem:
        problem = summary.problem
        add("## PROBLEM")
        add(
            f"task={problem.task_type.value} target={problem.target_column} "
            f"metric={problem.primary_metric} confidence={problem.confidence}"
        )
        add(f"rationale: {problem.rationale}")
        add(f"objective: {problem.business_objective}")
        add("")

    if summary.plan:
        add("## PLAN")
        add(summary.plan.summary)
        for step in summary.plan.ordered():
            add(f"{step.order}. [{step.agent.value}] {step.title} — {step.objective}")
        add("")

    if summary.cleaning:
        add("## CLEANING DECISIONS")
        for decision in summary.cleaning.decisions[:30]:
            add(
                f"- {decision.action.value} on {decision.columns or 'table'}"
                f"{f' via {decision.strategy.value}' if decision.strategy else ''}: "
                f"{decision.rationale}"
            )
        add("")

    if summary.features:
        add("## FEATURE ENGINEERING")
        for decision in summary.features.decisions[:30]:
            add(f"- {decision.op.value} on {decision.input_columns}: {decision.rationale}")
        add("")

    if summary.model_selection:
        add("## MODEL SELECTION")
        add(summary.model_selection.reasoning)
        for candidate in summary.model_selection.candidates:
            add(
                f"- #{candidate.rank} {candidate.family.value} "
                f"({candidate.suitability}): {candidate.rationale}"
            )
        add("")

    if summary.experiments:
        log = summary.experiments
        add(f"## LEADERBOARD (metric={log.primary_metric}, higher_is_better={log.higher_is_better})")
        for result in sorted(
            log.results,
            key=lambda r: (r.primary_score is None, r.primary_score or 0.0),
            reverse=True,
        ):
            flag = " <-- BEST" if result.experiment_id == log.best_experiment_id else ""
            if result.failed:
                add(f"- {result.family.value}: FAILED ({result.error}){flag}")
                continue
            metrics = ", ".join(f"{m.name}={m.value:.6g}" for m in result.metrics)
            add(
                f"- {result.family.value}{' [tuned]' if result.tuned else ''}: "
                f"{result.primary_metric}={result.primary_score} "
                f"({metrics}) trained in {result.train_seconds:.2f}s{flag}"
            )
        add("")

    if summary.tuning:
        tuning = summary.tuning
        add("## TUNING")
        if tuning.ran:
            add(
                f"method={tuning.method.value} trials={tuning.n_trials_completed} "
                f"best={tuning.best_score} baseline={tuning.baseline_score} "
                f"improvement={tuning.improvement}"
            )
        else:
            add(f"skipped: {tuning.skipped_reason or 'not attempted'}")
        add("")

    if summary.explainability:
        add("## EXPLAINABILITY")
        for attribution in summary.explainability.global_attributions[:15]:
            add(f"- {attribution.feature}: {attribution.importance:.4f} ({attribution.direction})")
        for sentence in summary.explainability.plain_language_explanations[:8]:
            add(f"- {sentence}")
        add("")

    if summary.evaluation:
        evaluation = summary.evaluation
        add("## EVALUATION")
        add(
            f"grade={evaluation.overall_grade} acceptable={evaluation.acceptable} "
            f"action={evaluation.recommended_action} drift_risk={evaluation.drift_risk}"
        )
        add(f"rationale: {evaluation.verdict_rationale}")
        bias = evaluation.bias_variance
        add(
            f"fit: train={bias.train_score} valid={bias.validation_score} "
            f"test={bias.test_score} gap={bias.gap} verdict={bias.verdict}"
        )
        for interval in evaluation.confidence_intervals:
            add(
                f"CI {interval.metric}: {interval.point_estimate:.4g} "
                f"[{interval.lower:.4g}, {interval.upper:.4g}] at {interval.level:.0%}"
            )
        for weakness in evaluation.weaknesses[:8]:
            add(f"weakness: {weakness}")
        add("")

    if summary.insights:
        add("## BUSINESS INSIGHTS")
        add(summary.insights.executive_summary)
        for insight in summary.insights.insights[:10]:
            add(f"- {insight.headline} ({insight.confidence}): {insight.detail}")
        add("")

    if summary.report:
        add("## REPORT")
        add(summary.report.executive_summary)
        deployment = summary.report.deployment
        add(f"deployment: {deployment.pattern.value} — {deployment.rationale}")
        add("")

    if summary.steps:
        add("## STEP TRACE")
        for step in summary.steps:
            add(
                f"- {step.step_id} [{step.status.value}] "
                f"{step.duration_seconds:.1f}s{f' error={step.error}' if step.error else ''}"
            )
        add("")

    if summary.warnings:
        add("## WARNINGS")
        for warning in summary.warnings[:25]:
            add(f"- {warning}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[digest truncated]"
    return text


def answer_question(
    summary: RunSummary,
    question: str,
    *,
    settings: Settings | None = None,
) -> QuestionAnswer:
    """Answer a natural-language question about a finished run.

    Delegates to a dedicated Q&A agent if the agents package ships one; otherwise
    asks the model directly with the run digest as its only evidence.

    Args:
        summary: The run to answer about.
        question: The user's question.
        settings: Settings override.

    Returns:
        A grounded :class:`QuestionAnswer`.

    Raises:
        Exception: Propagates LLM failures (missing credentials, transport) so the
            caller can surface them; there is no useful offline answer.
    """
    delegated = _delegate_question(summary, question)
    if delegated is not None:
        return delegated

    from ..core.llm import get_llm_client

    client = get_llm_client(settings)
    digest = render_run_digest(summary)
    result = client.structured(
        output_model=QuestionAnswer,
        user=(
            f"{digest}\n\n---\n\nQuestion from the user:\n{question}\n\n"
            "Answer only from the record above."
        ),
        agent_instructions=_QA_INSTRUCTIONS,
        effort="high",
        max_tokens=8000,
    )
    return result.value


def _delegate_question(summary: RunSummary, question: str) -> QuestionAnswer | None:
    """Use the package's own Q&A agent if this install has one.

    Preferred over the local fallback because the real agent reasons over the
    frozen run context and the persisted event log, which makes its answers
    citable. Returns ``None`` only when no implementation is reachable — a
    delegated implementation that *fails* is allowed to raise, so the caller
    reports the real cause instead of quietly answering with less evidence.
    """
    facade = _question_facade()
    if facade is not None:
        value = facade.ask(summary.run_id, question)
        if isinstance(value, QuestionAnswer):
            return value

    for module_name in (
        "automl_architect.agents.question",
        "automl_architect.agents.qa",
        "automl_architect.agents",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attribute in ("answer_question", "ask", "answer"):
            candidate = getattr(module, attribute, None)
            if not callable(candidate) or inspect.isclass(candidate):
                continue
            try:
                value = candidate(summary, question)
            except TypeError:  # not the signature we guessed; try the next one
                continue
            if isinstance(value, QuestionAnswer):
                return value
    return None


def _question_facade() -> Any | None:
    """The library facade (``AutoMLArchitect``), which owns the QA wiring."""
    try:
        module = importlib.import_module("automl_architect.runner")
    except ImportError:
        return None
    facade_class = getattr(module, "AutoMLArchitect", None)
    if not inspect.isclass(facade_class):
        return None
    try:
        facade = facade_class(settings=get_settings())
    except Exception:
        logger.debug("could not construct the library facade for Q&A", exc_info=True)
        return None
    return facade if callable(getattr(facade, "ask", None)) else None


# ---------------------------------------------------------------------------
# Optional-feature detection (shared by /api/health and `amla doctor`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One optional capability, the modules it needs, and how to install them."""

    name: str
    modules: tuple[str, ...]
    install: str
    detail: str
    mode: str = "all"


#: Import name -> pip name, where they differ.
_PIP_NAMES = {
    "pptx": "python-pptx",
    "sklearn": "scikit-learn",
    "azure.storage.blob": "azure-storage-blob",
    "google.cloud.storage": "google-cloud-storage",
    "snowflake.sqlalchemy": "snowflake-sqlalchemy",
    "sse_starlette": "sse-starlette",
    "psycopg": "psycopg[binary]",
    "pymysql": "PyMySQL",
}

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "boosted trees",
        ("xgboost", "lightgbm"),
        'pip install "automl-architect[boost]"',
        "XGBoost/LightGBM candidates in the model zoo",
        mode="any",
    ),
    FeatureSpec(
        "catboost",
        ("catboost",),
        "pip install catboost",
        "CatBoost candidate; optional, the zoo works without it",
    ),
    FeatureSpec(
        "hyperparameter tuning",
        ("optuna",),
        'pip install "automl-architect[tuning]"',
        "Optuna TPE and Bayesian search",
    ),
    FeatureSpec(
        "SHAP explanations",
        ("shap",),
        'pip install "automl-architect[explain]"',
        "SHAP attributions; permutation importance is used as a fallback",
    ),
    FeatureSpec(
        "chart PNG export",
        ("kaleido",),
        'pip install "automl-architect[charts]"',
        "static images for PDF/PPTX; HTML charts work regardless",
    ),
    FeatureSpec(
        "PDF reports",
        ("reportlab",),
        'pip install "automl-architect[reports]"',
        "--format pdf",
    ),
    FeatureSpec(
        "PowerPoint reports",
        ("pptx",),
        'pip install "automl-architect[reports]"',
        "--format pptx",
    ),
    FeatureSpec(
        "HTML reports",
        ("markdown", "jinja2"),
        'pip install "automl-architect[reports]"',
        "--format html",
    ),
    FeatureSpec(
        "interactive charts",
        ("plotly",),
        "pip install plotly",
        "Plotly figures embedded in the HTML report",
    ),
    FeatureSpec(
        "HTTP API",
        ("fastapi", "uvicorn", "sse_starlette"),
        'pip install "automl-architect[api]"',
        "amla serve",
    ),
    FeatureSpec(
        "cloud object storage",
        ("boto3", "azure.storage.blob", "google.cloud.storage"),
        'pip install "automl-architect[cloud]"',
        "s3://, az://, and gs:// sources",
        mode="any",
    ),
    FeatureSpec(
        "SQL connectors",
        ("psycopg", "pymysql"),
        'pip install "automl-architect[sql]"',
        "postgres:// and mysql:// sources",
        mode="any",
    ),
    FeatureSpec(
        "Snowflake",
        ("snowflake.sqlalchemy",),
        'pip install "automl-architect[snowflake]"',
        "snowflake:// sources",
    ),
    FeatureSpec(
        "Kaggle datasets",
        ("kaggle",),
        'pip install "automl-architect[kaggle]"',
        "kaggle: dataset slugs",
    ),
    FeatureSpec(
        "MLflow tracking",
        ("mlflow",),
        'pip install "automl-architect[mlflow]"',
        "mirrors experiments into an MLflow tracking server",
    ),
)


def module_available(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it.

    Uses :func:`importlib.util.find_spec`, which is cheap and side-effect free
    for the common case. A dotted name whose parent is absent raises rather than
    returning ``None``, so both are treated as "missing".
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def pip_name(module: str) -> str:
    """The pip package name for an import name."""
    return _PIP_NAMES.get(module, module)


def detect_features() -> list[FeatureStatus]:
    """Report every optional capability as available or missing.

    Each entry carries the exact pip command that fixes it, which is the point:
    a diagnostic that says "missing" without saying "install this" makes the user
    guess at the mapping from feature to package.
    """
    out: list[FeatureStatus] = []
    for spec in FEATURE_SPECS:
        present = [module for module in spec.modules if module_available(module)]
        missing = [module for module in spec.modules if module not in present]
        available = bool(present) if spec.mode == "any" else not missing
        if available and missing and spec.mode == "any":
            note = f"{spec.detail} (using {', '.join(present)}; missing {', '.join(pip_name(m) for m in missing)})"
        elif available:
            note = spec.detail
        else:
            note = f"{spec.detail} — missing {', '.join(pip_name(m) for m in missing)}"
        out.append(
            FeatureStatus(
                name=spec.name,
                available=available,
                packages=[pip_name(module) for module in spec.modules],
                install_hint=spec.install,
                detail=note,
            )
        )
    return out


def python_version() -> str:
    """The running interpreter's version, e.g. ``3.11.9``."""
    return sys.version.split()[0]


# ---------------------------------------------------------------------------
# Locating rendered reports
# ---------------------------------------------------------------------------

#: Report format -> the attribute on ``ReportBundle`` that records its path.
REPORT_FORMAT_ATTRS: dict[str, str] = {
    "markdown": "markdown_path",
    "md": "markdown_path",
    "html": "html_path",
    "pdf": "pdf_path",
    "pptx": "pptx_path",
    "json": "json_path",
}


def report_candidates(run_id: str, summary: RunSummary, key: str) -> list[Path]:
    """Places a report of this format might live, best guess first.

    The path recorded on the bundle is tried both as written and as run-relative,
    because the workspace root may itself be a relative path: a writer that
    stored ``workspace/runs/<id>/reports/report.html`` and one that stored
    ``reports/report.html`` are both plausible, and neither should 404.
    """
    from ..storage.artifacts import REPORTS_SUBDIR, run_dir

    root = run_dir(run_id)
    suffix = ".md" if key in ("markdown", "md") else f".{key}"
    out: list[Path] = []

    bundle = summary.report_bundle
    attribute = REPORT_FORMAT_ATTRS.get(key)
    recorded = getattr(bundle, attribute, None) if (bundle and attribute) else None
    if recorded:
        raw = Path(str(recorded))
        out.append(raw)
        if not raw.is_absolute():
            out.append(root / raw)

    out.append(root / REPORTS_SUBDIR / f"report{suffix}")
    out.append(root / "report" / f"report{suffix}")
    out.append(root / f"report{suffix}")
    # Last resort: any report-ish file of the right type the writer left behind.
    if root.is_dir():
        out.extend(
            path
            for path in sorted(root.rglob(f"*{suffix}"))
            if path.is_file() and "report" in path.name.lower()
        )
    return out


def locate_report(run_id: str, summary: RunSummary, key: str) -> Path | None:
    """Resolve a report file inside the run directory, or ``None`` if absent.

    Even a path the report writer recorded itself is re-validated against the run
    directory: a writer bug must not turn into an arbitrary file read when the
    API serves the result.
    """
    from ..storage.artifacts import ArtifactAccessError, resolve_artifact, run_dir

    root = run_dir(run_id).resolve()
    for path in report_candidates(run_id, summary, key):
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(root).as_posix()
            return resolve_artifact(run_id, relative)
        except (ArtifactAccessError, ValueError, OSError):
            logger.warning("report path %s is outside run %s; refusing", path, run_id)
    return None


def available_report_formats(run_id: str, summary: RunSummary) -> list[str]:
    """Which report formats actually exist on disk for a run."""
    return [
        key
        for key in ("markdown", "html", "pdf", "pptx", "json")
        if locate_report(run_id, summary, key) is not None
    ]


__all__ = [
    "FEATURE_SPECS",
    "REPORT_FORMAT_ATTRS",
    "FeatureSpec",
    "RunControl",
    "RunHandle",
    "RunManager",
    "TERMINAL_STATUSES",
    "active_control",
    "answer_question",
    "available_report_formats",
    "detect_features",
    "locate_report",
    "get_run_manager",
    "module_available",
    "pip_name",
    "python_version",
    "register_run_executor",
    "render_run_digest",
    "reset_run_manager",
    "resolve_run_executor",
    "set_active_control",
]
