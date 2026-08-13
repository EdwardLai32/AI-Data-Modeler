"""Run event bus.

Every meaningful action emits a :class:`RunEvent`. That log is the substrate for
three separate features, which is why it is a first-class object rather than
logging calls: the live UI stream, the audit trail in the report, and the
natural-language interface's evidence base ("why did accuracy drop?" is answered
by replaying events, not by guessing).

The bus is synchronous and thread-safe. Subscribers must not block — the API
layer bridges to async by pushing into a queue.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from .schemas import (
    AgentName,
    EventKind,
    Param,
    RunEvent,
    dict_to_params,
)

logger = logging.getLogger(__name__)

Subscriber = Callable[[RunEvent], None]


class EventBus:
    """Ordered, replayable event log for a single run."""

    def __init__(self, run_id: str, *, max_history: int = 10_000) -> None:
        self.run_id = run_id
        self._events: list[RunEvent] = []
        self._subscribers: list[Subscriber] = []
        self._lock = threading.RLock()
        self._sequence = 0
        self._max_history = max_history

    # -- subscription -----------------------------------------------------

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a listener. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    # -- emission ---------------------------------------------------------

    def emit(
        self,
        kind: EventKind,
        message: str = "",
        *,
        agent: AgentName | None = None,
        step_id: str | None = None,
        payload: dict[str, Any] | list[Param] | None = None,
        duration_seconds: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cache_read_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> RunEvent:
        if isinstance(payload, dict):
            params = dict_to_params(payload)
        else:
            params = list(payload or [])

        with self._lock:
            self._sequence += 1
            event = RunEvent(
                run_id=self.run_id,
                sequence=self._sequence,
                kind=kind,
                agent=agent,
                step_id=step_id,
                message=message,
                payload=params,
                duration_seconds=duration_seconds,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens,
                cost_usd=cost_usd,
            )
            self._events.append(event)
            if len(self._events) > self._max_history:
                # Keep the tail; the persisted copy in storage is authoritative.
                del self._events[: len(self._events) - self._max_history]
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # a broken listener must not fail the run
                logger.exception("event subscriber raised; continuing")

        return event

    # convenience wrappers -------------------------------------------------

    def log(self, message: str, **kwargs: Any) -> RunEvent:
        return self.emit(EventKind.LOG, message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> RunEvent:
        logger.warning("[%s] %s", self.run_id, message)
        return self.emit(EventKind.WARNING, message, **kwargs)

    def thinking(self, agent: AgentName, summary: str, **kwargs: Any) -> RunEvent:
        return self.emit(EventKind.AGENT_THINKING, summary, agent=agent, **kwargs)

    def decision(self, agent: AgentName, summary: str, **kwargs: Any) -> RunEvent:
        return self.emit(EventKind.AGENT_DECISION, summary, agent=agent, **kwargs)

    def metric(self, name: str, value: float, **kwargs: Any) -> RunEvent:
        return self.emit(
            EventKind.METRIC_RECORDED,
            f"{name}={value:.6g}",
            payload={"metric": name, "value": value},
            **kwargs,
        )

    def artifact(self, path: str, kind: str = "file", **kwargs: Any) -> RunEvent:
        return self.emit(
            EventKind.ARTIFACT_WRITTEN,
            f"wrote {kind}: {path}",
            payload={"path": path, "artifact_kind": kind},
            **kwargs,
        )

    # -- reading ----------------------------------------------------------

    @property
    def events(self) -> list[RunEvent]:
        with self._lock:
            return list(self._events)

    def since(self, sequence: int) -> list[RunEvent]:
        """Events after ``sequence``. Used by reconnecting stream clients."""
        with self._lock:
            return [e for e in self._events if e.sequence > sequence]

    def of_kind(self, *kinds: EventKind) -> list[RunEvent]:
        wanted = set(kinds)
        with self._lock:
            return [e for e in self._events if e.kind in wanted]

    def __iter__(self) -> Iterator[RunEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class NullEventBus(EventBus):
    """Discards events. Convenient for unit tests and one-off library calls."""

    def __init__(self) -> None:
        super().__init__(run_id="null", max_history=0)

    def emit(self, kind: EventKind, message: str = "", **kwargs: Any) -> RunEvent:  # type: ignore[override]
        self._sequence += 1
        return RunEvent(
            run_id=self.run_id, sequence=self._sequence, kind=kind, message=message
        )
