"""In-memory dataframe source, for library use and for tests.

``DataSource.options`` is ``list[Param]`` — string keys and string values — so a
dataframe cannot be embedded in it directly. Instead the frame is parked in a
process-local registry and the source carries its token. That keeps the source
JSON-serialisable (it still round-trips into a ``RunSummary``) while letting a
Python caller skip the filesystem entirely::

    from automl_architect.ingestion.dataframe import make_dataframe_source

    source = make_dataframe_source(df, name="churn_extract")
    frame, result = load_source(source)

Tokens are only meaningful inside the process that created them; a summary
replayed elsewhere will report a clear "frame is no longer registered" error
rather than silently loading different data.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

import pandas as pd

from ..core.errors import ConfigurationError
from ..core.schemas import DataSource, Param, SourceKind
from .base import Connector, LoadOutcome, cap_rows, register

#: Option key holding the registry token.
FRAME_KEY_OPTION = "frame_key"

_FRAMES: dict[str, Any] = {}
_LOCK = threading.Lock()


def register_frame(frame: Any, key: str | None = None) -> str:
    """Park a dataframe in the process-local registry.

    Args:
        frame: The dataframe (or anything ``pd.DataFrame`` accepts).
        key: Explicit token. Generated when omitted.

    Returns:
        The token to put on the :class:`DataSource`.
    """
    token = key or f"frame_{uuid4().hex[:12]}"
    with _LOCK:
        _FRAMES[token] = frame
    return token


def get_frame(key: str) -> Any | None:
    """Return a registered frame, or ``None`` when the token is unknown."""
    with _LOCK:
        return _FRAMES.get(key)


def release_frame(key: str) -> None:
    """Forget one registered frame. Safe to call with an unknown token."""
    with _LOCK:
        _FRAMES.pop(key, None)


def clear_frames() -> None:
    """Forget every registered frame. Intended for test teardown."""
    with _LOCK:
        _FRAMES.clear()


def registered_frame_keys() -> list[str]:
    """Tokens currently held by the registry."""
    with _LOCK:
        return sorted(_FRAMES)


def make_dataframe_source(
    frame: Any,
    *,
    name: str = "",
    key: str | None = None,
    copy: bool = True,
) -> DataSource:
    """Build a ``SourceKind.DATAFRAME`` source for an in-memory frame.

    Args:
        frame: The dataframe to analyse.
        name: Human-readable label recorded in ``uri`` for the report.
        key: Explicit registry token.
        copy: Whether the connector should copy the frame before the pipeline
            mutates it. Leave ``True`` unless the caller is done with the frame.

    Returns:
        A source the router can load.
    """
    token = register_frame(frame, key)
    options = [Param(key=FRAME_KEY_OPTION, value=token)]
    if not copy:
        options.append(Param(key="copy", value="false"))
    return DataSource(
        kind=SourceKind.DATAFRAME,
        uri=name or f"dataframe://{token}",
        options=options,
    )


@register(SourceKind.DATAFRAME)
class DataFrameConnector(Connector):
    """Reads a frame that the caller already has in memory."""

    def _lookup_key(self) -> str:
        """Find the registry token on the source.

        Returns:
            The token.

        Raises:
            ConfigurationError: No token is present anywhere on the source.
        """
        token = self.opt_str(FRAME_KEY_OPTION, "key", "frame", "frame_id", "name")
        if token:
            return token
        uri = self.source.uri or ""
        if uri.startswith("dataframe://"):
            return uri[len("dataframe://") :]
        if uri and get_frame(uri) is not None:
            return uri
        raise ConfigurationError(
            "A dataframe source needs a registry token. Build it with "
            "automl_architect.ingestion.dataframe.make_dataframe_source(df)."
        )

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Fetch the registered frame.

        Args:
            max_rows: Row cap.

        Returns:
            The frame (copied by default) plus notes.

        Raises:
            ConfigurationError: The token is unknown or holds non-tabular data.
        """
        key = self._lookup_key()
        raw = get_frame(key)
        if raw is None:
            raise ConfigurationError(
                f"No in-memory frame is registered under {key!r}. In-memory sources "
                "cannot be replayed in a different process; re-register the frame."
            )

        if isinstance(raw, pd.DataFrame):
            frame = raw
        else:
            try:
                frame = pd.DataFrame(raw)
            except Exception as exc:
                raise ConfigurationError(
                    f"Registered object under {key!r} is {type(raw).__name__}, which is "
                    f"not tabular: {exc}"
                ) from exc
            self.note(f"Converted {type(raw).__name__} to a DataFrame.")

        if self.opt_bool("copy", default=True):
            frame = frame.copy()

        if self.opt_bool("reset_index", default=False):
            frame = frame.reset_index(drop=True)

        frame, truncated = cap_rows(frame, max_rows)
        return self.outcome(frame, truncated=truncated, detail="in-memory dataframe")


__all__ = [
    "DataFrameConnector",
    "FRAME_KEY_OPTION",
    "clear_frames",
    "get_frame",
    "make_dataframe_source",
    "register_frame",
    "registered_frame_keys",
    "release_frame",
]
