"""REST API connector: JSON over HTTP, paginated, flattened into a frame.

Supports the two pagination styles that cover most business APIs — numeric
``page``/``offset`` and opaque cursors — plus an explicit ``next`` URL when the
API returns one. Every mode is capped by ``max_pages`` so a broken cursor cannot
turn ingestion into an infinite loop, and the cap is reported as truncation
rather than passed off as a complete dataset.

Auth tokens come from the environment variable named in ``DataSource.secret_env``
and are placed in a header. They are never written to notes, logs, or the
:class:`~automl_architect.core.schemas.IngestionResult`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..core.errors import ConfigurationError, IngestionError
from ..core.schemas import SourceKind
from .base import (
    Connector,
    LoadOutcome,
    cap_rows,
    coerce_datetime_columns,
    lazy_import,
    redact_uri,
    register,
    scrub_secrets,
)
from .files import JSON_DATA_KEYS, dig_path, records_to_frame

logger = logging.getLogger(__name__)

#: Hard ceiling on pagination, whatever the caller asks for.
MAX_PAGES_CEILING = 200

#: Default page count when the source configures pagination without a cap.
DEFAULT_MAX_PAGES = 20

_AUTH_HEADER_DEFAULTS = {
    "bearer": ("Authorization", "Bearer {token}"),
    "token": ("Authorization", "Token {token}"),
    "api_key": ("X-API-Key", "{token}"),
    "apikey": ("X-API-Key", "{token}"),
    "basic": ("Authorization", "Basic {token}"),
    "raw": ("Authorization", "{token}"),
}


@register(SourceKind.REST_API)
class RestApiConnector(Connector):
    """Fetches JSON from an HTTP endpoint and normalises it into a frame.

    Recognised options (all optional):

    ``method``
        ``GET`` (default) or ``POST``.
    ``headers``, ``params``, ``json``
        JSON objects merged into the request. ``json`` is the POST body.
    ``records_path``
        Dot path to the record list, e.g. ``"data.items"``. Without it the
        connector looks for a list under the usual keys (``data``, ``results``,
        ...) and falls back to treating the document as one record.
    ``auth_scheme``, ``auth_header``
        How to present the credential: ``bearer`` (default), ``api_key``,
        ``basic``, ``token``, ``raw``, or ``none``.
    ``page_param``/``page_size_param``/``page_size``/``start_page``
        Numeric page pagination.
    ``offset_param``/``limit_param``
        Offset pagination; the offset advances by the records received.
    ``cursor_param``/``cursor_path``
        Cursor pagination; ``cursor_path`` locates the next cursor in the body.
    ``next_url_path``
        Dot path to an absolute URL for the next page.
    ``max_pages``, ``timeout``
        Safety limits.
    """

    def _auth_headers(self) -> dict[str, str]:
        """Build the auth header from the environment, if a credential exists."""
        scheme = (self.opt_str("auth_scheme", "auth", default="bearer") or "bearer").lower()
        if scheme in {"none", "off", "false"}:
            return {}
        token = self.secret("token", "api_key", "apikey", "key", "secret", "bearer")
        if not token:
            return {}
        header, template = _AUTH_HEADER_DEFAULTS.get(
            scheme, _AUTH_HEADER_DEFAULTS["bearer"]
        )
        header = self.opt_str("auth_header", "auth_header_name", default=header) or header
        return {header: template.format(token=token)}

    def _extract(self, payload: Any, path: str | None) -> tuple[list[Any], bool]:
        """Pull the record list out of one response body.

        Returns:
            ``(records, found_list)``; ``found_list`` is False when the body was
            a bare object that had to be treated as a single record.
        """
        if path:
            located = dig_path(payload, path)
            if located is None:
                self.note(f"records_path {path!r} not present in the response body.")
            elif isinstance(located, list):
                return located, True
            elif isinstance(located, dict):
                return [located], False
        if isinstance(payload, list):
            return payload, True
        if isinstance(payload, dict):
            for key in JSON_DATA_KEYS:
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate, True
            return [payload], False
        return [], False

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Call the endpoint, following pagination, and flatten the records.

        Args:
            max_rows: Row cap; pagination stops once it is reached.

        Returns:
            The flattened frame plus notes describing pages fetched.

        Raises:
            ConfigurationError: No URL was supplied.
            IngestionError: The request failed or returned non-JSON.
        """
        httpx = lazy_import("httpx", "REST API ingestion")
        url = (self.source.uri or self.opt_str("url", "endpoint", default="") or "").strip()
        if not url:
            raise ConfigurationError(
                "A rest_api source needs the endpoint URL in DataSource.uri."
            )

        method = (self.opt_str("method", default="GET") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise ConfigurationError(
                f"Unsupported HTTP method {method!r}; use GET or POST."
            )

        headers: dict[str, Any] = {"Accept": "application/json"}
        headers.update(self.opt_dict("headers"))
        headers.update(self._auth_headers())

        params: dict[str, Any] = dict(self.opt_dict("params", "query_params"))
        body: dict[str, Any] | None = self.opt_dict("json", "body", "json_body") or None
        timeout = float(self.opt("timeout", "timeout_seconds", default=60.0) or 60.0)

        records_path = self.opt_str("records_path", "data_path", "path")
        page_param = self.opt_str("page_param")
        page_size_param = self.opt_str("page_size_param", "per_page_param")
        page_size = self.opt_int("page_size", "per_page")
        offset_param = self.opt_str("offset_param")
        limit_param = self.opt_str("limit_param")
        cursor_param = self.opt_str("cursor_param")
        cursor_path = self.opt_str("cursor_path", "next_cursor_path")
        next_url_path = self.opt_str("next_url_path", "next_path")
        paginate_in_body = (
            self.opt_str("paginate_in", default="params") or "params"
        ).lower() == "body"

        max_pages = min(
            self.opt_int("max_pages", "page_limit", default=DEFAULT_MAX_PAGES)
            or DEFAULT_MAX_PAGES,
            MAX_PAGES_CEILING,
        )
        row_target = None if max_rows is None else max_rows + 1

        page = self.opt_int("start_page", default=1) or 1
        offset = self.opt_int("start_offset", default=0) or 0
        if page_size and page_size_param:
            params[page_size_param] = page_size
        if page_size and limit_param:
            params[limit_param] = page_size

        records: list[Any] = []
        next_url: str | None = url
        pages_fetched = 0
        hit_page_cap = False

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            while next_url and pages_fetched < max_pages:
                page_params = dict(params)
                page_body = dict(body) if body else None
                slot = (
                    page_body
                    if (paginate_in_body and page_body is not None)
                    else page_params
                )
                if page_param:
                    slot[page_param] = page
                if offset_param:
                    slot[offset_param] = offset

                payload = self._request(
                    client, method, next_url, page_params, page_body, headers
                )
                pages_fetched += 1

                page_records, was_list = self._extract(payload, records_path)
                records.extend(page_records)

                if not page_records or not was_list:
                    break
                if row_target is not None and len(records) >= row_target:
                    break

                page += 1
                offset += len(page_records)
                if next_url_path:
                    located = dig_path(payload, next_url_path)
                    next_url = str(located) if located else None
                elif cursor_param:
                    cursor = (
                        dig_path(payload, cursor_path) if cursor_path else None
                    )
                    if not cursor:
                        next_url = None
                    else:
                        params[cursor_param] = cursor
                elif page_param or offset_param:
                    if page_size and len(page_records) < page_size:
                        next_url = None  # short page means the last page
                else:
                    next_url = None

                if next_url and pages_fetched >= max_pages:
                    hit_page_cap = True

        if pages_fetched > 1:
            self.note(f"Fetched {pages_fetched} page(s) from the API.")
        if hit_page_cap:
            self.note(
                f"Stopped at the {max_pages}-page cap; more data may be available. "
                "Raise the 'max_pages' option to fetch more."
            )
        if not records:
            self.note("The API returned no records.")
            return self.outcome(pd.DataFrame(), detail="rest_api")

        frame = records_to_frame(
            records, sep=str(self.opt_str("nested_sep", default=".") or ".")
        )
        if self.opt_bool("parse_dates", default=True) and not frame.empty:
            parsed = coerce_datetime_columns(frame)
            if parsed:
                self.note(
                    f"Parsed {len(parsed)} field(s) as datetimes: {', '.join(parsed[:5])}."
                )

        frame, truncated = cap_rows(frame, max_rows)
        return self.outcome(
            frame, truncated=truncated or hit_page_cap, detail="rest_api"
        )

    def _request(
        self,
        client: Any,
        method: str,
        url: str,
        params: dict[str, Any],
        body: dict[str, Any] | None,
        headers: dict[str, Any],
    ) -> Any:
        """Perform one HTTP call and decode the JSON body.

        Raises:
            IngestionError: Transport failure, error status, or non-JSON body.
        """
        try:
            response = client.request(
                method,
                url,
                params=params or None,
                json=body if method == "POST" else None,
                headers=headers,
            )
            response.raise_for_status()
        except Exception as exc:
            detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            raise IngestionError(
                scrub_secrets(
                    f"REST request to {redact_uri(url)} failed: {type(exc).__name__}: "
                    f"{detail[:300]}",
                    self.resolved_secrets,
                )
            ) from exc
        try:
            return response.json()
        except ValueError as exc:
            content_type = response.headers.get("content-type", "unknown")
            raise IngestionError(
                f"Response from {redact_uri(url)} is not JSON (content-type "
                f"{content_type}): {response.text[:200]!r}"
            ) from exc


__all__ = ["DEFAULT_MAX_PAGES", "MAX_PAGES_CEILING", "RestApiConnector"]
