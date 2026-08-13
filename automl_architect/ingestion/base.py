"""Connector contract, registry, and the shared post-load pipeline.

A connector's only job is to turn a :class:`~automl_architect.core.schemas.DataSource`
into a dataframe. Everything that happens *after* the bytes arrive — header
normalisation, structural validation, schema description, memory accounting — is
shared here, so all thirteen source kinds produce an identically-shaped
:class:`~automl_architect.core.schemas.IngestionResult` and no connector can
forget to validate.

Two invariants are enforced in this module rather than left to each connector:

* **Credentials never leave the process.** Secrets are resolved from environment
  variable *names* carried on the source, held only in locals, and scrubbed out
  of any exception text via :func:`scrub_secrets`. :func:`redact_source` masks
  the copy of the source that is embedded in the result.
* **Optional dependencies degrade with a hint.** :func:`lazy_import` raises
  :class:`~automl_architect.core.errors.MissingDependencyError` carrying the
  exact ``pip install`` line instead of letting an ``ImportError`` escape.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..core.errors import (
    ConfigurationError,
    IngestionError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from ..core.schemas import (
    DataSource,
    IngestionResult,
    Param,
    SchemaField,
    SourceKind,
    params_to_dict,
)

logger = logging.getLogger(__name__)

#: How many example values each :class:`SchemaField` carries.
MAX_SAMPLE_VALUES = 3

#: Rows inspected when guessing dtypes or checking for mixed types.
INSPECT_ROWS = 5_000

#: Substrings that mark an option or env-var name as sensitive.
SECRET_NAME_HINTS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "credential",
    "private",
    "sas",
    "signature",
    "apikey",
    "api_key",
    "access_key",
    "authorization",
)

#: Query-string keys that carry pre-signed credentials in cloud URLs.
_SECRET_QUERY_KEYS = {
    "sig",
    "signature",
    "sas",
    "token",
    "access_token",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "awsaccesskeyid",
    "key",
    "apikey",
    "api_key",
}

_REDACTED = "***"

# Imported for their side effect of registering connectors. Loaded lazily so
# that importing this module does not drag in httpx, duckdb, or pyarrow.
_CONNECTOR_MODULES = ("dataframe", "files", "sql", "cloud", "rest", "kaggle")


# ---------------------------------------------------------------------------
# Connector contract
# ---------------------------------------------------------------------------


@dataclass
class LoadOutcome:
    """What a connector hands back to the router.

    Attributes:
        frame: The loaded dataframe (or anything ``pd.DataFrame`` accepts).
        notes: Human-readable warnings collected while reading, e.g. "fell back
            to latin-1". These become ``IngestionResult.validation_warnings``.
        truncated: True when a row cap stopped the read short of the full table.
        detail: Short, credential-free description of what was actually read,
            e.g. ``"s3 via boto3"``. Logged, never used for control flow.
        text_delimited: True when the frame came from delimited text. Only then
            does a single-column result implicate the delimiter, so validation
            uses this to avoid telling a SQL user to check their separator.
    """

    frame: Any
    notes: list[str] = field(default_factory=list)
    truncated: bool = False
    detail: str = ""
    text_delimited: bool = False


class Connector(ABC):
    """Base class for every data source reader.

    Subclasses declare which :class:`SourceKind` values they serve via the
    :func:`register` decorator and implement :meth:`load`. They should raise
    :class:`~automl_architect.core.errors.MissingDependencyError` for absent
    optional packages and :class:`~automl_architect.core.errors.ConfigurationError`
    for a source that cannot be interpreted, and otherwise let the router turn
    unexpected failures into :class:`~automl_architect.core.errors.IngestionError`.
    """

    kinds: ClassVar[tuple[SourceKind, ...]] = ()

    def __init__(self, source: DataSource) -> None:
        """Store the source and decode its ``options`` into real Python values.

        Args:
            source: The source description to read.
        """
        self.source = source
        self.options: dict[str, Any] = params_to_dict(source.options)
        self.notes: list[str] = []
        #: Concrete secret values resolved during the load. Used only to scrub
        #: them back out of error messages; never serialised.
        self.resolved_secrets: list[str] = []

    # -- option access ----------------------------------------------------

    def opt(self, *names: str, default: Any = None) -> Any:
        """Return the first present option among ``names``.

        Args:
            *names: Option keys to try, in priority order. Matching is
                case-insensitive.
            default: Value to return when no key is present.

        Returns:
            The decoded option value, or ``default``.
        """
        lowered = {str(k).lower(): v for k, v in self.options.items()}
        for name in names:
            key = name.lower()
            if key in lowered and lowered[key] is not None:
                return lowered[key]
        return default

    def opt_str(self, *names: str, default: str | None = None) -> str | None:
        """Return an option coerced to ``str`` (or ``default`` when absent)."""
        value = self.opt(*names)
        if value is None:
            return default
        return value if isinstance(value, str) else str(value)

    def opt_bool(self, *names: str, default: bool = False) -> bool:
        """Return an option coerced to ``bool``."""
        value = self.opt(*names)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def opt_int(self, *names: str, default: int | None = None) -> int | None:
        """Return an option coerced to ``int``, or ``default`` if unparseable."""
        value = self.opt(*names)
        if value is None or isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def opt_dict(self, *names: str) -> dict[str, Any]:
        """Return an option that should be a mapping, else an empty dict."""
        value = self.opt(*names)
        return dict(value) if isinstance(value, dict) else {}

    # -- helpers ----------------------------------------------------------

    def note(self, message: str) -> None:
        """Record a load-time warning that will surface in the result."""
        if message and message not in self.notes:
            self.notes.append(message)

    def secret(
        self, *hints: str, required: bool = False, allow_single: bool = True
    ) -> str | None:
        """Resolve one credential from the source's ``secret_env`` names.

        Args:
            *hints: Substrings that identify the wanted variable, e.g.
                ``"password", "pwd"``.
            required: Raise :class:`ConfigurationError` when nothing resolves.
            allow_single: Accept the source's only ``secret_env`` entry when no
                hint matches. Pass ``False`` when reading one of several
                different credentials, so a secret cannot be mistaken for an id.

        Returns:
            The credential value, or ``None`` when absent and not required.

        Raises:
            ConfigurationError: ``required`` is set and no value was found.
        """
        value = resolve_secret(
            self.source.secret_env,
            *hints,
            required=required,
            allow_single=allow_single,
        )
        if value:
            self.resolved_secrets.append(value)
        return value

    def outcome(
        self, frame: Any, *, truncated: bool = False, detail: str = ""
    ) -> LoadOutcome:
        """Wrap ``frame`` together with the notes accumulated so far."""
        return LoadOutcome(
            frame=frame, notes=list(self.notes), truncated=truncated, detail=detail
        )

    @abstractmethod
    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Read the source.

        Args:
            max_rows: Optional cap on rows returned. Connectors should read one
                extra row where cheap (see :func:`fetch_limit`) so that
                truncation can be reported accurately.

        Returns:
            The loaded frame plus load-time notes.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[SourceKind, type[Connector]] = {}
_BUILTINS_LOADED = False


def register(*kinds: SourceKind) -> Callable[[type[Connector]], type[Connector]]:
    """Class decorator mapping one or more :class:`SourceKind` to a connector.

    Args:
        *kinds: The source kinds this connector serves.

    Returns:
        The decorator, which records the class and returns it unchanged.

    Raises:
        ValueError: No kinds were supplied.
    """
    if not kinds:
        raise ValueError("register() requires at least one SourceKind")

    def decorate(cls: type[Connector]) -> type[Connector]:
        cls.kinds = tuple(kinds)
        for kind in kinds:
            previous = _REGISTRY.get(kind)
            if previous is not None and previous is not cls:
                logger.debug(
                    "replacing connector for %s: %s -> %s",
                    kind.value,
                    previous.__name__,
                    cls.__name__,
                )
            _REGISTRY[kind] = cls
        return cls

    return decorate


def _load_builtin_connectors() -> None:
    """Import the shipped connector modules so the registry is populated."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True  # Set first: a partial failure must not re-import.
    for name in _CONNECTOR_MODULES:
        try:
            importlib.import_module(f"{__package__}.{name}")
        except Exception as exc:  # pragma: no cover - defensive
            # One broken connector module must not make every source kind
            # unusable, so this degrades to a narrower registry.
            logger.warning("connector module %r failed to import: %s", name, exc)


def registered_kinds() -> list[SourceKind]:
    """Source kinds that currently have a connector, in enum order."""
    _load_builtin_connectors()
    return [kind for kind in SourceKind if kind in _REGISTRY]


def get_connector(kind: SourceKind) -> type[Connector]:
    """Look up the connector class for ``kind``.

    Args:
        kind: The requested source kind.

    Returns:
        The registered connector class.

    Raises:
        UnsupportedSourceError: Nothing is registered for ``kind``; the message
            lists the kinds that *are* available.
    """
    _load_builtin_connectors()
    cls = _REGISTRY.get(kind)
    if cls is None:
        available = ", ".join(k.value for k in registered_kinds()) or "none"
        raise UnsupportedSourceError(
            f"No connector registered for source kind '{getattr(kind, 'value', kind)}'. "
            f"Registered kinds: {available}."
        )
    return cls


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def lazy_import(module: str, feature: str, extra: str | None = None) -> Any:
    """Import an optional module, or raise a dependency error with a hint.

    Args:
        module: Importable module path, e.g. ``"s3fs"`` or ``"azure.storage.blob"``.
        feature: What the caller was trying to do, for the message.
        extra: ``pip install`` target when it differs from ``module``.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: The module is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module, feature, extra) from exc


def module_available(module: str) -> bool:
    """Whether ``module`` can be imported, without raising on failure."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def resolve_secret(
    env_names: Sequence[str],
    *hints: str,
    required: bool = False,
    allow_single: bool = True,
) -> str | None:
    """Read one credential out of the environment by variable name.

    The source carries *names*, never values, so this is the only place a
    credential enters the process.

    Args:
        env_names: Candidate environment variable names, from
            ``DataSource.secret_env``.
        *hints: Substrings identifying the wanted variable. An all-caps hint is
            also tried as a literal variable name, which covers the conventional
            ``AWS_SECRET_ACCESS_KEY``-style setup where the source lists nothing.
        required: Raise instead of returning ``None`` when unresolved.
        allow_single: When no hint matches and the source lists exactly one
            variable, treat it as the answer. Correct for a source with one
            credential (a database password, an API token); wrong — and disabled
            — where several distinct credentials are read from the same source,
            since it would map a secret key onto an access-key id.

    Returns:
        The credential value, or ``None``.

    Raises:
        ConfigurationError: ``required`` and nothing resolved.
    """
    names = [n for n in env_names if n]
    chosen: str | None = None
    if hints:
        for name in names:
            low = name.lower()
            if any(hint.lower() in low for hint in hints):
                chosen = name
                break
    if chosen is None and (not hints or (allow_single and len(names) == 1)):
        chosen = names[0] if names else None

    # A hint may also name an environment variable directly, which is the
    # conventional case (AWS_SECRET_ACCESS_KEY, KAGGLE_KEY, ...).
    candidates = [chosen] if chosen else []
    candidates.extend(h for h in hints if h.isupper())

    for name in candidates:
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value

    if required:
        wanted = ", ".join(n for n in candidates if n) or ", ".join(hints) or "a secret"
        raise ConfigurationError(
            f"Missing credential: set one of the environment variables [{wanted}]. "
            "Credential values are read from the environment only and are never "
            "stored on the DataSource."
        )
    return None


#: A URI embedded in prose. Bounded by whitespace and quoting characters so the
#: surrounding sentence is never treated as part of the URI.
_URI_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>]+")

#: Punctuation that ends a sentence rather than a URI.
_URI_TRAILERS = ".,;:!?)]}"


def _redact_uri_match(match: re.Match[str]) -> str:
    """Redact one URI found inside a longer message, keeping its punctuation."""
    token = match.group(0)
    trailing = ""
    while token and token[-1] in _URI_TRAILERS:
        trailing = token[-1] + trailing
        token = token[:-1]
    return redact_uri(token) + trailing


def scrub_secrets(message: str, secrets: Iterable[str]) -> str:
    """Replace concrete secret values in ``message`` with ``***``.

    Applied to every exception message that crosses the ingestion boundary so a
    driver that echoes a connection string cannot leak a password into logs or
    into a run summary.

    Only the URI-shaped substrings are handed to :func:`redact_uri`; running it
    over the whole message would treat everything after the first ``?`` as a
    query string and percent-encode the prose that follows it.

    Args:
        message: Text to sanitise.
        secrets: Values to mask. Short values (< 4 chars) are skipped to avoid
            mangling unrelated text.

    Returns:
        The sanitised text.
    """
    out = message
    for secret in secrets:
        if secret and len(secret) >= 4:
            out = out.replace(secret, _REDACTED)
    return _URI_IN_TEXT.sub(_redact_uri_match, out) if "://" in out else out


def redact_uri(uri: str) -> str:
    """Mask userinfo passwords and signed query parameters inside a URI.

    Args:
        uri: A URI, connection string, or arbitrary text containing one.

    Returns:
        The URI with credentials replaced by ``***``.
    """
    if not uri:
        return uri
    text = re.sub(r"(?<=://)([^/:\s@]+):([^/@\s]+)@", rf"\1:{_REDACTED}@", uri)
    if "?" not in text:
        return text
    try:
        parts = urlsplit(text)
    except ValueError:  # pragma: no cover - malformed input
        return text
    if not parts.query:
        return text
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if not pairs or not any(k.lower() in _SECRET_QUERY_KEYS for k, _ in pairs):
        # Nothing to mask, so leave the query string byte-for-byte alone rather
        # than round-tripping it through urlencode.
        return text
    cleaned = [
        (k, _REDACTED if k.lower() in _SECRET_QUERY_KEYS else v) for k, v in pairs
    ]
    # safe="*" keeps the redaction marker readable instead of "%2A%2A%2A".
    return urlunsplit(parts._replace(query=urlencode(cleaned, safe="*")))


def is_secret_name(name: str) -> bool:
    """Whether an option key looks like it holds a credential."""
    low = str(name).lower()
    return any(hint in low for hint in SECRET_NAME_HINTS)


def redact_source(source: DataSource) -> DataSource:
    """Return a copy of ``source`` safe to persist in a run summary.

    Masks credentials inside ``uri`` and blanks any option whose key looks
    secret. ``secret_env`` is left intact: those are variable *names*, which are
    useful for reproducing a run and carry no secret material.

    Args:
        source: The original source.

    Returns:
        A redacted copy. The original is never mutated.
    """
    options: list[Param] = []
    for param in source.options:
        if is_secret_name(param.key):
            options.append(Param(key=param.key, value=_REDACTED))
        elif "://" in param.value:
            options.append(Param(key=param.key, value=redact_uri(param.value)))
        else:
            options.append(param)
    return source.model_copy(
        update={"uri": redact_uri(source.uri), "options": options}, deep=True
    )


# ---------------------------------------------------------------------------
# Row caps
# ---------------------------------------------------------------------------


def fetch_limit(max_rows: int | None) -> int | None:
    """Rows a connector should request in order to detect truncation.

    Reading one row beyond the cap is what lets ``truncated`` be a fact rather
    than a guess: ``len(frame) > max_rows`` proves more data existed.

    Args:
        max_rows: The caller's cap, or ``None`` for no cap.

    Returns:
        ``max_rows + 1``, or ``None``.
    """
    if max_rows is None or max_rows < 0:
        return None
    return max_rows + 1


def cap_rows(frame: pd.DataFrame, max_rows: int | None) -> tuple[pd.DataFrame, bool]:
    """Trim ``frame`` to ``max_rows`` and report whether anything was dropped.

    Args:
        frame: The loaded frame.
        max_rows: Row cap, or ``None``.

    Returns:
        ``(frame, truncated)``.
    """
    if max_rows is None or max_rows < 0 or len(frame) <= max_rows:
        return frame, False
    return frame.iloc[:max_rows].copy(), True


# ---------------------------------------------------------------------------
# Frame coercion helpers shared by text-ish connectors
# ---------------------------------------------------------------------------

_GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DATEISH = re.compile(
    r"""
    ^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}        # 2024-01-31, 2024/01/31
  | ^\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}      # 31/01/2024
  | ^\s*\d{8}T                               # 20240131T...
  | \d{1,2}:\d{2}(?::\d{2})?                 # any clock time
  | ^\s*[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}    # January 31, 2024
  | ^\s*\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}    # 31 Jan 2024
    """,
    re.VERBOSE,
)


def _text_columns(frame: pd.DataFrame) -> list[str]:
    """Column labels holding strings or arbitrary objects."""
    out: list[str] = []
    for name in frame.columns:
        series = frame[name]
        if getattr(series, "ndim", 1) != 1:  # duplicate labels select a frame
            continue
        if is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        if is_datetime64_any_dtype(series):
            continue
        if is_string_dtype(series) or is_object_dtype(series):
            out.append(name)
    return out


def coerce_grouped_numbers(
    frame: pd.DataFrame, thousands: str = ",", decimal: str = "."
) -> list[str]:
    """Convert ``"1,234.5"``-style text columns to numeric, in place.

    Only columns where *every* non-null sampled value carries the grouping
    separator pattern are converted, so identifiers and free text are left
    alone.

    Args:
        frame: Frame to modify in place.
        thousands: Grouping separator to strip.
        decimal: Decimal separator; when not ``"."`` it is normalised first.

    Returns:
        Names of the columns that were converted.
    """
    if not thousands:
        return []
    converted: list[str] = []
    pattern = _GROUPED_NUMBER
    if thousands != ",":
        pattern = re.compile(
            r"^[+-]?\d{1,3}(?:"
            + re.escape(thousands)
            + r"\d{3})+(?:"
            + re.escape(decimal)
            + r"\d+)?$"
        )
    for name in _text_columns(frame):
        series = frame[name]
        sample = series.dropna().astype(str).head(INSPECT_ROWS)
        if sample.empty:
            continue
        hits = sample.str.match(pattern)
        if not bool(hits.all()) or not bool(hits.any()):
            continue
        text = series.astype("string").str.replace(thousands, "", regex=False)
        if decimal != ".":
            text = text.str.replace(decimal, ".", regex=False)
        numeric = pd.to_numeric(text, errors="coerce")
        if numeric.notna().sum() >= series.notna().sum():
            frame[name] = numeric
            converted.append(str(name))
    return converted


def coerce_datetime_columns(
    frame: pd.DataFrame, *, min_parse_rate: float = 0.9, dayfirst: bool = False
) -> list[str]:
    """Parse text columns that look like timestamps, in place.

    A column is converted only when it *looks* like a date (contains a date or
    time separator) and at least ``min_parse_rate`` of a sample actually parses.
    The two-stage test is deliberate: ``pd.to_datetime`` will happily turn an
    integer-like identifier into a nonsense timestamp, which would silently
    destroy a key column.

    Args:
        frame: Frame to modify in place.
        min_parse_rate: Fraction of sampled non-null values that must parse.
        dayfirst: Interpret ambiguous ``01/02/2024`` as day-first.

    Returns:
        Names of the columns that were converted.
    """
    converted: list[str] = []
    for name in _text_columns(frame):
        series = frame[name]
        sample = series.dropna().astype(str).head(INSPECT_ROWS)
        if len(sample) < 3:
            continue
        looks = sample.str.contains(_DATEISH, regex=True, na=False)
        if float(looks.mean()) < min_parse_rate:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
            if float(probe.notna().mean()) < min_parse_rate:
                continue
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        # Refuse the conversion if it would nullify values that were present.
        if parsed.notna().sum() < series.notna().sum() * min_parse_rate:
            continue
        frame[name] = parsed
        converted.append(str(name))
    return converted


# ---------------------------------------------------------------------------
# Normalisation, validation, schema
# ---------------------------------------------------------------------------


def ensure_frame(obj: Any) -> pd.DataFrame:
    """Coerce a connector's payload into a real :class:`pandas.DataFrame`.

    Args:
        obj: A dataframe, series, mapping, or record iterable.

    Returns:
        A dataframe.

    Raises:
        IngestionError: The payload cannot be interpreted as tabular data.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        raise IngestionError("Connector returned no data.")
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    try:
        return pd.DataFrame(obj)
    except Exception as exc:
        raise IngestionError(
            f"Connector returned {type(obj).__name__}, which is not tabular: {exc}"
        ) from exc


def normalise_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip whitespace from headers and make duplicate names unique.

    Deliberately minimal: no lower-casing and no character mangling, because
    every downstream agent and report refers to columns by the name the user
    knows them by.

    Args:
        frame: The loaded frame.

    Returns:
        ``(frame, warnings)``. The input frame is not mutated when renaming is
        required; a shallow copy is returned instead.
    """
    warnings_out: list[str] = []
    original = [c for c in frame.columns]
    stringified = 0
    stripped: list[str] = []
    for label in original:
        if not isinstance(label, str):
            stringified += 1
            label = str(label)
        stripped.append(label.strip())

    whitespace_hits = [
        str(o) for o, s in zip(original, stripped) if isinstance(o, str) and o != s
    ]
    if whitespace_hits:
        warnings_out.append(
            f"Stripped surrounding whitespace from {len(whitespace_hits)} column "
            f"name(s): {_preview(whitespace_hits)}."
        )
    if stringified:
        warnings_out.append(f"Converted {stringified} non-text column label(s) to text.")

    # Name empty labels positionally before de-duplicating, so several blanks do
    # not all collapse onto one generated name.
    blanks = 0
    for index, label in enumerate(stripped):
        if label == "":
            stripped[index] = f"column_{index + 1}"
            blanks += 1
    if blanks:
        warnings_out.append(
            f"Replaced {blanks} blank column name(s) with positional names."
        )

    final, dupes = _dedupe(stripped)
    if dupes:
        warnings_out.append(
            f"Duplicate column name(s) made unique: {_preview(sorted(dupes))}."
        )

    if final != original:
        frame = frame.copy(deep=False)
        frame.columns = pd.Index(final, dtype=object)
    return frame, warnings_out


def _dedupe(names: list[str]) -> tuple[list[str], set[str]]:
    """Suffix repeated names with ``_2``, ``_3``, ... avoiding new collisions."""
    seen: dict[str, int] = {}
    taken = set(names)
    out: list[str] = []
    duplicated: set[str] = set()
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
            continue
        duplicated.add(name)
        counter = seen[name] + 1
        candidate = f"{name}_{counter}"
        while candidate in taken or candidate in out:
            counter += 1
            candidate = f"{name}_{counter}"
        seen[name] = counter
        taken.add(candidate)
        out.append(candidate)
    return out, duplicated


def _preview(values: Sequence[str], limit: int = 5) -> str:
    """Render up to ``limit`` names for a message."""
    shown = [str(v) for v in values[:limit]]
    suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(repr(v) for v in shown) + suffix


_UNNAMED = re.compile(r"^(unnamed(:\s*\d+)?(_level_\d+)?|index|level_\d+)$", re.I)


def validate_frame(
    frame: pd.DataFrame, *, delimited: bool = False
) -> tuple[list[str], list[str]]:
    """Structurally validate a loaded frame.

    Args:
        frame: The frame, already header-normalised.
        delimited: True when the data came from delimited text, which makes a
            single-column result a likely delimiter mis-sniff.

    Returns:
        ``(errors, warnings)``. Errors mean the frame is unusable for modelling
        (no rows, no columns); warnings flag things a human should look at but
        that do not stop the run.
    """
    errors: list[str] = []
    warns: list[str] = []

    n_rows, n_cols = int(frame.shape[0]), int(frame.shape[1])
    if n_cols == 0:
        errors.append("Loaded frame has zero columns; nothing to analyse.")
    if n_rows == 0:
        errors.append("Loaded frame has zero rows; nothing to analyse.")
    if errors:
        return errors, warns

    if n_cols == 1 and delimited:
        warns.append(
            f"Only one column was parsed ({frame.columns[0]!r}); the delimiter may "
            "have been mis-detected. Pass a 'sep' option to override."
        )
    elif n_cols == 1:
        warns.append(
            f"The source produced a single column ({frame.columns[0]!r}), which is "
            "rarely enough to model."
        )

    empty = [
        str(c)
        for c in frame.columns
        if getattr(frame[c], "ndim", 1) == 1 and bool(frame[c].isna().all())
    ]
    if empty:
        warns.append(f"{len(empty)} fully-empty column(s): {_preview(empty)}.")

    unnamed = [str(c) for c in frame.columns if _UNNAMED.match(str(c).strip())]
    if unnamed:
        warns.append(
            f"Column(s) {_preview(unnamed)} look like an exported row index rather "
            "than a feature; consider excluding them."
        )

    mixed: list[str] = []
    for name in frame.columns:
        series = frame[name]
        if getattr(series, "ndim", 1) != 1 or not is_object_dtype(series):
            continue
        kind = infer_dtype(series.head(INSPECT_ROWS), skipna=True)
        if kind.startswith("mixed") or kind == "unknown-array":
            mixed.append(f"{name} ({kind})")
    if mixed:
        warns.append(
            f"Mixed-type column(s) detected: {_preview(mixed)}. Downstream casts may "
            "coerce or drop values."
        )

    return errors, warns


def describe_schema(frame: pd.DataFrame) -> list[SchemaField]:
    """Build one :class:`SchemaField` per column.

    Args:
        frame: The normalised frame.

    Returns:
        Field descriptions with dtype, nullability, and up to three samples.
    """
    fields: list[SchemaField] = []
    for name in frame.columns:
        series = frame[name]
        if getattr(series, "ndim", 1) != 1:  # pragma: no cover - post-dedupe
            series = series.iloc[:, 0]
        fields.append(
            SchemaField(
                name=str(name),
                inferred_dtype=_dtype_label(series),
                nullable=bool(series.isna().any()),
                sample_values=_samples(series),
            )
        )
    return fields


def _dtype_label(series: pd.Series) -> str:
    """Readable dtype, refined for object columns via value inspection."""
    label = str(series.dtype)
    if is_object_dtype(series):
        inferred = infer_dtype(series.head(INSPECT_ROWS), skipna=True)
        if inferred and inferred != "empty":
            return f"object[{inferred}]"
    return label


def _samples(series: pd.Series, limit: int = MAX_SAMPLE_VALUES) -> list[str]:
    """First ``limit`` non-null values rendered as short strings."""
    out: list[str] = []
    for value in series.dropna().head(limit).tolist():
        text = str(value)
        out.append(text if len(text) <= 60 else text[:57] + "...")
    return out


def frame_bytes(frame: pd.DataFrame) -> int:
    """Deep memory footprint of ``frame`` in bytes, 0 if unmeasurable."""
    try:
        return int(frame.memory_usage(deep=True).sum())
    except Exception:  # pragma: no cover - exotic extension dtypes
        return 0


def build_result(
    source: DataSource,
    frame: pd.DataFrame,
    *,
    load_seconds: float,
    truncated: bool = False,
    notes: Iterable[str] = (),
    dataset_id: str | None = None,
    delimited: bool = False,
) -> IngestionResult:
    """Assemble the :class:`IngestionResult` for an already-normalised frame.

    Args:
        source: The source that was read. It is redacted before being embedded.
        frame: The normalised, validated frame.
        load_seconds: Wall-clock seconds the read took.
        truncated: Whether a row cap applied.
        notes: Extra warnings from the connector and normalisation step.
        dataset_id: Override for the generated dataset id.
        delimited: Whether the data came from delimited text.

    Returns:
        A fully populated result.
    """
    errors, warns = validate_frame(frame, delimited=delimited)
    combined: list[str] = []
    for message in list(notes) + warns:
        if message and message not in combined:
            combined.append(message)

    kwargs: dict[str, Any] = {}
    if dataset_id:
        kwargs["dataset_id"] = dataset_id
    return IngestionResult(
        source=redact_source(source),
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        schema_fields=describe_schema(frame),
        validation_errors=errors,
        validation_warnings=combined,
        bytes_in_memory=frame_bytes(frame),
        load_seconds=round(float(load_seconds), 4),
        truncated=bool(truncated),
        **kwargs,
    )


__all__ = [
    "Connector",
    "LoadOutcome",
    "MAX_SAMPLE_VALUES",
    "build_result",
    "cap_rows",
    "coerce_datetime_columns",
    "coerce_grouped_numbers",
    "describe_schema",
    "ensure_frame",
    "fetch_limit",
    "frame_bytes",
    "get_connector",
    "is_secret_name",
    "lazy_import",
    "module_available",
    "normalise_columns",
    "redact_source",
    "redact_uri",
    "register",
    "registered_kinds",
    "resolve_secret",
    "scrub_secrets",
    "validate_frame",
]
