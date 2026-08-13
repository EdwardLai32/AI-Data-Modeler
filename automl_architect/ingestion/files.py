"""File-based connectors: CSV, Excel, JSON, and Parquet.

The CSV path is the one that earns its keep. Real-world CSVs arrive with a
semicolon delimiter, a UTF-8 BOM, Latin-1 mojibake, quoted thousands
separators, and dates in five formats, so the reader sniffs the delimiter from a
byte sample, walks an encoding ladder, and only *then* hands the file to pandas.
Every fallback it takes is recorded as a note so the choice shows up in the
ingestion result instead of silently changing the data.

The module-level readers (:func:`read_any` and friends) are reused by the cloud
and Kaggle connectors, which download an object and then need exactly this
logic applied to the bytes.
"""

from __future__ import annotations

import bz2
import codecs
import csv
import gzip
import io
import json
import logging
import lzma
import re
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pandas as pd

from ..core.errors import ConfigurationError, IngestionError
from ..core.schemas import SourceKind
from .base import (
    Connector,
    LoadOutcome,
    cap_rows,
    coerce_datetime_columns,
    coerce_grouped_numbers,
    fetch_limit,
    lazy_import,
    redact_uri,
    register,
)

logger = logging.getLogger(__name__)

#: Encoding ladder. ``latin-1`` never raises, so it is the terminal fallback.
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

#: Delimiters considered when sniffing a CSV.
DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|")

#: Bytes sampled for delimiter/encoding detection.
SAMPLE_BYTES = 256 * 1024

#: Keys searched for a record list inside a nested JSON object.
JSON_DATA_KEYS: tuple[str, ...] = (
    "data",
    "records",
    "results",
    "rows",
    "items",
    "value",
    "payload",
    "entries",
)

_EXTENSION_FORMATS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".psv": "csv",
    ".dat": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".geojson": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".parq": "parquet",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".xlsb": "excel",
    ".ods": "excel",
}

_COMPRESSION_EXTENSIONS = {".gz", ".bz2", ".xz", ".zip", ".zst", ".zstd", ".lzma"}

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"PK\x03\x04", "zip"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
)

_HTTP = re.compile(r"^https?://", re.I)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_ESCAPES = {"\\t": "\t", "\\|": "|", "\\s": " ", "tab": "\t", "\\\\": "\\"}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(name: str, explicit: str | None = None) -> str:
    """Decide which reader to use for a path or URI.

    Args:
        name: File name, path, or URI. Query strings are ignored.
        explicit: Caller override, e.g. from a ``format`` option.

    Returns:
        One of ``"csv"``, ``"excel"``, ``"json"``, ``"parquet"``.

    Raises:
        ConfigurationError: ``explicit`` is not a supported format.
    """
    if explicit:
        fmt = explicit.strip().lower()
        aliases = {
            "xlsx": "excel",
            "xls": "excel",
            "spreadsheet": "excel",
            "jsonl": "json",
            "ndjson": "json",
            "tsv": "csv",
            "text": "csv",
            "pq": "parquet",
        }
        fmt = aliases.get(fmt, fmt)
        if fmt not in {"csv", "excel", "json", "parquet"}:
            raise ConfigurationError(
                f"Unsupported file format {explicit!r}. Use csv, excel, json, or parquet."
            )
        return fmt

    stem = name.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    suffixes = [s.lower() for s in Path(stem).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION_EXTENSIONS:
        suffixes.pop()
    if suffixes and suffixes[-1] in _EXTENSION_FORMATS:
        return _EXTENSION_FORMATS[suffixes[-1]]
    return "csv"


# ---------------------------------------------------------------------------
# Byte acquisition
# ---------------------------------------------------------------------------


def _http_bytes(url: str, options: dict[str, Any], timeout: float = 60.0) -> bytes:
    """Download a URL into memory with httpx, following redirects."""
    httpx = lazy_import("httpx", "reading data over HTTP")
    headers = options.get("headers") if isinstance(options.get("headers"), dict) else {}
    try:
        response = httpx.get(
            url, timeout=timeout, follow_redirects=True, headers=headers or None
        )
        response.raise_for_status()
    except Exception as exc:
        raise IngestionError(
            f"HTTP fetch failed for {redact_uri(url)}: {type(exc).__name__}: {exc}"
        ) from exc
    return response.content


def _fsspec_bytes(uri: str, storage_options: dict[str, Any]) -> bytes:
    """Read a whole remote object through fsspec."""
    fsspec = lazy_import("fsspec", f"reading {uri.split('://', 1)[0]} URIs", "fsspec")
    with fsspec.open(uri, mode="rb", **storage_options) as handle:
        return handle.read()


def _decompress(data: bytes, kind: str, notes: list[str]) -> bytes:
    """Fully decompress ``data``, or return it unchanged if we cannot."""
    try:
        if kind == "gzip":
            return gzip.decompress(data)
        if kind == "bz2":
            return bz2.decompress(data)
        if kind in ("xz", "lzma"):
            return lzma.decompress(data)
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [n for n in archive.namelist() if not n.endswith("/")]
                if not members:
                    raise IngestionError("Zip archive contains no files.")
                if len(members) > 1:
                    notes.append(
                        f"Zip archive holds {len(members)} files; read {members[0]!r}."
                    )
                return archive.read(members[0])
        if kind == "zstd":
            zstd = lazy_import("zstandard", "reading zstd-compressed files", "zstandard")
            return zstd.ZstdDecompressor().decompress(data)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not decompress {kind} payload: {exc}") from exc
    return data


def _file_uri_to_path(text: str) -> Path:
    """Convert a ``file://`` URI to a local path.

    Slicing the scheme off by hand is wrong on both platforms: it leaves the
    leading slash of ``file:///C:/data.csv`` in place on Windows and leaves
    percent-escapes (``%20`` for a space) undecoded everywhere.
    :func:`urllib.request.url2pathname` handles both.

    Args:
        text: A ``file://`` URI.

    Returns:
        The local path it refers to.
    """
    parts = urlsplit(text)
    host = parts.netloc
    if host and host.lower() != "localhost":
        # A UNC share: file://server/share/x -> \\server\share\x
        return Path(f"//{host}{url2pathname(parts.path)}")
    return Path(url2pathname(parts.path))


def _detect_compression(head: bytes) -> str | None:
    """Identify a compression container from its magic bytes."""
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    return None


def _materialise(
    target: Any,
    storage_options: dict[str, Any] | None,
    options: dict[str, Any],
    notes: list[str],
    *,
    decompress: bool = True,
) -> tuple[Any, bytes]:
    """Turn any supported target into something pandas can read, plus a sample.

    Local uncompressed files are left as paths so pandas can stream them; every
    other case is pulled into memory (and decompressed) because sniffing a
    delimiter and retrying an encoding both need seekable bytes.

    Args:
        target: Path, URI, bytes, or file-like object.
        storage_options: fsspec options for remote URIs.
        options: Connector options (``headers``, ``timeout``).
        notes: Mutable note list appended to on fallbacks.
        decompress: Unwrap gzip/bz2/xz/zip containers. Must be off for binary
            formats that *are* containers — an ``.xlsx`` is a zip archive, and
            unwrapping it would hand pandas a fragment of XML.

    Returns:
        ``(readable, sample_bytes)`` where ``readable`` is a :class:`Path` or a
        :class:`io.BytesIO`.

    Raises:
        IngestionError: The target does not exist or cannot be fetched.
    """
    data: bytes | None = None

    if isinstance(target, (bytes, bytearray, memoryview)):
        data = bytes(target)
    elif hasattr(target, "read"):
        try:
            target.seek(0)
        except Exception:  # pragma: no cover - non-seekable stream
            pass
        raw = target.read()
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    else:
        text = str(target)
        if _HTTP.match(text):
            timeout = float(options.get("timeout") or 60.0)
            data = _http_bytes(text, options, timeout=timeout)
        elif _SCHEME.match(text) and not text.lower().startswith("file://"):
            data = _fsspec_bytes(text, storage_options or {})
        else:
            path = (
                _file_uri_to_path(text)
                if text.lower().startswith("file://")
                else Path(text)
            )
            if not path.exists():
                raise IngestionError(f"File not found: {path}")
            if not path.is_file():
                raise IngestionError(f"Not a file: {path}")
            with path.open("rb") as handle:
                head = handle.read(SAMPLE_BYTES)
            compression = _detect_compression(head) if decompress else None
            if compression is None:
                return path, head
            data = _decompress(path.read_bytes(), compression, notes)
            notes.append(f"Decompressed {compression} input in memory.")

    if data is None:  # pragma: no cover - defensive
        raise IngestionError("Nothing to read from the given target.")

    compression = _detect_compression(data[:16]) if decompress else None
    if compression is not None:
        data = _decompress(data, compression, notes)
        notes.append(f"Decompressed {compression} input in memory.")
    return io.BytesIO(data), data[:SAMPLE_BYTES]


def _rewind(readable: Any) -> Any:
    """Reset a buffer so a failed parse attempt can be retried."""
    if hasattr(readable, "seek"):
        readable.seek(0)
    return readable


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _decode_sample(sample: bytes, encodings: Sequence[str]) -> tuple[str, str]:
    """Decode a byte sample with the first encoding that accepts it.

    Args:
        sample: Raw head of the file.
        encodings: Candidate encodings, in preference order.

    Returns:
        ``(text, encoding)``.
    """
    if sample.startswith(codecs.BOM_UTF8):
        # Plain utf-8 "succeeds" on BOM'd bytes but leaves a U+FEFF at the front,
        # which corrupts the first header and breaks json.loads outright.
        encodings = ["utf-8-sig", *[e for e in encodings if e != "utf-8-sig"]]
    # The tail of a fixed-size sample can split a multi-byte character, so trim
    # the last few bytes rather than mis-blaming the encoding.
    for encoding in encodings:
        for trim in range(0, 4):
            body = sample[: len(sample) - trim] if trim else sample
            try:
                return body.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
    return sample.decode("latin-1", errors="replace"), "latin-1"


def _complete_lines(text: str, limit: int = 60) -> str:
    """First ``limit`` whole lines of ``text``, dropping a partial last line."""
    lines = text.splitlines()
    if len(lines) > 1:
        lines = lines[:limit]
        if not text.endswith(("\n", "\r")):
            lines = lines[:-1] or lines
    return "\n".join(lines)


def _score_delimiter(sample: str, delimiter: str) -> tuple[int, int]:
    """Rate a delimiter by field-count consistency across sampled rows.

    Returns:
        ``(consistent_rows, fields_per_row)`` — higher is better, and a
        single-field result scores zero because it proves nothing.
    """
    try:
        rows = [
            row
            for row in csv.reader(io.StringIO(sample), delimiter=delimiter)
            if row and any(cell.strip() for cell in row)
        ][:50]
    except csv.Error:
        return (0, 0)
    if len(rows) < 2:
        # A header-only file still tells us something: more fields is better.
        return (0, len(rows[0]) if rows else 0)
    width = len(rows[0])
    if width < 2:
        return (0, width)
    consistent = sum(1 for row in rows if len(row) == width)
    return (consistent, width)


def _sniff_delimiter(sample: str, notes: list[str]) -> str:
    """Determine the field delimiter of a CSV sample.

    Tries :class:`csv.Sniffer` first, then falls back to scoring each candidate
    by how consistently it produces the same field count per row.
    """
    body = _complete_lines(sample)
    if not body.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(body, delimiters="".join(DELIMITER_CANDIDATES))
        if dialect.delimiter in DELIMITER_CANDIDATES:
            return dialect.delimiter
    except csv.Error:
        pass

    scored = sorted(
        ((_score_delimiter(body, d), d) for d in DELIMITER_CANDIDATES),
        key=lambda item: (item[0][0], item[0][1]),
        reverse=True,
    )
    (consistent, width), best = scored[0]
    if consistent >= 2 and width >= 2:
        notes.append(f"Delimiter sniffed as {best!r} by field-count consistency.")
        return best
    notes.append("Could not sniff a delimiter; defaulted to ','.")
    return ","


def _unescape_sep(value: str) -> str:
    """Interpret ``"\\t"``-style option values as the character they name."""
    return _ESCAPES.get(value, value)


def _read_csv(
    target: Any,
    options: dict[str, Any],
    max_rows: int | None,
    storage_options: dict[str, Any] | None,
    notes: list[str],
) -> pd.DataFrame:
    """Read a delimited text file, sniffing delimiter and encoding."""
    readable, sample = _materialise(target, storage_options, options, notes)

    explicit_encoding = options.get("encoding")
    encodings: tuple[str, ...] = (
        (str(explicit_encoding),) + TEXT_ENCODINGS
        if explicit_encoding
        else TEXT_ENCODINGS
    )
    text, sample_encoding = _decode_sample(sample, encodings)

    raw_sep = options.get("sep") or options.get("delimiter")
    if raw_sep is not None:
        sep = _unescape_sep(str(raw_sep))
    else:
        sep = _sniff_delimiter(text, notes)

    thousands = options.get("thousands", ",")
    thousands = None if thousands in (None, "", False) else str(thousands)
    decimal = str(options.get("decimal", "."))
    if thousands == decimal:
        thousands = None

    kwargs: dict[str, Any] = {
        "sep": sep,
        "thousands": thousands,
        "decimal": decimal,
        "nrows": fetch_limit(max_rows),
    }
    if len(sep) > 1:
        kwargs["engine"] = "python"
    for key in (
        "header",
        "skiprows",
        "skipfooter",
        "usecols",
        "na_values",
        "comment",
        "quotechar",
        "escapechar",
        "true_values",
        "false_values",
        "index_col",
        "lineterminator",
        "dtype",
    ):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    if kwargs.get("skipfooter"):
        kwargs["engine"] = "python"
        kwargs.pop("lineterminator", None)

    ordered = list(dict.fromkeys(encodings))
    # The sample decoded under `sample_encoding`; try that first, then the rest.
    ordered.sort(key=lambda enc: 0 if enc == sample_encoding else 1)

    last_error: Exception | None = None
    for encoding in ordered:
        try:
            frame = pd.read_csv(_rewind(readable), encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            last_error = exc
            try:
                frame = pd.read_csv(
                    _rewind(readable),
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                    **{k: v for k, v in kwargs.items() if k != "engine"},
                )
            except Exception:
                continue
            notes.append(
                "Malformed CSV rows were skipped after a parse error "
                f"({str(exc).splitlines()[0][:120]})."
            )
        except pd.errors.EmptyDataError:
            notes.append("The file contains no parseable rows or columns.")
            return pd.DataFrame()
        if encoding != ordered[0] or encoding == "latin-1":
            notes.append(f"Decoded with {encoding!r}.")
        if encoding == "latin-1" and "utf-8" in ordered:
            notes.append(
                "utf-8 decoding failed, so latin-1 was used; non-ASCII text may be "
                "mis-transliterated."
            )
        return frame

    raise IngestionError(
        "Could not decode the file with any of "
        f"{ordered}: {type(last_error).__name__}: {last_error}"
    )


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def _read_excel(
    target: Any,
    options: dict[str, Any],
    max_rows: int | None,
    storage_options: dict[str, Any] | None,
    notes: list[str],
) -> pd.DataFrame:
    """Read one sheet of a workbook, defaulting to the first."""
    if _is_passthrough_uri(target, storage_options):
        source: Any = target
    else:
        source, _ = _materialise(
            target, storage_options, options, notes, decompress=False
        )

    sheet: Any = options.get("sheet")
    if sheet is None:
        sheet = options.get("sheet_name")
    if sheet is None:
        sheet = 0
    elif isinstance(sheet, str) and sheet.strip().lstrip("-").isdigit():
        sheet = int(sheet.strip())

    kwargs: dict[str, Any] = {"nrows": fetch_limit(max_rows)}
    for key in ("header", "skiprows", "usecols", "na_values", "index_col", "dtype"):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    if storage_options and _is_passthrough_uri(target, storage_options):
        kwargs["storage_options"] = storage_options

    available: list[str] = []
    try:
        with pd.ExcelFile(_rewind(source)) as book:
            available = [str(name) for name in book.sheet_names]
    except Exception as exc:  # pragma: no cover - remote or exotic engines
        logger.debug("could not enumerate sheets: %s", exc)

    if isinstance(sheet, str) and available and sheet not in available:
        match = next((n for n in available if n.lower() == sheet.lower()), None)
        if match is not None:
            sheet = match
        else:
            notes.append(
                f"Sheet {sheet!r} not found; read {available[0]!r} instead "
                f"(available: {', '.join(available[:10])})."
            )
            sheet = available[0]
    if isinstance(sheet, int) and available and sheet >= len(available):
        notes.append(
            f"Sheet index {sheet} is out of range; read {available[0]!r} instead."
        )
        sheet = 0
    if len(available) > 1:
        label = available[sheet] if isinstance(sheet, int) else sheet
        notes.append(
            f"Workbook has {len(available)} sheets; read {label!r}. "
            "Pass a 'sheet' option to select another."
        )

    frame = pd.read_excel(_rewind(source), sheet_name=sheet, **kwargs)
    if isinstance(frame, dict):  # sheet_name=None was passed through
        first = next(iter(frame))
        notes.append(f"Multiple sheets requested; kept {first!r}.")
        frame = frame[first]
    return frame


def _is_passthrough_uri(target: Any, storage_options: dict[str, Any] | None) -> bool:
    """Whether pandas can read ``target`` itself via fsspec storage options."""
    return (
        storage_options is not None
        and isinstance(target, str)
        and bool(_SCHEME.match(target))
        and not _HTTP.match(target)
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def dig_path(payload: Any, path: str) -> Any:
    """Follow a dotted path into nested JSON, returning ``None`` if absent.

    List indices are supported as numeric segments, so ``"data.0.rows"`` works.

    Args:
        payload: Decoded JSON.
        path: Dot-separated path, e.g. ``"response.items"``.

    Returns:
        The located value, or ``None``.
    """
    current = payload
    for part in str(path).split("."):
        if part == "":
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def records_to_frame(
    records: Any, *, sep: str = ".", max_records: int | None = None
) -> pd.DataFrame:
    """Normalise JSON-ish records into a flat frame.

    Args:
        records: A list of mappings, a list of scalars, or a single mapping.
        sep: Separator used when flattening nested keys.
        max_records: Optional cap applied before normalising.

    Returns:
        A flat dataframe.
    """
    if isinstance(records, dict):
        records = [records]
    if isinstance(records, list) and max_records is not None:
        records = records[:max_records]
    if isinstance(records, list) and records and not isinstance(records[0], dict):
        return pd.DataFrame({"value": records})
    if not isinstance(records, list):
        return pd.DataFrame()
    return pd.json_normalize(records, sep=sep)


def _read_json_lines(text: str, limit: int | None, notes: list[str]) -> list[Any]:
    """Parse JSON Lines, tolerating (and counting) unparseable lines."""
    records: list[Any] = []
    failures = 0
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped or stripped in ("[", "]"):
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            failures += 1
            continue
        if limit is not None and len(records) >= limit:
            break
    if failures:
        notes.append(f"Skipped {failures} unparseable JSON Lines record(s).")
    return records


def _read_json(
    target: Any,
    options: dict[str, Any],
    max_rows: int | None,
    storage_options: dict[str, Any] | None,
    notes: list[str],
) -> pd.DataFrame:
    """Read records, a nested object with a data key, or JSON Lines."""
    readable, _ = _materialise(target, storage_options, options, notes)
    raw = readable.read_bytes() if isinstance(readable, Path) else _rewind(readable).read()
    if isinstance(raw, str):  # pragma: no cover - defensive
        raw = raw.encode("utf-8")
    encodings = (
        (str(options["encoding"]),) + TEXT_ENCODINGS
        if options.get("encoding")
        else TEXT_ENCODINGS
    )
    text, encoding = _decode_sample(raw, encodings)
    if encoding not in ("utf-8", "utf-8-sig"):
        notes.append(f"Decoded JSON with {encoding!r}.")

    limit = fetch_limit(max_rows)
    sep = str(options.get("nested_sep") or options.get("record_sep") or ".")
    path = options.get("records_path") or options.get("data_path") or options.get("path")

    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        records = _read_json_lines(text, limit, notes)
        if not records:
            raise IngestionError(
                f"Payload is neither JSON nor JSON Lines: {exc.msg} (line {exc.lineno})"
            ) from exc
        notes.append("Parsed input as JSON Lines.")
        return records_to_frame(records, sep=sep, max_records=limit)

    if path:
        located = dig_path(payload, str(path))
        if located is None:
            notes.append(f"records_path {path!r} not found; using the document root.")
        else:
            payload = located

    if isinstance(payload, list):
        return records_to_frame(payload, sep=sep, max_records=limit)

    if isinstance(payload, dict):
        for key in JSON_DATA_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list) and candidate:
                notes.append(f"Extracted records from the {key!r} key.")
                return records_to_frame(candidate, sep=sep, max_records=limit)
        columnar = [v for v in payload.values() if isinstance(v, list)]
        if columnar and len(columnar) == len(payload):
            lengths = {len(v) for v in columnar}
            if len(lengths) == 1:
                notes.append("Interpreted the object as column-oriented JSON.")
                frame = pd.DataFrame(payload)
                return frame.head(limit) if limit else frame
        notes.append(
            "JSON document is a single object with no record list; flattened it "
            "into one row."
        )
        return records_to_frame(payload, sep=sep)

    raise IngestionError(
        f"JSON payload of type {type(payload).__name__} cannot be read as a table."
    )


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def _read_parquet(
    target: Any,
    options: dict[str, Any],
    max_rows: int | None,
    storage_options: dict[str, Any] | None,
    notes: list[str],
) -> pd.DataFrame:
    """Read a Parquet file, reading only the needed row groups when capped."""
    columns = options.get("columns")
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",") if c.strip()]

    limit = fetch_limit(max_rows)
    if limit is not None:
        frame = _read_parquet_head(target, columns, limit, storage_options, options, notes)
        if frame is not None:
            return frame

    if _is_passthrough_uri(target, storage_options):
        return pd.read_parquet(
            target,
            columns=columns,
            storage_options=storage_options,
            engine="pyarrow",
        )
    readable, _ = _materialise(
        target, storage_options, options, notes, decompress=False
    )
    return pd.read_parquet(_rewind(readable), columns=columns, engine="pyarrow")


def _read_parquet_head(
    target: Any,
    columns: list[str] | None,
    limit: int,
    storage_options: dict[str, Any] | None,
    options: dict[str, Any],
    notes: list[str],
) -> pd.DataFrame | None:
    """Stream the first ``limit`` rows via pyarrow, or ``None`` if unsupported."""
    try:
        pq = lazy_import("pyarrow.parquet", "reading Parquet files", "pyarrow")
        pa = lazy_import("pyarrow", "reading Parquet files", "pyarrow")
        if _is_passthrough_uri(target, storage_options):
            handle: Any = target
        else:
            handle, _ = _materialise(
                target, storage_options, options, notes, decompress=False
            )
            handle = _rewind(handle)
        parquet_file = pq.ParquetFile(handle)
        batches: list[Any] = []
        collected = 0
        for batch in parquet_file.iter_batches(
            batch_size=min(limit, 65_536), columns=columns
        ):
            batches.append(batch)
            collected += batch.num_rows
            if collected >= limit:
                break
        if not batches:
            return pa.Table.from_batches(
                [], schema=parquet_file.schema_arrow
            ).to_pandas()
        table = pa.Table.from_batches(batches)
        return table.slice(0, limit).to_pandas()
    except Exception as exc:
        logger.debug("parquet streaming read failed, falling back: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_READERS = {
    "csv": _read_csv,
    "excel": _read_excel,
    "json": _read_json,
    "parquet": _read_parquet,
}

#: Formats whose values arrive as text and therefore benefit from type guessing.
_TEXTUAL = {"csv", "excel", "json"}


def read_any(
    target: Any,
    *,
    fmt: str,
    options: dict[str, Any] | None = None,
    max_rows: int | None = None,
    storage_options: dict[str, Any] | None = None,
    notes: Iterable[str] = (),
) -> LoadOutcome:
    """Read a file-like target in the given format.

    Shared entry point for the file connectors and for the cloud/Kaggle
    connectors, which fetch an object first and then delegate here.

    Args:
        target: Path, URI, bytes, or file-like object.
        fmt: ``csv``, ``excel``, ``json``, or ``parquet``.
        options: Connector options (``sep``, ``encoding``, ``sheet``, ...).
        max_rows: Row cap.
        storage_options: fsspec options for remote URIs.
        notes: Notes accumulated by the caller, carried into the outcome.

    Returns:
        The loaded frame plus notes and the truncation flag.

    Raises:
        ConfigurationError: ``fmt`` is not supported.
        IngestionError: The target could not be read.
    """
    opts = dict(options or {})
    collected = list(notes)
    reader = _READERS.get(fmt)
    if reader is None:
        raise ConfigurationError(
            f"Unsupported format {fmt!r}; expected one of {sorted(_READERS)}."
        )

    frame = reader(target, opts, max_rows, storage_options, collected)
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)

    if fmt in _TEXTUAL and not frame.empty:
        thousands = opts.get("thousands", ",")
        if thousands not in (None, "", False):
            changed = coerce_grouped_numbers(
                frame, str(thousands), str(opts.get("decimal", "."))
            )
            if changed:
                collected.append(
                    f"Parsed thousands separators in {len(changed)} column(s): "
                    f"{', '.join(changed[:5])}."
                )
        if _wants_dates(opts):
            parsed = coerce_datetime_columns(
                frame, dayfirst=bool(opts.get("dayfirst", False))
            )
            if parsed:
                collected.append(
                    f"Parsed {len(parsed)} text column(s) as datetimes: "
                    f"{', '.join(parsed[:5])}."
                )

    frame, truncated = cap_rows(frame, max_rows)
    return LoadOutcome(
        frame=frame,
        notes=collected,
        truncated=truncated,
        detail=fmt,
        text_delimited=fmt == "csv",
    )


def _wants_dates(options: dict[str, Any]) -> bool:
    """Whether automatic datetime inference is enabled (default: yes)."""
    flag = options.get("parse_dates", options.get("infer_datetimes", True))
    if isinstance(flag, bool):
        return flag
    return str(flag).strip().lower() not in {"0", "false", "no", "off", "none"}


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------


class _FileConnector(Connector):
    """Shared plumbing for the local/URL file connectors."""

    fmt: str = "csv"

    def target(self) -> str:
        """Resolve the path or URL to read.

        Returns:
            The resolved location.

        Raises:
            ConfigurationError: Neither ``uri`` nor a path option was given.
        """
        location = self.source.uri or self.opt_str("path", "file", "filepath", "url")
        if not location:
            raise ConfigurationError(
                f"A {self.source.kind.value} source needs a path in DataSource.uri "
                "(or a 'path' option)."
            )
        return location

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Read the file and return the frame with any load-time notes."""
        location = self.target()
        fmt = detect_format(location, self.opt_str("format"))
        if fmt != self.fmt:
            self.note(
                f"Source kind is {self.source.kind.value} but the target looks like "
                f"{fmt}; read it as {fmt}."
            )
        outcome = read_any(
            location,
            fmt=fmt,
            options=self.options,
            max_rows=max_rows,
            notes=self.notes,
        )
        return outcome


@register(SourceKind.CSV)
class CsvConnector(_FileConnector):
    """Delimited text files, with delimiter and encoding detection."""

    fmt = "csv"


@register(SourceKind.EXCEL)
class ExcelConnector(_FileConnector):
    """Excel workbooks; reads a named or indexed sheet, first sheet by default."""

    fmt = "excel"


@register(SourceKind.JSON)
class JsonConnector(_FileConnector):
    """JSON records, a nested object with a data key, or JSON Lines."""

    fmt = "json"


@register(SourceKind.PARQUET)
class ParquetConnector(_FileConnector):
    """Parquet files via pyarrow."""

    fmt = "parquet"


__all__ = [
    "CsvConnector",
    "DELIMITER_CANDIDATES",
    "ExcelConnector",
    "JSON_DATA_KEYS",
    "JsonConnector",
    "ParquetConnector",
    "TEXT_ENCODINGS",
    "detect_format",
    "dig_path",
    "read_any",
    "records_to_frame",
]
