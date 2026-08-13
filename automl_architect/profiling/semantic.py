"""Column-kind and semantic-type inference.

Nothing here trusts a dtype. A CSV read gives ``str`` to dates, codes, booleans,
and free prose alike, so every non-numeric column is examined by value: parsed as
a date, matched against semantic regexes, and measured for token count and
cardinality. The result is a :class:`ColumnInference` carrying both the coarse
:class:`~automl_architect.core.schemas.ColumnKind` the rest of the pipeline
branches on and the fine-grained ``detected_semantic_type`` an agent can reason
about.

Every threshold is a module-level constant rather than a literal buried in a
branch: the numbers are part of the audit trail, and a reviewer needs to be able
to see and change them in one place.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.schemas import ColumnKind

# --- tunable thresholds ---------------------------------------------------

SEMANTIC_SAMPLE_SIZE = 2_000
"""Values examined for regex/date/token work. Inference does not need more."""

SEMANTIC_MATCH_THRESHOLD = 0.85
DATETIME_PARSE_THRESHOLD = 0.85
DATE_SHAPE_THRESHOLD = 0.50
NEAR_UNIQUE_RATIO = 0.95
MIN_ROWS_FOR_NEAR_UNIQUE = 20
DISCRETE_MAX_UNIQUE = 25
DISCRETE_MAX_RATIO = 0.05
TEXT_MIN_TOKENS = 3.0
TEXT_STRONG_TOKENS = 6.0
TEXT_MIN_CARDINALITY_RATIO = 0.30
TEXT_MIN_UNIQUE = 50
ORDINAL_MAX_UNIQUE = 12
NUMERIC_AS_TEXT_THRESHOLD = 0.95
NUMERIC_AS_TEXT_MIN_UNIQUE = 10

LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)
EPOCH_SECONDS_RANGE = (5.0e8, 4.0e9)  # 1985-11 .. 2096-10
EPOCH_MILLIS_RANGE = (5.0e11, 4.0e12)
YEAR_RANGE = (1800.0, 2200.0)

# --- name hints -----------------------------------------------------------

ID_NAME_TOKENS = frozenset(
    {
        "id", "ids", "ident", "uuid", "guid", "key", "keys", "code", "codes",
        "hash", "index", "idx", "no", "num", "number", "ref", "reference", "sku",
        "isbn", "ssn", "account", "acct", "identifier", "pk", "serial",
        "token", "barcode", "imei", "vin",
    }
)

GEO_NAME_TOKENS = frozenset(
    {
        "lat", "latitude", "lon", "lng", "long", "longitude", "geo", "coord",
        "coords", "coordinate", "coordinates", "postal", "postcode", "zip",
        "zipcode", "country", "countries", "city", "cities", "state", "province",
        "region", "county", "district", "municipality", "address", "street",
        "location", "place", "borough", "neighbourhood", "neighborhood",
    }
)

LATITUDE_NAME_TOKENS = frozenset({"lat", "latitude", "lats"})
LONGITUDE_NAME_TOKENS = frozenset({"lon", "lng", "long", "longitude", "lons"})
POSTAL_NAME_TOKENS = frozenset({"postal", "postcode", "zip", "zipcode", "pincode"})

DATE_NAME_TOKENS = frozenset(
    {
        "date", "dates", "time", "times", "timestamp", "datetime", "day", "days",
        "month", "year", "dt", "created", "updated", "modified", "opened",
        "closed", "period", "dob", "birthday", "birthdate", "start", "end",
        "expiry", "expires", "expiration", "signup", "joined", "seen", "at",
    }
)

TEXT_NAME_TOKENS = frozenset(
    {
        "comment", "comments", "description", "desc", "text", "review",
        "reviews", "note", "notes", "message", "msg", "body", "summary",
        "title", "feedback", "content", "remark", "remarks", "abstract",
        "bio", "biography", "narrative", "story", "tweet", "post", "reason",
    }
)

CURRENCY_NAME_TOKENS = frozenset(
    {
        "price", "prices", "cost", "costs", "amount", "revenue", "salary",
        "wage", "fee", "fees", "payment", "balance", "usd", "eur", "gbp",
        "charge", "charges", "income", "spend", "budget", "profit", "sales",
        "mrr", "arr", "value",
    }
)

PERCENT_NAME_TOKENS = frozenset(
    {"pct", "percent", "percentage", "rate", "ratio", "share", "utilisation", "utilization"}
)

ORDINAL_NAME_TOKENS = frozenset(
    {
        "level", "levels", "grade", "grades", "rating", "ratings", "rank",
        "tier", "stage", "severity", "priority", "satisfaction", "quality",
        "size", "sizes", "class", "band", "bracket", "segment", "seniority",
        "education", "degree", "risk",
    }
)

# Column names that imply knowledge of the outcome. Used by leakage detection,
# which requires a statistical signal on top of the name — see leakage.py.
POST_OUTCOME_NAME_FRAGMENTS: tuple[str, ...] = (
    "churn", "cancel", "refund", "exit", "outcome", "label", "target",
    "result", "_after", "post_", "resolved", "closed_reason", "final",
    "actual", "settled", "chargeback", "fraud_flag", "is_fraud", "y_true",
)

# --- value vocabularies ---------------------------------------------------

BOOLEAN_VOCABULARIES: tuple[frozenset[str], ...] = (
    frozenset({"true", "false"}),
    frozenset({"t", "f"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"1", "0"}),
    frozenset({"on", "off"}),
    frozenset({"present", "absent"}),
    frozenset({"pass", "fail"}),
    frozenset({"success", "failure"}),
    frozenset({"enabled", "disabled"}),
    frozenset({"active", "inactive"}),
)

ORDINAL_VOCABULARIES: tuple[tuple[str, ...], ...] = (
    ("low", "medium", "high"),
    ("low", "med", "high"),
    ("very low", "low", "medium", "high", "very high"),
    ("small", "medium", "large"),
    ("xs", "s", "m", "l", "xl", "xxl"),
    ("poor", "fair", "good", "excellent"),
    ("terrible", "bad", "average", "good", "great"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("cold", "warm", "hot"),
    ("beginner", "intermediate", "advanced", "expert"),
    ("bronze", "silver", "gold", "platinum", "diamond"),
    ("free", "basic", "standard", "premium", "enterprise"),
    ("primary", "secondary", "tertiary"),
    ("high school", "bachelor", "master", "phd"),
    ("junior", "mid", "senior", "lead", "principal"),
    ("weekly", "monthly", "quarterly", "yearly"),
    ("minor", "moderate", "major", "critical"),
    ("first", "second", "third", "fourth"),
)

# --- semantic regexes -----------------------------------------------------

SEMANTIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[A-Za-z]{2,}$"),
    "url": re.compile(r"^(?:https?|ftp)://\S+$|^www\.\S+\.\S+$", re.IGNORECASE),
    "uuid": re.compile(
        r"^\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?$",
        re.IGNORECASE,
    ),
    "ipv4": re.compile(
        r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
    ),
    "phone": re.compile(r"^\+?\d[\d\s().\-]{5,18}\d$"),
    "postal_code": re.compile(
        r"^\d{5}(?:-\d{4})?$"  # US
        r"|^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$"  # UK
        r"|^[A-Z]\d[A-Z]\s?\d[A-Z]\d$"  # CA
        r"|^\d{4}\s?[A-Z]{2}$",  # NL
        re.IGNORECASE,
    ),
    "currency": re.compile(
        r"^[-+(]?\s?[$€£¥₹]\s?[\d,]+(?:\.\d{1,4})?\)?$"
        r"|^[-+]?[\d,]+(?:\.\d{1,4})?\s?(?:USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY)$",
        re.IGNORECASE,
    ),
    "percentage": re.compile(r"^[-+]?\d+(?:\.\d+)?\s?%$"),
}

_PURE_DIGITS = re.compile(r"^\d+$")
_NUMERIC_TEXT = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_DATE_SHAPE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2020-01-31, 31/01/2020
    r"|\d{1,2}:\d{2}"  # 13:45
    r"|^\d{8}$"  # 20200131
    r"|\d{4}-\d{2}(?![\d-])"  # 2020-01 month stamps
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)

_KIND_SURVIVES_CONSTANT = frozenset(
    {ColumnKind.DATETIME, ColumnKind.TEXT, ColumnKind.GEO, ColumnKind.UNKNOWN}
)

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WHITESPACE = re.compile(r"\S+")


@dataclass
class ColumnInference:
    """The inferred identity of one column.

    Attributes:
        kind: Coarse kind the rest of the pipeline branches on.
        looks_like_id: Near-unique and named or shaped like a key.
        looks_like_geo: Geographic by name and (for coordinates) by value range.
        looks_like_text: Free prose rather than a short label.
        looks_like_datetime: Parses as a timestamp, or is an epoch integer.
        detected_semantic_type: Fine-grained type, e.g. ``email``, ``latitude``.
        parsed_datetime: The parsed timestamp series when a string column turned
            out to be dates; ``None`` otherwise. Lets the caller compute
            datetime statistics without re-parsing.
        is_boolean_like: Two-valued in a true/false vocabulary.
        n_non_null: Non-null count used by the inference.
        n_unique: Distinct non-null values used by the inference.
        cardinality_ratio: ``n_unique / n_non_null``.
        mean_token_count: Mean whitespace-token count, for string columns.
        reason: One sentence recording why this kind was chosen.
    """

    kind: ColumnKind
    looks_like_id: bool = False
    looks_like_geo: bool = False
    looks_like_text: bool = False
    looks_like_datetime: bool = False
    detected_semantic_type: str | None = None
    parsed_datetime: Any | None = None
    is_boolean_like: bool = False
    n_non_null: int = 0
    n_unique: int = 0
    cardinality_ratio: float = 0.0
    mean_token_count: float | None = None
    reason: str = ""


def name_tokens(name: str) -> set[str]:
    """Lower-cased word tokens of a column name, splitting camelCase too.

    Args:
        name: Raw column label.

    Returns:
        Set of tokens, e.g. ``customerSignupDate`` -> ``{customer, signup, date}``.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", str(name))
    return {token.lower() for token in _TOKEN_SPLIT.split(spaced) if token}


def has_name_hint(name: str, vocabulary: frozenset[str]) -> bool:
    """Whether any token of ``name`` appears in ``vocabulary``."""
    return bool(name_tokens(name) & vocabulary)


def suspicious_outcome_name(name: str) -> str | None:
    """The post-outcome fragment matched by a column name, if any.

    A match alone is never evidence of leakage — see
    :func:`automl_architect.profiling.leakage.detect_leakage`, which requires a
    statistical signal as well.

    Args:
        name: Column name.

    Returns:
        The matched fragment, or ``None``.
    """
    low = str(name).lower()
    for fragment in POST_OUTCOME_NAME_FRAGMENTS:
        if fragment in low:
            return fragment
    return None


def string_sample(series: Any, limit: int = SEMANTIC_SAMPLE_SIZE) -> list[str]:
    """Up to ``limit`` non-null values rendered as stripped strings."""
    try:
        non_null = series.dropna()
    except AttributeError:  # not a Series
        return []
    if len(non_null) > limit:
        non_null = non_null.iloc[:limit]
    out: list[str] = []
    for value in non_null.to_numpy(dtype=object, copy=False):
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def match_fraction(values: list[str], pattern: re.Pattern[str]) -> float:
    """Fraction of ``values`` fully matching ``pattern``."""
    if not values:
        return 0.0
    hits = sum(1 for value in values if pattern.match(value))
    return hits / len(values)


def detect_semantic_type(values: list[str], name: str) -> str | None:
    """Best-matching semantic type for a sample of string values.

    Args:
        values: Sampled non-null values as strings.
        name: Column name, used to disambiguate ambiguous patterns such as a
            bare five-digit postal code versus a five-digit integer.

    Returns:
        A semantic type label (``email``, ``url``, ``ipv4``, ``uuid``,
        ``phone``, ``postal_code``, ``currency``, ``percentage``,
        ``numeric_as_text``) or ``None``.
    """
    if not values:
        return None

    # Ordered most-specific first; uuid before phone so a digit-rich uuid is not
    # mistaken for a phone number.
    for label in ("uuid", "email", "url", "ipv4", "currency", "percentage", "postal_code", "phone"):
        pattern = SEMANTIC_PATTERNS[label]
        if match_fraction(values, pattern) < SEMANTIC_MATCH_THRESHOLD:
            continue
        if label == "postal_code":
            all_digits = all(_PURE_DIGITS.match(value) for value in values)
            # "12345" is only a postal code if the name says so; otherwise it is
            # an integer that happens to have five digits.
            if all_digits and not has_name_hint(name, POSTAL_NAME_TOKENS):
                continue
        if label == "phone":
            digit_counts = [sum(char.isdigit() for char in value) for value in values]
            if min(digit_counts) < 7:
                continue
        return label

    if (
        match_fraction(values, _NUMERIC_TEXT) >= NUMERIC_AS_TEXT_THRESHOLD
        and len(set(values)) >= NUMERIC_AS_TEXT_MIN_UNIQUE
    ):
        return "numeric_as_text"
    return None


def try_parse_datetime(
    series: Any,
    name: str = "",
    *,
    threshold: float = DATETIME_PARSE_THRESHOLD,
) -> tuple[Any | None, float]:
    """Attempt to parse a non-datetime series as timestamps.

    Parsing is only attempted when the values *look* like dates or the column is
    named like one. Without that gate ``pd.to_datetime`` cheerfully turns plain
    integers into years and every numeric code column becomes a date.

    Args:
        series: Candidate series (string, object, or categorical).
        name: Column name, consulted for date-ish tokens.
        threshold: Minimum fraction of non-null values that must parse.

    Returns:
        ``(parsed_series, success_fraction)``. ``parsed_series`` is ``None``
        unless the success fraction met ``threshold``.
    """
    values = string_sample(series)
    if not values:
        return None, 0.0

    shape_fraction = sum(1 for value in values if _DATE_SHAPE.search(value)) / len(values)
    name_hint = has_name_hint(name, DATE_NAME_TOKENS)
    mostly_numeric = all(_PURE_DIGITS.match(value) for value in values)
    if shape_fraction < DATE_SHAPE_THRESHOLD and not (name_hint and not mostly_numeric):
        return None, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = None
        for kwargs in ({"format": "mixed"}, {}):
            try:
                parsed = pd.to_datetime(series, errors="coerce", **kwargs)
                break
            except (ValueError, TypeError, OverflowError):
                continue
    if parsed is None:
        return None, 0.0

    try:
        n_non_null = int(series.notna().sum())
        n_parsed = int(parsed.notna().sum())
    except (AttributeError, TypeError):
        return None, 0.0
    if n_non_null == 0:
        return None, 0.0

    fraction = n_parsed / n_non_null
    return (parsed if fraction >= threshold else None), fraction


def _numeric_array(series: Any) -> np.ndarray:
    """Finite float view of a series, empty when it is not numeric at all."""
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except (ValueError, TypeError):
        return np.empty(0, dtype="float64")
    array = numeric.to_numpy(dtype="float64", na_value=np.nan, copy=False)
    return array[np.isfinite(array)]


def _is_integral(array: np.ndarray) -> bool:
    return array.size > 0 and bool(np.all(np.mod(array, 1.0) == 0.0))


def _lowered_uniques(values: list[str], limit: int = 64) -> set[str] | None:
    """Lower-cased distinct values, or ``None`` if there are too many to match."""
    uniques = {value.lower() for value in values}
    if len(uniques) > limit:
        return None
    return uniques


def _match_boolean_vocabulary(uniques: set[str] | None) -> bool:
    if not uniques or len(uniques) > 2:
        return False
    return any(uniques <= vocabulary for vocabulary in BOOLEAN_VOCABULARIES)


def _match_ordinal_vocabulary(uniques: set[str] | None) -> tuple[str, ...] | None:
    if not uniques or len(uniques) < 2 or len(uniques) > ORDINAL_MAX_UNIQUE:
        return None
    for vocabulary in ORDINAL_VOCABULARIES:
        if uniques <= set(vocabulary):
            return vocabulary
    return None


def _token_stats(values: list[str]) -> tuple[float | None, float | None]:
    """Mean whitespace-token count and mean character length of a sample."""
    if not values:
        return None, None
    tokens = [len(_WHITESPACE.findall(value)) for value in values]
    lengths = [len(value) for value in values]
    return float(np.mean(tokens)), float(np.mean(lengths))


def infer_column(
    name: str,
    series: Any,
    *,
    n_unique: int | None = None,
    n_non_null: int | None = None,
) -> ColumnInference:
    """Infer the kind and semantic type of one column.

    Args:
        name: Column name (used for hints only, never as sole evidence).
        series: The column, ideally the profiling sample rather than a 10M-row
            original.
        n_unique: Distinct non-null count measured on the *full* column. Passed
            in so cardinality decisions use the true value while the value-level
            work runs on the sample.
        n_non_null: Non-null count measured on the full column.

    Returns:
        A :class:`ColumnInference`. Never raises: an unexpected dtype degrades to
        :attr:`ColumnKind.UNKNOWN` with the failure recorded in ``reason``.
    """
    try:
        inference = _infer_column(name, series, n_unique=n_unique, n_non_null=n_non_null)
    except Exception as exc:  # pragma: no cover - defensive: one column must not stop a profile
        return ColumnInference(
            kind=ColumnKind.UNKNOWN,
            reason=f"inference failed ({type(exc).__name__}: {exc})",
        )

    # A single-valued column collapses to CONSTANT *unless* its inferred kind
    # still tells the pipeline something it must act on: one repeated sentence is
    # still free text that needs a text pipeline, and one repeated timestamp is
    # still the temporal column. The ``is_constant`` statistic carries the
    # degeneracy in every case, so nothing is lost.
    if inference.n_unique <= 1 and inference.kind not in _KIND_SURVIVES_CONSTANT:
        inference.reason = (
            f"a single distinct value across {inference.n_non_null:,} non-null rows "
            f"(otherwise reads as {inference.kind.value}: {inference.reason})"
        )
        inference.kind = ColumnKind.CONSTANT
    return inference


def _infer_column(
    name: str,
    series: Any,
    *,
    n_unique: int | None,
    n_non_null: int | None,
) -> ColumnInference:
    dtype = series.dtype
    non_null = series.dropna()
    total_non_null = int(n_non_null if n_non_null is not None else len(non_null))
    try:
        distinct = int(n_unique if n_unique is not None else non_null.nunique())
    except TypeError:  # unhashable cell values
        distinct = int(len(non_null))
    ratio = distinct / total_non_null if total_non_null else 0.0

    base = ColumnInference(
        kind=ColumnKind.UNKNOWN,
        n_non_null=total_non_null,
        n_unique=distinct,
        cardinality_ratio=ratio,
    )

    if total_non_null == 0:
        base.kind = ColumnKind.UNKNOWN
        base.reason = "every value is missing, so nothing can be inferred"
        return base

    near_unique = (
        total_non_null >= MIN_ROWS_FOR_NEAR_UNIQUE and ratio >= NEAR_UNIQUE_RATIO
    )

    if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
        base.kind = ColumnKind.CATEGORICAL_ORDINAL
        base.reason = "pandas ordered Categorical dtype declares an explicit order"
        return base

    if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.PeriodDtype
    ):
        base.kind = ColumnKind.DATETIME
        base.looks_like_datetime = True
        base.reason = f"native temporal dtype ({dtype})"
        return base

    if pd.api.types.is_timedelta64_dtype(dtype):
        base.kind = ColumnKind.NUMERIC_CONTINUOUS
        base.detected_semantic_type = "duration"
        base.reason = "timedelta dtype, treated as a continuous duration"
        return base

    if pd.api.types.is_bool_dtype(dtype):
        base.kind = ColumnKind.BOOLEAN
        base.is_boolean_like = True
        base.reason = "native boolean dtype"
        return base

    # is_numeric_dtype is True for bool, so the bool check above must come first.
    if pd.api.types.is_numeric_dtype(dtype):
        return _infer_numeric(base, name, series, near_unique=near_unique)

    return _infer_string_like(base, name, series, near_unique=near_unique)


def _infer_numeric(
    base: ColumnInference,
    name: str,
    series: Any,
    *,
    near_unique: bool,
) -> ColumnInference:
    array = _numeric_array(series)
    if array.size == 0:
        base.kind = ColumnKind.NUMERIC_CONTINUOUS
        base.reason = "numeric dtype with no finite values"
        return base

    integral = _is_integral(array)
    minimum = float(array.min())
    maximum = float(array.max())
    tokens = name_tokens(name)

    if base.n_unique == 2 and set(np.unique(array)).issubset({0.0, 1.0}):
        base.kind = ColumnKind.BOOLEAN
        base.is_boolean_like = True
        base.reason = "numeric column taking only the two values 0 and 1"
        return base

    is_latitude = bool(tokens & LATITUDE_NAME_TOKENS) and (
        LATITUDE_RANGE[0] <= minimum and maximum <= LATITUDE_RANGE[1]
    )
    is_longitude = bool(tokens & LONGITUDE_NAME_TOKENS) and (
        LONGITUDE_RANGE[0] <= minimum and maximum <= LONGITUDE_RANGE[1]
    )
    if is_latitude or is_longitude:
        base.kind = ColumnKind.GEO
        base.looks_like_geo = True
        base.detected_semantic_type = "latitude" if is_latitude else "longitude"
        base.reason = (
            f"name suggests a coordinate and every value falls inside the valid "
            f"{base.detected_semantic_type} range ([{minimum:g}, {maximum:g}])"
        )
        return base

    monotonic_ints = integral and bool(np.all(np.diff(array) > 0))
    if near_unique and (bool(tokens & ID_NAME_TOKENS) or monotonic_ints):
        base.kind = ColumnKind.IDENTIFIER
        base.looks_like_id = True
        base.reason = (
            f"near-unique numeric column (cardinality ratio "
            f"{base.cardinality_ratio:.3f})"
            + (
                " with an identifier-like name"
                if tokens & ID_NAME_TOKENS
                else " whose values increase strictly monotonically, like a row key"
            )
        )
        return base

    semantic: str | None = None
    if integral and EPOCH_SECONDS_RANGE[0] <= minimum and maximum <= EPOCH_SECONDS_RANGE[1] and (
        tokens & DATE_NAME_TOKENS or tokens & {"epoch", "unix"}
    ):
        semantic = "epoch_seconds"
        base.looks_like_datetime = True
    elif integral and EPOCH_MILLIS_RANGE[0] <= minimum and maximum <= EPOCH_MILLIS_RANGE[1] and (
        tokens & DATE_NAME_TOKENS or tokens & {"epoch", "unix"}
    ):
        semantic = "epoch_millis"
        base.looks_like_datetime = True
    elif integral and tokens & {"year", "yr"} and YEAR_RANGE[0] <= minimum and maximum <= YEAR_RANGE[1]:
        semantic = "year"
        base.looks_like_datetime = True
    elif tokens & PERCENT_NAME_TOKENS and minimum >= 0.0 and maximum <= 100.0:
        semantic = "percentage"
    elif tokens & CURRENCY_NAME_TOKENS:
        semantic = "currency"
    base.detected_semantic_type = semantic

    if integral and (
        base.n_unique <= DISCRETE_MAX_UNIQUE
        or base.cardinality_ratio <= DISCRETE_MAX_RATIO
    ):
        base.kind = ColumnKind.NUMERIC_DISCRETE
        base.reason = (
            f"integer-valued with only {base.n_unique:,} distinct values "
            f"(cardinality ratio {base.cardinality_ratio:.4f}), so it behaves as a count "
            "or an encoded level rather than a continuous measurement"
        )
        return base

    base.kind = ColumnKind.NUMERIC_CONTINUOUS
    base.reason = (
        f"{'integer' if integral else 'real'}-valued with {base.n_unique:,} distinct "
        f"values spanning [{minimum:g}, {maximum:g}]"
    )
    return base


def _infer_string_like(
    base: ColumnInference,
    name: str,
    series: Any,
    *,
    near_unique: bool,
) -> ColumnInference:
    values = string_sample(series)
    tokens = name_tokens(name)
    uniques = _lowered_uniques(values)
    mean_tokens, mean_length = _token_stats(values)
    base.mean_token_count = mean_tokens

    if _match_boolean_vocabulary(uniques):
        base.kind = ColumnKind.BOOLEAN
        base.is_boolean_like = True
        base.reason = f"two values drawn from a boolean vocabulary ({sorted(uniques or [])})"
        return base

    parsed, parse_fraction = try_parse_datetime(series, name)
    if parsed is not None:
        base.kind = ColumnKind.DATETIME
        base.looks_like_datetime = True
        base.parsed_datetime = parsed
        base.detected_semantic_type = "datetime_string"
        base.reason = (
            f"{parse_fraction:.1%} of non-null values parse as timestamps despite the "
            f"{series.dtype} dtype"
        )
        return base

    semantic = detect_semantic_type(values, name)
    base.detected_semantic_type = semantic

    if semantic == "uuid":
        base.kind = ColumnKind.IDENTIFIER
        base.looks_like_id = True
        base.reason = "values match the UUID format, which is an opaque key"
        return base

    if semantic in {"email", "phone", "url"} and near_unique:
        base.kind = ColumnKind.IDENTIFIER
        base.looks_like_id = True
        base.reason = (
            f"near-unique {semantic} values (cardinality ratio "
            f"{base.cardinality_ratio:.3f}) identify individual records"
        )
        return base

    if semantic == "postal_code" or (tokens & GEO_NAME_TOKENS):
        base.kind = ColumnKind.GEO
        base.looks_like_geo = True
        base.reason = (
            "values match a postal-code format"
            if semantic == "postal_code"
            else f"name refers to a geographic entity ({sorted(tokens & GEO_NAME_TOKENS)})"
        )
        return base

    is_texty = mean_tokens is not None and (
        mean_tokens >= TEXT_STRONG_TOKENS
        or (
            mean_tokens >= TEXT_MIN_TOKENS
            and (
                base.cardinality_ratio >= TEXT_MIN_CARDINALITY_RATIO
                or base.n_unique >= TEXT_MIN_UNIQUE
            )
        )
    )
    if is_texty:
        base.kind = ColumnKind.TEXT
        base.looks_like_text = True
        base.reason = (
            f"averages {mean_tokens:.1f} whitespace tokens over "
            f"{base.n_unique:,} distinct values (mean length {mean_length:.0f} chars), "
            "which reads as free text rather than a label"
        )
        return base

    if near_unique and (
        bool(tokens & ID_NAME_TOKENS) or base.n_unique == base.n_non_null
    ):
        base.kind = ColumnKind.IDENTIFIER
        base.looks_like_id = True
        base.reason = (
            f"cardinality ratio {base.cardinality_ratio:.3f} with "
            + (
                "an identifier-like name"
                if tokens & ID_NAME_TOKENS
                else "one distinct value per row"
            )
        )
        return base

    ordinal_vocabulary = _match_ordinal_vocabulary(uniques)
    if ordinal_vocabulary is not None:
        base.kind = ColumnKind.CATEGORICAL_ORDINAL
        base.reason = (
            "values are a subset of the known ordered vocabulary "
            f"{list(ordinal_vocabulary)}"
        )
        return base
    if tokens & ORDINAL_NAME_TOKENS and base.n_unique <= ORDINAL_MAX_UNIQUE:
        base.kind = ColumnKind.CATEGORICAL_ORDINAL
        base.reason = (
            f"name implies a ranked scale ({sorted(tokens & ORDINAL_NAME_TOKENS)}) over "
            f"only {base.n_unique} levels"
        )
        return base

    base.kind = ColumnKind.CATEGORICAL_NOMINAL
    base.looks_like_text = bool(tokens & TEXT_NAME_TOKENS) and (mean_tokens or 0) >= 2.0
    base.reason = (
        f"{base.n_unique:,} distinct short labels (mean "
        f"{mean_tokens if mean_tokens is not None else 0:.1f} tokens) with no detected order"
    )
    return base


__all__ = [
    "BOOLEAN_VOCABULARIES",
    "ColumnInference",
    "ORDINAL_VOCABULARIES",
    "POST_OUTCOME_NAME_FRAGMENTS",
    "SEMANTIC_PATTERNS",
    "detect_semantic_type",
    "has_name_hint",
    "infer_column",
    "match_fraction",
    "name_tokens",
    "string_sample",
    "suspicious_outcome_name",
    "try_parse_datetime",
]
