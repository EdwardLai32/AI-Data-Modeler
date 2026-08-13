"""Per-column and whole-frame statistics.

Every function here is deterministic, returns plain Python/pydantic values, and
degrades to ``None`` rather than raising on a column it cannot measure. That
contract matters because the profiler runs these across arbitrary user data:
a single unhashable cell or an all-NaN column must not take down the run.

The association measures (:func:`cramers_v`, :func:`correlation_ratio`) live here
rather than in the profiler because leakage detection needs them too, and both
callers must use the identical definition for the numbers in a report to agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.schemas import (
    CategoryCount,
    ColumnKind,
    OutlierSummary,
    Quantiles,
    TargetSummary,
)

# --- tunable thresholds ---------------------------------------------------

TOP_K_VALUES = 12
OUTLIER_IQR_MULTIPLIER = 1.5
NEAR_ZERO_VARIANCE_FREQ_RATIO = 19.0
NEAR_ZERO_VARIANCE_UNIQUE_RATIO = 0.10
NEAR_ZERO_COEFFICIENT_OF_VARIATION = 1e-4
MIXED_TYPE_SAMPLE = 2_000
MAX_VALUE_LABEL_CHARS = 100
IMBALANCE_RATIO_THRESHOLD = 3.0
QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
GAP_MULTIPLIER = 1.5


@dataclass
class FrameOverview:
    """Whole-table integrity counts.

    Attributes:
        memory_bytes: Deep memory footprint including the index.
        n_duplicate_rows: Fully duplicated rows (first occurrence excluded).
        duplicate_fraction: ``n_duplicate_rows / n_rows``.
        total_missing_cells: Count of null cells across the frame.
        missing_cell_fraction: ``total_missing_cells / (n_rows * n_columns)``.
        notes: Degradation notes, e.g. an unhashable column blocking the
            duplicate scan.
    """

    memory_bytes: int = 0
    n_duplicate_rows: int = 0
    duplicate_fraction: float = 0.0
    total_missing_cells: int = 0
    missing_cell_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)


def safe_float(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` for NaN/inf/non-numeric input."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def coerce_numeric_text(series: Any) -> Any:
    """Numeric view of a column whose numbers were exported as text.

    Thousands separators are stripped first, since ``pd.to_numeric`` rejects
    ``"1,234"``. A column that survives this is one the cleaning agent should
    cast, and reporting its distribution is what lets the agent argue for the
    cast rather than guess at it.

    Args:
        series: A string/object column.

    Returns:
        A float series with unparseable entries as NaN.
    """
    try:
        text = series.astype("string").str.replace(",", "", regex=False)
        return pd.to_numeric(text, errors="coerce")
    except (TypeError, ValueError):  # pragma: no cover
        return pd.Series(dtype="float64")


def _numeric_series(series: Any) -> Any:
    """Numeric view of a series with non-numeric entries coerced to NaN."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype("float64")
    try:
        return pd.to_numeric(series, errors="coerce")
    except (ValueError, TypeError):
        return pd.Series(dtype="float64")


def label_of(value: Any) -> str:
    """Render a cell value as a bounded, printable label."""
    text = "<NA>" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > MAX_VALUE_LABEL_CHARS:
        text = text[: MAX_VALUE_LABEL_CHARS - 3] + "..."
    return text


def basic_counts(series: Any, n_rows: int) -> dict[str, Any]:
    """Missingness, cardinality, and memory for one column.

    These are measured on the *full* column even when the rest of the profile
    samples, because they are cheap single passes and because cardinality drives
    kind inference — a sampled ``n_unique`` would misclassify identifiers.

    Args:
        series: The full column.
        n_rows: True row count of the frame.

    Returns:
        Dict with ``n_missing``, ``missing_fraction``, ``n_unique``,
        ``cardinality_ratio``, ``is_constant``, ``memory_bytes``, ``n_non_null``.
    """
    n_missing = int(series.isna().sum())
    n_non_null = max(0, int(len(series)) - n_missing)
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:  # unhashable values (lists, dicts) — fall back to a string view
        try:
            n_unique = int(series.dropna().astype("string").nunique())
        except Exception:  # pragma: no cover - genuinely unmeasurable
            n_unique = n_non_null
    try:
        memory_bytes = int(series.memory_usage(index=False, deep=True))
    except (TypeError, ValueError):  # pragma: no cover
        memory_bytes = 0
    return {
        "n_missing": n_missing,
        "missing_fraction": (n_missing / n_rows) if n_rows else 0.0,
        "n_unique": n_unique,
        "cardinality_ratio": (n_unique / n_non_null) if n_non_null else 0.0,
        "is_constant": n_unique <= 1,
        "memory_bytes": memory_bytes,
        "n_non_null": n_non_null,
    }


def numeric_summary(series: Any) -> dict[str, Any]:
    """Central tendency, spread, shape, and sign statistics.

    Args:
        series: Numeric (or numeric-coercible) column.

    Returns:
        Dict with ``mean``, ``std``, ``variance``, ``minimum``, ``maximum``,
        ``skewness``, ``kurtosis``, ``zero_fraction``, ``negative_fraction``.
        Any statistic that cannot be computed is ``None``.
    """
    numeric = _numeric_series(series).dropna()
    empty = {
        "mean": None, "std": None, "variance": None, "minimum": None,
        "maximum": None, "skewness": None, "kurtosis": None,
        "zero_fraction": None, "negative_fraction": None,
    }
    if len(numeric) == 0:
        return empty

    count = int(len(numeric))
    out = dict(empty)
    out["mean"] = safe_float(numeric.mean())
    out["minimum"] = safe_float(numeric.min())
    out["maximum"] = safe_float(numeric.max())
    out["zero_fraction"] = float((numeric == 0).sum()) / count
    out["negative_fraction"] = float((numeric < 0).sum()) / count
    if count >= 2:
        out["std"] = safe_float(numeric.std())
        out["variance"] = safe_float(numeric.var())
    else:
        out["std"] = 0.0
        out["variance"] = 0.0
    # skew needs 3 points, excess kurtosis needs 4; pandas returns NaN otherwise
    # and safe_float turns that into None.
    if count >= 3:
        out["skewness"] = safe_float(numeric.skew())
    if count >= 4:
        out["kurtosis"] = safe_float(numeric.kurt())
    return out


def quantile_summary(series: Any) -> Quantiles | None:
    """All seven reported quantiles, or ``None`` for a non-numeric column."""
    numeric = _numeric_series(series).dropna()
    if len(numeric) == 0:
        return None
    try:
        values = numeric.quantile(list(QUANTILE_LEVELS))
    except (ValueError, TypeError):  # pragma: no cover
        return None
    return Quantiles(
        p01=safe_float(values.iloc[0]),
        p05=safe_float(values.iloc[1]),
        p25=safe_float(values.iloc[2]),
        p50=safe_float(values.iloc[3]),
        p75=safe_float(values.iloc[4]),
        p95=safe_float(values.iloc[5]),
        p99=safe_float(values.iloc[6]),
    )


def outlier_summary(
    series: Any, *, multiplier: float = OUTLIER_IQR_MULTIPLIER
) -> OutlierSummary | None:
    """Tukey IQR fences and the count of values outside them.

    A zero IQR (more than half the mass on one value) yields no outliers rather
    than flagging every distinct value, which is the degenerate result the naive
    formula gives.

    Args:
        series: Numeric column.
        multiplier: IQR multiplier for the fences.

    Returns:
        An :class:`OutlierSummary`, or ``None`` if the column is not numeric.
    """
    numeric = _numeric_series(series).dropna()
    if len(numeric) < 4:
        return None
    q1 = safe_float(numeric.quantile(0.25))
    q3 = safe_float(numeric.quantile(0.75))
    if q1 is None or q3 is None:
        return None
    iqr = q3 - q1
    if iqr <= 0:
        return OutlierSummary(
            method="iqr", n_outliers=0, fraction=0.0, lower_bound=q1, upper_bound=q3
        )
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    n_outliers = int(((numeric < lower) | (numeric > upper)).sum())
    return OutlierSummary(
        method="iqr",
        n_outliers=n_outliers,
        fraction=n_outliers / len(numeric),
        lower_bound=lower,
        upper_bound=upper,
    )


def top_value_counts(series: Any, k: int = TOP_K_VALUES) -> list[CategoryCount]:
    """The ``k`` most frequent values with their share of non-null rows."""
    try:
        counts = series.value_counts(dropna=True)
    except TypeError:  # unhashable values
        try:
            counts = series.dropna().astype("string").value_counts()
        except Exception:  # pragma: no cover
            return []
    total = int(counts.sum())
    if total == 0:
        return []
    out: list[CategoryCount] = []
    for value, count in counts.head(k).items():
        out.append(
            CategoryCount(
                value=label_of(value), count=int(count), fraction=int(count) / total
            )
        )
    return out


def frequency_ratio(series: Any) -> float | None:
    """Ratio of the most common value's count to the second most common.

    Kuhn's near-zero-variance diagnostic: a large ratio means one value dominates
    and the column carries almost no usable signal.
    """
    try:
        counts = series.value_counts(dropna=True)
    except TypeError:
        return None
    if len(counts) < 2:
        return math.inf if len(counts) == 1 else None
    second = float(counts.iloc[1])
    if second <= 0:  # pragma: no cover
        return math.inf
    return float(counts.iloc[0]) / second


def near_zero_variance(
    series: Any, *, n_unique: int, n_non_null: int, numeric_stats: dict[str, Any] | None
) -> bool:
    """Whether a column is effectively constant for modelling purposes.

    Two independent tests, either of which fires:

    1.  Kuhn's frequency-ratio rule — one value dominates (>19:1 over the runner
        up) and distinct values are under 10% of rows.
    2.  For numerics, a coefficient of variation below 1e-4, which catches
        columns that vary only in floating-point noise.
    """
    if n_non_null == 0 or n_unique <= 1:
        return True
    unique_ratio = n_unique / n_non_null
    ratio = frequency_ratio(series)
    if (
        ratio is not None
        and ratio > NEAR_ZERO_VARIANCE_FREQ_RATIO
        and unique_ratio < NEAR_ZERO_VARIANCE_UNIQUE_RATIO
    ):
        return True
    if numeric_stats:
        std = numeric_stats.get("std")
        mean = numeric_stats.get("mean")
        if std is not None and std == 0.0:
            return True
        if std is not None and mean not in (None, 0.0):
            if abs(std / mean) < NEAR_ZERO_COEFFICIENT_OF_VARIATION:
                return True
    return False


def string_summary(series: Any) -> dict[str, Any]:
    """Character-length and whitespace-token statistics for a text-ish column.

    Returns:
        Dict with ``mean_string_length``, ``max_string_length``,
        ``mean_token_count``, ``n_blank`` (values that are empty or whitespace),
        and ``n_untrimmed`` (values with leading/trailing whitespace).
    """
    out: dict[str, Any] = {
        "mean_string_length": None,
        "max_string_length": None,
        "mean_token_count": None,
        "n_blank": 0,
        "n_untrimmed": 0,
    }
    try:
        text = series.dropna().astype("string")
    except (TypeError, ValueError):  # pragma: no cover
        return out
    if len(text) == 0:
        return out
    lengths = text.str.len()
    out["mean_string_length"] = safe_float(lengths.mean())
    maximum = safe_float(lengths.max())
    out["max_string_length"] = int(maximum) if maximum is not None else None
    # str.count on a non-whitespace pattern avoids materialising a list column,
    # which matters on wide frames.
    out["mean_token_count"] = safe_float(text.str.count(r"\S+").mean())
    stripped = text.str.strip()
    out["n_blank"] = int((stripped.str.len() == 0).sum())
    out["n_untrimmed"] = int((stripped != text).sum())
    return out


def _pretty_timedelta(delta: Any) -> str:
    """Compact human label for a modal gap, e.g. ``1D``, ``15min``, ``2h``."""
    try:
        seconds = float(pd.Timedelta(delta).total_seconds())
    except (ValueError, TypeError):  # pragma: no cover
        return str(delta)
    if seconds <= 0:
        return "0s"
    for size, unit in ((86_400 * 365, "Y"), (86_400 * 30, "M"), (86_400 * 7, "W"), (86_400, "D"), (3600, "h"), (60, "min")):
        if seconds >= size and abs(seconds / size - round(seconds / size)) < 0.05:
            return f"{int(round(seconds / size))}{unit}"
    return f"{seconds:g}s"


def datetime_summary(series: Any, *, order_series: Any | None = None) -> dict[str, Any]:
    """Range, frequency, gap count, and monotonicity of a temporal column.

    Args:
        series: Parsed datetime column used for range/frequency work.
        order_series: Optional column whose original row order should be used for
            the monotonicity test. Defaults to ``series``.

    Returns:
        Dict with ``min_timestamp``, ``max_timestamp``, ``inferred_frequency``,
        ``n_gaps``, ``is_monotonic``, ``is_monotonic_increasing``, ``span_days``.
    """
    out: dict[str, Any] = {
        "min_timestamp": None,
        "max_timestamp": None,
        "inferred_frequency": None,
        "n_gaps": None,
        "is_monotonic": None,
        "is_monotonic_increasing": None,
        "span_days": None,
    }
    try:
        values = series.dropna()
    except AttributeError:  # pragma: no cover
        return out
    if len(values) == 0:
        return out

    ordered = (order_series if order_series is not None else series).dropna()
    try:
        increasing = bool(ordered.is_monotonic_increasing)
        decreasing = bool(ordered.is_monotonic_decreasing)
    except (TypeError, ValueError):  # pragma: no cover
        increasing = decreasing = False
    out["is_monotonic_increasing"] = increasing
    out["is_monotonic"] = increasing or decreasing

    sorted_unique = pd.DatetimeIndex(pd.Series(values.unique()).sort_values())
    out["min_timestamp"] = sorted_unique[0].isoformat()
    out["max_timestamp"] = sorted_unique[-1].isoformat()
    out["span_days"] = safe_float(
        (sorted_unique[-1] - sorted_unique[0]).total_seconds() / 86_400
    )

    if len(sorted_unique) < 2:
        out["n_gaps"] = 0
        return out

    try:
        inferred = pd.infer_freq(sorted_unique)
    except (ValueError, TypeError):
        inferred = None

    diffs = pd.Series(sorted_unique).diff().dropna()
    modal = None
    if len(diffs):
        modes = diffs.mode()
        if len(modes):
            modal = modes.iloc[0]
    if inferred:
        out["inferred_frequency"] = inferred
    elif modal is not None:
        # No calendar-regular frequency, but the modal spacing still tells an
        # agent whether this is daily, hourly, or event-driven data.
        out["inferred_frequency"] = f"irregular (modal gap {_pretty_timedelta(modal)})"

    if modal is not None and pd.Timedelta(modal).total_seconds() > 0:
        out["n_gaps"] = int((diffs > GAP_MULTIPLIER * modal).sum())
    else:
        out["n_gaps"] = 0
    return out


def mixed_type_names(series: Any, sample_size: int = MIXED_TYPE_SAMPLE) -> list[str]:
    """Distinct Python type names present in an object column.

    Only ``object`` columns can be genuinely mixed under pandas 3, where strings
    get their own dtype; everything else returns an empty list.
    """
    if not pd.api.types.is_object_dtype(series.dtype):
        return []
    values = series.dropna()
    if len(values) > sample_size:
        values = values.iloc[:sample_size]
    if len(values) == 0:
        return []
    try:
        names = {type(value).__name__ for value in values.to_numpy(dtype=object, copy=False)}
    except (TypeError, ValueError):  # pragma: no cover
        return []
    # int/float mixing is benign — pandas would have unified them if it mattered.
    if names <= {"int", "float", "bool"}:
        return []
    return sorted(names)


def dataframe_overview(df: Any) -> FrameOverview:
    """Memory, duplicate rows, and missing-cell counts for the whole frame."""
    overview = FrameOverview()
    n_rows = int(len(df))
    n_columns = int(df.shape[1])

    try:
        overview.memory_bytes = int(df.memory_usage(index=True, deep=True).sum())
    except (TypeError, ValueError):  # pragma: no cover
        overview.notes.append("memory footprint could not be measured")

    try:
        overview.total_missing_cells = int(df.isna().to_numpy().sum())
    except (TypeError, ValueError):  # pragma: no cover
        overview.notes.append("missing-cell count could not be measured")
    cells = n_rows * n_columns
    overview.missing_cell_fraction = (
        overview.total_missing_cells / cells if cells else 0.0
    )

    try:
        overview.n_duplicate_rows = int(df.duplicated().sum())
    except (TypeError, ValueError):
        # Unhashable cells (lists, dicts) break hashing; a stringified view still
        # answers the question, just more slowly.
        try:
            overview.n_duplicate_rows = int(
                df.astype("string").duplicated().sum()
            )
            overview.notes.append(
                "duplicate rows were counted on a stringified copy of the frame"
            )
        except Exception:  # pragma: no cover
            overview.notes.append("duplicate rows could not be counted")
    overview.duplicate_fraction = (
        overview.n_duplicate_rows / n_rows if n_rows else 0.0
    )
    return overview


def build_target_summary(
    name: str,
    series: Any,
    kind: ColumnKind,
    *,
    classification: bool,
    max_classes: int = 50,
) -> TargetSummary:
    """Summarise the target column, including class balance.

    Args:
        name: Target column name.
        series: The full target column.
        kind: Inferred kind of the target.
        classification: Whether to compute class counts and imbalance.
        max_classes: Cap on reported class counts.

    Returns:
        A populated :class:`TargetSummary`.
    """
    summary = TargetSummary(name=name, kind=kind)
    summary.n_missing = int(series.isna().sum())
    non_null = series.dropna()

    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
        series.dtype
    ):
        stats = numeric_summary(series)
        summary.mean = stats["mean"]
        summary.std = stats["std"]
        summary.skewness = stats["skewness"]

    if not classification:
        return summary

    try:
        counts = non_null.value_counts()
    except TypeError:  # pragma: no cover
        return summary
    summary.n_classes = int(len(counts))
    total = int(counts.sum())
    if total == 0:
        return summary
    summary.class_counts = [
        CategoryCount(value=label_of(value), count=int(count), fraction=int(count) / total)
        for value, count in counts.head(max_classes).items()
    ]
    minority = float(counts.min())
    if minority > 0:
        ratio = float(counts.max()) / minority
        summary.imbalance_ratio = ratio
        summary.is_imbalanced = ratio > IMBALANCE_RATIO_THRESHOLD
    return summary


# --- association measures -------------------------------------------------


def cramers_v(left: Any, right: Any) -> float | None:
    """Bias-corrected Cramer's V between two categorical columns.

    Uses the Bergsma correction, without which V is inflated on small samples
    with many levels — exactly the shape where a spurious "strong association"
    would mislead a leakage judgement.

    Args:
        left: First categorical column.
        right: Second categorical column, positionally aligned with ``left``.

    Returns:
        V in ``[0, 1]``, or ``None`` when the contingency table is degenerate.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return None
    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)
    if left_values.shape != right_values.shape:
        return None
    frame = pd.DataFrame(
        {"left": pd.Series(left_values), "right": pd.Series(right_values)}
    ).dropna()
    if len(frame) < 4:
        return None
    try:
        table = pd.crosstab(frame["left"], frame["right"])
    except (TypeError, ValueError):  # pragma: no cover
        return None
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    n = float(table.to_numpy().sum())
    if n <= 1:
        return None
    try:
        chi2 = float(chi2_contingency(table.to_numpy(), correction=False)[0])
    except (ValueError, ZeroDivisionError):  # pragma: no cover
        return None
    rows, cols = table.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corrected = cols - ((cols - 1) ** 2) / (n - 1)
    denominator = max(min(cols_corrected - 1, rows_corrected - 1), 1e-12)
    return float(min(1.0, math.sqrt(phi2_corrected / denominator)))


def correlation_ratio(categories: Any, values: Any) -> float | None:
    """Bias-corrected correlation ratio between a categorical and numeric column.

    The raw ``eta`` — the square root of the variance share explained by group
    membership — is badly upward-biased when levels are numerous relative to
    rows, and reaches exactly 1.0 for a column with one level per row. Reporting
    that would label every identifier column a perfect predictor. This returns
    Kelley's epsilon instead, which subtracts the variance a random grouping of
    the same shape would explain, and is ``None`` when there is no within-group
    variance left to estimate at all. The result is on the same 0-1 scale as an
    absolute correlation, so it can be ranked alongside one.

    Args:
        categories: Grouping column.
        values: Numeric column, positionally aligned with ``categories``.

    Returns:
        Bias-corrected eta in ``[0, 1]``, or ``None`` when it is undefined.
    """
    category_values = np.asarray(categories, dtype=object)
    numeric_values = np.asarray(values)
    if category_values.shape != numeric_values.shape:
        return None
    cats = pd.Series(category_values)
    nums = pd.to_numeric(pd.Series(numeric_values), errors="coerce")
    frame = pd.DataFrame({"c": cats, "v": nums}).dropna()
    n = len(frame)
    k = int(frame["c"].nunique()) if n else 0
    if n < 4 or k < 2 or n - k < 1:
        return None
    grand_mean = float(frame["v"].mean())
    grouped = frame.groupby("c", observed=True)["v"]
    between = float((((grouped.mean() - grand_mean) ** 2) * grouped.count()).sum())
    total = float(((frame["v"] - grand_mean) ** 2).sum())
    if total <= 0:
        return None
    mean_square_within = (total - between) / (n - k)
    epsilon_squared = (between - (k - 1) * mean_square_within) / total
    return float(min(1.0, math.sqrt(max(0.0, epsilon_squared))))


def correlation_matrices(
    numeric_df: Any, *, min_periods: int = 8
) -> tuple[Any | None, Any | None]:
    """Pearson and Spearman matrices for a numeric frame.

    Spearman is computed as Pearson on ranks rather than via
    ``method="spearman"``: ranking once up front is dramatically cheaper than
    pandas' pairwise rank computation on wide frames, and gives the identical
    coefficient for average-ranked ties.

    Args:
        numeric_df: Frame of numeric columns.
        min_periods: Minimum overlapping observations per pair.

    Returns:
        ``(pearson, spearman)`` frames, either of which may be ``None``.
    """
    if numeric_df is None or numeric_df.shape[1] < 2:
        return None, None
    pearson: Any | None
    spearman: Any | None
    try:
        pearson = numeric_df.corr(method="pearson", min_periods=min_periods)
    except (ValueError, TypeError):  # pragma: no cover
        pearson = None
    try:
        spearman = numeric_df.rank(numeric_only=True).corr(
            method="pearson", min_periods=min_periods
        )
    except (ValueError, TypeError):  # pragma: no cover
        spearman = None
    return pearson, spearman


def _paired_numeric(left: Any, right: Any) -> Any | None:
    """Two positionally-aligned numeric columns as a frame, or ``None``.

    Both inputs are stripped to raw arrays first so a pandas index can never
    join rows that do not correspond, and a length mismatch is rejected outright
    rather than being padded with NaN — either would silently return a
    coefficient measured on the wrong pairs.
    """
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        return None
    frame = pd.DataFrame(
        {
            "a": pd.to_numeric(pd.Series(left_values), errors="coerce"),
            "b": pd.to_numeric(pd.Series(right_values), errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return None
    return frame


def pearson(left: Any, right: Any) -> float | None:
    """Pearson correlation of two numeric columns, or ``None``."""
    frame = _paired_numeric(left, right)
    if frame is None:
        return None
    return safe_float(frame["a"].corr(frame["b"]))


def spearman(left: Any, right: Any) -> float | None:
    """Spearman rank correlation of two numeric columns, or ``None``."""
    frame = _paired_numeric(left, right)
    if frame is None:
        return None
    return safe_float(frame["a"].rank().corr(frame["b"].rank()))


__all__ = [
    "FrameOverview",
    "basic_counts",
    "build_target_summary",
    "coerce_numeric_text",
    "correlation_matrices",
    "correlation_ratio",
    "cramers_v",
    "dataframe_overview",
    "datetime_summary",
    "frequency_ratio",
    "label_of",
    "mixed_type_names",
    "near_zero_variance",
    "numeric_summary",
    "outlier_summary",
    "pearson",
    "quantile_summary",
    "safe_float",
    "spearman",
    "string_summary",
    "top_value_counts",
]
