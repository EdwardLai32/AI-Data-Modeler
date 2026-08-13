"""The deterministic profiling engine.

:func:`profile_dataframe` is the only entry point the pipeline calls. It produces
the :class:`~automl_architect.core.schemas.DatasetProfile` that every agent
reasons over, which makes it the anti-hallucination foundation of the whole
system: an agent can only ground a claim in a number that appears here.

Three design rules follow from that:

*   **Sample for cost, report the truth.** Expensive passes (correlations,
    outliers, mutual information, regex work) run on at most
    ``settings.max_profile_rows`` rows, while row counts, missingness, and
    cardinality are always measured on the full frame. ``n_rows`` is never the
    sample size.
*   **One column may never take down the profile.** Every per-column computation
    is wrapped; a failure degrades that column to
    :attr:`ColumnKind.UNKNOWN` and records an INFO quality issue.
*   **Mixed-type tables are first-class.** Pearson and Spearman cover
    numeric pairs, bias-corrected Cramer's V covers categorical pairs, and the
    correlation ratio covers the mixed case, so a table of strings and dates is
    not silently reported as having no relationships.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from ..core.errors import ValidationError
from ..core.schemas import (
    ColumnKind,
    ColumnProfile,
    CorrelationPair,
    DataQualityIssue,
    DatasetProfile,
    LeakageFinding,
    Severity,
)
from . import stats as st
from .leakage import (
    TargetAssociation,
    detect_leakage,
    detect_quality_issues,
    score_target_associations,
    target_is_classification_like,
)
from .semantic import ColumnInference, infer_column, try_parse_datetime

logger = logging.getLogger(__name__)

# --- budgets --------------------------------------------------------------

MAX_NUMERIC_CORRELATION_COLUMNS = 150
MAX_CATEGORICAL_CORRELATION_COLUMNS = 30
MAX_CATEGORICAL_CORRELATION_LEVELS = 50
MULTICOLLINEARITY_THRESHOLD = 0.95
MULTICOLLINEARITY_DECIMALS = 2
MAX_TOP_CORRELATIONS = 40
MAX_TARGET_CORRELATIONS = 60
MIN_CORRELATION_PERIODS = 8
NONLINEARITY_GAP = 0.10
TOP_VALUES_MAX_UNIQUE = 200
IDENTIFIER_TOP_VALUES = 3

_NUMERIC_KINDS = frozenset(
    {
        ColumnKind.NUMERIC_CONTINUOUS,
        ColumnKind.NUMERIC_DISCRETE,
        ColumnKind.BOOLEAN,
    }
)
_CATEGORICAL_KINDS = frozenset(
    {
        ColumnKind.CATEGORICAL_NOMINAL,
        ColumnKind.CATEGORICAL_ORDINAL,
        ColumnKind.BOOLEAN,
        ColumnKind.GEO,
    }
)
_ALWAYS_TOP_VALUE_KINDS = frozenset(
    {
        ColumnKind.CATEGORICAL_NOMINAL,
        ColumnKind.CATEGORICAL_ORDINAL,
        ColumnKind.BOOLEAN,
        ColumnKind.GEO,
        ColumnKind.TEXT,
        ColumnKind.IDENTIFIER,
        ColumnKind.CONSTANT,
        ColumnKind.NUMERIC_DISCRETE,
    }
)


def dataset_fingerprint(df: Any) -> str:
    """Stable content-derived id for a frame.

    The profile contract does not pass a ``dataset_id`` in, so one is derived
    from the structure (shape, column names, dtypes). Deterministic for the same
    table, which keeps a re-profiled run comparable.

    Args:
        df: The dataframe.

    Returns:
        An id of the form ``ds_<12 hex chars>``.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(str(df.shape).encode("utf-8"))
    for name, dtype in zip(df.columns, df.dtypes, strict=False):
        digest.update(f"{name}:{dtype}|".encode("utf-8"))
    return f"ds_{digest.hexdigest()[:12]}"


def _unique_names(columns: Any) -> tuple[list[str], list[str]]:
    """De-duplicated display names, plus the duplicated originals.

    Profiles are keyed by name, so two columns called ``value`` would collide.
    The second occurrence becomes ``value__2``.
    """
    seen: dict[str, int] = {}
    names: list[str] = []
    duplicates: list[str] = []
    for raw in columns:
        name = str(raw)
        if name in seen:
            seen[name] += 1
            duplicates.append(name)
            names.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 1
            names.append(name)
    return names, sorted(set(duplicates))


def _wants_top_values(kind: ColumnKind, n_unique: int) -> bool:
    return kind in _ALWAYS_TOP_VALUE_KINDS or n_unique <= TOP_VALUES_MAX_UNIQUE


def is_collinear(coefficient: float) -> bool:
    """Whether a coefficient counts as multicollinear.

    The fence is 0.95, applied to ``|r|`` rounded to two decimals. Rounding first
    is deliberate: a pair at 0.9454 is one a reader would write down as 0.95, and
    dropping it on the third decimal hides a real redundancy behind
    floating-point hair-splitting.
    """
    return round(abs(coefficient), MULTICOLLINEARITY_DECIMALS) >= MULTICOLLINEARITY_THRESHOLD


def _timestamp_bounds(series: Any) -> tuple[str, str] | None:
    """``(min, max)`` as ISO strings for an already-temporal series, else ``None``.

    Deliberately a single min/max pass rather than
    :func:`~automl_architect.profiling.stats.datetime_summary`: this is used to
    correct the reported range of a sampled column, and it must not cost a sort of
    the full table. ``None`` for anything that would need parsing, so the caller
    can fall back to the sample.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.PeriodDtype):
        series = series.dt.to_timestamp()
    elif not pd.api.types.is_datetime64_any_dtype(dtype):
        return None
    try:
        low = series.min()
        high = series.max()
    except (TypeError, ValueError):  # pragma: no cover
        return None
    if pd.isna(low) or pd.isna(high):
        return None
    return pd.Timestamp(low).isoformat(), pd.Timestamp(high).isoformat()


def _build_column_profile(
    name: str,
    full_series: Any,
    sample_series: Any,
    *,
    n_rows: int,
    dt_series: Any | None,
) -> tuple[ColumnProfile, ColumnInference, dict[str, Any]]:
    """Profile one column. Returns the profile, its inference, and side facts."""
    counts = st.basic_counts(full_series, n_rows)
    inference = infer_column(
        name,
        sample_series,
        n_unique=counts["n_unique"],
        n_non_null=counts["n_non_null"],
    )

    profile = ColumnProfile(
        name=name,
        kind=inference.kind,
        dtype=str(full_series.dtype),
        n_missing=counts["n_missing"],
        missing_fraction=counts["missing_fraction"],
        n_unique=counts["n_unique"],
        cardinality_ratio=counts["cardinality_ratio"],
        is_constant=counts["is_constant"],
        memory_bytes=counts["memory_bytes"],
        looks_like_id=inference.looks_like_id,
        looks_like_geo=inference.looks_like_geo,
        looks_like_text=inference.looks_like_text,
        looks_like_datetime=inference.looks_like_datetime,
        detected_semantic_type=inference.detected_semantic_type,
    )

    side: dict[str, Any] = {}
    dtype = sample_series.dtype
    numeric_like = pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(
        dtype
    )

    is_timedelta = pd.api.types.is_timedelta64_dtype(dtype)
    # A column of numbers exported as text still *has* a distribution, and the
    # cleaning agent needs it to argue for the cast rather than assume one.
    numeric_source = sample_series if numeric_like else None
    if numeric_source is None and is_timedelta:
        # Durations are quantities; seconds is the unit every downstream model
        # can consume, and the dtype string still records the original unit.
        numeric_source = sample_series.dt.total_seconds()
    if numeric_source is None and inference.detected_semantic_type == "numeric_as_text":
        coerced = st.coerce_numeric_text(sample_series)
        if len(coerced) and float(coerced.notna().mean()) >= 0.9:
            numeric_source = coerced
            side["numeric_as_text"] = True

    numeric_stats: dict[str, Any] | None = None
    if numeric_source is not None:
        numeric_stats = st.numeric_summary(numeric_source)
        profile.mean = numeric_stats["mean"]
        profile.std = numeric_stats["std"]
        profile.variance = numeric_stats["variance"]
        profile.minimum = numeric_stats["minimum"]
        profile.maximum = numeric_stats["maximum"]
        profile.skewness = numeric_stats["skewness"]
        profile.kurtosis = numeric_stats["kurtosis"]
        profile.zero_fraction = numeric_stats["zero_fraction"]
        profile.negative_fraction = numeric_stats["negative_fraction"]
        profile.quantiles = st.quantile_summary(numeric_source)
        profile.outliers = st.outlier_summary(numeric_source)

    profile.is_near_zero_variance = st.near_zero_variance(
        sample_series,
        n_unique=counts["n_unique"],
        n_non_null=counts["n_non_null"],
        numeric_stats=numeric_stats,
    )

    text_like = not numeric_like and not (
        is_timedelta
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    )
    if text_like:
        string_stats = st.string_summary(sample_series)
        profile.mean_string_length = string_stats["mean_string_length"]
        profile.max_string_length = string_stats["max_string_length"]
        profile.mean_token_count = string_stats["mean_token_count"]
        side["n_blank"] = string_stats["n_blank"]
        side["n_untrimmed"] = string_stats["n_untrimmed"]

    if _wants_top_values(inference.kind, counts["n_unique"]):
        limit = (
            IDENTIFIER_TOP_VALUES
            if inference.kind is ColumnKind.IDENTIFIER
            else st.TOP_K_VALUES
        )
        profile.top_values = st.top_value_counts(sample_series, limit)

    temporal = None
    if pd.api.types.is_datetime64_any_dtype(dtype):
        temporal = sample_series
    elif isinstance(dtype, pd.PeriodDtype):
        temporal = sample_series.dt.to_timestamp()
    elif inference.parsed_datetime is not None:
        temporal = inference.parsed_datetime
    if temporal is not None:
        # Frequency and gap counts are meaningless on a randomly sampled frame —
        # every skipped row reads as a gap — so when the frame was sampled the
        # caller hands over a contiguous slice to measure spacing on instead.
        source = temporal
        if dt_series is not None:
            if pd.api.types.is_datetime64_any_dtype(dt_series.dtype):
                source = dt_series
            elif isinstance(dt_series.dtype, pd.PeriodDtype):
                source = dt_series.dt.to_timestamp()
            else:
                parsed, _ = try_parse_datetime(dt_series, name)
                if parsed is not None:
                    source = parsed
        dt_stats = st.datetime_summary(source, order_series=source)
        if dt_series is not None:
            # `source` is a contiguous head slice, which is what spacing needs and
            # exactly what the range must not be read from: on time-ordered data it
            # truncates max_timestamp at the slice boundary, understating the span an
            # agent uses to pick a horizon or a temporal cut. min/max are a single
            # cheap pass, so they are taken from the widest series available without
            # re-parsing.
            bounds = _timestamp_bounds(full_series)
            if bounds is None:
                bounds = _timestamp_bounds(temporal)
            if bounds is not None:
                dt_stats["min_timestamp"], dt_stats["max_timestamp"] = bounds
        profile.min_timestamp = dt_stats["min_timestamp"]
        profile.max_timestamp = dt_stats["max_timestamp"]
        profile.inferred_frequency = dt_stats["inferred_frequency"]
        profile.n_gaps = dt_stats["n_gaps"]
        profile.is_monotonic = dt_stats["is_monotonic"]
        side["n_gaps"] = dt_stats["n_gaps"]
        side["monotonic_increasing"] = dt_stats["is_monotonic_increasing"]
        side["temporal_series"] = temporal

    mixed = st.mixed_type_names(sample_series)
    if mixed:
        side["mixed_types"] = mixed

    return profile, inference, side


def _numeric_frame(
    sample: Any,
    names: list[str],
    kinds: dict[str, ColumnKind],
    *,
    exclude: set[str],
) -> tuple[Any | None, bool]:
    """Frame of correlation-eligible numeric columns, and whether it was capped."""
    chosen: list[str] = []
    for name in names:
        if name in exclude:
            continue
        if kinds.get(name) not in _NUMERIC_KINDS:
            continue
        chosen.append(name)
    capped = len(chosen) > MAX_NUMERIC_CORRELATION_COLUMNS
    if capped:
        chosen = chosen[:MAX_NUMERIC_CORRELATION_COLUMNS]
    if len(chosen) < 2:
        return None, capped
    frame = sample[chosen].apply(
        lambda col: col.astype("float64")
        if pd.api.types.is_bool_dtype(col.dtype)
        else pd.to_numeric(col, errors="coerce")
    )
    return frame, capped


def _feature_correlations(
    sample: Any,
    names: list[str],
    kinds: dict[str, ColumnKind],
    profiles: dict[str, ColumnProfile],
    *,
    exclude: set[str],
) -> tuple[list[CorrelationPair], list[CorrelationPair], list[str]]:
    """Feature-to-feature correlations across mixed types.

    Returns:
        ``(top_correlations, highly_correlated_pairs, notes)``.
    """
    notes: list[str] = []
    # Raw tuples, not models: a 150-column numeric frame yields >11k pairs of
    # which 40 are reported, and building throwaway pydantic objects for the rest
    # dominates the pass.
    scored: list[tuple[float, str, str, float, str]] = []
    collinear: list[CorrelationPair] = []

    def consider(left: str, right: str, value: float, method: str) -> None:
        scored.append((abs(value), left, right, value, method))

    numeric_df, capped = _numeric_frame(sample, names, kinds, exclude=exclude)
    if capped:
        notes.append(
            f"numeric correlations were limited to the first "
            f"{MAX_NUMERIC_CORRELATION_COLUMNS} numeric columns"
        )
    if numeric_df is not None:
        pearson, spearman = st.correlation_matrices(
            numeric_df, min_periods=MIN_CORRELATION_PERIODS
        )
        columns = list(numeric_df.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                p = st.safe_float(pearson.at[left, right]) if pearson is not None else None
                s = st.safe_float(spearman.at[left, right]) if spearman is not None else None
                if p is not None:
                    consider(left, right, p, "pearson")
                # Spearman earns its own entry only when it disagrees materially
                # with Pearson, which is the signature of a non-linear pair.
                if s is not None and (p is None or abs(abs(s) - abs(p)) >= NONLINEARITY_GAP):
                    consider(left, right, s, "spearman")
                best = max(
                    ((abs(v), v, m) for v, m in ((p, "pearson"), (s, "spearman")) if v is not None),
                    default=None,
                )
                if best is not None and is_collinear(best[1]):
                    collinear.append(
                        CorrelationPair(
                            left=left, right=right, coefficient=best[1], method=best[2]
                        )
                    )

    categorical = [
        name
        for name in names
        if name not in exclude
        and kinds.get(name) in _CATEGORICAL_KINDS
        and profiles[name].n_unique <= MAX_CATEGORICAL_CORRELATION_LEVELS
        and not profiles[name].is_constant
    ]
    if len(categorical) > MAX_CATEGORICAL_CORRELATION_COLUMNS:
        notes.append(
            f"categorical association was limited to the {MAX_CATEGORICAL_CORRELATION_COLUMNS} "
            "lowest-cardinality categorical columns"
        )
        categorical = sorted(categorical, key=lambda n: profiles[n].n_unique)[
            :MAX_CATEGORICAL_CORRELATION_COLUMNS
        ]

    for i, left in enumerate(categorical):
        for right in categorical[i + 1 :]:
            value = st.cramers_v(
                sample[left].to_numpy(dtype=object), sample[right].to_numpy(dtype=object)
            )
            if value is None:
                continue
            consider(left, right, value, "cramers_v")
            if is_collinear(value):
                collinear.append(
                    CorrelationPair(
                        left=left, right=right, coefficient=value, method="cramers_v"
                    )
                )

    numeric_names = list(numeric_df.columns) if numeric_df is not None else []
    for category in categorical:
        if kinds.get(category) is ColumnKind.BOOLEAN and category in numeric_names:
            continue  # already covered by the numeric matrix
        for numeric_name in numeric_names:
            value = st.correlation_ratio(
                sample[category].to_numpy(dtype=object), sample[numeric_name].to_numpy()
            )
            if value is None:
                continue
            consider(category, numeric_name, value, "correlation_ratio")
            if is_collinear(value):
                collinear.append(
                    CorrelationPair(
                        left=category,
                        right=numeric_name,
                        coefficient=value,
                        method="correlation_ratio",
                    )
                )

    scored.sort(reverse=True)
    top = [
        CorrelationPair(left=left, right=right, coefficient=value, method=method)
        for _, left, right, value, method in scored[:MAX_TOP_CORRELATIONS]
    ]
    collinear.sort(key=lambda p: abs(p.coefficient), reverse=True)
    return top, collinear, notes


def _target_correlations(
    associations: Sequence[TargetAssociation], target: str
) -> list[CorrelationPair]:
    """Project the target-association pass onto correlation pairs.

    These are not recomputed from the frame on purpose. The association pass
    already measures exactly the right coefficient for each feature/target type
    combination — Pearson and Spearman for numeric pairs, the correlation ratio
    for mixed pairs, Cramer's V for categorical pairs — and deriving the reported
    correlations from those same numbers guarantees that the correlation an agent
    reads and the score the leakage detector acted on can never disagree.

    Args:
        associations: Output of ``score_target_associations``.
        target: Target column name, used as the right-hand side of each pair.

    Returns:
        Pairs sorted by descending absolute strength, capped for prompt budget.
    """
    out: list[CorrelationPair] = []
    for association in associations:
        name = association.column
        if association.pearson is not None:
            out.append(
                CorrelationPair(
                    left=name, right=target, coefficient=association.pearson, method="pearson"
                )
            )
        # Spearman earns a line only when it disagrees materially with Pearson;
        # that gap is what tells an agent the relationship is non-linear.
        if association.spearman is not None and (
            association.pearson is None
            or abs(abs(association.spearman) - abs(association.pearson)) >= NONLINEARITY_GAP
        ):
            out.append(
                CorrelationPair(
                    left=name, right=target, coefficient=association.spearman, method="spearman"
                )
            )
        if association.eta is not None:
            out.append(
                CorrelationPair(
                    left=name,
                    right=target,
                    coefficient=association.eta,
                    method="correlation_ratio",
                )
            )
        if association.cramers_v is not None:
            out.append(
                CorrelationPair(
                    left=name, right=target, coefficient=association.cramers_v, method="cramers_v"
                )
            )
    out.sort(key=lambda pair: (-abs(pair.coefficient), pair.left, pair.method))
    return out[:MAX_TARGET_CORRELATIONS]


def profile_dataframe(
    df: Any,
    *,
    target: str | None = None,
    settings: Settings | None = None,
    sample_rows: int | None = None,
    dataset_id: str | None = None,
    cv_folds: int = 5,
) -> DatasetProfile:
    """Compute the full deterministic profile of a dataframe.

    Args:
        df: The dataframe to profile. Not mutated.
        target: Name of the target column, when known. Enables the target
            summary, target correlations, and leakage detection. A name that is
            not in the frame is ignored with an INFO quality issue rather than
            raising, because the caller may be passing an operator's guess.
        settings: Process settings. Defaults to :func:`get_settings`; supplies
            ``max_profile_rows`` and ``default_random_state``.
        sample_rows: Overrides ``settings.max_profile_rows`` for the expensive
            passes. ``n_rows`` still reports the true row count.
        dataset_id: Id to stamp on the profile. Defaults to a structural
            fingerprint of the frame.
        cv_folds: Planned cross-validation folds, used only to judge whether a
            target class is too rare to stratify on.

    Returns:
        A fully populated :class:`DatasetProfile`.

    Raises:
        ValidationError: If ``df`` is not a pandas DataFrame.
    """
    started = time.perf_counter()
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"profile_dataframe expects a pandas DataFrame, got {type(df).__name__}"
        )

    settings = settings or get_settings()
    n_rows = int(len(df))
    n_columns = int(df.shape[1])
    notes: list[str] = []

    names, duplicated_names = _unique_names(df.columns)
    if duplicated_names:
        notes.append(
            "duplicate column labels were renamed with a numeric suffix for profiling: "
            + ", ".join(duplicated_names)
        )

    if target is not None and target not in set(map(str, df.columns)):
        notes.append(
            f"the requested target column '{target}' is not in the frame, so the target "
            "summary, target correlations, and leakage detection were skipped"
        )
        target = None
    if target is not None and duplicated_names and str(target) in duplicated_names:
        notes.append(
            f"the target column '{target}' is duplicated in the frame; the first "
            "occurrence was used"
        )

    cap = int(sample_rows or settings.max_profile_rows)
    sampled = n_rows > cap > 0
    if sampled:
        # Random rows keep distributions honest; sort_index restores relative
        # order so a downstream monotonicity check is not scrambled.
        sample = df.sample(n=cap, random_state=settings.default_random_state).sort_index()
        # A contiguous head preserves the true spacing between timestamps, which
        # a random sample destroys.
        temporal_source = df.head(cap)
        notes.append(
            f"expensive statistics (correlations, outliers, mutual information, regex "
            f"detection) were computed on a random sample of {cap:,} of {n_rows:,} rows; "
            "counts, missingness, and cardinality are exact"
        )
    else:
        sample = df
        temporal_source = df

    overview = st.dataframe_overview(df)

    profiles: list[ColumnProfile] = []
    profiles_by_name: dict[str, ColumnProfile] = {}
    kinds: dict[str, ColumnKind] = {}
    mixed_types: dict[str, list[str]] = {}
    blank_strings: dict[str, int] = {}
    untrimmed_strings: dict[str, int] = {}
    datetime_gaps: dict[str, int] = {}
    unsorted_datetimes: list[str] = []
    numeric_as_text: list[str] = []
    temporal_frames: dict[str, Any] = {}

    for position, name in enumerate(names):
        try:
            full_series = df.iloc[:, position]
            sample_series = sample.iloc[:, position]
            dt_series = temporal_source.iloc[:, position] if sampled else None
            profile, inference, side = _build_column_profile(
                name,
                full_series,
                sample_series,
                n_rows=n_rows,
                dt_series=dt_series,
            )
        except Exception as exc:
            logger.warning("profiling column %r failed: %s", name, exc)
            notes.append(
                f"column `{name}` could not be profiled ({type(exc).__name__}: {exc}); "
                "it is reported as unknown"
            )
            profile = ColumnProfile(
                name=name, kind=ColumnKind.UNKNOWN, dtype=str(df.dtypes.iloc[position])
            )
            inference = ColumnInference(kind=ColumnKind.UNKNOWN, reason="profiling failed")
            side = {}

        profiles.append(profile)
        profiles_by_name[name] = profile
        kinds[name] = profile.kind
        # The schema has nowhere to carry *why* a kind was chosen, but the
        # reasoning is the audit trail for a misclassification, so it goes to the
        # log rather than being discarded.
        logger.debug("column %r -> %s: %s", name, profile.kind.value, inference.reason)

        if side.get("mixed_types"):
            mixed_types[name] = side["mixed_types"]
        if side.get("n_blank"):
            blank_strings[name] = int(side["n_blank"])
        if side.get("n_untrimmed"):
            untrimmed_strings[name] = int(side["n_untrimmed"])
        if side.get("n_gaps"):
            datetime_gaps[name] = int(side["n_gaps"])
        if "monotonic_increasing" in side and side["monotonic_increasing"] is False:
            unsorted_datetimes.append(name)
        if side.get("numeric_as_text"):
            numeric_as_text.append(name)
        if side.get("temporal_series") is not None:
            temporal_frames[name] = side["temporal_series"]

    # A sample keyed by the de-duplicated display names makes every downstream
    # lookup (correlations, leakage) unambiguous.
    work = sample.copy(deep=False)
    work.columns = names
    for name, parsed in temporal_frames.items():
        if not pd.api.types.is_datetime64_any_dtype(work[name].dtype):
            work[name] = parsed

    temporal_columns = [p.name for p in profiles if p.kind is ColumnKind.DATETIME]
    geo_columns = [p.name for p in profiles if p.kind is ColumnKind.GEO]
    text_columns = [p.name for p in profiles if p.kind is ColumnKind.TEXT]
    identifier_columns = [p.name for p in profiles if p.kind is ColumnKind.IDENTIFIER]
    constant_columns = [p.name for p in profiles if p.is_constant]

    exclude_from_feature_corr = set(constant_columns) | set(identifier_columns)
    if target:
        exclude_from_feature_corr.add(target)

    try:
        top_correlations, collinear, corr_notes = _feature_correlations(
            work, names, kinds, profiles_by_name, exclude=exclude_from_feature_corr
        )
        notes.extend(corr_notes)
    except Exception as exc:  # pragma: no cover - degrade
        logger.warning("correlation pass failed: %s", exc)
        notes.append(f"correlation analysis failed ({type(exc).__name__}: {exc})")
        top_correlations, collinear = [], []

    target_summary = None
    target_correlations: list[CorrelationPair] = []
    leakage_findings: list[LeakageFinding] = []
    if target:
        target_kind = kinds.get(target, ColumnKind.UNKNOWN)
        target_profile = profiles_by_name.get(target)
        target_n_unique = target_profile.n_unique if target_profile else 0
        classification = target_is_classification_like(target_kind, target_n_unique)
        try:
            target_summary = st.build_target_summary(
                target,
                df.iloc[:, names.index(target)],
                target_kind,
                classification=classification,
            )
        except Exception as exc:  # pragma: no cover - degrade
            logger.warning("target summary failed: %s", exc)
            notes.append(f"target summary failed ({type(exc).__name__}: {exc})")

        try:
            associations = score_target_associations(
                work,
                target,
                kinds,
                classification=classification,
                random_state=settings.default_random_state,
            )
            target_correlations = _target_correlations(associations, target)
            leakage_findings = detect_leakage(
                work,
                target,
                associations,
                kinds,
                classification=classification,
                target_n_unique=target_n_unique,
            )
        except Exception as exc:  # pragma: no cover - degrade
            logger.warning("target association pass failed: %s", exc)
            notes.append(
                f"target correlations and leakage detection failed "
                f"({type(exc).__name__}: {exc})"
            )

    try:
        quality_issues = detect_quality_issues(
            columns=profiles,
            overview=overview,
            n_rows=n_rows,
            n_columns=n_columns,
            target=target_summary,
            highly_correlated=collinear,
            mixed_types=mixed_types,
            blank_strings=blank_strings,
            untrimmed_strings=untrimmed_strings,
            unsorted_datetimes=unsorted_datetimes,
            datetime_gaps=datetime_gaps,
            numeric_as_text=numeric_as_text,
            cv_folds=cv_folds,
            notes=notes,
        )
    except Exception as exc:  # pragma: no cover - degrade
        logger.warning("quality detection failed: %s", exc)
        quality_issues = [
            DataQualityIssue(
                code="profiling_note",
                severity=Severity.INFO,
                columns=[],
                detail=f"quality detection failed ({type(exc).__name__}: {exc})",
            )
        ]

    return DatasetProfile(
        dataset_id=dataset_id or dataset_fingerprint(df),
        n_rows=n_rows,
        n_columns=n_columns,
        memory_bytes=overview.memory_bytes,
        n_duplicate_rows=overview.n_duplicate_rows,
        duplicate_fraction=overview.duplicate_fraction,
        total_missing_cells=overview.total_missing_cells,
        missing_cell_fraction=overview.missing_cell_fraction,
        columns=profiles,
        target=target_summary,
        top_correlations=top_correlations,
        target_correlations=target_correlations,
        highly_correlated_pairs=collinear,
        leakage_findings=leakage_findings,
        quality_issues=quality_issues,
        temporal_columns=temporal_columns,
        geo_columns=geo_columns,
        text_columns=text_columns,
        identifier_columns=identifier_columns,
        constant_columns=constant_columns,
        profile_seconds=round(time.perf_counter() - started, 4),
    )


def summarise_profile(profile: DatasetProfile) -> str:
    """A short human-readable digest, for logs and self-tests.

    The prompt-facing rendering lives in
    :mod:`automl_architect.core.context`; this is deliberately terser.

    Args:
        profile: A computed profile.

    Returns:
        A multi-line summary string.
    """
    lines = [
        f"dataset {profile.dataset_id}: {profile.n_rows:,} rows x {profile.n_columns} columns "
        f"({profile.memory_bytes / 1024:.1f} KB) in {profile.profile_seconds:.2f}s",
        f"missing cells: {profile.total_missing_cells:,} "
        f"({profile.missing_cell_fraction:.2%}); duplicate rows: "
        f"{profile.n_duplicate_rows:,} ({profile.duplicate_fraction:.2%})",
    ]
    counts: dict[str, int] = {}
    for column in profile.columns:
        counts[column.kind.value] = counts.get(column.kind.value, 0) + 1
    lines.append("kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if profile.target:
        target = profile.target
        detail = f"target `{target.name}` ({target.kind.value})"
        if target.n_classes is not None:
            detail += f", {target.n_classes} classes"
        if target.imbalance_ratio is not None:
            detail += f", imbalance {target.imbalance_ratio:.2f}:1"
        lines.append(detail)
    lines.append(
        f"leakage findings: {len(profile.leakage_findings)}; "
        f"quality issues: {len(profile.quality_issues)}; "
        f"collinear pairs: {len(profile.highly_correlated_pairs)}"
    )
    return "\n".join(lines)


__all__ = ["dataset_fingerprint", "profile_dataframe", "summarise_profile"]
