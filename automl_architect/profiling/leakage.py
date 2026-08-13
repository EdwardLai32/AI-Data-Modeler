"""Target-leakage and data-quality detection.

Leakage is the failure mode that most reliably produces a beautiful, useless
model, so this module is deliberately opinionated about what counts as evidence:

*   **Statistics first, names second.** A column called ``churn_reason`` is not
    leakage because of its name. It is leakage because its association with the
    target is near-perfect *and* its name says the outcome was already known.
    Name-only matches are never reported, and every reason string says so
    explicitly, because the downstream agent is told to trust these reasons.
*   **Association is measured on a bounded 0-1 scale** regardless of the target
    type, so a single severity ladder applies to mutual information, AUC, and
    correlation alike. Mutual information for classification is normalised by the
    target's entropy (the uncertainty coefficient); for regression it is mapped
    through the Gaussian identity ``r = sqrt(1 - e^(-2I))``.
*   **Unreliable measures are dropped, not reported.** Mutual information against
    a near-unique categorical feature approaches the target entropy by
    memorisation alone; scoring that as leakage would flag every identifier
    column in every dataset. Those cases fall back to bias-corrected Cramer's V.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..core.schemas import (
    ColumnKind,
    ColumnProfile,
    CorrelationPair,
    DataQualityIssue,
    LeakageFinding,
    Severity,
    TargetSummary,
)
from .semantic import suspicious_outcome_name
from .stats import (
    NEAR_ZERO_VARIANCE_FREQ_RATIO,
    FrameOverview,
    correlation_ratio,
    cramers_v,
    safe_float,
)

# --- leakage thresholds ---------------------------------------------------

LEAKAGE_CRITICAL = 0.995
LEAKAGE_HIGH = 0.97
LEAKAGE_MEDIUM = 0.90
SUSPICIOUS_NAME_MIN_SCORE = 0.60
MONOTONE_TRANSFORM_RHO = 0.9995
MIN_ROWS_FOR_ASSOCIATION = 20
MI_MAX_ROWS = 20_000
MI_MAX_ROWS_WIDE = 5_000
WIDE_FRAME_COLUMNS = 120
HIGH_CARDINALITY_MI_LEVELS = 20
HIGH_CARDINALITY_MI_RATIO = 0.10

# --- quality thresholds ---------------------------------------------------

MISSING_CRITICAL = 0.90
MISSING_HIGH = 0.50
MISSING_MEDIUM = 0.20
MISSING_LOW = 0.05
DUPLICATE_HIGH = 0.20
DUPLICATE_MEDIUM = 0.05
SKEW_MEDIUM = 2.0
SKEW_HIGH = 5.0
HIGH_CARDINALITY_UNIQUE = 50
HIGH_CARDINALITY_RATIO = 0.50
OUTLIER_HEAVY_FRACTION = 0.10
TINY_DATASET_ROWS = 50
SMALL_DATASET_ROWS = 200
WIDE_RATIO_MEDIUM = 0.50
IMBALANCE_HIGH = 10.0
IMBALANCE_MEDIUM = 3.0
BLANK_STRING_FRACTION = 0.01

_CLASSIFICATION_KINDS = frozenset(
    {
        ColumnKind.BOOLEAN,
        ColumnKind.CATEGORICAL_NOMINAL,
        ColumnKind.CATEGORICAL_ORDINAL,
    }
)
_SKIP_ASSOCIATION_KINDS = frozenset({ColumnKind.CONSTANT, ColumnKind.UNKNOWN})


@dataclass
class TargetAssociation:
    """How strongly one feature is associated with the target.

    Attributes:
        column: Feature name.
        score: Best available association on a 0-1 scale.
        method: Which measure produced ``score``.
        mutual_info: Raw mutual information in nats, when computed.
        normalised_mi: ``mutual_info`` mapped onto 0-1.
        auc: ROC AUC of the single feature against a binary target.
        pearson: Pearson correlation with a numeric target.
        spearman: Spearman correlation with a numeric target.
        cramers_v: Bias-corrected Cramer's V against a categorical target.
        eta: Correlation ratio for mixed categorical/numeric pairs.
        mi_reliable: False when mutual information was rejected as inflated.
        notes: Degradation notes worth surfacing in a reason string.
    """

    column: str
    score: float = 0.0
    method: str = "none"
    mutual_info: float | None = None
    normalised_mi: float | None = None
    auc: float | None = None
    pearson: float | None = None
    spearman: float | None = None
    cramers_v: float | None = None
    eta: float | None = None
    mi_reliable: bool = True
    notes: list[str] = field(default_factory=list)


def target_is_classification_like(kind: ColumnKind, n_unique: int) -> bool:
    """Whether a target should be scored as classes rather than as a quantity.

    Args:
        kind: Inferred kind of the target column.
        n_unique: Distinct non-null values in the target.

    Returns:
        True for boolean/categorical targets and for low-cardinality integer
        targets. This is a scoring decision only — the authoritative task type
        comes from the Problem Identification Agent.
    """
    if kind in _CLASSIFICATION_KINDS:
        return True
    if kind is ColumnKind.NUMERIC_DISCRETE and n_unique <= 20:
        return True
    return False


def _entropy_nats(codes: np.ndarray) -> float:
    valid = codes[codes >= 0]
    if valid.size == 0:
        return 0.0
    counts = np.bincount(valid)
    probabilities = counts[counts > 0] / valid.size
    return float(-(probabilities * np.log(probabilities)).sum())


def _aligned_pair(frame: Any, feature: str, target: str) -> Any:
    """Rows where both columns are present, as a two-column frame."""
    pair = pd.DataFrame(
        {"x": frame[feature].reset_index(drop=True), "y": frame[target].reset_index(drop=True)}
    )
    return pair.dropna()


def _is_numeric(series: Any) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _as_float(series: Any) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64").to_numpy(dtype="float64", na_value=np.nan)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series.astype("int64").to_numpy(dtype="float64")
    return pd.to_numeric(series, errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )


def _mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    discrete_feature: bool,
    classification: bool,
    random_state: int,
) -> float | None:
    """Mutual information of one feature with the target, in nats."""
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        return None
    matrix = x.reshape(-1, 1)
    try:
        if classification:
            value = mutual_info_classif(
                matrix,
                y,
                discrete_features=[discrete_feature],
                random_state=random_state,
            )[0]
        else:
            value = mutual_info_regression(
                matrix,
                y,
                discrete_features=[discrete_feature],
                random_state=random_state,
            )[0]
    except (ValueError, TypeError, MemoryError):
        return None
    return safe_float(value)


def _mi_subsample(size: int, max_rows: int, random_state: int) -> np.ndarray | None:
    """Deterministic row indices for the mutual-information pass, or ``None``.

    ``None`` means "use every row"; the caller skips the indexing entirely.
    """
    if size <= max_rows:
        return None
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(size, size=max_rows, replace=False))


def _roc_auc(x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # pragma: no cover
        return None
    try:
        auc = float(roc_auc_score(y, x))
    except (ValueError, TypeError):
        return None
    return max(auc, 1.0 - auc)


def score_target_associations(
    frame: Any,
    target: str,
    kinds: dict[str, ColumnKind],
    *,
    classification: bool,
    random_state: int = 42,
    max_rows: int | None = None,
    skip_columns: Iterable[str] = (),
) -> list[TargetAssociation]:
    """Measure every feature's association with the target.

    Args:
        frame: Sampled dataframe containing the target and features.
        target: Target column name.
        kinds: Inferred kind per column, used to choose the right measure.
        classification: Score against classes (mutual information / AUC /
            Cramer's V) rather than against a quantity (correlations).
        random_state: Seed for the kNN estimator inside mutual information.
        max_rows: Row cap for the mutual-information pass. Defaults to a value
            chosen from the frame width, since the estimator is the slowest part
            of profiling.
        skip_columns: Columns to leave unscored, e.g. already-dropped columns.

    Returns:
        One :class:`TargetAssociation` per scored feature, strongest first.
    """
    if target not in frame.columns:
        return []

    skip = set(skip_columns) | {target}
    if max_rows is None:
        max_rows = (
            MI_MAX_ROWS_WIDE if frame.shape[1] > WIDE_FRAME_COLUMNS else MI_MAX_ROWS
        )

    results: list[TargetAssociation] = []
    for column in frame.columns:
        if column in skip:
            continue
        kind = kinds.get(column, ColumnKind.UNKNOWN)
        if kind in _SKIP_ASSOCIATION_KINDS:
            continue
        try:
            association = _score_one(
                frame,
                column,
                target,
                kind=kind,
                classification=classification,
                random_state=random_state,
                max_rows=max_rows,
            )
        except Exception as exc:  # pragma: no cover - never abort the profile
            association = TargetAssociation(
                column=column,
                notes=[f"association scoring failed ({type(exc).__name__}: {exc})"],
            )
        if association is not None:
            results.append(association)

    results.sort(key=lambda a: a.score, reverse=True)
    return results


def _score_one(
    frame: Any,
    column: str,
    target: str,
    *,
    kind: ColumnKind,
    classification: bool,
    random_state: int,
    max_rows: int,
) -> TargetAssociation | None:
    association = TargetAssociation(column=column)
    pair = _aligned_pair(frame, column, target)
    if len(pair) < MIN_ROWS_FOR_ASSOCIATION:
        association.notes.append(
            f"only {len(pair)} rows have both this column and the target present"
        )
        return association
    x_series = pair["x"]
    y_series = pair["y"]
    if x_series.nunique() < 2:
        association.notes.append("constant once rows with a missing target are removed")
        return association

    feature_numeric = _is_numeric(x_series) or pd.api.types.is_datetime64_any_dtype(
        x_series.dtype
    )
    discrete_feature = not feature_numeric or kind in {
        ColumnKind.NUMERIC_DISCRETE,
        ColumnKind.BOOLEAN,
    }

    # Every discrete-feature measure degrades once there are too few observations
    # per level: mutual information approaches the target entropy by memorisation,
    # and Cramer's V converges on a constant that depends only on the table shape,
    # not on the relationship. Both are then unusable, so they are suppressed
    # rather than reported as a number an agent would take at face value.
    n_levels = int(x_series.nunique())
    discrete_reliable = not (
        discrete_feature
        and (
            n_levels
            > max(HIGH_CARDINALITY_MI_LEVELS, HIGH_CARDINALITY_MI_RATIO * len(pair))
            or kind is ColumnKind.IDENTIFIER
        )
    )
    association.mi_reliable = discrete_reliable
    if not discrete_reliable:
        association.notes.append(
            f"discrete association measures suppressed: {n_levels:,} levels over "
            f"{len(pair):,} rows leaves too few observations per level to distinguish "
            "a real relationship from memorisation"
        )

    if feature_numeric:
        x_values = _as_float(x_series)
    else:
        x_values = pd.factorize(x_series, use_na_sentinel=False)[0].astype("float64")
    finite = np.isfinite(x_values)
    if not finite.all():
        x_values = x_values[finite]
        # x_series is masked in step with x_values: Cramer's V and the correlation
        # ratio below are handed the raw values, and a series left one row longer
        # than its partner is silently index-aligned into the wrong pairs rather
        # than rejected.
        x_series = x_series[finite]
        y_series = y_series[finite]
    if x_values.size < MIN_ROWS_FOR_ASSOCIATION:
        association.notes.append("too few finite feature values to score")
        return association

    # Only the kNN mutual-information estimator is expensive enough to need a row
    # cap. Correlations, AUC, and Cramer's V stay on the full sample so the number
    # reported as a correlation is the same number the leakage score used.
    candidates: list[tuple[float, str]] = []

    if classification:
        y_codes = pd.factorize(y_series, use_na_sentinel=False)[0]
        entropy = _entropy_nats(y_codes)
        keep = _mi_subsample(x_values.size, max_rows, random_state)
        mi = _mutual_information(
            x_values if keep is None else x_values[keep],
            y_codes if keep is None else y_codes[keep],
            discrete_feature=discrete_feature,
            classification=True,
            random_state=random_state,
        )
        association.mutual_info = mi
        if mi is not None and entropy > 0:
            association.normalised_mi = float(min(1.0, max(0.0, mi / entropy)))
            if association.mi_reliable:
                candidates.append((association.normalised_mi, "normalised_mutual_info"))

        classes = pd.unique(y_codes)
        if len(classes) == 2 and feature_numeric:
            auc = _roc_auc(x_values, y_codes)
            association.auc = auc
            if auc is not None:
                candidates.append((2.0 * abs(auc - 0.5), "roc_auc_single_feature"))
        if feature_numeric:
            eta = correlation_ratio(y_series.to_numpy(dtype=object), x_values)
            association.eta = eta
            if eta is not None:
                candidates.append((eta, "correlation_ratio"))
        elif discrete_reliable:
            v = cramers_v(x_series.to_numpy(dtype=object), y_series.to_numpy(dtype=object))
            association.cramers_v = v
            if v is not None:
                candidates.append((v, "cramers_v"))
    else:
        y_values = _as_float(y_series)
        finite_y = np.isfinite(y_values)
        if not finite_y.all():
            x_values = x_values[finite_y]
            x_series = x_series[finite_y]
            y_values = y_values[finite_y]
            y_series = y_series[finite_y]
        if x_values.size < MIN_ROWS_FOR_ASSOCIATION:
            association.notes.append("too few finite paired values to score")
            return association

        keep = _mi_subsample(x_values.size, max_rows, random_state)
        mi = _mutual_information(
            x_values if keep is None else x_values[keep],
            y_values if keep is None else y_values[keep],
            discrete_feature=discrete_feature,
            classification=False,
            random_state=random_state,
        )
        association.mutual_info = mi
        if mi is not None:
            # Gaussian identity: I = -0.5*ln(1-r^2) => |r| = sqrt(1-e^(-2I)).
            association.normalised_mi = float(
                min(1.0, math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * max(0.0, mi)))))
            )
            if association.mi_reliable:
                candidates.append((association.normalised_mi, "normalised_mutual_info"))

        if feature_numeric:
            frame_xy = pd.DataFrame({"a": x_values, "b": y_values})
            if frame_xy["a"].nunique() >= 2 and frame_xy["b"].nunique() >= 2:
                association.pearson = safe_float(frame_xy["a"].corr(frame_xy["b"]))
                association.spearman = safe_float(
                    frame_xy["a"].rank().corr(frame_xy["b"].rank())
                )
                if association.pearson is not None:
                    candidates.append((abs(association.pearson), "pearson"))
                if association.spearman is not None:
                    candidates.append((abs(association.spearman), "spearman"))
        elif discrete_reliable:
            eta = correlation_ratio(x_series.to_numpy(dtype=object), y_values)
            association.eta = eta
            if eta is not None:
                candidates.append((eta, "correlation_ratio"))

    if candidates:
        score, method = max(candidates, key=lambda item: item[0])
        association.score = float(min(1.0, max(0.0, score)))
        association.method = method
    return association


def _duplicates_target(x: Any, y: Any) -> bool:
    """Whether a feature is the target under a different name."""
    try:
        if x.equals(y):
            return True
    except (TypeError, ValueError):  # pragma: no cover
        pass
    if _is_numeric(x) and _is_numeric(y):
        left = _as_float(x)
        right = _as_float(y)
        if left.shape != right.shape:
            return False
        both = np.isfinite(left) & np.isfinite(right)
        if both.sum() == 0:
            return False
        return bool(np.allclose(left[both], right[both], rtol=1e-9, atol=1e-12))
    try:
        return bool(
            x.astype("string").fillna("<NA>").reset_index(drop=True).equals(
                y.astype("string").fillna("<NA>").reset_index(drop=True)
            )
        )
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _severity_for(score: float) -> Severity | None:
    if score >= LEAKAGE_CRITICAL:
        return Severity.CRITICAL
    if score >= LEAKAGE_HIGH:
        return Severity.HIGH
    if score >= LEAKAGE_MEDIUM:
        return Severity.MEDIUM
    return None


_SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def detect_leakage(
    frame: Any,
    target: str,
    associations: Sequence[TargetAssociation],
    kinds: dict[str, ColumnKind],
    *,
    classification: bool,
    target_n_unique: int,
) -> list[LeakageFinding]:
    """Turn measured associations into explained leakage findings.

    Four detectors run, and a column flagged by several is reported once with the
    highest severity and every reason concatenated:

    1.  Near-perfect single-feature association with the target.
    2.  A column that duplicates the target exactly.
    3.  A monotone transform of a continuous target (rank correlation ~1 while
        the association is not an exact duplicate).
    4.  A post-outcome column *name* combined with a strong measured signal.

    Args:
        frame: Sampled dataframe.
        target: Target column name.
        associations: Output of :func:`score_target_associations`.
        kinds: Inferred kind per column.
        classification: Whether the target was scored as classes.
        target_n_unique: Distinct target values, used to gate the
            monotone-transform test to genuinely continuous targets.

    Returns:
        Findings ordered by descending severity then descending score.
    """
    findings: dict[str, LeakageFinding] = {}

    def record(
        column: str, score: float, method: str, severity: Severity, reason: str
    ) -> None:
        existing = findings.get(column)
        if existing is None:
            findings[column] = LeakageFinding(
                column=column,
                score=round(float(score), 6),
                method=method,
                severity=severity,
                reason=reason,
            )
            return
        if _SEVERITY_ORDER[severity] > _SEVERITY_ORDER[existing.severity]:
            existing.severity = severity
        if score > existing.score:
            existing.score = round(float(score), 6)
            existing.method = method
        if reason not in existing.reason:
            existing.reason = f"{existing.reason} Also: {reason}"

    target_series = frame[target] if target in frame.columns else None

    for association in associations:
        column = association.column
        kind = kinds.get(column, ColumnKind.UNKNOWN)
        score = association.score
        fragment = suspicious_outcome_name(column)

        if target_series is not None and column in frame.columns:
            if _duplicates_target(frame[column], target_series):
                record(
                    column,
                    1.0,
                    "exact_duplicate",
                    Severity.CRITICAL,
                    f"`{column}` is value-for-value identical to the target `{target}` "
                    "on every observed row, so it is a copy of the label rather than a "
                    "feature. Training on it produces a perfect score that cannot "
                    "reproduce at prediction time.",
                )
                continue

        severity = _severity_for(score)
        if severity is not None:
            method_label = association.method.replace("_", " ")
            record(
                column,
                score,
                association.method,
                severity,
                f"A single feature explains the target almost completely "
                f"({method_label} = {score:.4f} on a 0-1 scale). Genuine predictors "
                f"rarely reach this level; the usual cause is that `{column}` is "
                f"recorded at the same time as, or after, `{target}`. Confirm it would "
                "be observable before the outcome is known."
                + (
                    f" Its name also contains '{fragment}', which points the same way."
                    if fragment
                    else ""
                ),
            )

        if (
            not classification
            and target_n_unique > 10
            and association.spearman is not None
            and abs(association.spearman) >= MONOTONE_TRANSFORM_RHO
        ):
            pearson_note = (
                f" Pearson is {association.pearson:.4f}, so the relationship is "
                "monotone but not linear — typically a log, rank, or bucketed copy "
                "of the target."
                if association.pearson is not None
                and abs(association.pearson) < MONOTONE_TRANSFORM_RHO
                else " The relationship is essentially an exact rescaling."
            )
            record(
                column,
                abs(association.spearman),
                "spearman_monotone",
                Severity.CRITICAL,
                f"`{column}` is a monotone transform of the target: rank correlation "
                f"{association.spearman:.6f}.{pearson_note}",
            )

        if fragment and severity is None and score >= SUSPICIOUS_NAME_MIN_SCORE:
            record(
                column,
                score,
                association.method or "name_plus_signal",
                Severity.MEDIUM if score < LEAKAGE_HIGH else Severity.HIGH,
                f"The name of `{column}` contains '{fragment}', which suggests knowledge "
                f"of the outcome, AND it is strongly predictive "
                f"({association.method.replace('_', ' ')} = {score:.4f}). The name alone "
                "would not be evidence of leakage; the measured signal is what makes "
                "this worth checking.",
            )

        if kind is ColumnKind.IDENTIFIER and score >= LEAKAGE_MEDIUM:
            record(
                column,
                score,
                association.method,
                Severity.MEDIUM,
                f"`{column}` looks like a record identifier yet still predicts the "
                f"target at {score:.4f}. That normally means the rows were sorted or "
                "assigned by outcome, so the id encodes information no future record "
                "will carry.",
            )

    ordered = sorted(
        findings.values(),
        key=lambda f: (_SEVERITY_ORDER[f.severity], f.score),
        reverse=True,
    )
    return ordered


# --- quality issues -------------------------------------------------------


def _issue(
    code: str, severity: Severity, columns: list[str], detail: str
) -> DataQualityIssue:
    return DataQualityIssue(
        code=code, severity=severity, columns=columns, detail=detail
    )


def _render(entries: Sequence[tuple[str, str]], limit: int = 20) -> str:
    """``[(name, label)]`` -> ``"label, label (+n more)"``."""
    shown = [label for _, label in entries[:limit]]
    suffix = f" (+{len(entries) - len(shown)} more)" if len(entries) > len(shown) else ""
    return ", ".join(shown) + suffix


def _add_grouped(
    issues: list[DataQualityIssue],
    code: str,
    buckets: dict[Severity, list[tuple[str, str]]],
    describe: str,
) -> None:
    """Emit one issue per severity bucket, listing the affected columns.

    Entries are ``(column_name, rendered_label)`` pairs so the machine-readable
    ``columns`` list never has to be recovered by parsing the prose — a column
    name containing a space would break that.
    """
    for severity in (
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
        Severity.LOW,
        Severity.INFO,
    ):
        entries = buckets.get(severity)
        if not entries:
            continue
        issues.append(
            _issue(
                code,
                severity,
                [name for name, _ in entries],
                f"{describe}: {_render(entries)}",
            )
        )


def detect_quality_issues(
    *,
    columns: Sequence[ColumnProfile],
    overview: FrameOverview,
    n_rows: int,
    n_columns: int,
    target: TargetSummary | None = None,
    highly_correlated: Sequence[CorrelationPair] = (),
    mixed_types: dict[str, list[str]] | None = None,
    blank_strings: dict[str, int] | None = None,
    untrimmed_strings: dict[str, int] | None = None,
    unsorted_datetimes: Sequence[str] = (),
    datetime_gaps: dict[str, int] | None = None,
    numeric_as_text: Sequence[str] = (),
    cv_folds: int = 5,
    notes: Sequence[str] = (),
) -> list[DataQualityIssue]:
    """Assemble the data-quality issue list from measured column statistics.

    Args:
        columns: Completed column profiles.
        overview: Whole-frame integrity counts.
        n_rows: True row count.
        n_columns: Column count.
        target: Target summary, when a target is known.
        highly_correlated: Pairs above the multicollinearity threshold.
        mixed_types: Column name -> Python type names found in it.
        blank_strings: Column name -> count of empty/whitespace-only values.
        untrimmed_strings: Column name -> count of values with stray whitespace.
        unsorted_datetimes: Temporal columns not in ascending row order.
        datetime_gaps: Temporal column name -> count of gaps beyond the modal
            spacing.
        numeric_as_text: Columns holding numbers that were stored as strings.
        cv_folds: Planned cross-validation folds, used to judge rare classes.
        notes: Free-form degradation notes to surface as INFO issues.

    Returns:
        Issues in no particular order; the prompt renderer sorts by severity.
    """
    issues: list[DataQualityIssue] = []
    mixed_types = mixed_types or {}
    blank_strings = blank_strings or {}
    untrimmed_strings = untrimmed_strings or {}
    datetime_gaps = datetime_gaps or {}

    missing_buckets: dict[Severity, list[tuple[str, str]]] = {}
    skew_buckets: dict[Severity, list[tuple[str, str]]] = {}
    cardinality_buckets: dict[Severity, list[tuple[str, str]]] = {}
    outlier_buckets: dict[Severity, list[tuple[str, str]]] = {}
    constant_columns: list[str] = []
    all_missing: list[str] = []
    nzv_columns: list[str] = []
    identifier_columns: list[str] = []

    for column in columns:
        if column.n_missing >= n_rows and n_rows > 0:
            all_missing.append(column.name)
        elif column.missing_fraction >= MISSING_LOW:
            if column.missing_fraction >= MISSING_CRITICAL:
                severity = Severity.CRITICAL
            elif column.missing_fraction >= MISSING_HIGH:
                severity = Severity.HIGH
            elif column.missing_fraction >= MISSING_MEDIUM:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            missing_buckets.setdefault(severity, []).append(
                (column.name, f"`{column.name}` ({column.missing_fraction:.1%})")
            )

        if column.name in all_missing:
            # Already reported as entirely null. Saying it is also "dominated by one
            # value" would be false — it has no values — and the reader acts on
            # these strings, so the weaker claim is dropped rather than duplicated.
            pass
        elif column.is_constant:
            constant_columns.append(column.name)
        elif column.is_near_zero_variance:
            nzv_columns.append(column.name)

        if column.skewness is not None and abs(column.skewness) >= SKEW_MEDIUM:
            severity = (
                Severity.MEDIUM if abs(column.skewness) < SKEW_HIGH else Severity.HIGH
            )
            skew_buckets.setdefault(severity, []).append(
                (column.name, f"`{column.name}` (skew {column.skewness:.2f})")
            )

        if (
            column.kind
            in {ColumnKind.CATEGORICAL_NOMINAL, ColumnKind.CATEGORICAL_ORDINAL, ColumnKind.GEO}
            and column.n_unique >= HIGH_CARDINALITY_UNIQUE
        ):
            severity = (
                Severity.HIGH
                if column.cardinality_ratio >= HIGH_CARDINALITY_RATIO
                else Severity.MEDIUM
            )
            cardinality_buckets.setdefault(severity, []).append(
                (
                    column.name,
                    f"`{column.name}` ({column.n_unique:,} levels, ratio "
                    f"{column.cardinality_ratio:.3f})",
                )
            )

        if column.outliers and column.outliers.fraction >= OUTLIER_HEAVY_FRACTION:
            outlier_buckets.setdefault(Severity.LOW, []).append(
                (
                    column.name,
                    f"`{column.name}` ({column.outliers.fraction:.1%} beyond the IQR fences)",
                )
            )

        if column.kind is ColumnKind.IDENTIFIER:
            identifier_columns.append(column.name)

    _add_grouped(issues, "high_missingness", missing_buckets, "Columns with missing values")
    _add_grouped(issues, "skewed_numeric", skew_buckets, "Strongly skewed numeric columns")
    _add_grouped(
        issues,
        "high_cardinality_categorical",
        cardinality_buckets,
        "High-cardinality categoricals (one-hot encoding would explode the feature space)",
    )
    _add_grouped(issues, "outlier_heavy", outlier_buckets, "Columns with heavy tails")

    if all_missing:
        issues.append(
            _issue(
                "all_missing_column",
                Severity.HIGH,
                all_missing,
                f"{len(all_missing)} column(s) are entirely null and carry no information: "
                + ", ".join(f"`{name}`" for name in all_missing[:20]),
            )
        )
    if constant_columns:
        issues.append(
            _issue(
                "constant_column",
                Severity.MEDIUM,
                constant_columns,
                f"{len(constant_columns)} column(s) hold a single distinct value and cannot "
                "contribute to any model: "
                + ", ".join(f"`{name}`" for name in constant_columns[:20]),
            )
        )
    if nzv_columns:
        issues.append(
            _issue(
                "near_zero_variance",
                Severity.LOW,
                nzv_columns,
                f"{len(nzv_columns)} column(s) are dominated by one value (frequency ratio "
                f"> {NEAR_ZERO_VARIANCE_FREQ_RATIO:g}:1) so they behave almost like "
                "constants: " + ", ".join(f"`{name}`" for name in nzv_columns[:20]),
            )
        )
    if identifier_columns:
        issues.append(
            _issue(
                "identifier_column",
                Severity.INFO,
                identifier_columns,
                "Identifier-like columns should be excluded from features (they memorise "
                "rows rather than generalise): "
                + ", ".join(f"`{name}`" for name in identifier_columns[:20]),
            )
        )

    if overview.n_duplicate_rows > 0:
        fraction = overview.duplicate_fraction
        severity = (
            Severity.HIGH
            if fraction >= DUPLICATE_HIGH
            else Severity.MEDIUM
            if fraction >= DUPLICATE_MEDIUM
            else Severity.LOW
        )
        issues.append(
            _issue(
                "duplicate_rows",
                severity,
                [],
                f"{overview.n_duplicate_rows:,} fully duplicated rows ({fraction:.2%} of the "
                "table). Duplicates leak between train and test splits and inflate scores.",
            )
        )

    if numeric_as_text:
        issues.append(
            _issue(
                "numeric_stored_as_text",
                Severity.MEDIUM,
                list(numeric_as_text),
                "Columns hold numbers stored as strings, so they would be one-hot "
                "encoded as categories unless cast first: "
                + ", ".join(f"`{name}`" for name in list(numeric_as_text)[:20]),
            )
        )

    for name, type_names in mixed_types.items():
        issues.append(
            _issue(
                "mixed_types",
                Severity.MEDIUM,
                [name],
                f"`{name}` mixes Python types {type_names}, so pandas fell back to object "
                "dtype. It needs a deliberate cast before any numeric or categorical "
                "treatment.",
            )
        )

    blank_entries = [
        (name, f"`{name}` ({count:,})")
        for name, count in blank_strings.items()
        if n_rows and count / n_rows >= BLANK_STRING_FRACTION
    ]
    if blank_entries:
        issues.append(
            _issue(
                "blank_strings",
                Severity.LOW,
                [name for name, _ in blank_entries],
                "Empty or whitespace-only strings are present and are NOT counted as "
                "missing by pandas: " + _render(blank_entries),
            )
        )
    untrimmed_entries = [
        (name, f"`{name}` ({count:,})")
        for name, count in untrimmed_strings.items()
        if count
    ]
    if untrimmed_entries:
        issues.append(
            _issue(
                "untrimmed_strings",
                Severity.LOW,
                [name for name, _ in untrimmed_entries],
                "Values carry leading/trailing whitespace, which splits categories that "
                "should be identical: " + _render(untrimmed_entries),
            )
        )

    gap_entries = [
        (name, f"`{name}` ({count:,} gaps)")
        for name, count in datetime_gaps.items()
        if count
    ]
    if gap_entries:
        issues.append(
            _issue(
                "datetime_gaps",
                Severity.MEDIUM,
                [name for name, _ in gap_entries],
                "Temporal columns skip periods relative to their modal spacing, so lag and "
                "rolling features cannot assume a regular grid: " + _render(gap_entries),
            )
        )
    if unsorted_datetimes:
        issues.append(
            _issue(
                "unsorted_datetime",
                Severity.LOW,
                list(unsorted_datetimes),
                "Temporal columns are not in ascending row order: "
                + ", ".join(f"`{name}`" for name in unsorted_datetimes[:20])
                + ". Any temporal split must sort first.",
            )
        )

    if highly_correlated:
        pairs = ", ".join(
            f"`{pair.left}`~`{pair.right}` ({pair.coefficient:.3f})"
            for pair in list(highly_correlated)[:12]
        )
        involved = sorted({p.left for p in highly_correlated} | {p.right for p in highly_correlated})
        issues.append(
            _issue(
                "multicollinearity",
                Severity.MEDIUM,
                involved,
                f"{len(highly_correlated)} feature pair(s) are almost perfectly correlated, "
                "which destabilises linear coefficients and splits tree importances: "
                + pairs,
            )
        )

    if n_rows and n_rows < TINY_DATASET_ROWS:
        issues.append(
            _issue(
                "tiny_dataset",
                Severity.HIGH,
                [],
                f"Only {n_rows:,} rows. Held-out estimates will have very wide confidence "
                "intervals; prefer cross-validation over a single split and favour "
                "high-bias models.",
            )
        )
    elif n_rows and n_rows < SMALL_DATASET_ROWS:
        issues.append(
            _issue(
                "small_dataset",
                Severity.MEDIUM,
                [],
                f"Only {n_rows:,} rows, so complex models will overfit and metric estimates "
                "will be noisy.",
            )
        )

    n_features = max(0, n_columns - (1 if target else 0))
    if n_rows and n_features >= n_rows:
        issues.append(
            _issue(
                "wide_dataset_p_over_n",
                Severity.HIGH,
                [],
                f"{n_features:,} features for {n_rows:,} rows (p >= n). Regularisation or "
                "dimensionality reduction is mandatory, not optional.",
            )
        )
    elif n_rows and n_features > WIDE_RATIO_MEDIUM * n_rows:
        issues.append(
            _issue(
                "wide_dataset",
                Severity.MEDIUM,
                [],
                f"{n_features:,} features for {n_rows:,} rows, so the feature-to-row ratio "
                "leaves little data per parameter.",
            )
        )

    if target is not None:
        if target.n_missing:
            fraction = target.n_missing / n_rows if n_rows else 0.0
            issues.append(
                _issue(
                    "target_missing_values",
                    Severity.HIGH if fraction >= MISSING_LOW else Severity.MEDIUM,
                    [target.name],
                    f"The target `{target.name}` is missing on {target.n_missing:,} rows "
                    f"({fraction:.2%}). Those rows cannot be used for supervised training.",
                )
            )
        singles = [c.value for c in target.class_counts if c.count == 1]
        if singles:
            issues.append(
                _issue(
                    "single_row_class",
                    Severity.HIGH,
                    [target.name],
                    f"Target classes with exactly one row: {singles[:20]}. Stratified "
                    "splitting and cross-validation both fail on these; they must be "
                    "merged or dropped.",
                )
            )
        rare = [
            f"{c.value} ({c.count})"
            for c in target.class_counts
            if 1 < c.count < cv_folds
        ]
        if rare:
            issues.append(
                _issue(
                    "rare_target_class",
                    Severity.MEDIUM,
                    [target.name],
                    f"Target classes with fewer rows than the {cv_folds} planned CV folds: "
                    f"{rare[:20]}.",
                )
            )
        if target.imbalance_ratio is not None and target.is_imbalanced:
            severity = (
                Severity.HIGH
                if target.imbalance_ratio >= IMBALANCE_HIGH
                else Severity.MEDIUM
            )
            issues.append(
                _issue(
                    "imbalanced_target",
                    severity,
                    [target.name],
                    f"Class imbalance ratio {target.imbalance_ratio:.1f}:1 "
                    f"(majority/minority). Accuracy will be misleading; use "
                    "precision/recall, PR-AUC, or balanced class weights.",
                )
            )
        if target.n_classes is not None and target.n_classes == 1:
            issues.append(
                _issue(
                    "constant_target",
                    Severity.CRITICAL,
                    [target.name],
                    f"The target `{target.name}` has a single class, so there is nothing to "
                    "learn.",
                )
            )

    for note in list(notes) + list(overview.notes):
        issues.append(_issue("profiling_note", Severity.INFO, [], note))

    return issues


__all__ = [
    "TargetAssociation",
    "detect_leakage",
    "detect_quality_issues",
    "score_target_associations",
    "target_is_classification_like",
]
