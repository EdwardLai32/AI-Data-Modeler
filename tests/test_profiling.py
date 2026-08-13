"""Profiler correctness on frames whose properties are known by construction.

The profiler is the foundation of the trust story: every number an agent reasons
over comes from here, so a wrong statistic is not a cosmetic bug — it produces a
confidently-argued wrong decision downstream. These tests therefore check values
against independently-computed pandas results rather than against themselves,
and they build frames where the right answer is arithmetic rather than opinion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_architect.core.schemas import ColumnKind, DatasetProfile, Severity

from .conftest import CHURN_LEAK, CHURN_TARGET, HOUSE_TARGET, import_or_skip

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def profile_dataframe():  # noqa: ANN201 - a function handle from a sibling module
    """The profiler entry point, or a skip if the profiling module is not written."""
    module = import_or_skip(
        "automl_architect.profiling.profiler",
        "automl_architect.profiling",
        feature="profiling/profiler.py",
    )
    if not hasattr(module, "profile_dataframe"):
        pytest.skip("profiling module exposes no profile_dataframe()")
    return module.profile_dataframe


# ---------------------------------------------------------------------------
# Purpose-built frames
# ---------------------------------------------------------------------------


@pytest.fixture
def known_frame() -> pd.DataFrame:
    """A 100-row frame where every statistic is known in advance.

    Column contents by design:
        ``ident``: 100 unique strings -> identifier.
        ``constant``: one value -> constant, zero variance.
        ``right_skew``: exponential -> positive skewness.
        ``left_skew``: mirrored exponential -> negative skewness.
        ``symmetric``: 0..99 -> skewness ~0.
        ``with_missing``: exactly 25 NaN.
        ``all_missing``: 100 NaN.
        ``category``: 3 levels at 60/30/10.
        ``flag``: boolean.
        ``email``: an addressable semantic type.
        ``free_text``: multi-sentence prose.
    """
    rng = np.random.default_rng(7)
    n = 100
    with_missing = pd.Series(np.arange(n, dtype="float64"))
    with_missing.iloc[:25] = np.nan
    return pd.DataFrame(
        {
            "ident": [f"ID-{i:04d}" for i in range(n)],
            "constant": ["same"] * n,
            "right_skew": rng.exponential(scale=2.0, size=n),
            "left_skew": -rng.exponential(scale=2.0, size=n),
            "symmetric": np.arange(n, dtype="float64"),
            "with_missing": with_missing,
            "all_missing": pd.Series([np.nan] * n, dtype="float64"),
            "category": ["a"] * 60 + ["b"] * 30 + ["c"] * 10,
            "flag": [True, False] * 50,
            "email": [f"user{i}@example.com" for i in range(n)],
            "free_text": [
                "The subscriber called about a billing discrepancy and asked for a credit."
                for _ in range(n)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Shape and coverage
# ---------------------------------------------------------------------------


def test_reports_exact_shape(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    assert isinstance(profile, DatasetProfile)
    assert profile.n_rows == len(known_frame)
    assert profile.n_columns == known_frame.shape[1]


def test_profiles_every_column_exactly_once(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """A missed column is invisible to every agent downstream."""
    profile = profile_dataframe(known_frame)
    names = [c.name for c in profile.columns]
    assert names == list(known_frame.columns) or sorted(names) == sorted(known_frame.columns)
    assert len(names) == len(set(names))


def test_column_lookup_matches_list(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    for column in profile.columns:
        assert profile.column(column.name) is column


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def test_missing_counts_are_exact(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    assert profile.column("with_missing").n_missing == 25
    assert profile.column("with_missing").missing_fraction == pytest.approx(0.25)
    assert profile.column("all_missing").n_missing == 100
    assert profile.column("all_missing").missing_fraction == pytest.approx(1.0)
    assert profile.column("symmetric").n_missing == 0


def test_total_missing_cells_matches_pandas(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    expected = int(known_frame.isna().sum().sum())
    assert profile.total_missing_cells == expected
    assert profile.missing_cell_fraction == pytest.approx(
        expected / (known_frame.shape[0] * known_frame.shape[1])
    )


def test_all_nan_column_does_not_crash(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """A fully-empty column is a real thing in exported data; it must degrade."""
    profile = profile_dataframe(known_frame)
    column = profile.column("all_missing")
    assert column is not None
    assert column.missing_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------------------------


def test_numeric_stats_match_independent_pandas(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """Mean/std/min/max are compared against pandas, not against the profiler."""
    profile = profile_dataframe(known_frame)
    for name in ("right_skew", "symmetric", "with_missing"):
        series = known_frame[name].dropna()
        column = profile.column(name)
        assert column.mean == pytest.approx(float(series.mean()), rel=1e-6)
        assert column.minimum == pytest.approx(float(series.min()), rel=1e-6)
        assert column.maximum == pytest.approx(float(series.max()), rel=1e-6)
        # std may be either ddof; both are within a few percent at n=75+.
        assert column.std == pytest.approx(float(series.std()), rel=0.02)


def test_quantiles_are_monotonic(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    quantiles = profile.column("symmetric").quantiles
    assert quantiles is not None
    values = [quantiles.p01, quantiles.p05, quantiles.p25, quantiles.p50, quantiles.p75, quantiles.p95, quantiles.p99]
    present = [v for v in values if v is not None]
    assert present == sorted(present)
    assert quantiles.p50 == pytest.approx(float(known_frame["symmetric"].median()), rel=1e-6)


def test_skewness_signs_are_correct(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """Sign of skewness drives the mean-vs-median imputation argument."""
    profile = profile_dataframe(known_frame)
    assert profile.column("right_skew").skewness > 0.5
    assert profile.column("left_skew").skewness < -0.5
    assert abs(profile.column("symmetric").skewness) < 0.1


def test_zero_and_negative_fractions(profile_dataframe) -> None:
    frame = pd.DataFrame({"mixed": [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    column = profile_dataframe(frame).column("mixed")
    assert column.zero_fraction == pytest.approx(0.2)
    assert column.negative_fraction == pytest.approx(0.2)


def test_outlier_summary_bounds_bracket_the_data(profile_dataframe) -> None:
    """IQR bounds must be ordered and the count must match them."""
    values = list(range(100)) + [10_000, 20_000]
    frame = pd.DataFrame({"v": values})
    column = profile_dataframe(frame).column("v")
    assert column.outliers is not None
    assert column.outliers.n_outliers >= 2
    if column.outliers.lower_bound is not None and column.outliers.upper_bound is not None:
        assert column.outliers.lower_bound < column.outliers.upper_bound
    assert 0.0 <= column.outliers.fraction <= 1.0


def test_constant_column_flagged(profile_dataframe, known_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(known_frame)
    assert profile.column("constant").is_constant is True
    assert "constant" in profile.constant_columns


def test_near_zero_variance_flagged(profile_dataframe) -> None:
    """999 identical values and one outlier is a near-useless feature."""
    frame = pd.DataFrame({"almost_constant": [1.0] * 999 + [2.0]})
    column = profile_dataframe(frame).column("almost_constant")
    assert column.is_near_zero_variance is True


# ---------------------------------------------------------------------------
# Kinds and semantic detection
# ---------------------------------------------------------------------------


def test_identifier_column_detected(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """A key with cardinality ratio 1.0 must be flagged, or a tree memorises it."""
    profile = profile_dataframe(known_frame)
    column = profile.column("ident")
    assert column.looks_like_id is True
    assert column.cardinality_ratio == pytest.approx(1.0)
    assert "ident" in profile.identifier_columns


def test_boolean_kind_detected(profile_dataframe, known_frame: pd.DataFrame) -> None:
    assert profile_dataframe(known_frame).column("flag").kind is ColumnKind.BOOLEAN


def test_categorical_top_values_are_ordered_and_exact(
    profile_dataframe, known_frame: pd.DataFrame
) -> None:
    profile = profile_dataframe(known_frame)
    top = profile.column("category").top_values
    assert [t.value for t in top][:3] == ["a", "b", "c"]
    assert [t.count for t in top][:3] == [60, 30, 10]
    assert top[0].fraction == pytest.approx(0.6)


def test_free_text_column_detected(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """Free text needs a different pipeline from a category, so it must be told apart."""
    profile = profile_dataframe(known_frame)
    column = profile.column("free_text")
    assert column.looks_like_text is True or column.kind is ColumnKind.TEXT
    assert column.mean_string_length is not None and column.mean_string_length > 30
    assert "free_text" in profile.text_columns


def test_email_semantic_type_detected(profile_dataframe, known_frame: pd.DataFrame) -> None:
    detected = profile_dataframe(known_frame).column("email").detected_semantic_type
    assert detected is not None and "email" in detected.lower()


def test_datetime_column_detected(profile_dataframe) -> None:
    frame = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=90, freq="D")})
    profile = profile_dataframe(frame)
    column = profile.column("when")
    assert column.kind is ColumnKind.DATETIME
    assert "when" in profile.temporal_columns
    assert column.min_timestamp is not None and "2024-01-01" in column.min_timestamp
    assert column.is_monotonic is True


def test_datetime_gaps_counted(profile_dataframe) -> None:
    """A gap in a daily series is the difference between a forecast and a fantasy."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D").to_list()
    del dates[20:27]  # a one-week outage
    profile = profile_dataframe(pd.DataFrame({"when": dates}))
    assert (profile.column("when").n_gaps or 0) >= 1


def test_duplicate_rows_counted(profile_dataframe) -> None:
    base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    frame = pd.concat([base, base.iloc[[0, 1]]], ignore_index=True)
    profile = profile_dataframe(frame)
    assert profile.n_duplicate_rows == 2
    assert profile.duplicate_fraction == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# Target summary
# ---------------------------------------------------------------------------


def test_target_none_leaves_summary_empty(profile_dataframe, known_frame: pd.DataFrame) -> None:
    assert profile_dataframe(known_frame, target=None).target is None


def test_classification_target_summary(profile_dataframe, churn_df: pd.DataFrame) -> None:
    """The churn label is 26% positive: two classes, imbalance ratio ~2.85."""
    profile = profile_dataframe(churn_df, target=CHURN_TARGET)
    target = profile.target
    assert target is not None
    assert target.name == CHURN_TARGET
    assert target.n_classes == 2
    assert sum(c.count for c in target.class_counts) == len(churn_df)
    assert target.imbalance_ratio == pytest.approx(2.85, abs=0.15)


def test_strong_imbalance_is_flagged(profile_dataframe) -> None:
    """The contract says ratio > 3 is meaningfully imbalanced; 19:1 must trip it."""
    frame = pd.DataFrame({"x": range(200), "y": [0] * 190 + [1] * 10})
    target = profile_dataframe(frame, target="y").target
    assert target is not None
    assert target.imbalance_ratio == pytest.approx(19.0, rel=0.01)
    assert target.is_imbalanced is True


def test_regression_target_summary(profile_dataframe, house_df: pd.DataFrame) -> None:
    """A right-skewed price target reports positive skewness, not a class count."""
    target = profile_dataframe(house_df, target=HOUSE_TARGET).target
    assert target is not None
    assert target.mean == pytest.approx(float(house_df[HOUSE_TARGET].mean()), rel=1e-6)
    assert target.skewness is not None and target.skewness > 1.0


# ---------------------------------------------------------------------------
# Correlations and leakage
# ---------------------------------------------------------------------------


def test_collinear_pair_reported(profile_dataframe, house_df: pd.DataFrame) -> None:
    """``sqft_living`` and ``sqft_above`` correlate at 0.945 by construction."""
    profile = profile_dataframe(house_df, target=HOUSE_TARGET)
    pairs = {frozenset((p.left, p.right)) for p in profile.highly_correlated_pairs}
    assert frozenset(("sqft_living", "sqft_above")) in pairs


def test_target_correlations_are_sorted_by_strength(
    profile_dataframe, house_df: pd.DataFrame
) -> None:
    profile = profile_dataframe(house_df, target=HOUSE_TARGET)
    strengths = [abs(p.coefficient) for p in profile.target_correlations]
    assert strengths == sorted(strengths, reverse=True)
    for pair in profile.target_correlations:
        assert -1.0 <= pair.coefficient <= 1.0


def test_leakage_column_is_found(profile_dataframe, churn_df: pd.DataFrame) -> None:
    """churn.csv ships one true leak; the detector must find it.

    ``cancellation_tickets`` reaches AUC 0.9975 against the target while the best
    legitimate feature reaches 0.627, so this is the easiest possible positive
    case. Failing it means leakage detection is not functioning at all.
    """
    profile = profile_dataframe(churn_df, target=CHURN_TARGET)
    flagged = {f.column for f in profile.leakage_findings}
    assert CHURN_LEAK in flagged, f"leak not detected; flagged={flagged}"
    finding = next(f for f in profile.leakage_findings if f.column == CHURN_LEAK)
    assert finding.score >= 0.8
    assert finding.severity in (Severity.HIGH, Severity.CRITICAL)
    assert finding.reason.strip()


def test_leakage_does_not_flag_legitimate_features(
    profile_dataframe, churn_df: pd.DataFrame
) -> None:
    """False positives cost real signal, so the bar has to stay high."""
    profile = profile_dataframe(churn_df, target=CHURN_TARGET)
    flagged = {f.column for f in profile.leakage_findings}
    for legitimate in ("tenure_months", "monthly_charges", "satisfaction_score", "support_tickets"):
        assert legitimate not in flagged, f"{legitimate} wrongly flagged as leakage"
    assert CHURN_TARGET not in flagged, "the target itself must never be a leakage finding"


def test_leak_outranks_every_other_finding(profile_dataframe, churn_df: pd.DataFrame) -> None:
    profile = profile_dataframe(churn_df, target=CHURN_TARGET)
    if len(profile.leakage_findings) > 1:
        top = max(profile.leakage_findings, key=lambda f: f.score)
        assert top.column == CHURN_LEAK


# ---------------------------------------------------------------------------
# Quality issues
# ---------------------------------------------------------------------------


def test_quality_issues_are_actionable(profile_dataframe, known_frame: pd.DataFrame) -> None:
    """Each issue names a code, a severity, and enough detail to act on."""
    profile = profile_dataframe(known_frame)
    assert profile.quality_issues, "a frame with a constant and an all-NaN column has issues"
    for issue in profile.quality_issues:
        assert issue.code.strip()
        assert issue.detail.strip()
        assert isinstance(issue.severity, Severity)
        for column in issue.columns:
            assert column in known_frame.columns


# ---------------------------------------------------------------------------
# Determinism and sampling
# ---------------------------------------------------------------------------


def _stable_view(profile: DatasetProfile) -> dict:
    payload = profile.model_dump(mode="json")
    for volatile in ("profiled_at", "profile_seconds", "dataset_id"):
        payload.pop(volatile, None)
    return payload


def test_profiling_is_deterministic(profile_dataframe, churn_df: pd.DataFrame) -> None:
    """Two profiles of one frame must be identical, or nothing downstream is reproducible."""
    first = profile_dataframe(churn_df, target=CHURN_TARGET)
    second = profile_dataframe(churn_df, target=CHURN_TARGET)
    assert _stable_view(first) == _stable_view(second)


def test_sampling_still_reports_true_row_count(profile_dataframe, churn_df: pd.DataFrame) -> None:
    """Statistics may sample; the reported shape may not.

    An agent told "500 rows" would size its plan for 500 rows. The settings
    docstring is explicit: the full row count is always reported.
    """
    profile = profile_dataframe(churn_df, target=CHURN_TARGET, sample_rows=500)
    assert profile.n_rows == len(churn_df)
    assert profile.n_columns == churn_df.shape[1]


def test_profile_seconds_recorded(profile_dataframe, known_frame: pd.DataFrame) -> None:
    assert profile_dataframe(known_frame).profile_seconds >= 0.0


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_single_row_frame(profile_dataframe) -> None:
    profile = profile_dataframe(pd.DataFrame({"a": [1.0], "b": ["x"]}), target="a")
    assert profile.n_rows == 1
    assert profile.n_columns == 2


def test_single_column_frame(profile_dataframe) -> None:
    profile = profile_dataframe(pd.DataFrame({"only": [1, 2, 3, 4]}))
    assert profile.n_columns == 1
    assert not profile.highly_correlated_pairs


def test_empty_frame_degrades(profile_dataframe) -> None:
    """Zero rows must produce a profile that says so, not divide by zero."""
    profile = profile_dataframe(pd.DataFrame({"a": pd.Series(dtype="float64")}))
    assert profile.n_rows == 0
    assert profile.missing_cell_fraction in (0.0, pytest.approx(0.0))


def test_duplicate_column_names_do_not_crash(profile_dataframe) -> None:
    """Exported CSVs really do contain repeated headers."""
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    profile = profile_dataframe(frame)
    assert profile.n_columns == 2


def test_wide_frame_profiles_every_column(profile_dataframe) -> None:
    """The context renderer abbreviates above 60 columns; the profile must not."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(50, 120)), columns=[f"f{i}" for i in range(120)])
    profile = profile_dataframe(frame)
    assert profile.n_columns == 120
    assert len(profile.columns) == 120
