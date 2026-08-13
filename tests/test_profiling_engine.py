"""Branch coverage for the profiling engine's inference and detection paths.

``test_profiling.py`` checks that the numbers are right. This module checks that
every *branch* fires: each column kind, each semantic regex, each quality code,
and each of the four leakage detectors — including the negative case that a
suspicious column name with no statistical signal is deliberately not reported.

Run directly (``python -m tests.test_profiling_engine`` from the repo root) to
print the profile. The ``-m`` form is required: invoking the file by path puts
``tests/`` on ``sys.path`` instead of the repo root, so ``automl_architect``
would not import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_architect.core.context import render_profile
from automl_architect.core.schemas import ColumnKind, DatasetProfile, Severity
from automl_architect.profiling import (
    correlation_ratio,
    cramers_v,
    profile_dataframe,
    summarise_profile,
)

N = 600


def build_messy_frame() -> pd.DataFrame:
    """A wide customer table exercising every inference branch at once."""
    rng = np.random.default_rng(11)
    churned = rng.random(N) < 0.10  # imbalanced boolean target

    # Daily stamps with a hole every sixth row, so gap detection has something.
    days = np.arange(N) + (np.arange(N) // 6) * 3
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(days, unit="D")

    tenure = rng.integers(1, 60, size=N)
    monthly = rng.lognormal(mean=3.6, sigma=0.9, size=N).round(2)  # heavy right tail
    total_spend = (monthly * tenure).round(2)

    missing_score = rng.normal(50.0, 12.0, size=N)
    missing_score[rng.random(N) < 0.62] = np.nan

    vocabulary = [
        "the", "agent", "resolved", "my", "billing", "question", "quickly",
        "but", "app", "kept", "crashing", "during", "checkout", "again",
    ]
    notes = [
        " ".join(rng.choice(vocabulary, size=int(rng.integers(8, 20))))
        for _ in range(N)
    ]

    frame = pd.DataFrame(
        {
            # --- identifiers
            "customer_id": [f"CUST-{i:07d}" for i in range(N)],
            "row_index": np.arange(N),
            "account_uuid": [f"{i:08x}-1234-5678-9abc-{i:012x}" for i in range(N)],
            "contact_email": [f"user{i}@example.com" for i in range(N)],
            # --- temporal (one hiding inside strings)
            "signup_date": signup.strftime("%Y-%m-%d"),
            "last_seen_at": signup + pd.to_timedelta(rng.integers(0, 400, N), unit="D"),
            # --- numeric
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_spend": total_spend,
            "total_spend_usd": total_spend * 2.0,  # perfectly collinear
            "missing_score": missing_score,
            "support_calls": rng.poisson(1.2, size=N),
            # --- categorical
            "region": rng.choice(["north", "south", "east", "west"], size=N),
            "satisfaction": rng.choice(["low", "medium", "high"], size=N),
            "plan_tier": rng.choice(["bronze", "silver", "gold"], size=N),
            "country": rng.choice(["US", "GB", "DE", "FR"], size=N),
            "zip_code": [f"{90000 + int(v):05d}" for v in rng.integers(0, 900, N)],
            # --- geographic coordinates
            "latitude": rng.uniform(24.5, 49.0, size=N).round(5),
            "longitude": rng.uniform(-124.0, -67.0, size=N).round(5),
            # --- semantic strings
            "homepage_url": [f"https://site{i % 97}.example.com/profile" for i in range(N)],
            "last_login_ip": [
                f"{rng.integers(1, 255)}.{rng.integers(0, 255)}."
                f"{rng.integers(0, 255)}.{rng.integers(1, 255)}"
                for _ in range(N)
            ],
            "support_phone": [f"+1 (415) 555-{i % 10000:04d}" for i in range(N)],
            "list_price": [f"${(i % 400) + 0.99:,.2f}" for i in range(N)],
            "discount_pct": [f"{i % 60}%" for i in range(N)],
            # --- free text
            "support_notes": notes,
            # --- degenerate
            "data_version": "v1",
            "legacy_flag": rng.random(N) < 0.008,
            "dirty_category": rng.choice([" alpha ", "beta", "", "gamma"], size=N),
            "mixed_column": [i if i % 3 else f"code_{i}" for i in range(N)],
            # --- target and planted leakage
            "churned": churned,
            "churn_reason": np.where(
                churned, rng.choice(["price", "service", "competitor"], size=N), "still_active"
            ),
            "risk_after_review": churned * 2.4 + rng.normal(0, 1.15, size=N),
            "cancel_risk_score": churned * 1.7 + rng.normal(0, 1.0, size=N),
            # --- suspicious names with NO signal: must not be reported
            "outcome_survey_channel": rng.choice(["crm", "web", "app"], size=N),
            "label_source_system": rng.choice(["batch", "stream"], size=N),
        }
    )
    frame.loc[frame.sample(n=18, random_state=3).index, "region"] = np.nan
    return pd.concat([frame, frame.iloc[:24]], ignore_index=True)  # 24 duplicate rows


def build_wide_regression_frame() -> pd.DataFrame:
    """A tiny wide regression table with a monotone-transform leak."""
    rng = np.random.default_rng(23)
    n = 40
    price = rng.gamma(shape=3.0, scale=90_000.0, size=n).round(2)
    data: dict[str, object] = {f"feature_{i:02d}": rng.normal(size=n) for i in range(45)}
    data["area_sqft"] = (price / 210.0 + rng.normal(0, 15, size=n)).round(1)
    data["log_price"] = np.log(price)  # monotone transform of the target
    data["price"] = price
    return pd.DataFrame(data)


def build_rare_class_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    labels = ["a"] * 60 + ["b"] * 30 + ["c"] * 3 + ["d"]
    return pd.DataFrame(
        {
            "grade": labels,
            "score": rng.normal(size=len(labels)),
            "amount": rng.gamma(2.0, 5.0, size=len(labels)),
        }
    )


# --- helpers --------------------------------------------------------------


def kinds_of(profile: DatasetProfile) -> dict[str, ColumnKind]:
    return {column.name: column.kind for column in profile.columns}


def codes_of(profile: DatasetProfile) -> set[str]:
    return {issue.code for issue in profile.quality_issues}


def leaks_of(profile: DatasetProfile) -> dict[str, Severity]:
    return {finding.column: finding.severity for finding in profile.leakage_findings}


_MESSY: DatasetProfile | None = None


def messy_profile() -> DatasetProfile:
    """Profile the messy frame once; it is the fixture for most checks here."""
    global _MESSY
    if _MESSY is None:
        _MESSY = profile_dataframe(build_messy_frame(), target="churned")
    return _MESSY


# --- kind inference -------------------------------------------------------


def test_every_column_kind_branch_fires() -> None:
    kinds = kinds_of(messy_profile())
    expected = {
        "customer_id": ColumnKind.IDENTIFIER,
        "row_index": ColumnKind.IDENTIFIER,
        "account_uuid": ColumnKind.IDENTIFIER,
        "contact_email": ColumnKind.IDENTIFIER,
        "signup_date": ColumnKind.DATETIME,  # dates hidden in strings
        "last_seen_at": ColumnKind.DATETIME,
        "monthly_charges": ColumnKind.NUMERIC_CONTINUOUS,
        "support_calls": ColumnKind.NUMERIC_DISCRETE,
        "region": ColumnKind.GEO,  # a geographic name hint outranks "just a label"
        "outcome_survey_channel": ColumnKind.CATEGORICAL_NOMINAL,
        "satisfaction": ColumnKind.CATEGORICAL_ORDINAL,
        "plan_tier": ColumnKind.CATEGORICAL_ORDINAL,
        "churned": ColumnKind.BOOLEAN,
        "legacy_flag": ColumnKind.BOOLEAN,
        "support_notes": ColumnKind.TEXT,
        "latitude": ColumnKind.GEO,
        "longitude": ColumnKind.GEO,
        "country": ColumnKind.GEO,
        "zip_code": ColumnKind.GEO,
        "data_version": ColumnKind.CONSTANT,
    }
    wrong = {name: kinds[name] for name, want in expected.items() if kinds[name] is not want}
    assert not wrong, f"misclassified: {wrong}"
    # Every kind in the expectation set is represented, so no branch is untested.
    assert set(expected.values()) <= set(kinds.values())


def test_semantic_column_groups() -> None:
    profile = messy_profile()
    assert set(profile.temporal_columns) == {"signup_date", "last_seen_at"}
    assert profile.text_columns == ["support_notes"]
    assert {"latitude", "longitude", "country", "zip_code"} <= set(profile.geo_columns)
    assert {"customer_id", "row_index", "account_uuid"} <= set(profile.identifier_columns)
    assert "data_version" in profile.constant_columns


def test_semantic_regex_detection() -> None:
    semantic = {c.name: c.detected_semantic_type for c in messy_profile().columns}
    assert semantic["contact_email"] == "email"
    assert semantic["homepage_url"] == "url"
    assert semantic["last_login_ip"] == "ipv4"
    assert semantic["account_uuid"] == "uuid"
    assert semantic["support_phone"] == "phone"
    assert semantic["zip_code"] == "postal_code"
    assert semantic["list_price"] == "currency"
    assert semantic["discount_pct"] == "percentage"
    assert semantic["latitude"] == "latitude"
    assert semantic["longitude"] == "longitude"


def test_numeric_statistics_are_complete() -> None:
    charges = messy_profile().column("monthly_charges")
    assert charges is not None
    for attribute in (
        "mean", "std", "variance", "minimum", "maximum", "skewness", "kurtosis",
        "zero_fraction", "negative_fraction",
    ):
        assert getattr(charges, attribute) is not None, attribute
    assert charges.skewness > 1.0, "a lognormal column must read as right-skewed"
    assert charges.quantiles is not None
    q = charges.quantiles
    values = [q.p01, q.p05, q.p25, q.p50, q.p75, q.p95, q.p99]
    assert None not in values, "all seven quantiles are required"
    assert values == sorted(values)
    assert charges.outliers is not None and charges.outliers.n_outliers > 0
    assert charges.outliers.lower_bound is not None
    assert charges.outliers.upper_bound is not None
    assert charges.memory_bytes > 0


def test_string_and_text_statistics() -> None:
    notes = messy_profile().column("support_notes")
    assert notes is not None
    assert notes.mean_token_count is not None and notes.mean_token_count > 5
    assert notes.mean_string_length is not None and notes.mean_string_length > 30
    assert notes.max_string_length is not None and notes.max_string_length > 20


def test_datetime_statistics() -> None:
    signup = messy_profile().column("signup_date")
    assert signup is not None
    assert signup.min_timestamp and signup.max_timestamp
    assert signup.min_timestamp < signup.max_timestamp
    assert signup.inferred_frequency, "a frequency label is always derivable"
    assert signup.n_gaps and signup.n_gaps > 0
    assert signup.looks_like_datetime is True
    # The 24 appended duplicate rows restart the calendar, so row order is not
    # ascending and the temporal-split warning must fire.
    assert signup.is_monotonic is False
    assert "unsorted_datetime" in codes_of(messy_profile())

    ordered = profile_dataframe(
        pd.DataFrame({"when": pd.date_range("2024-01-01", periods=90, freq="D")})
    ).column("when")
    assert ordered is not None
    assert ordered.is_monotonic is True
    assert ordered.inferred_frequency == "D"
    assert ordered.n_gaps == 0


def test_variance_flags() -> None:
    profile = messy_profile()
    constant = profile.column("data_version")
    assert constant is not None and constant.is_constant and constant.is_near_zero_variance
    flag = profile.column("legacy_flag")
    assert flag is not None and flag.is_near_zero_variance and not flag.is_constant


def test_top_values_sum_to_one() -> None:
    region = messy_profile().column("region")
    assert region is not None and region.top_values
    assert abs(sum(v.fraction for v in region.top_values) - 1.0) < 1e-9


# --- target summary ------------------------------------------------------


def test_imbalanced_boolean_target() -> None:
    target = messy_profile().target
    assert target is not None
    assert target.name == "churned" and target.n_classes == 2
    assert target.imbalance_ratio is not None and target.imbalance_ratio > 3.0
    assert target.is_imbalanced is True
    assert "imbalanced_target" in codes_of(messy_profile())


# --- correlations --------------------------------------------------------


def test_correlations_span_mixed_types() -> None:
    profile = messy_profile()
    methods = {pair.method for pair in profile.top_correlations}
    assert "pearson" in methods
    assert {"cramers_v", "correlation_ratio"} & methods, "mixed-type pairs must be covered"

    collinear = {tuple(sorted((p.left, p.right))) for p in profile.highly_correlated_pairs}
    assert ("total_spend", "total_spend_usd") in collinear
    assert "multicollinearity" in codes_of(profile)

    assert profile.target_correlations
    strengths = [abs(p.coefficient) for p in profile.target_correlations]
    assert strengths == sorted(strengths, reverse=True)
    assert all(p.right == "churned" for p in profile.target_correlations)
    assert all(-1.0 <= p.coefficient <= 1.0 for p in profile.target_correlations)


# --- leakage -------------------------------------------------------------


def test_cramers_v_is_bias_corrected() -> None:
    """Uncorrected V reaches 1.0 for a unique-per-row column against any target."""
    rng = np.random.default_rng(4)
    n = 400
    unique = [f"k{i}" for i in range(n)]
    labels = rng.choice(["yes", "no"], size=n)
    inflated = cramers_v(unique, labels)
    assert inflated is not None and inflated < 0.25, "memorisation must not read as 1.0"

    real = rng.choice(["a", "b"], size=n)
    assert cramers_v(real, np.where(real == "a", "yes", "no")) == pytest.approx(1.0, abs=1e-6)


def test_near_perfect_predictor_is_flagged() -> None:
    profile = messy_profile()
    leaks = leaks_of(profile)
    assert "churn_reason" in leaks
    assert leaks["churn_reason"] in {Severity.HIGH, Severity.CRITICAL}
    assert all(0.0 <= f.score <= 1.0 for f in profile.leakage_findings)
    assert all(f.reason.strip() and f.method.strip() for f in profile.leakage_findings)


def test_suspicious_name_without_signal_is_not_leakage() -> None:
    """The stated rule: a name match alone is never evidence."""
    leaks = leaks_of(messy_profile())
    assert "outcome_survey_channel" not in leaks
    assert "label_source_system" not in leaks
    assert "churned" not in leaks, "the target is never its own leak"


def test_name_plus_signal_detector_explains_itself() -> None:
    profile = messy_profile()
    named = [
        f for f in profile.leakage_findings
        if f.column in {"cancel_risk_score", "risk_after_review"}
    ]
    assert named, "a suspicious name with real signal must be reported"
    assert any(
        "name alone" in f.reason or "points the same way" in f.reason for f in named
    ), "the reason must say the name alone would not qualify"


def test_high_cardinality_columns_do_not_fake_a_perfect_association() -> None:
    """The bug this guards: a one-level-per-row column explains 100% of any
    numeric target's variance by memorisation, and an uncorrected correlation
    ratio reports that as 1.0 — a CRITICAL leakage finding on the primary key of
    every table. Bias correction must return nothing instead."""
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame(
        {
            "property_id": [f"P-{i:05d}" for i in range(n)],
            "noise_code": rng.integers(0, 200, size=n).astype(str),
            "size_sqft": rng.normal(1500, 300, size=n),
            "sale_price": rng.gamma(3.0, 90_000.0, size=n),
        }
    )
    profile = profile_dataframe(frame, target="sale_price")
    flagged = {f.column for f in profile.leakage_findings}
    assert "property_id" not in flagged, "an identifier is not a leak just by being unique"
    assert "noise_code" not in flagged
    assert correlation_ratio([f"id{i}" for i in range(n)], rng.normal(size=n)) is None
    random_eta = correlation_ratio(rng.integers(0, 50, n).astype(str), rng.normal(size=n))
    assert random_eta is not None and random_eta < 0.2, "a random grouping explains nothing"

    groups = rng.integers(0, 3, n)
    real_eta = correlation_ratio(groups.astype(str), groups * 3.0 + rng.normal(size=n))
    assert real_eta is not None and real_eta > 0.8, "a real relationship must survive"


def test_duplicate_of_target_is_critical() -> None:
    frame = build_rare_class_frame()
    frame["grade_copy"] = frame["grade"]
    profile = profile_dataframe(frame, target="grade")
    finding = next(f for f in profile.leakage_findings if f.column == "grade_copy")
    assert finding.severity is Severity.CRITICAL
    assert "identical" in finding.reason


def test_monotone_transform_of_target_is_critical() -> None:
    profile = profile_dataframe(build_wide_regression_frame(), target="price")
    finding = next(f for f in profile.leakage_findings if f.column == "log_price")
    assert finding.severity is Severity.CRITICAL
    assert "monotone" in finding.reason.lower()


# --- quality issues ------------------------------------------------------


def test_all_quality_codes_fire() -> None:
    codes = codes_of(messy_profile())
    for expected in (
        "high_missingness",
        "constant_column",
        "near_zero_variance",
        "duplicate_rows",
        "skewed_numeric",
        "high_cardinality_categorical",
        "mixed_types",
        "imbalanced_target",
        "datetime_gaps",
        "identifier_column",
        "blank_strings",
        "untrimmed_strings",
        "multicollinearity",
    ):
        assert expected in codes, f"missing quality code {expected}; got {sorted(codes)}"


def test_issue_columns_are_real_columns() -> None:
    frame = build_messy_frame()
    for issue in messy_profile().quality_issues:
        for column in issue.columns:
            assert column in frame.columns, f"{issue.code} names unknown column {column}"


def test_frame_level_integrity_counts() -> None:
    profile = messy_profile()
    assert profile.n_rows == N + 24
    assert profile.n_duplicate_rows == 24
    assert profile.duplicate_fraction > 0
    assert profile.total_missing_cells > 0
    assert 0.0 < profile.missing_cell_fraction < 1.0
    assert profile.memory_bytes > 0
    assert profile.profile_seconds > 0
    assert profile.dataset_id.startswith("ds_")


def test_p_over_n_and_tiny_dataset() -> None:
    codes = codes_of(profile_dataframe(build_wide_regression_frame(), target="price"))
    assert "wide_dataset_p_over_n" in codes
    assert "tiny_dataset" in codes


def test_rare_and_single_row_classes() -> None:
    codes = codes_of(profile_dataframe(build_rare_class_frame(), target="grade"))
    assert "single_row_class" in codes
    assert "rare_target_class" in codes


def test_continuous_target_has_no_class_counts() -> None:
    target = profile_dataframe(build_wide_regression_frame(), target="price").target
    assert target is not None
    assert target.n_classes is None
    assert target.mean is not None and target.std is not None and target.skewness is not None


# --- degradation ---------------------------------------------------------


def test_sampling_reports_true_row_count() -> None:
    frame = build_rare_class_frame()
    profile = profile_dataframe(frame, target="grade", sample_rows=30)
    assert profile.n_rows == len(frame)
    assert any("random sample" in issue.detail for issue in profile.quality_issues)


def test_unknown_target_and_pathological_columns_degrade() -> None:
    frame = pd.DataFrame(
        {
            "all_null": [None] * 30,
            "unhashable": [[1, 2]] * 30,
            "ok": list(range(30)),
        }
    )
    profile = profile_dataframe(frame, target="does_not_exist")
    assert profile.target is None
    assert profile.n_columns == 3
    assert any("does_not_exist" in issue.detail for issue in profile.quality_issues)
    assert "all_missing_column" in codes_of(profile)


def test_exotic_dtypes_still_yield_statistics() -> None:
    """Timedelta, Period, and numbers-stored-as-text all support real statistics."""
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame(
        {
            "duration": pd.to_timedelta(rng.integers(1, 5000, size=n), unit="s"),
            "month_period": pd.period_range("2020-01", periods=n, freq="M"),
            "amount_text": [f"{rng.integers(1000, 99999):,}.{rng.integers(10, 99)}" for _ in range(n)],
        }
    )
    profile = profile_dataframe(frame)

    duration = profile.column("duration")
    assert duration is not None
    assert duration.kind is ColumnKind.NUMERIC_CONTINUOUS
    assert duration.mean is not None and duration.std is not None
    assert duration.mean_string_length is None, "a duration is not a string"

    period = profile.column("month_period")
    assert period is not None
    assert period.kind is ColumnKind.DATETIME
    assert period.min_timestamp is not None and period.inferred_frequency

    amount = profile.column("amount_text")
    assert amount is not None
    assert amount.detected_semantic_type == "numeric_as_text"
    assert amount.mean is not None and amount.quantiles is not None
    assert "numeric_stored_as_text" in codes_of(profile)


def test_duplicate_labels_are_disambiguated() -> None:
    frame = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["value", "value"])
    profile = profile_dataframe(frame)
    assert [c.name for c in profile.columns] == ["value", "value__2"]


def test_renders_into_the_prompt_context() -> None:
    profile = messy_profile()
    text = render_profile(profile)
    assert "MEASURED DATASET FACTS" in text
    assert "Target-leakage candidates" in text
    assert "churn_reason" in text
    assert render_profile(profile) == text, "rendering must be deterministic"


if __name__ == "__main__":
    import sys
    import traceback

    result = messy_profile()
    print(summarise_profile(result))
    print("\n--- kinds ---")
    for column in result.columns:
        print(
            f"  {column.name:24} {column.kind.value:22} "
            f"semantic={column.detected_semantic_type or '-':16} "
            f"unique={column.n_unique:<6} missing={column.missing_fraction:.1%}"
        )
    print("\n--- leakage ---")
    for item in result.leakage_findings:
        print(f"  [{item.severity.value:8}] {item.column:22} {item.score:.4f} ({item.method})")
        print(f"      {item.reason}")
    print("\n--- quality issues ---")
    for item in sorted(result.quality_issues, key=lambda i: i.code):
        print(f"  [{item.severity.value:8}] {item.code}: {item.detail[:140]}")
    print("\n--- top feature correlations ---")
    for pair in result.top_correlations[:10]:
        print(f"  {pair.left:20} ~ {pair.right:20} {pair.coefficient:+.4f} ({pair.method})")
    print("\n--- target correlations ---")
    for pair in result.target_correlations[:10]:
        print(f"  {pair.left:20} -> {pair.right:10} {pair.coefficient:+.4f} ({pair.method})")

    failures = 0
    checks = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    print()
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {check.__name__}")
            traceback.print_exc()
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(1 if failures else 0)
