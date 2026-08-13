"""Generate the example datasets shipped with AutoML Architect.

Every dataset here is synthetic but built to behave like real data, because the
point of an example dataset in this project is to exercise a specific code path.
Nothing is random at runtime: a fixed seed plus integer/float rounding means
re-running this script byte-for-byte reproduces the CSVs, so the examples in the
README and the tests that assert on them stay honest.

What each dataset is for:

``churn.csv``
    Binary classification with a genuinely imbalanced target (~26% positive),
    mixed dtypes, a right-skewed income column with missing values, a
    high-cardinality identifier, a categorical with a sub-1% rare level, and one
    deliberate leakage column (``cancellation_tickets``) so the leakage detector
    has something true to find rather than only false positives to avoid.

``house_prices.csv``
    Regression with a right-skewed (log-normal) target, a deliberately
    near-collinear feature pair, and a handful of luxury-property outliers that
    move the mean far off the median.

``sales_timeseries.csv``
    Panel time series: a date column, four store series under one group key,
    linear trend plus weekly seasonality, promotion effects, and two calendar
    gaps so gap detection and frequency inference have real work to do.

Run it with::

    python examples/generate_datasets.py            # writes next to this file
    python examples/generate_datasets.py --out /tmp # writes elsewhere
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 20260728
CHURN_ROWS = 3000
HOUSE_ROWS = 1600
TS_START = "2023-01-01"
TS_END = "2024-12-31"
TS_STORES = ("S001", "S002", "S003", "S004")

#: Deliberate leakage column in ``churn.csv``. Named here so tests and docs
#: reference one constant instead of a string literal in three places.
CHURN_LEAKAGE_COLUMN = "cancellation_tickets"
CHURN_TARGET = "churned"
CHURN_POSITIVE_RATE = 0.26

HOUSE_TARGET = "sale_price"
TS_TARGET = "units_sold"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bernoulli_at_rate(
    logit: np.ndarray, uniforms: np.ndarray, target_rate: float
) -> np.ndarray:
    """Draw labels from ``logit`` with the *realised* positive rate pinned.

    Solving for an intercept that sets ``mean(sigmoid(logit + b))`` leaves the
    realised rate a coin-flip away from target (±0.8pp at n=3000), which is
    enough to make a documented "26% positive" wrong. Bisecting against the
    already-drawn uniforms instead fixes the realised count exactly while
    keeping the label's dependence on the features intact.
    """
    low, high = -25.0, 25.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if (uniforms < _sigmoid(logit + mid)).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (uniforms < _sigmoid(logit + (low + high) / 2.0)).astype(int)


def _choice(rng: np.random.Generator, values: list[str], probs: list[float], n: int) -> np.ndarray:
    weights = np.asarray(probs, dtype=float)
    return rng.choice(values, size=n, p=weights / weights.sum())


# ---------------------------------------------------------------------------
# churn
# ---------------------------------------------------------------------------


def make_churn(n_rows: int = CHURN_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the imbalanced binary-classification example.

    Args:
        n_rows: Number of customers to generate.
        seed: Base seed; the generator is derived from it deterministically.

    Returns:
        A dataframe whose ``churned`` column is roughly 26% positive and which
        contains exactly one leakage column, ``cancellation_tickets``.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_rows)
    contract = _choice(
        rng,
        ["month_to_month", "one_year", "two_year"],
        [0.55, 0.26, 0.19],
        n_rows,
    )
    # A rare level at ~0.4% so rare-category handling has a real case to hit.
    payment_method = _choice(
        rng,
        ["credit_card", "bank_transfer", "electronic_check", "mailed_check", "crypto_wallet"],
        [0.34, 0.25, 0.28, 0.129, 0.004],
        n_rows,
    )
    internet = _choice(rng, ["fiber", "dsl", "none"], [0.44, 0.40, 0.16], n_rows)
    region = _choice(rng, ["north", "south", "east", "west"], [0.3, 0.27, 0.23, 0.2], n_rows)

    base_charge = np.where(internet == "fiber", 74.0, np.where(internet == "dsl", 52.0, 21.0))
    monthly_charges = np.round(base_charge + rng.normal(0, 9.5, n_rows), 2).clip(15.0, 140.0)
    # total_charges is tenure * monthly plus drift: strongly but not perfectly
    # correlated with both, which is what a multicollinearity check should see.
    total_charges = np.round(
        monthly_charges * tenure * rng.normal(1.0, 0.045, n_rows), 2
    ).clip(0.0, None)

    support_tickets = rng.poisson(
        np.where(contract == "month_to_month", 1.9, 0.9), size=n_rows
    )
    paperless = rng.random(n_rows) < 0.59
    autopay = rng.random(n_rows) < np.where(contract == "two_year", 0.78, 0.42)

    # Right-skewed income: log-normal, so mean >> median and the median is the
    # defensible imputation choice. ~9% missing, missing-at-random.
    annual_income = np.round(rng.lognormal(mean=10.85, sigma=0.62, size=n_rows), 0)
    income_missing = rng.random(n_rows) < 0.09
    annual_income_col: Any = annual_income.astype("float64")
    annual_income_col[income_missing] = np.nan

    satisfaction = np.round(rng.normal(6.8, 1.9, n_rows), 1).clip(1.0, 10.0)

    logit = (
        -0.031 * tenure
        + 0.0165 * (monthly_charges - 60.0)
        + 0.29 * support_tickets
        - 0.33 * satisfaction
        + np.where(contract == "month_to_month", 1.05, np.where(contract == "one_year", 0.1, -0.72))
        + np.where(internet == "fiber", 0.44, 0.0)
        + np.where(autopay, -0.36, 0.0)
        + rng.normal(0, 0.55, n_rows)
    )
    churned = _bernoulli_at_rate(logit, rng.random(n_rows), CHURN_POSITIVE_RATE)

    # --- the deliberate leak -------------------------------------------------
    # Cancellation tickets are only ever filed *after* a customer decides to
    # leave, so this column cannot exist at prediction time. It is near-perfectly
    # separating (a 1.5% false-positive rate keeps it from being a literal copy
    # of the label), which is exactly the signature a leakage detector hunts for.
    leak = np.where(
        churned == 1,
        rng.integers(1, 6, size=n_rows),
        (rng.random(n_rows) < 0.015).astype(int),
    )

    signup_day = rng.integers(0, 1460, size=n_rows)
    signup_date = pd.Timestamp("2020-01-01") + pd.to_timedelta(signup_day, unit="D")

    frame = pd.DataFrame(
        {
            # High-cardinality identifier: unique per row, no predictive content.
            "customer_id": [f"CUST-{i:06d}" for i in rng.permutation(n_rows)],
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tenure_months": tenure.astype("int64"),
            "contract_type": contract,
            "payment_method": payment_method,
            "internet_service": internet,
            "region": region,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "annual_income": annual_income_col,
            "support_tickets": support_tickets.astype("int64"),
            "satisfaction_score": satisfaction,
            "has_paperless_billing": paperless,
            "is_autopay": autopay,
            CHURN_LEAKAGE_COLUMN: leak.astype("int64"),
            CHURN_TARGET: churned.astype("int64"),
        }
    )
    return frame


# ---------------------------------------------------------------------------
# house prices
# ---------------------------------------------------------------------------


def make_house_prices(n_rows: int = HOUSE_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the regression example: skewed target, collinearity, outliers.

    Args:
        n_rows: Number of properties to generate.
        seed: Base seed. Offset internally so it does not share a stream with
            :func:`make_churn`.

    Returns:
        A dataframe with a log-normal ``sale_price`` target.
    """
    rng = np.random.default_rng(seed + 1)

    grade = rng.integers(3, 13, size=n_rows)
    sqft_above = np.round(rng.lognormal(7.35, 0.36, n_rows), 0).clip(420, 9000)
    # Basements exist on ~35% of homes; sqft_living = above + basement, which
    # makes living/above collinear at ~0.95 without being identical.
    basement = np.where(
        rng.random(n_rows) < 0.35, np.round(sqft_above * rng.uniform(0.15, 0.40, n_rows)), 0.0
    )
    sqft_living = sqft_above + basement
    lot_size = np.round(sqft_living * rng.lognormal(1.1, 0.5, n_rows), 0).clip(800, 200_000)

    bedrooms = np.clip(np.round(sqft_living / 620 + rng.normal(0, 0.6, n_rows)), 1, 9).astype(int)
    bathrooms = np.round(np.clip(bedrooms * 0.65 + rng.normal(0, 0.45, n_rows), 1.0, 7.0) * 2) / 2
    year_built = rng.integers(1900, 2023, size=n_rows)
    renovated = np.where(rng.random(n_rows) < 0.17, rng.integers(1985, 2024, size=n_rows), 0)
    floors = np.clip(np.round(rng.normal(1.6, 0.55, n_rows) * 2) / 2, 1.0, 3.5)
    condition = np.clip(rng.integers(1, 6, size=n_rows), 1, 5)
    waterfront = (rng.random(n_rows) < 0.021).astype(int)
    neighborhood = _choice(
        rng,
        ["riverside", "hilltop", "downtown", "eastgate", "lakeview", "old_town"],
        [0.14, 0.19, 0.21, 0.24, 0.09, 0.13],
        n_rows,
    )
    hood_premium = pd.Series(neighborhood).map(
        {
            "riverside": 0.14,
            "hilltop": 0.22,
            "downtown": 0.08,
            "eastgate": -0.05,
            "lakeview": 0.31,
            "old_town": -0.12,
        }
    ).to_numpy()

    log_price = (
        11.62
        + 0.00021 * sqft_living
        + 0.081 * grade
        + 0.045 * condition
        + 0.0022 * (year_built - 1900)
        + 0.38 * waterfront
        + hood_premium
        + 0.000_0009 * lot_size
        + rng.normal(0, 0.21, n_rows)
    )
    sale_price = np.round(np.exp(log_price), -2)

    # A handful of genuine luxury outliers. Real housing data has these, and a
    # cleaning agent has to decide between clipping them and keeping them. The
    # sqft bump is applied to living *and* above by the same factor so the
    # collinear pair stays collinear.
    n_outliers = 14
    outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    price_factor = rng.uniform(2.8, 5.2, n_outliers)
    sqft_factor = rng.uniform(1.8, 2.6, n_outliers)
    sale_price[outlier_idx] = np.round(sale_price[outlier_idx] * price_factor, -2)
    sqft_living[outlier_idx] = sqft_living[outlier_idx] * sqft_factor
    sqft_above[outlier_idx] = sqft_above[outlier_idx] * sqft_factor

    lot_col: Any = lot_size.astype("float64")
    lot_col[rng.random(n_rows) < 0.055] = np.nan  # unrecorded lot size

    return pd.DataFrame(
        {
            "property_id": [f"P{i:05d}" for i in range(1, n_rows + 1)],
            "neighborhood": neighborhood,
            "sqft_living": np.round(sqft_living, 0).astype("int64"),
            "sqft_above": np.round(sqft_above, 0).astype("int64"),
            "lot_size_sqft": lot_col,
            "bedrooms": bedrooms.astype("int64"),
            "bathrooms": bathrooms,
            "floors": floors,
            "grade": grade.astype("int64"),
            "condition": condition.astype("int64"),
            "year_built": year_built.astype("int64"),
            "year_renovated": renovated.astype("int64"),
            "waterfront": waterfront.astype("int64"),
            HOUSE_TARGET: sale_price,
        }
    )


# ---------------------------------------------------------------------------
# sales time series
# ---------------------------------------------------------------------------


def make_sales_timeseries(
    start: str = TS_START,
    end: str = TS_END,
    stores: tuple[str, ...] = TS_STORES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build the panel time-series example.

    Args:
        start: First calendar date, inclusive.
        end: Last calendar date, inclusive.
        stores: Group keys; each gets its own level, trend, and promo intensity.
        seed: Base seed. Offset internally.

    Returns:
        A long-format dataframe sorted by ``(store_id, date)`` with two calendar
        gaps punched out.
    """
    rng = np.random.default_rng(seed + 2)
    dates = pd.date_range(start, end, freq="D")
    weekday_lift = np.array([0.94, 0.90, 0.95, 1.02, 1.18, 1.34, 1.12])  # Mon..Sun

    rows: list[pd.DataFrame] = []
    for index, store in enumerate(stores):
        n = len(dates)
        level = 180.0 + 55.0 * index
        trend = np.linspace(0.0, 40.0 + 14.0 * index, n)
        weekly = weekday_lift[dates.dayofweek.to_numpy()]
        # Annual seasonality on top of the weekly cycle: retail has both.
        annual = 1.0 + 0.11 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 20) / 365.25)
        promo = (rng.random(n) < 0.14 + 0.03 * index).astype(int)
        temperature = np.round(
            12.0 + 11.0 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 110) / 365.25)
            + rng.normal(0, 2.6, n),
            1,
        )
        units = (
            (level + trend) * weekly * annual * (1.0 + 0.26 * promo)
            + 1.4 * (temperature - 12.0)
            + rng.normal(0, 11.0, n)
        )
        units = np.maximum(np.round(units, 0), 0.0)
        unit_price = np.round(4.25 + 0.4 * index + rng.normal(0, 0.12, n), 2)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "store_id": store,
                    "region": ["north", "south", "east", "west"][index % 4],
                    "promo_flag": promo.astype("int64"),
                    "is_holiday": np.isin(
                        dates.strftime("%m-%d"), ["01-01", "07-04", "11-28", "12-25"]
                    ).astype("int64"),
                    "temperature_c": temperature,
                    "unit_price": unit_price,
                    TS_TARGET: units.astype("int64"),
                    "revenue": np.round(units * unit_price, 2),
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)

    # Two deliberate gaps: a week-long outage at S002 and a three-day one at
    # S004. Gap detection should find both without flagging the panel structure.
    gap_one = pd.date_range("2023-07-10", "2023-07-16").strftime("%Y-%m-%d")
    gap_two = pd.date_range("2024-03-05", "2024-03-07").strftime("%Y-%m-%d")
    drop = (frame["store_id"].eq("S002") & frame["date"].isin(gap_one)) | (
        frame["store_id"].eq("S004") & frame["date"].isin(gap_two)
    )
    frame = frame.loc[~drop].reset_index(drop=True)
    return frame.sort_values(["store_id", "date"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

BUILDERS: dict[str, Any] = {
    "churn.csv": make_churn,
    "house_prices.csv": make_house_prices,
    "sales_timeseries.csv": make_sales_timeseries,
}


def write_all(out_dir: Path) -> dict[str, Path]:
    """Write every example CSV into ``out_dir``.

    Args:
        out_dir: Destination directory; created if absent.

    Returns:
        Mapping of filename to the path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, builder in BUILDERS.items():
        frame = builder()
        path = out_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        written[name] = path
    return written


def main() -> None:
    """CLI wrapper: write the CSVs and print a one-line summary of each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write the CSVs into (default: alongside this script).",
    )
    args = parser.parse_args()

    for name, path in write_all(args.out).items():
        frame = pd.read_csv(path)
        print(f"{name}: {len(frame):,} rows x {frame.shape[1]} cols -> {path}")
        if name == "churn.csv":
            rate = frame[CHURN_TARGET].mean()
            print(
                f"    positive rate={rate:.3%}  "
                f"income missing={frame['annual_income'].isna().mean():.2%}  "
                f"leak col={CHURN_LEAKAGE_COLUMN}"
            )
        elif name == "house_prices.csv":
            print(
                f"    target skew={frame[HOUSE_TARGET].skew():.2f}  "
                f"median={frame[HOUSE_TARGET].median():,.0f}  "
                f"max={frame[HOUSE_TARGET].max():,.0f}"
            )
        else:
            print(
                f"    {frame['store_id'].nunique()} series  "
                f"{frame['date'].min()} .. {frame['date'].max()}"
            )


if __name__ == "__main__":
    main()
