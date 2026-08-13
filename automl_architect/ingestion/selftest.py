"""Executable smoke test for the ingestion layer.

Run it with::

    python -m automl_architect.ingestion.selftest

It writes CSV, JSON, JSON Lines, Parquet, Excel, and DuckDB fixtures to a temp
directory, loads each through :func:`~automl_architect.ingestion.router.load_source`,
and prints the resulting :class:`~automl_architect.core.schemas.IngestionResult`.
The awkward cases are covered on purpose: a semicolon-delimited Latin-1 file with
quoted thousands separators, duplicate headers, whitespace-padded headers, a
fully-empty column, an ``Unnamed: 0`` index column, and a gzipped CSV.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.schemas import DataSource, IngestionResult, Param, SourceKind
from .dataframe import make_dataframe_source
from .router import available_kinds, infer_source, load_source


def _fixture_frame(rows: int = 40) -> pd.DataFrame:
    """A small frame with a date column, a category, and a numeric target."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "signup_date": pd.date_range("2024-01-01", periods=rows, freq="D").astype(str),
            "region": ["EU", "US", "APAC", "EU"] * (rows // 4),
            "monthly_spend": [round(100 + i * 13.5, 2) for i in range(rows)],
            "churned": [i % 3 == 0 for i in range(rows)],
        }
    )


def _write_fixtures(root: Path) -> list[tuple[str, DataSource]]:
    """Create every fixture and return ``(label, source)`` pairs to load."""
    frame = _fixture_frame()
    cases: list[tuple[str, DataSource]] = []

    plain_csv = root / "plain.csv"
    frame.to_csv(plain_csv, index=False)
    cases.append(("csv / clean", DataSource(kind=SourceKind.CSV, uri=str(plain_csv))))

    # Semicolon delimiter, Latin-1 text, thousands separators, padded headers,
    # a duplicate header, an all-empty column, and an exported index column.
    messy = root / "messy.csv"
    lines = [
        'Unnamed: 0;" region ";region;"spend";notes;empty_col',
        '0;EU;EU;"1,234.50";café;',
        '1;US;US;"2,000.00";naïve;',
        '2;APAC;APAC;"310.25";Zürich;',
    ]
    messy.write_bytes(("\n".join(lines) + "\n").encode("latin-1"))
    cases.append(("csv / messy latin-1", DataSource(kind=SourceKind.CSV, uri=str(messy))))

    gz = root / "plain.csv.gz"
    gz.write_bytes(gzip.compress(plain_csv.read_bytes()))
    cases.append(("csv / gzipped", DataSource(kind=SourceKind.CSV, uri=str(gz))))

    records = root / "records.json"
    records.write_text(
        json.dumps(json.loads(frame.to_json(orient="records"))), encoding="utf-8"
    )
    cases.append(("json / records", DataSource(kind=SourceKind.JSON, uri=str(records))))

    nested = root / "nested.json"
    nested.write_text(
        json.dumps(
            {
                "meta": {"page": 1},
                "data": [
                    {"id": 1, "customer": {"name": "Ada", "tier": "gold"}, "spend": 10.5},
                    {
                        "id": 2,
                        "customer": {"name": "Linus", "tier": "silver"},
                        "spend": 7.25,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cases.append(
        ("json / nested data key", DataSource(kind=SourceKind.JSON, uri=str(nested)))
    )

    lines_file = root / "events.jsonl"
    lines_file.write_text(
        "\n".join(
            json.dumps({"event": f"e{i}", "at": f"2024-03-{i + 1:02d}", "value": i})
            for i in range(5)
        ),
        encoding="utf-8",
    )
    cases.append(
        ("json / json lines", DataSource(kind=SourceKind.JSON, uri=str(lines_file)))
    )

    parquet = root / "frame.parquet"
    frame.to_parquet(parquet, index=False)
    cases.append(("parquet", DataSource(kind=SourceKind.PARQUET, uri=str(parquet))))
    cases.append(
        (
            "parquet / max_rows=10",
            DataSource(kind=SourceKind.PARQUET, uri=str(parquet)),
        )
    )

    excel = root / "book.xlsx"
    with pd.ExcelWriter(excel) as writer:
        frame.to_excel(writer, sheet_name="summary", index=False)
        frame.head(5).to_excel(writer, sheet_name="detail", index=False)
    cases.append(("excel / first sheet", DataSource(kind=SourceKind.EXCEL, uri=str(excel))))
    cases.append(
        (
            "excel / named sheet",
            DataSource(
                kind=SourceKind.EXCEL,
                uri=str(excel),
                options=[Param(key="sheet", value="detail")],
            ),
        )
    )

    duck = root / "warehouse.duckdb"
    try:
        import duckdb

        connection = duckdb.connect(str(duck))
        connection.register("fixture_frame", frame)
        connection.execute(
            "CREATE OR REPLACE TABLE customers AS SELECT * FROM fixture_frame"
        )
        connection.close()
        cases.append(
            (
                "duckdb / table",
                DataSource(kind=SourceKind.DUCKDB, uri=str(duck), query="customers"),
            )
        )
    except Exception as exc:  # pragma: no cover - duckdb is a hard dependency
        print(f"  (skipped duckdb fixture: {type(exc).__name__}: {exc})")

    cases.append(
        (
            "duckdb / parquet scan",
            DataSource(
                kind=SourceKind.DUCKDB,
                query=(
                    "SELECT region, count(*) AS n FROM "
                    f"'{parquet.as_posix()}' GROUP BY region"
                ),
            ),
        )
    )

    cases.append(("dataframe / in-memory", make_dataframe_source(frame, name="fixture")))

    empty = root / "empty.csv"
    empty.write_text("a,b,c\n", encoding="utf-8")
    cases.append(
        (
            "csv / header only (expect error)",
            DataSource(kind=SourceKind.CSV, uri=str(empty)),
        )
    )

    return cases


def _report(label: str, frame: Any, result: IngestionResult) -> None:
    """Print one result compactly."""
    print(f"\n=== {label} ===")
    print(
        f"  {result.n_rows} rows x {result.n_columns} cols | "
        f"{result.bytes_in_memory:,} bytes | {result.load_seconds:.3f}s | "
        f"truncated={result.truncated} | dataset_id={result.dataset_id}"
    )
    print(f"  source: kind={result.source.kind.value} uri={result.source.uri}")
    for field in result.schema_fields[:8]:
        samples = ", ".join(field.sample_values) or "-"
        print(
            f"    - {field.name}: {field.inferred_dtype} "
            f"(nullable={field.nullable}) [{samples}]"
        )
    if len(result.schema_fields) > 8:
        print(f"    ... {len(result.schema_fields) - 8} more field(s)")
    for message in result.validation_errors:
        print(f"  ERROR   {message}")
    for message in result.validation_warnings:
        print(f"  WARNING {message}")
    assert list(frame.columns) == [f.name for f in result.schema_fields], (
        "schema fields must mirror the frame's columns"
    )


def main() -> int:
    """Run every fixture and return a process exit code."""
    # The router logs each finding; printing them again below would double up.
    logging.getLogger("automl_architect.ingestion").setLevel(logging.CRITICAL)
    print(f"registered kinds: {', '.join(k.value for k in available_kinds())}")
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="automl_ingest_") as tmp:
        root = Path(tmp)
        for label, source in _write_fixtures(root):
            max_rows = 10 if "max_rows" in label else None
            try:
                frame, result = load_source(source, max_rows=max_rows)
            except Exception as exc:
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"\n=== {label} ===\n  FAILED {type(exc).__name__}: {exc}")
                continue
            _report(label, frame, result)
            _check(label, frame, result, failures)

        # infer_source: a bare path must resolve to the right connector.
        inferred = infer_source(str(root / "plain.csv"))
        frame, result = load_source(inferred)
        print(
            f"\n=== infer_source ===\n  kind={inferred.kind.value} "
            f"rows={result.n_rows} cols={result.n_columns}"
        )
        if inferred.kind is not SourceKind.CSV or result.n_rows != 40:
            failures.append("infer_source did not resolve the CSV path correctly")

    if failures:
        print("\nFAILURES:")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nAll ingestion self-tests passed.")
    return 0


def _check(
    label: str, frame: Any, result: IngestionResult, failures: list[str]
) -> None:
    """Assert the case-specific expectations for one fixture."""
    warnings_text = " ".join(result.validation_warnings).lower()

    if label == "csv / messy latin-1":
        if "region_2" not in frame.columns:
            failures.append(f"{label}: duplicate header was not made unique")
        if "region" not in frame.columns:
            failures.append(f"{label}: whitespace-padded header was not stripped")
        if "duplicate column name" not in warnings_text:
            failures.append(f"{label}: missing duplicate-column warning")
        if "latin-1" not in warnings_text:
            failures.append(f"{label}: encoding fallback was not reported")
        if "fully-empty" not in warnings_text:
            failures.append(f"{label}: missing empty-column warning")
        if "row index" not in warnings_text:
            failures.append(f"{label}: missing Unnamed index-column warning")
        if str(frame["spend"].dtype) not in {"float64", "int64"}:
            failures.append(
                f"{label}: thousands separator not parsed (spend is "
                f"{frame['spend'].dtype})"
            )
        if "café" not in set(frame["notes"]):
            failures.append(f"{label}: latin-1 text was not decoded correctly")

    if label.startswith("csv / clean") or label.startswith("csv / gzipped"):
        if result.n_rows != 40 or result.n_columns != 5:
            failures.append(
                f"{label}: expected 40x5, got {result.n_rows}x{result.n_columns}"
            )
        if not str(frame["signup_date"].dtype).startswith("datetime"):
            failures.append(f"{label}: signup_date was not parsed as a datetime")
        if str(frame["customer_id"].dtype).startswith("datetime"):
            failures.append(f"{label}: customer_id must not be coerced to a datetime")

    if label == "json / nested data key":
        if "customer.name" not in frame.columns:
            failures.append(f"{label}: nested keys were not flattened")

    if label == "json / json lines":
        if result.n_rows != 5:
            failures.append(f"{label}: expected 5 rows, got {result.n_rows}")

    if "max_rows" in label:
        if result.n_rows != 10 or not result.truncated:
            failures.append(
                f"{label}: expected 10 rows with truncated=True, got "
                f"{result.n_rows}/{result.truncated}"
            )

    if label.startswith("excel / named"):
        if result.n_rows != 5:
            failures.append(f"{label}: expected the 5-row 'detail' sheet")

    if label.startswith("duckdb / parquet"):
        if "region" not in frame.columns or result.n_rows != 3:
            failures.append(f"{label}: unexpected group-by result {result.n_rows}")

    if "expect error" in label:
        if not result.validation_errors:
            failures.append(f"{label}: an empty frame must produce a validation error")

    if label.startswith("dataframe"):
        if result.n_rows != 40:
            failures.append(f"{label}: expected the registered frame's 40 rows")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
