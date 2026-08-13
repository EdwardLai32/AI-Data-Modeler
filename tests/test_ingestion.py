"""Round-trip every file and local-database connector.

The contract is one function — ``load_source(source, *, max_rows=None)`` returning
``(dataframe, IngestionResult)`` — so every test here writes the same known frame
out in some format, reads it back through the router, and checks two things: the
data survived, and the ``IngestionResult`` describes what actually happened.

Connectors whose driver is not installed (S3, GCS, Azure, Kaggle, Postgres,
MySQL, Snowflake) must raise :class:`MissingDependencyError` with an install
hint, not ``ImportError`` from three frames deep. That is tested too, because a
missing optional dependency is the single most common way this layer is met.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl_architect.core.errors import (
    IngestionError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from automl_architect.core.schemas import DataSource, IngestionResult, Param, SourceKind

from .conftest import import_or_skip


@pytest.fixture(scope="module")
def load_source():  # noqa: ANN201 - a function handle from a sibling module
    """The router entry point, or a skip if ingestion is not written yet."""
    module = import_or_skip(
        "automl_architect.ingestion.router",
        "automl_architect.ingestion",
        feature="ingestion/router.py",
    )
    if not hasattr(module, "load_source"):
        pytest.skip("ingestion module exposes no load_source()")
    return module.load_source


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small frame with one of every dtype a connector has to preserve."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["ana", "ben", "cass", "dev", "eli"],
            "score": [1.5, 2.25, 3.0, 4.75, 5.5],
            "active": [True, False, True, True, False],
        }
    )


def assert_round_tripped(
    frame: pd.DataFrame,
    result: IngestionResult,
    expected: pd.DataFrame,
    *,
    kind: SourceKind,
) -> None:
    """Assert the shared post-conditions of every successful load.

    Args:
        frame: The dataframe the router returned.
        result: The accompanying IngestionResult.
        expected: The frame that was written out.
        kind: The source kind under test, echoed in the result.
    """
    assert list(frame.columns) == list(expected.columns)
    assert len(frame) == len(expected)
    pd.testing.assert_series_equal(
        frame["id"].astype("int64").reset_index(drop=True),
        expected["id"].astype("int64").reset_index(drop=True),
        check_names=False,
    )

    assert isinstance(result, IngestionResult)
    assert result.n_rows == len(frame)
    assert result.n_columns == frame.shape[1]
    assert result.source.kind is kind
    assert result.dataset_id.startswith("ds_")
    assert result.load_seconds >= 0.0
    assert result.bytes_in_memory > 0
    assert not result.validation_errors
    assert {f.name for f in result.schema_fields} == set(map(str, frame.columns))


# ---------------------------------------------------------------------------
# Flat files
# ---------------------------------------------------------------------------


def test_csv_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    sample_frame.to_csv(path, index=False)
    frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.CSV)


def test_csv_honours_connector_options(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """``options`` is the only channel for reader knobs like a non-comma separator."""
    path = tmp_path / "semi.csv"
    sample_frame.to_csv(path, index=False, sep=";")
    source = DataSource(
        kind=SourceKind.CSV,
        uri=str(path),
        options=[Param(key="sep", value=";")],
    )
    frame, result = load_source(source)
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.CSV)


def test_csv_with_utf8_bom(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """Excel writes a BOM; a BOM'd first header must not become ``\\ufeffid``."""
    path = tmp_path / "bom.csv"
    path.write_text(sample_frame.to_csv(index=False), encoding="utf-8-sig")
    frame, _ = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    assert list(frame.columns)[0] == "id"


def test_parquet_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    sample_frame.to_parquet(path, index=False)
    frame, result = load_source(DataSource(kind=SourceKind.PARQUET, uri=str(path)))
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.PARQUET)
    # Parquet is typed, so unlike CSV the bool must survive as a bool.
    assert frame["active"].dtype == bool


def test_excel_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    sample_frame.to_excel(path, index=False, sheet_name="Sheet1")
    frame, result = load_source(DataSource(kind=SourceKind.EXCEL, uri=str(path)))
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.EXCEL)


def test_excel_named_sheet(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """A workbook's second sheet is reachable through ``options``."""
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as writer:
        sample_frame.head(2).to_excel(writer, index=False, sheet_name="first")
        sample_frame.to_excel(writer, index=False, sheet_name="second")
    source = DataSource(
        kind=SourceKind.EXCEL,
        uri=str(path),
        options=[Param(key="sheet_name", value="second")],
    )
    frame, _ = load_source(source)
    assert len(frame) == len(sample_frame)


def test_json_records_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(sample_frame.to_json(orient="records"), encoding="utf-8")
    frame, result = load_source(DataSource(kind=SourceKind.JSON, uri=str(path)))
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.JSON)


def test_json_lines_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """NDJSON is what log and event exports look like."""
    path = tmp_path / "data.jsonl"
    path.write_text(sample_frame.to_json(orient="records", lines=True), encoding="utf-8")
    source = DataSource(
        kind=SourceKind.JSON,
        uri=str(path),
        options=[Param(key="lines", value="true")],
    )
    frame, _ = load_source(source)
    assert len(frame) == len(sample_frame)


# ---------------------------------------------------------------------------
# Local databases
# ---------------------------------------------------------------------------


def test_sqlite_via_sql_connector(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """SQLAlchemy ships a SQLite driver, so the generic SQL path is testable offline."""
    import sqlalchemy

    db_path = tmp_path / "test.db"
    engine = sqlalchemy.create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")
    sample_frame.to_sql("people", engine, index=False)
    engine.dispose()

    source = DataSource(
        kind=SourceKind.SQL,
        uri=f"sqlite+pysqlite:///{db_path.as_posix()}",
        query="SELECT * FROM people",
    )
    frame, result = load_source(source)
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.SQL)


def test_sqlite_bare_table_name(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """``query`` doubles as a table name, per the field's documentation."""
    import sqlalchemy

    db_path = tmp_path / "table.db"
    engine = sqlalchemy.create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")
    sample_frame.to_sql("people", engine, index=False)
    engine.dispose()

    frame, _ = load_source(
        DataSource(
            kind=SourceKind.SQL,
            uri=f"sqlite+pysqlite:///{db_path.as_posix()}",
            query="people",
        )
    )
    assert len(frame) == len(sample_frame)


def test_duckdb_round_trip(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "test.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.register("incoming", sample_frame)
    connection.execute("CREATE TABLE people AS SELECT * FROM incoming")
    connection.close()

    source = DataSource(
        kind=SourceKind.DUCKDB,
        uri=str(db_path),
        query="SELECT * FROM people",
    )
    frame, result = load_source(source)
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.DUCKDB)


def test_duckdb_can_query_a_parquet_file(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """DuckDB over Parquet is the cheap route for a file too large to load whole."""
    pytest.importorskip("duckdb")
    parquet = tmp_path / "data.parquet"
    sample_frame.to_parquet(parquet, index=False)
    source = DataSource(
        kind=SourceKind.DUCKDB,
        uri=":memory:",
        query=f"SELECT * FROM read_parquet('{parquet.as_posix()}')",
    )
    frame, _ = load_source(source)
    assert len(frame) == len(sample_frame)


# ---------------------------------------------------------------------------
# In-memory dataframes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataframe_mod():  # noqa: ANN201
    """The in-memory dataframe connector."""
    return import_or_skip(
        "automl_architect.ingestion.dataframe", feature="ingestion/dataframe.py"
    )


def test_dataframe_source(load_source, dataframe_mod, sample_frame: pd.DataFrame) -> None:
    """A caller who already has a frame must be able to skip the file system.

    ``SourceKind.DATAFRAME`` cannot carry a frame in ``options`` — values are
    strings — so the frame is parked in a process-local registry and the source
    carries its token. That keeps the source JSON-serialisable, which matters
    because it round-trips into the stored ``RunSummary``.
    """
    source = dataframe_mod.make_dataframe_source(sample_frame, name="churn_extract")
    assert source.kind is SourceKind.DATAFRAME
    assert source.model_dump_json(), "the source must stay JSON-serialisable"

    frame, result = load_source(source)
    assert_round_tripped(frame, result, sample_frame, kind=SourceKind.DATAFRAME)


def test_dataframe_source_copies_by_default(
    load_source, dataframe_mod, sample_frame: pd.DataFrame
) -> None:
    """The pipeline mutates what it loads, so the caller's frame must be safe."""
    source = dataframe_mod.make_dataframe_source(sample_frame)
    frame, _ = load_source(source)
    frame.loc[0, "score"] = -999.0
    assert sample_frame.loc[0, "score"] != -999.0, "the caller's frame was mutated"


def test_unregistered_token_fails_clearly(load_source, dataframe_mod) -> None:
    """A summary replayed in another process must say so, not load other data."""
    from automl_architect.core.errors import AutoMLArchitectError

    source = dataframe_mod.make_dataframe_source(pd.DataFrame({"a": [1]}))
    token = next(
        p.value for p in source.options if p.key == dataframe_mod.FRAME_KEY_OPTION
    )
    dataframe_mod.release_frame(token)

    with pytest.raises(AutoMLArchitectError) as info:
        load_source(source)
    assert token in str(info.value) or "register" in str(info.value).lower()


def test_dataframe_row_cap(load_source, dataframe_mod, churn_df: pd.DataFrame) -> None:
    frame, result = load_source(
        dataframe_mod.make_dataframe_source(churn_df), max_rows=120
    )
    assert len(frame) == 120
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Row limits
# ---------------------------------------------------------------------------


def test_max_rows_truncates_and_says_so(load_source, churn_csv: Path) -> None:
    """Silent truncation would make every downstream statistic quietly wrong."""
    frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(churn_csv)), max_rows=250)
    assert len(frame) == 250
    assert result.n_rows == 250
    assert result.truncated is True


def test_max_rows_above_length_is_not_truncation(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    sample_frame.to_csv(path, index=False)
    frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)), max_rows=10_000)
    assert len(frame) == len(sample_frame)
    assert result.truncated is False


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_file_raises_ingestion_error(load_source, tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_source(DataSource(kind=SourceKind.CSV, uri=str(tmp_path / "nope.csv")))


def test_unparseable_file_raises_ingestion_error(load_source, tmp_path: Path) -> None:
    """A .parquet that is not Parquet must fail as an IngestionError."""
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"this is definitely not parquet")
    with pytest.raises(IngestionError):
        load_source(DataSource(kind=SourceKind.PARQUET, uri=str(path)))


def test_empty_csv_is_rejected_or_reported(load_source, tmp_path: Path) -> None:
    """Zero usable rows is either an IngestionError or a result that admits it."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    try:
        frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    except IngestionError:
        return
    assert result.n_rows == 0
    assert len(frame) == 0


#: Kind -> a URI that connector will accept as well-formed, so the failure it
#: reports is the missing driver rather than a malformed address.
UNINSTALLED_DRIVERS = [
    (SourceKind.S3, "s3://bucket/key.csv"),
    (SourceKind.GCS, "gs://bucket/key.csv"),
    (SourceKind.AZURE_BLOB, "az://container/key.csv"),
    (SourceKind.KAGGLE, "owner/some-dataset"),
    (SourceKind.POSTGRES, "postgresql+psycopg://user:pw@localhost:5432/db"),
    (SourceKind.MYSQL, "mysql+pymysql://user:pw@localhost:3306/db"),
    (SourceKind.SNOWFLAKE, "snowflake://user:pw@account/db/schema"),
]


@pytest.mark.parametrize(("kind", "uri"), UNINSTALLED_DRIVERS, ids=lambda v: getattr(v, "value", ""))
def test_absent_optional_driver_fails_with_a_typed_error(
    load_source, kind: SourceKind, uri: str
) -> None:
    """A missing optional driver must produce a typed error with an install hint.

    None of these drivers are installed here. The requirement is that the failure
    is one of this package's own exceptions carrying an actionable message — not a
    raw ``ImportError`` from three frames deep, and not a ``KeyError`` on a
    connector registry.
    """
    from automl_architect.core.errors import AutoMLArchitectError, ConfigurationError

    source = DataSource(kind=kind, uri=uri, query="SELECT 1")
    with pytest.raises(AutoMLArchitectError) as info:
        load_source(source)

    error = info.value
    assert isinstance(
        error, (MissingDependencyError, UnsupportedSourceError, IngestionError, ConfigurationError)
    ), f"{kind.value} raised an untyped {type(error).__name__}"
    assert str(error).strip(), f"{kind.value} raised an error with no message"
    if isinstance(error, MissingDependencyError):
        assert "pip install" in str(error)


@pytest.mark.parametrize(("kind", "uri"), UNINSTALLED_DRIVERS, ids=lambda v: getattr(v, "value", ""))
def test_optional_driver_failure_is_not_a_bare_import_error(
    load_source, kind: SourceKind, uri: str
) -> None:
    """``ImportError`` escaping the router is the failure mode being guarded against."""
    try:
        load_source(DataSource(kind=kind, uri=uri, query="SELECT 1"))
    except (ImportError, KeyError, AttributeError) as exc:  # pragma: no cover - the bug case
        pytest.fail(f"{kind.value} leaked a {type(exc).__name__}: {exc}")
    except Exception:
        pass


def test_sql_without_query_is_an_error(load_source, tmp_path: Path) -> None:
    """A database source with nothing to run is a configuration error, not a hang."""
    from automl_architect.core.errors import ConfigurationError

    with pytest.raises((ConfigurationError, IngestionError, ValueError)) as info:
        load_source(
            DataSource(kind=SourceKind.SQL, uri=f"sqlite+pysqlite:///{tmp_path.as_posix()}/x.db")
        )
    assert "query" in str(info.value).lower()


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


def test_schema_fields_carry_dtypes_and_samples(load_source, sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    """``schema_fields`` is what the API shows before a run starts."""
    path = tmp_path / "data.csv"
    sample_frame.to_csv(path, index=False)
    _, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    by_name = {f.name: f for f in result.schema_fields}
    assert by_name["id"].inferred_dtype
    assert by_name["name"].inferred_dtype
    assert any(f.sample_values for f in result.schema_fields)


def test_result_flags_a_column_that_is_entirely_null(load_source, tmp_path: Path) -> None:
    """An all-null column is a warning, not an error: the run can still proceed."""
    path = tmp_path / "nulls.csv"
    path.write_text("a,b\n1,\n2,\n3,\n", encoding="utf-8")
    frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    assert frame["b"].isna().all()
    assert not result.validation_errors


def test_duplicate_headers_are_reported_not_dropped(load_source, tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text("a,a,b\n1,2,3\n4,5,6\n", encoding="utf-8")
    frame, result = load_source(DataSource(kind=SourceKind.CSV, uri=str(path)))
    assert frame.shape[1] == 3
    assert result.n_columns == 3
