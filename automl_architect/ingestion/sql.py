"""Database connectors: generic SQL, PostgreSQL, MySQL, Snowflake, Databricks, DuckDB.

Everything except DuckDB goes through SQLAlchemy, so one code path covers five
dialects and the URL is assembled by :meth:`sqlalchemy.engine.URL.create` — which
quotes credentials correctly, unlike string concatenation. DuckDB uses its own
Python API because ``duckdb-engine`` is an extra dependency and the brief
requires DuckDB to work out of the box.

Three rules shape this module:

* **A credential is a local variable and nothing else.** It is read from the
  environment variable named in ``DataSource.secret_env``, handed to SQLAlchemy,
  and scrubbed out of any exception message before it can reach a log or a run
  summary. The rendered URL is never logged.
* **Row caps are dialect-agnostic.** ``pandas.read_sql_query(chunksize=...)``
  stops after the first chunk, so no ``LIMIT`` / ``TOP`` / ``FETCH FIRST``
  dialect matrix is needed.
* **Reads only.** A statement whose leading keyword mutates data is refused
  outright; ingestion has no business writing to a user's warehouse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..core.errors import ConfigurationError, IngestionError, MissingDependencyError
from ..core.schemas import SourceKind
from .base import (
    Connector,
    LoadOutcome,
    cap_rows,
    fetch_limit,
    lazy_import,
    module_available,
    register,
    scrub_secrets,
)

logger = logging.getLogger(__name__)

#: Bare table references, optionally schema- or catalog-qualified.
_IDENTIFIER = re.compile(
    r'^[\w$]+(\.[\w$]+){0,2}$|^"[^"]+"(\.("[^"]+"|[\w$]+)){0,2}$'
)

_MUTATING = {
    "insert",
    "update",
    "delete",
    "merge",
    "drop",
    "alter",
    "truncate",
    "create",
    "replace",
    "grant",
    "revoke",
    "copy",
    "call",
    "execute",
    "vacuum",
    "attach",
    "install",
    "load",
    "set",
}

#: Candidate SQLAlchemy drivers per kind, in preference order:
#: ``(python module, driver name, pip target)``.
_DRIVERS: dict[SourceKind, tuple[tuple[str, str, str], ...]] = {
    SourceKind.POSTGRES: (
        ("psycopg", "postgresql+psycopg", "psycopg[binary]"),
        ("psycopg2", "postgresql+psycopg2", "psycopg2-binary"),
        ("pg8000", "postgresql+pg8000", "pg8000"),
    ),
    SourceKind.MYSQL: (
        ("pymysql", "mysql+pymysql", "PyMySQL"),
        ("MySQLdb", "mysql+mysqldb", "mysqlclient"),
        ("mysql.connector", "mysql+mysqlconnector", "mysql-connector-python"),
    ),
    SourceKind.SNOWFLAKE: (
        ("snowflake.sqlalchemy", "snowflake", "snowflake-sqlalchemy"),
    ),
    SourceKind.DATABRICKS: (
        ("databricks.sqlalchemy", "databricks", "databricks-sqlalchemy"),
        ("databricks.sql", "databricks", "databricks-sql-connector[sqlalchemy]"),
    ),
}

_DEFAULT_PORTS = {SourceKind.POSTGRES: 5432, SourceKind.MYSQL: 3306}

_FEATURE_LABEL = {
    SourceKind.POSTGRES: "PostgreSQL ingestion",
    SourceKind.MYSQL: "MySQL ingestion",
    SourceKind.SNOWFLAKE: "Snowflake ingestion",
    SourceKind.DATABRICKS: "Databricks ingestion",
    SourceKind.SQL: "SQL ingestion",
}


@dataclass
class _Endpoint:
    """A connectable target plus a credential-free label for messages."""

    url: Any
    label: str


def normalise_statement(query: str) -> str:
    """Turn a table name or query into a read-only SELECT statement.

    Args:
        query: A bare (optionally qualified) table name, or a full SQL query.

    Returns:
        SQL safe to execute.

    Raises:
        ConfigurationError: The statement's leading keyword would mutate data.
    """
    text = query.strip().rstrip(";").strip()
    if not text:
        raise ConfigurationError("Empty SQL statement.")
    if _IDENTIFIER.match(text):
        return f"SELECT * FROM {text}"
    first = re.split(r"\s|\(", text, maxsplit=1)[0].lower()
    if first in _MUTATING:
        raise ConfigurationError(
            f"Refusing to run a '{first.upper()}' statement during ingestion; "
            "only SELECT / WITH queries and bare table names are accepted."
        )
    return text


def _resolve_driver(kind: SourceKind) -> tuple[str, str]:
    """Pick an installed driver for ``kind``.

    Returns:
        ``(drivername, module)`` for the first installed candidate.

    Raises:
        MissingDependencyError: No candidate driver is installed.
    """
    candidates = _DRIVERS.get(kind, ())
    for module, drivername, _extra in candidates:
        if module_available(module):
            return drivername, module
    module, _drivername, extra = candidates[0]
    raise MissingDependencyError(module, _FEATURE_LABEL.get(kind, "SQL ingestion"), extra)


class SqlConnector(Connector):
    """Reads a table or query through SQLAlchemy."""

    def _build_endpoint(self) -> _Endpoint:
        """Assemble the connection URL from the source.

        Returns:
            The endpoint, with credentials resolved from the environment.

        Raises:
            ConfigurationError: Not enough information to connect.
            MissingDependencyError: The dialect's driver is not installed.
        """
        sa = lazy_import("sqlalchemy", "SQL ingestion")
        kind = self.source.kind
        uri = (self.source.uri or self.opt_str("uri", "url", default="") or "").strip()

        password = self.secret("password", "passwd", "pwd", "secret", "token")
        username = self.secret("user", "username", "login", allow_single=False)

        if "://" in uri:
            try:
                url = sa.engine.make_url(uri)
            except Exception as exc:
                raise ConfigurationError(
                    f"Could not parse the connection URL for {kind.value}: {exc}"
                ) from exc
            if kind in _DRIVERS and "+" not in url.drivername:
                drivername, _module = _resolve_driver(kind)
                url = url.set(drivername=drivername)
            if not url.password and password:
                url = url.set(password=password)
            if not url.username and username:
                url = url.set(username=username)
            return _Endpoint(url=url, label=self._label(url))

        if kind is SourceKind.SQL:
            raise ConfigurationError(
                "A generic 'sql' source needs a full SQLAlchemy URL in DataSource.uri, "
                "e.g. 'postgresql+psycopg://host/db'. Use the postgres/mysql/snowflake/"
                "databricks/duckdb kinds to have the URL built from options instead."
            )

        drivername, _module = _resolve_driver(kind)
        host = self.opt_str("host", "hostname", "server", "account", "server_hostname")
        database = self.opt_str("database", "db", "catalog", "dbname")
        if not host:
            raise ConfigurationError(
                f"A {kind.value} source needs a 'host' option (or a full URL in uri)."
            )

        query: dict[str, str] = {}
        for key in ("warehouse", "role", "schema", "http_path", "catalog", "sslmode"):
            value = self.opt_str(key)
            if value:
                query[key] = value

        if kind is SourceKind.SNOWFLAKE:
            # snowflake-sqlalchemy encodes the schema as a second path segment.
            schema = query.pop("schema", None)
            if schema and database:
                database = f"{database}/{schema}"
        if kind is SourceKind.DATABRICKS and not username:
            username = "token"

        url = sa.engine.URL.create(
            drivername=drivername,
            username=username,
            password=password,
            host=host,
            port=self.opt_int("port", default=_DEFAULT_PORTS.get(kind)),
            database=database,
            query=query,
        )
        return _Endpoint(url=url, label=self._label(url))

    @staticmethod
    def _label(url: Any) -> str:
        """Credential-free description of a connection, for messages."""
        host = getattr(url, "host", None)
        database = getattr(url, "database", None) or ""
        target = f"//{host}/{database}" if host else database
        return f"{getattr(url, 'drivername', 'sql')}:{target}"

    def _statement(self) -> str:
        """Resolve the SELECT to run."""
        query = self.source.query or self.opt_str("query", "sql", "table", "table_name")
        if not query:
            raise ConfigurationError(
                f"A {self.source.kind.value} source needs DataSource.query set to a "
                "table name or a SELECT statement."
            )
        return normalise_statement(str(query))

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Execute the statement and return the first ``max_rows`` rows.

        Args:
            max_rows: Row cap, applied with a server-side cursor chunk rather
                than dialect-specific SQL.

        Returns:
            The result frame plus notes.

        Raises:
            IngestionError: Connection or execution failed. The message is
                scrubbed of credential values.
        """
        sa = lazy_import("sqlalchemy", "SQL ingestion")
        endpoint = self._build_endpoint()
        statement = self._statement()
        connect_args = self.opt_dict("connect_args")
        timeout = self.opt_int("timeout_seconds", "timeout")

        try:
            engine = sa.create_engine(
                endpoint.url,
                pool_pre_ping=True,
                connect_args=connect_args or {},
            )
        except MissingDependencyError:
            raise
        except Exception as exc:
            # Two distinct absences land here: SQLAlchemy has no dialect plugin
            # (NoSuchModuleError), or the dialect exists but its DBAPI driver is
            # not installed (ImportError from dialect.import_dbapi). Both are the
            # same problem for the user, so both get an install hint.
            if type(exc).__name__ == "NoSuchModuleError" or isinstance(exc, ImportError):
                dialect = str(endpoint.url.drivername)
                package = getattr(exc, "name", None) or dialect
                raise MissingDependencyError(
                    package,
                    f"{dialect} ingestion",
                    _pip_hint_for_dialect(dialect),
                ) from exc
            raise IngestionError(
                self._scrub(f"Could not create an engine for {endpoint.label}: {exc}")
            ) from exc

        limit = fetch_limit(max_rows)
        try:
            with engine.connect() as connection:
                if timeout:
                    connection = connection.execution_options(timeout=timeout)
                frame = _read_sql(connection, sa.text(statement), limit)
        except Exception as exc:
            raise IngestionError(
                self._scrub(
                    f"SQL read from {endpoint.label} failed: {type(exc).__name__}: {exc}"
                )
            ) from exc
        finally:
            engine.dispose()

        frame, truncated = cap_rows(frame, max_rows)
        self.note(f"Read via {endpoint.label}.")
        return self.outcome(frame, truncated=truncated, detail=endpoint.label)

    def _scrub(self, message: str) -> str:
        """Strip any resolved credential value out of ``message``."""
        return scrub_secrets(message, self.resolved_secrets)


def _pip_hint_for_dialect(drivername: str) -> str:
    """Best-guess pip target for an unknown SQLAlchemy dialect."""
    base = drivername.split("+")[0].lower()
    known = {
        "postgresql": "psycopg[binary]",
        "postgres": "psycopg[binary]",
        "mysql": "PyMySQL",
        "mariadb": "PyMySQL",
        "mssql": "pyodbc",
        "oracle": "oracledb",
        "snowflake": "snowflake-sqlalchemy",
        "databricks": "databricks-sqlalchemy",
        "duckdb": "duckdb-engine",
        "bigquery": "sqlalchemy-bigquery",
        "redshift": "sqlalchemy-redshift",
        "trino": "trino[sqlalchemy]",
        "clickhouse": "clickhouse-sqlalchemy",
    }
    return known.get(base, base)


def _read_sql(connection: Any, statement: Any, limit: int | None) -> pd.DataFrame:
    """Run a statement, taking only the first chunk when a limit applies."""
    if limit is None:
        return pd.read_sql_query(statement, connection)
    chunks = pd.read_sql_query(statement, connection, chunksize=limit)
    try:
        for chunk in chunks:
            return chunk
    finally:
        # Closing the generator releases the server-side cursor now rather than
        # whenever the garbage collector gets to it.
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    # No rows at all: re-run unchunked purely to recover the column names.
    return pd.read_sql_query(statement, connection)


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

_FILE_READERS = {
    ".csv": "read_csv_auto",
    ".tsv": "read_csv_auto",
    ".txt": "read_csv_auto",
    ".parquet": "read_parquet",
    ".pq": "read_parquet",
    ".json": "read_json_auto",
    ".ndjson": "read_json_auto",
    ".jsonl": "read_json_auto",
}


@register(SourceKind.DUCKDB)
class DuckDbConnector(Connector):
    """Reads a DuckDB database, or any file DuckDB can scan.

    Uses the bundled ``duckdb`` package directly rather than the optional
    ``duckdb-engine`` SQLAlchemy dialect, so this kind never needs an extra
    install. An empty ``uri`` opens an in-memory database, which is the useful
    mode for querying Parquet/CSV globs::

        DataSource(kind=SourceKind.DUCKDB,
                   query="SELECT * FROM 'sales/*.parquet' WHERE region = 'EU'")
    """

    def _database(self) -> str:
        """Resolve the database path (``:memory:`` when none is given)."""
        uri = (
            self.source.uri or self.opt_str("database", "path", default="") or ""
        ).strip()
        if uri.startswith("duckdb:///"):
            uri = uri[len("duckdb:///") :]
        elif uri.startswith("duckdb://"):
            uri = uri[len("duckdb://") :]
        if not uri or uri in {":memory:", "memory"}:
            return ":memory:"
        return uri

    def _statement(self, database: str) -> str:
        """Resolve the SELECT, synthesising a file scan when possible."""
        query = self.source.query or self.opt_str("query", "sql", "table", "table_name")
        if query:
            return normalise_statement(str(query))

        suffix = "." + database.rsplit(".", 1)[-1].lower() if "." in database else ""
        reader = _FILE_READERS.get(suffix)
        if reader and database != ":memory:":
            escaped = database.replace("'", "''")
            self.note(f"No query given; scanned the file with DuckDB {reader}().")
            return f"SELECT * FROM {reader}('{escaped}')"
        raise ConfigurationError(
            "A duckdb source needs DataSource.query set to a table name or a SELECT "
            "statement (or a uri pointing at a csv/parquet/json file)."
        )

    def load(self, max_rows: int | None = None) -> LoadOutcome:
        """Run the statement against DuckDB.

        Args:
            max_rows: Row cap, applied as a ``LIMIT`` around the statement.

        Returns:
            The result frame plus notes.

        Raises:
            IngestionError: The query failed.
        """
        duckdb = lazy_import("duckdb", "DuckDB ingestion")
        database = self._database()
        statement = self._statement(database)

        limit = fetch_limit(max_rows)
        if limit is not None:
            statement = f"SELECT * FROM ({statement}) AS _capped LIMIT {int(limit)}"

        scanning_file = database != ":memory:" and any(
            database.lower().endswith(ext) for ext in _FILE_READERS
        )
        connect_to = ":memory:" if scanning_file else database
        read_only = self.opt_bool("read_only", default=connect_to != ":memory:")

        connection = None
        try:
            try:
                connection = duckdb.connect(connect_to, read_only=read_only)
            except Exception as exc:
                if not read_only:
                    raise
                # A brand-new file cannot be opened read-only; retry writable.
                logger.debug("duckdb read-only connect failed, retrying: %s", exc)
                connection = duckdb.connect(connect_to, read_only=False)
            frame = connection.execute(statement).fetch_df()
        except Exception as exc:
            raise IngestionError(
                scrub_secrets(
                    f"DuckDB read failed ({database}): {type(exc).__name__}: {exc}",
                    self.resolved_secrets,
                )
            ) from exc
        finally:
            if connection is not None:
                connection.close()

        frame, truncated = cap_rows(frame, max_rows)
        return self.outcome(frame, truncated=truncated, detail=f"duckdb:{database}")


# Registered after DuckDbConnector so the decorator order reads top-down.
SqlConnector = register(
    SourceKind.SQL,
    SourceKind.POSTGRES,
    SourceKind.MYSQL,
    SourceKind.SNOWFLAKE,
    SourceKind.DATABRICKS,
)(SqlConnector)


__all__ = [
    "DuckDbConnector",
    "SqlConnector",
    "normalise_statement",
]
