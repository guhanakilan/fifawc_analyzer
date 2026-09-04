"""Optional read-only SQL Server source for recent training data.

The project brief says: "Do not require direct SQL Server or stored-procedure
access for this version." This path is therefore strictly optional — the
application installs, starts and runs a complete retraining cycle with no
driver present and no configuration file, and file upload is unaffected.

What it will not do, by construction:

  * no writes — the statement is built here and is always a SELECT
  * no stored procedures — the brief excludes them and nothing here can call one
  * no arbitrary SQL from the browser — the caller chooses a *configured* source
    by name and supplies a date range, never a query
  * no credentials in the repo, in an export, or in the audit trail

Windows integrated authentication is the only mode: the connection borrows the
account already running the application, so there is no secret to store or leak.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger("nova_enhancer")

# A configured source names a table or view and its date column. Both are
# validated against this before they reach a statement — the only defence that
# matters, since neither is ever supplied by a browser.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")

MAX_ROWS = 2_000_000


class SqlSourceError(RuntimeError):
    """Configuration, connection or query failure, with nothing sensitive in it."""


@dataclass
class SqlSource:
    name: str
    server: str
    database: str
    table: str                      # table or view, optionally schema-qualified
    date_column: str
    driver: str = "ODBC Driver 17 for SQL Server"
    columns: list[str] = field(default_factory=list)   # empty means all
    max_rows: int = MAX_ROWS

    def validate(self) -> None:
        if not IDENTIFIER.match(self.table):
            raise SqlSourceError(
                f"Source '{self.name}' names an object that is not a plain identifier. "
                "Give a table or view name, optionally schema-qualified."
            )
        if not IDENTIFIER.match(self.date_column):
            raise SqlSourceError(
                f"Source '{self.name}' has a date column that is not a plain identifier."
            )
        for column in self.columns:
            if not IDENTIFIER.match(column):
                raise SqlSourceError(
                    f"Source '{self.name}' lists a column that is not a plain identifier: {column}"
                )

    def connection_string(self) -> str:
        """Integrated auth only — there is no password to place here."""
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            "Trusted_Connection=yes;"
            "Encrypt=yes;TrustServerCertificate=yes;"
        )

    def describe(self) -> dict:
        """Safe to show and to log: names an object, never a credential."""
        return {
            "name": self.name,
            "server": self.server,
            "database": self.database,
            "table": self.table,
            "date_column": self.date_column,
            "columns": self.columns or ["(all)"],
            "auth": "windows_integrated",
        }


def config_path() -> Path:
    """Where sources are configured. Deliberately outside the workspace."""
    override = os.environ.get("NOVA_ENHANCER_SQL_CONFIG")
    if override:
        return Path(override)
    from ..config import APP_ROOT
    return APP_ROOT / "sql_sources.json"


def load_sources() -> list[SqlSource]:
    """Configured sources, or an empty list. A missing file is normal."""
    path = config_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SqlSourceError(f"The SQL source configuration could not be read: {exc}") from exc

    entries = raw.get("sources") if isinstance(raw, dict) else raw
    sources = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            source = SqlSource(
                name=str(entry["name"]),
                server=str(entry["server"]),
                database=str(entry["database"]),
                table=str(entry["table"]),
                date_column=str(entry["date_column"]),
                driver=str(entry.get("driver", "ODBC Driver 17 for SQL Server")),
                columns=[str(c) for c in entry.get("columns", [])],
                max_rows=int(entry.get("max_rows", MAX_ROWS)),
            )
        except KeyError as exc:
            raise SqlSourceError(f"A configured SQL source is missing {exc}.") from exc
        source.validate()
        sources.append(source)
    return sources


def find_source(name: str) -> SqlSource:
    for source in load_sources():
        if source.name == name:
            return source
    raise SqlSourceError(f"No SQL source named '{name}' is configured.")


def driver_available() -> tuple[bool, str]:
    """Whether pyodbc is installed. Absence is a normal state, not an error."""
    try:
        import pyodbc  # noqa: F401
    except ImportError:
        return False, (
            "pyodbc is not installed, so the SQL source is unavailable. This is optional: "
            "upload a Parquet or CSV export instead, or install pyodbc and an ODBC driver "
            "to enable it."
        )
    return True, "pyodbc is available."


def build_select(source: SqlSource, date_from: date | None, date_to: date | None) -> tuple[str, list]:
    """A parameterised SELECT over a configured object. Never a procedure call.

    The table, column list and date column come from local configuration and are
    identifier-validated; the date bounds are bound parameters. Nothing a caller
    can send reaches the statement text.
    """
    source.validate()
    columns = ", ".join(source.columns) if source.columns else "*"
    statement = f"SELECT TOP {int(source.max_rows)} {columns} FROM {source.table}"

    clauses, params = [], []
    if date_from is not None:
        clauses.append(f"{source.date_column} >= ?")
        params.append(date_from)
    if date_to is not None:
        clauses.append(f"{source.date_column} < DATEADD(day, 1, ?)")
        params.append(date_to)
    if clauses:
        statement += " WHERE " + " AND ".join(clauses)
    statement += f" ORDER BY {source.date_column}"
    return statement, params


def fetch(source: SqlSource, date_from: date | None, date_to: date | None):
    """Run the SELECT and return a DataFrame. Raises SqlSourceError on failure."""
    available, reason = driver_available()
    if not available:
        raise SqlSourceError(reason)

    import pandas as pd
    import pyodbc

    statement, params = build_select(source, date_from, date_to)
    try:
        with pyodbc.connect(source.connection_string(), readonly=True, timeout=30) as connection:
            frame = pd.read_sql(statement, connection, params=params)
    except Exception as exc:  # noqa: BLE001 — driver errors vary wildly
        # Never surface the connection string: it names the server and database.
        raise SqlSourceError(
            f"The query against source '{source.name}' failed: {type(exc).__name__}. "
            "Check that the server is reachable and that your Windows account can read "
            "that object."
        ) from exc

    if len(frame) >= source.max_rows:
        logger.warning(
            "SQL source %s returned the row cap (%s); the window may be truncated.",
            source.name, source.max_rows,
        )
    return frame


def status() -> dict:
    """What the SQL path can do right now, and why not when it cannot."""
    available, reason = driver_available()
    try:
        sources = load_sources()
        error = None
    except SqlSourceError as exc:
        sources, error = [], str(exc)
    return {
        "driver_available": available,
        "driver_detail": reason,
        "config_path": str(config_path()),
        "sources": [s.describe() for s in sources],
        "config_error": error,
        "auth": "windows_integrated",
        "note": (
            "This path is optional. The application runs a complete retraining cycle "
            "from an uploaded file with no driver and no configuration present."
        ),
    }
