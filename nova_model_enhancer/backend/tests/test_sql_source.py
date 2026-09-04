"""The optional SQL Server source.

Two things are worth testing here: that its absence is a normal state rather
than a failure, and that the statement it builds cannot become anything other
than a read of a configured object.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from backend.services import sql_source
from backend.services.sql_source import SqlSource, SqlSourceError


def _source(**overrides):
    base = dict(
        name="recent", server="SQLPROD01", database="NoVA",
        table="dbo.vw_recent_inventory", date_column="UpdatedDateTimeGMT",
    )
    base.update(overrides)
    return SqlSource(**base)


# ── Optional by construction ─────────────────────────────────────────────────

def test_no_configuration_is_a_normal_state_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_ENHANCER_SQL_CONFIG", str(tmp_path / "absent.json"))
    assert sql_source.load_sources() == []
    report = sql_source.status()
    assert report["sources"] == []
    assert report["config_error"] is None
    assert "optional" in report["note"]


def test_a_missing_driver_is_reported_as_optional_not_broken():
    available, reason = sql_source.driver_available()
    if not available:
        assert "optional" in reason.lower()
        assert "upload" in reason.lower(), "the fallback must be named"


def test_an_unreadable_config_names_the_problem(tmp_path, monkeypatch):
    path = tmp_path / "sql_sources.json"
    path.write_text("{ not json")
    monkeypatch.setenv("NOVA_ENHANCER_SQL_CONFIG", str(path))
    with pytest.raises(SqlSourceError, match="could not be read"):
        sql_source.load_sources()


def test_sources_load_from_configuration(tmp_path, monkeypatch):
    path = tmp_path / "sql_sources.json"
    path.write_text(json.dumps({"sources": [{
        "name": "recent", "server": "SQLPROD01", "database": "NoVA",
        "table": "dbo.vw_recent", "date_column": "UpdatedDateTimeGMT",
        "columns": ["AccountID", "SubTask"],
    }]}))
    monkeypatch.setenv("NOVA_ENHANCER_SQL_CONFIG", str(path))
    sources = sql_source.load_sources()
    assert len(sources) == 1 and sources[0].name == "recent"
    assert sql_source.find_source("recent").table == "dbo.vw_recent"
    with pytest.raises(SqlSourceError, match="No SQL source named"):
        sql_source.find_source("nope")


# ── The statement is always a read of a configured object ────────────────────

def test_the_statement_is_a_select_with_bound_dates():
    statement, params = build = sql_source.build_select(
        _source(), date(2026, 1, 1), date(2026, 3, 31)
    )
    assert statement.lstrip().upper().startswith("SELECT")
    assert "dbo.vw_recent_inventory" in statement
    assert statement.count("?") == 2, "dates must be bound, never interpolated"
    assert params == [date(2026, 1, 1), date(2026, 3, 31)]
    assert "2026" not in statement, "no date value may appear in the statement text"


def test_either_bound_may_be_omitted():
    only_from, params = sql_source.build_select(_source(), date(2026, 1, 1), None)
    assert only_from.count("?") == 1 and params == [date(2026, 1, 1)]
    neither, params = sql_source.build_select(_source(), None, None)
    assert "WHERE" not in neither and params == []


def test_the_upper_bound_includes_its_whole_day():
    statement, _ = sql_source.build_select(_source(), None, date(2026, 3, 31))
    assert "DATEADD(day, 1, ?)" in statement, "a date-only bound must include that day"


def test_a_row_cap_is_always_applied():
    statement, _ = sql_source.build_select(_source(max_rows=5000), None, None)
    assert "TOP 5000" in statement


@pytest.mark.parametrize("table", [
    "dbo.vw_recent; DROP TABLE users",
    "EXEC sp_something",
    "dbo.vw_recent WHERE 1=1 --",
    "'; DELETE FROM inventory; --",
    "dbo.a.b.c.d",
])
def test_an_object_name_that_is_not_an_identifier_is_refused(table):
    """Configuration is local, but a typo must not become a statement."""
    with pytest.raises(SqlSourceError):
        _source(table=table).validate()


@pytest.mark.parametrize("column", ["Updated; DROP", "a b", "1col"])
def test_a_date_column_that_is_not_an_identifier_is_refused(column):
    with pytest.raises(SqlSourceError):
        _source(date_column=column).validate()


def test_a_projected_column_that_is_not_an_identifier_is_refused():
    with pytest.raises(SqlSourceError):
        _source(columns=["AccountID", "(SELECT password FROM users)"]).validate()


def test_no_statement_this_module_builds_can_write_or_call_a_procedure():
    statement, _ = sql_source.build_select(_source(), date(2026, 1, 1), None)
    lowered = statement.lower()
    for forbidden in ("insert", "update ", "delete", "drop", "exec", "merge", "truncate"):
        assert forbidden not in lowered


# ── Credentials ──────────────────────────────────────────────────────────────

def test_the_connection_string_carries_no_secret():
    connection = _source().connection_string()
    assert "Trusted_Connection=yes" in connection
    for forbidden in ("PWD=", "PASSWORD=", "UID="):
        assert forbidden not in connection.upper()


def test_the_description_is_safe_to_log_and_display():
    described = _source().describe()
    assert described["auth"] == "windows_integrated"
    assert "password" not in json.dumps(described).lower()
