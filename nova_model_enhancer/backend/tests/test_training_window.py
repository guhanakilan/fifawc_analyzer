"""Item 5 — the explicit training window.

The window decides which rows the model ever sees, so it has to be exact at the
boundaries and honest in the manifest about what it removed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.services.snapshot import SnapshotDecisions, SnapshotError, build_snapshot


def _frame(n=120, start="2026-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "AccountID": [f"A{i:04d}" for i in range(n)],
        "UpdatedDateTimeGMT": dates,
        "SubTask": ["Called Insurance" if i % 2 else "Portal Status Check" for i in range(n)],
        "ARComments": [""] * n,
    })


def _decisions(**overrides):
    base = dict(
        date_column="UpdatedDateTimeGMT",
        target_mode="derive_from_subtask",
        target_column=None,
        target_encoding={},
        dedup_mode="none",
        dedup_keys=[],
        subtask_mappings=[
            {"name": "Called Insurance", "flag": "Voice"},
            {"name": "Portal Status Check", "flag": "Non-Voice"},
        ],
        subtask_keywords=[],
        allow_unmapped_default=False,
        historical_window_days=None,
        approver="tester",
    )
    base.update(overrides)
    return SnapshotDecisions(**base)


def _build(tmp_path, decisions, df=None):
    path = tmp_path / "data.parquet"
    (df if df is not None else _frame()).to_parquet(path, index=False)
    return build_snapshot(
        [("combined", path)], decisions, tmp_path / "snaps", "SNAP_WINDOW", {"cfg": "test"}
    )


def test_range_keeps_only_rows_inside_it(tmp_path):
    manifest = _build(tmp_path, _decisions(date_from="2026-02-01", date_to="2026-02-28"))
    assert manifest["row_counts"]["final"] == 28
    assert manifest["exclusions"]["training_window_mode"] == "explicit_range"


def test_upper_bound_includes_the_whole_of_its_last_day(tmp_path):
    """A date-only 'to' must not silently exclude that day's rows."""
    df = _frame(n=3)
    df.loc[1, "UpdatedDateTimeGMT"] = pd.Timestamp("2026-01-02 23:59:00")
    manifest = _build(tmp_path, _decisions(date_to="2026-01-02"), df=df)
    assert manifest["row_counts"]["final"] == 2


def test_open_ended_lower_bound_is_allowed(tmp_path):
    manifest = _build(tmp_path, _decisions(date_from="2026-04-01"))
    # 1 Jan + 120 days runs to 30 Apr, so April has 30 rows.
    assert manifest["row_counts"]["final"] == 30


def test_an_explicit_range_overrides_days_back(tmp_path):
    manifest = _build(
        tmp_path, _decisions(date_from="2026-01-01", date_to="2026-01-10", historical_window_days=5)
    )
    assert manifest["row_counts"]["final"] == 10
    assert manifest["exclusions"]["training_window_mode"] == "explicit_range"


def test_days_back_still_works_for_a_job_saved_before_ranges(tmp_path):
    manifest = _build(tmp_path, _decisions(historical_window_days=10))
    assert manifest["exclusions"]["training_window_mode"] == "days_back"
    assert manifest["row_counts"]["final"] == 11  # inclusive of the cutoff day


def test_an_empty_window_fails_loudly_rather_than_freezing_nothing(tmp_path):
    with pytest.raises(SnapshotError) as exc:
        _build(tmp_path, _decisions(date_from="2030-01-01"))
    assert "no rows" in str(exc.value).lower()


def test_the_manifest_records_the_window_that_was_applied(tmp_path):
    manifest = _build(tmp_path, _decisions(date_from="2026-02-01", date_to="2026-02-10"))
    window = manifest["exclusions"]["training_window"]
    assert window == {"from": "2026-02-01", "to": "2026-02-10"}
    assert manifest["exclusions"]["rows_outside_window"] == 110
