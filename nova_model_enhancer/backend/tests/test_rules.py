"""Item 4 — the readiness rules engine.

These cover the two things that make a rules engine trustworthy: that a rule
fires when its condition is met, and that it stays silent (rather than
guessing) when the fact it needs cannot be measured.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.services import rules
from backend.services.rules_config import DEFAULT_RULES, default_rules, merge_rules


# ── Configuration ────────────────────────────────────────────────────────────

def test_defaults_are_not_shared_between_callers():
    """A caller mutating its rules must not alter the defaults for the next one."""
    first = default_rules()
    first["interventions"][0]["action"] = "off"
    assert default_rules()["interventions"][0]["action"] == "block"


def test_merge_keeps_rules_absent_from_a_stored_older_set():
    """A saved set from an earlier version must not drop rules added since."""
    stored = {"interventions": [{"id": "too_few_rows", "threshold": 100}]}
    merged = merge_rules(stored)
    assert len(merged["interventions"]) == len(DEFAULT_RULES["interventions"])
    rule = next(r for r in merged["interventions"] if r["id"] == "too_few_rows")
    assert rule["threshold"] == 100.0


def test_merge_ignores_an_unknown_rule_id():
    merged = merge_rules({"interventions": [{"id": "nonsense", "action": "block"}]})
    assert all(r["id"] != "nonsense" for r in merged["interventions"])


def test_merge_rejects_a_bad_threshold_without_raising():
    merged = merge_rules({"interventions": [{"id": "too_few_rows", "threshold": "lots"}]})
    rule = next(r for r in merged["interventions"] if r["id"] == "too_few_rows")
    assert rule["threshold"] == 500


# ── Interventions ────────────────────────────────────────────────────────────

def test_clean_facts_produce_no_findings():
    findings = rules.evaluate_interventions(default_rules(), {
        "row_count": 6000, "unmapped_subtasks": 0, "missing_required": [],
        "label_is_model_output": False, "duplicate_pct": 0.0,
        "unparseable_date_pct": 0.0, "class_balance_shift_pts": 0.4,
        "new_grouped_level_pct": 0.0,
    })
    assert findings == []


def test_each_blocking_rule_fires_on_its_own_condition():
    findings = rules.evaluate_interventions(default_rules(), {
        "row_count": 400, "unmapped_subtasks": 1, "missing_required": ["PayerName"],
        "label_is_model_output": True, "duplicate_pct": 0.0,
        "unparseable_date_pct": 0.0, "class_balance_shift_pts": 0.0,
        "new_grouped_level_pct": 0.0,
    })
    fired = {f["id"] for f in findings}
    assert fired == {
        "new_subtask", "missing_required_column", "model_output_as_label", "too_few_rows",
    }
    assert all(f["action"] == "block" for f in findings)


def test_warnings_fire_above_threshold_and_not_at_it():
    above = rules.evaluate_interventions(default_rules(), {
        "row_count": 6000, "duplicate_pct": 5.1, "unparseable_date_pct": 1.1,
        "class_balance_shift_pts": 10.1, "new_grouped_level_pct": 2.1,
    })
    assert {f["id"] for f in above} == {
        "duplicate_rows", "unparseable_dates", "class_balance_shift", "new_grouped_level",
    }

    at_threshold = rules.evaluate_interventions(default_rules(), {
        "row_count": 6000, "duplicate_pct": 5.0, "unparseable_date_pct": 1.0,
        "class_balance_shift_pts": 10.0, "new_grouped_level_pct": 2.0,
    })
    assert at_threshold == []


def test_an_unmeasurable_fact_does_not_fire_a_rule():
    """None means "could not measure", which must never read as "passed" or "failed"."""
    findings = rules.evaluate_interventions(default_rules(), {
        "row_count": 6000, "duplicate_pct": None, "unparseable_date_pct": None,
        "class_balance_shift_pts": None, "new_grouped_level_pct": None,
    })
    assert findings == []


def test_a_rule_switched_off_stops_firing():
    config = merge_rules({"interventions": [{"id": "too_few_rows", "action": "off"}]})
    findings = rules.evaluate_interventions(config, {"row_count": 1})
    assert findings == []


def test_blocks_are_listed_before_warnings():
    findings = rules.evaluate_interventions(default_rules(), {
        "row_count": 10, "duplicate_pct": 50.0,
    })
    assert [f["action"] for f in findings] == ["block", "warn"]
    assert len(rules.blocking(findings)) == 1


def test_a_threshold_override_changes_when_a_rule_fires():
    strict = merge_rules({"interventions": [{"id": "duplicate_rows", "threshold": 1.0}]})
    assert rules.evaluate_interventions(strict, {"row_count": 6000, "duplicate_pct": 2.0})
    assert not rules.evaluate_interventions(
        default_rules(), {"row_count": 6000, "duplicate_pct": 2.0}
    )


# ── Recommendations ──────────────────────────────────────────────────────────

@pytest.fixture
def frame():
    """DOSFrom is shuffled; UpdatedDateTimeGMT rises with row order."""
    rows = 400
    return pd.DataFrame({
        "UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "DOSFrom": pd.date_range("2024-01-01", periods=rows, freq="D")[::-1],
        "AccountID": [f"A{i:05d}" for i in range(rows)],
        "AmountBilled": [100.0 + i for i in range(rows)],
    })


def test_date_recommendation_prefers_the_column_matching_row_order(frame):
    """Picking DOSFrom over UpdatedDateTimeGMT splits on the wrong clock."""
    result = rules.recommend_date_column(
        frame, ["DOSFrom", "UpdatedDateTimeGMT"], default_rules()["recommendations"]["date_column"]
    )
    assert result["value"] == "UpdatedDateTimeGMT"
    assert result["evidence"]["ranked"][0]["monotonic"] is True


def test_date_recommendation_flags_a_close_call(frame):
    """Two equally good candidates must lower confidence, not pick silently."""
    frame = frame.copy()
    frame["DOSFrom"] = pd.date_range("2025-01-01", periods=len(frame), freq="D")
    result = rules.recommend_date_column(
        frame, ["DOSFrom", "UpdatedDateTimeGMT"], default_rules()["recommendations"]["date_column"]
    )
    assert result["evidence"]["close_call"] is True
    assert result["confidence"] in ("low", "medium")


def test_dedup_recommendation_reports_a_tie_rather_than_breaking_it(frame):
    """AccountID and PatientAcctNo both look like keys; deduping on the wrong
    one silently collapses distinct accounts."""
    frame = frame.copy()
    frame["PatientAcctNo"] = [f"P{i:05d}" for i in range(len(frame))]
    result = rules.recommend_dedup_key(
        frame, "UpdatedDateTimeGMT", default_rules()["recommendations"]["dedup_key"]
    )
    assert result["evidence"]["tied_with"] is not None
    assert result["confidence"] == "low"


def test_dedup_recommendation_declines_when_nothing_looks_like_a_key():
    frame = pd.DataFrame({"Status": ["open", "closed"] * 50, "Amount": [1.0, 2.0] * 50})
    result = rules.recommend_dedup_key(
        frame, None, default_rules()["recommendations"]["dedup_key"]
    )
    assert result["value"] == []
    assert "full-row" in result["why"]


def test_window_recommendation_uses_full_history_when_balance_is_stable():
    rows = 600
    frame = pd.DataFrame({
        "UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=rows, freq="D"),
    })
    labels = pd.Series([0, 1] * (rows // 2))
    result = rules.recommend_historical_window(
        frame, "UpdatedDateTimeGMT", labels,
        default_rules()["recommendations"]["historical_window"],
    )
    assert result["value"] is None
    assert "full history" in result["why"]


def test_window_recommendation_trims_history_when_balance_shifts():
    rows = 600
    frame = pd.DataFrame({
        "UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=rows, freq="D"),
    })
    # Older half almost all Voice, newer half balanced: the population changed.
    labels = pd.Series([0] * (rows // 2) + [0, 1] * (rows // 4))
    result = rules.recommend_historical_window(
        frame, "UpdatedDateTimeGMT", labels,
        default_rules()["recommendations"]["historical_window"],
    )
    assert isinstance(result["value"], int)
    assert result["value"] >= 30


def test_every_recommendation_carries_its_rule_and_evidence(frame):
    result = rules.recommend_date_column(
        frame, ["UpdatedDateTimeGMT"], default_rules()["recommendations"]["date_column"]
    )
    assert result["rule"]
    assert result["why"]
    assert result["source"] == "rules"
    assert result["evidence"]


# ── Label derivation ─────────────────────────────────────────────────────────
#
# The champion stores SubTask flags as the strings "Voice"/"Non-Voice"/
# "Keyword"/"Ignore". An earlier version of this code mapped those names
# straight through pd.to_numeric, which produced NaN for every row and silently
# switched off the class-balance and historical-window checks — the checks
# still "ran", they just never had anything to measure.

class _Configs:
    """Minimal stand-in carrying only what label derivation reads."""

    def __init__(self, mappings, keywords=()):
        self.subtask_mappings = mappings
        self.subtask_keywords = list(keywords)


def test_string_flags_produce_real_labels_not_nan():
    df = pd.DataFrame({
        "SubTask": ["Called Insurance", "Portal Status Check"] * 10,
        "ARComments": [""] * 20,
    })
    configs = _Configs([
        {"name": "Called Insurance", "flag": "Voice"},
        {"name": "Portal Status Check", "flag": "Non-Voice"},
    ])
    review = {"subtask_column": "SubTask", "unmapped": []}

    labelled, reason = rules.derive_labelled_frame(df, configs, review)
    assert labelled is not None, reason
    assert labelled["NonVoiceFlag"].notna().all()
    assert set(labelled["NonVoiceFlag"]) == {0, 1}
    assert float((labelled["NonVoiceFlag"] == 0).mean()) == pytest.approx(0.5)


def test_keyword_flag_resolves_against_comments():
    """A "Keyword" mapping is not a label — it is a lookup into ARComments."""
    df = pd.DataFrame({
        "SubTask": ["Follow Up"] * 4,
        "ARComments": ["called the payer", "portal updated", "spoke with rep", "no contact"],
    })
    configs = _Configs([{"name": "Follow Up", "flag": "Keyword"}], keywords=["called", "spoke with"])
    labelled, reason = rules.derive_labelled_frame(
        df, configs, {"subtask_column": "SubTask", "unmapped": []}
    )
    assert labelled is not None, reason
    assert labelled["NonVoiceFlag"].tolist() == [0, 1, 0, 1]


def test_ignore_flag_drops_rows_so_the_frame_is_returned_whole():
    """Dropping rows re-indexes, so labels must travel with their own frame."""
    df = pd.DataFrame({
        "SubTask": ["Day to Night", "Called Insurance", "Called Insurance"],
        "ARComments": [""] * 3,
        "UpdatedDateTimeGMT": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
    })
    configs = _Configs([
        {"name": "Day to Night", "flag": "Ignore"},
        {"name": "Called Insurance", "flag": "Voice"},
    ])
    labelled, _ = rules.derive_labelled_frame(
        df, configs, {"subtask_column": "SubTask", "unmapped": []}
    )
    assert len(labelled) == 2
    # Dates and labels still line up inside the returned frame.
    assert labelled["UpdatedDateTimeGMT"].min() == pd.Timestamp("2026-02-01")


def test_unmapped_subtasks_refuse_to_produce_labels():
    configs = _Configs([{"name": "Known", "flag": "Voice"}])
    labelled, reason = rules.derive_labelled_frame(
        pd.DataFrame({"SubTask": ["Known", "Brand New"]}),
        configs,
        {"subtask_column": "SubTask", "unmapped": [{"subtask": "Brand New", "rows": 1}]},
    )
    assert labelled is None
    assert "unmapped" in reason.lower()
