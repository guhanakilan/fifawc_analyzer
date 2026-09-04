"""Item 6 — data-driven weighting proposals.

The point of these rules is that two placements with different data get
different proposals. A rule that fires regardless of the data would be the
static default it replaced.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.services import weight_advisor as advisor


def _facts(**overrides):
    base = {
        "rows": 10000,
        "span_days": 400,
        "majority_class_pct": 52.0,
        "recent_share_pct": 10.0,
        "balance_drift_pts": 1.0,
        "rarest_subtask_pct": 12.0,
        "subtask_column": "SubTask",
        "correction_column": None,
        "error_column": None,
        "client_column": "FacilityName",
        "secondary_column": "PayerName",
    }
    base.update(overrides)
    return base


def _reason(proposal, component):
    return next(r for r in proposal["reasons"] if r["component"] == component)


def test_balanced_data_with_no_flags_proposes_no_weighting():
    proposal = advisor.propose(_facts())
    assert proposal["recommend_weighting"] is False
    assert proposal["strategy"]["enabled"] is False


def test_imbalance_enables_class_balancing():
    proposal = advisor.propose(_facts(majority_class_pct=82.0))
    assert proposal["strategy"]["components"]["class_balance"]["enabled"] is True
    assert "82.0%" in _reason(proposal, "class_balance")["why"]


def test_balance_drift_prefers_stronger_recency_than_mere_volume():
    """A shifted population is a better reason to up-weight than recent volume."""
    drifted = advisor.propose(_facts(balance_drift_pts=14.0))
    volume = advisor.propose(_facts(recent_share_pct=30.0))
    assert drifted["strategy"]["components"]["recency"]["weight"] == 2.0
    assert volume["strategy"]["components"]["recency"]["weight"] == 1.5


def test_flag_columns_bind_to_their_components():
    proposal = advisor.propose(
        _facts(correction_column="HumanCorrected", error_column="PreviousModelError")
    )
    components = proposal["strategy"]["components"]
    assert components["human_correction"]["column"] == "HumanCorrected"
    assert components["verified_error"]["column"] == "PreviousModelError"


def test_rare_subtask_is_up_weighted_only_when_actually_rare():
    assert advisor.propose(_facts(rarest_subtask_pct=0.4))[
        "strategy"]["components"]["rare_subtask"]["enabled"] is True
    assert advisor.propose(_facts(rarest_subtask_pct=8.0))[
        "strategy"]["components"]["rare_subtask"]["enabled"] is False


def test_too_little_data_overrides_every_other_rule():
    """Weighting a small sample mostly amplifies its noise."""
    proposal = advisor.propose(
        _facts(rows=400, majority_class_pct=95.0, correction_column="HumanCorrected")
    )
    assert proposal["recommend_weighting"] is False
    assert proposal["strategy"]["enabled"] is False
    assert "not enough data" in proposal["why"].lower()


def test_a_short_span_also_declines_weighting():
    proposal = advisor.propose(_facts(span_days=30, majority_class_pct=90.0))
    assert proposal["recommend_weighting"] is False


def test_every_component_states_a_reason_either_way():
    """A proposal has to be arguable, so silence is not an option."""
    proposal = advisor.propose(_facts(majority_class_pct=85.0))
    named = {r["component"] for r in proposal["reasons"]}
    assert named == set(proposal["strategy"]["components"])
    assert all(r["why"].strip() for r in proposal["reasons"])


# ── measure() against real frames ────────────────────────────────────────────

def _frame(n=1000, start="2025-01-01"):
    dates = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "UpdatedDateTimeGMT": dates,
        "SubTask": (["Common"] * (n - 5)) + (["VeryRare"] * 5),
        "FacilityName": ["North Clinic"] * n,
        "PayerName": ["Aetna"] * n,
        "HumanCorrected": [0] * n,
    })


def test_measure_finds_the_client_and_secondary_dimensions():
    df = _frame()
    facts = advisor.measure(
        df, df["UpdatedDateTimeGMT"], pd.Series([0, 1] * (len(df) // 2)),
        advisor.DEFAULT_THRESHOLDS,
    )
    assert facts["client_column"] == "FacilityName"
    assert facts["secondary_column"] == "PayerName"
    assert facts["correction_column"] == "HumanCorrected"


def test_measure_reports_the_rarest_subtask_share():
    df = _frame(n=1000)
    facts = advisor.measure(
        df, df["UpdatedDateTimeGMT"], pd.Series([0, 1] * 500), advisor.DEFAULT_THRESHOLDS
    )
    assert facts["rarest_subtask_pct"] == pytest.approx(0.5, abs=0.01)


def test_measure_survives_a_frame_with_no_optional_columns():
    df = pd.DataFrame({"UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=50, freq="D")})
    facts = advisor.measure(
        df, df["UpdatedDateTimeGMT"], pd.Series([0, 1] * 25), advisor.DEFAULT_THRESHOLDS
    )
    assert facts["client_column"] is None
    assert facts["rarest_subtask_pct"] is None
    # And a proposal can still be made from it.
    advisor.propose(facts)


def test_dimension_lookup_ignores_spacing_in_column_names():
    """norm_col keeps internal spaces, so "Payer Name" must still resolve.

    An upload that still carries inventory-style names ("Payer Name") is the
    normal case before the champion's rename map is applied, and a lookup that
    only matched "PayerName" would silently find no secondary dimension.
    """
    df = pd.DataFrame({
        "UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=10, freq="D"),
        "Payer Name": ["Aetna"] * 10,
        "Facility Name": ["North"] * 10,
    })
    facts = advisor.measure(
        df, df["UpdatedDateTimeGMT"], pd.Series([0, 1] * 5), advisor.DEFAULT_THRESHOLDS
    )
    assert facts["client_column"] == "Facility Name"
    assert facts["secondary_column"] == "Payer Name"
