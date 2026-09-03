"""Item 2 — the four-layer column lineage report."""

from __future__ import annotations

import pytest

from backend.services.data_profiler import column_lineage
from backend.services.nova_transform import NovaConfigs


@pytest.fixture
def configs():
    """A package shaped like the real PLC984 export."""
    return NovaConfigs({
        "column_map": {"column_map": [
            {"inventory": "Amount Billed", "production": "AmountBilled", "include": True},
            {"inventory": "Payer Name", "production": "PayerName", "include": True},
            {"inventory": "DOSFrom", "production": "DOSFrom", "include": True},
            {"inventory": "UpdatedDateTimeGMT", "production": "UpdatedDateTimeGMT", "include": True},
            {"inventory": "ARComments", "production": "ARComments", "include": True},
            {"inventory": "Task", "production": "Task", "include": True},
            {"inventory": "Secret", "production": "Secret", "include": False},
        ]},
        "column_config": {"matched_columns": ["AmountBilled", "PayerName", "ARComments"]},
        "feature_selection": {"selected_columns": ["amountbilled", "payername", "dosage_days"]},
        "derived_config": [{
            "output_col": "dosage_days", "col_type": "date_diff",
            "date_col": "updateddatetimegmt", "reference_col": "dosfrom",
        }],
    })


def _stage(report, column):
    return next(c["stage"] for c in report["columns"] if c["column"].lower() == column.lower())


def test_derived_column_is_never_reported_as_missing(configs):
    """dosage_days is computed, not uploaded — demanding it is a false alarm."""
    report = column_lineage(configs, ["AmountBilled", "PayerName", "DOSFrom", "UpdatedDateTimeGMT"])
    assert report["missing_required"] == []
    assert _stage(report, "dosage_days") == "derived"


def test_sources_of_a_derived_feature_are_required(configs):
    """DOSFrom is not a feature, but the model cannot be built without it."""
    report = column_lineage(configs, ["AmountBilled", "PayerName", "UpdatedDateTimeGMT"])
    assert report["missing_required"] == ["DOSFrom"]
    assert _stage(report, "DOSFrom") == "feeds_derived"


def test_column_dropped_between_matching_and_selection_is_flagged(configs):
    report = column_lineage(configs, ["AmountBilled", "PayerName", "ARComments"])
    assert "ARComments" in report["removed_during_build"]
    assert _stage(report, "ARComments") == "dropped_during_build"


def test_column_excluded_at_mapping_is_its_own_category(configs):
    report = column_lineage(configs, ["AmountBilled"])
    assert _stage(report, "Secret") == "excluded_at_mapping"


def test_inventory_names_are_renamed_before_comparison(configs):
    """An upload using inventory names must not read as wholesale drift."""
    report = column_lineage(configs, ["Amount Billed", "Payer Name", "DOSFrom", "UpdatedDateTimeGMT"])
    assert report["missing_required"] == []
    assert report["unexpected_columns"] == []


def test_genuinely_new_column_is_reported(configs):
    report = column_lineage(configs, ["AmountBilled", "PayerName", "DOSFrom",
                                      "UpdatedDateTimeGMT", "BrandNew"])
    assert report["unexpected_columns"] == ["BrandNew"]


def test_fitted_layer_is_optional_and_marked_as_such(configs):
    without = column_lineage(configs, ["AmountBilled"])
    assert without["layers"]["fitted_available"] is False

    with_fitted = column_lineage(
        configs, ["AmountBilled"],
        fitted_feature_names=["amountbilled", "payername_aetna", "payername_bcbs"],
    )
    assert with_fitted["layers"]["fitted_available"] is True
    payername = next(c for c in with_fitted["columns"] if c["column"].lower() == "payername")
    assert payername["fitted_feature_count"] == 2
