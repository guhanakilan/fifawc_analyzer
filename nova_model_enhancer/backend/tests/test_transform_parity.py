"""Transformation parity, split integrity and threshold/encoding behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nova_model_enhancer.backend.services import labeling, weighting
from nova_model_enhancer.backend.services.evaluator import (
    evaluate_gate,
    metrics_at_threshold,
    threshold_sweep,
)
from nova_model_enhancer.backend.services.nova_transform import (
    NovaConfigs,
    apply_bucketing,
    apply_dtype_config,
    apply_grouping,
    build_modelling_frame,
    build_rename_map,
    dedupe_columns,
    fit_transform_by_indices,
    norm_col,
    translate_feature_selection,
    unwrap_list,
)
from nova_model_enhancer.backend.services.splitter import (
    describe_split,
    temporal_split_indices,
)
from nova_model_enhancer.backend.tests import fixtures


# ── Column identity ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Sub-Task", "subtask"),
    ("  Amount  Billed ", "amount billed"),
    ("UpdatedDateTimeGMT", "updateddatetimegmt"),
    ("Payer/Name", "payername"),
])
def test_norm_col_matches_the_reference(raw, expected):
    assert norm_col(raw) == expected


def test_dedupe_columns_keeps_colliding_names_addressable():
    assert dedupe_columns(["subtask", "subtask", "amount"]) == ["subtask", "subtask_2", "amount"]


def test_rename_map_only_renames_when_the_name_actually_differs():
    column_map = [
        {"inventory": "Payer Name", "production": "PayerName", "include": True},
        {"inventory": "AgeDays", "production": "AgeDays", "include": True},
        {"inventory": "Dropped", "production": "Dropped2", "include": False},
    ]
    mapping = build_rename_map(column_map, ["Payer Name", "AgeDays", "Dropped"])
    assert mapping == {"Payer Name": "payername"}


# ── Config shape tolerance (defects D1/D2) ──────────────────────────────────

def test_wrapped_and_bare_config_shapes_both_parse():
    assert unwrap_list({"column_map": [{"a": 1}]}, "column_map") == [{"a": 1}]
    assert unwrap_list([{"a": 1}], "column_map") == [{"a": 1}]
    assert unwrap_list({"selected_columns": ["x"]}, "selected_columns") == ["x"]
    assert unwrap_list(None, "selected_columns") == []


# ── Transform steps ──────────────────────────────────────────────────────────

def test_dtype_config_refuses_to_guess_between_colliding_columns():
    frame = pd.DataFrame({"Sub-Task": ["a"], "SubTask": ["b"]})
    result = apply_dtype_config(frame, {"subtask": {"dtype": "float64", "fallback": 0}})
    # Both normalise to "subtask"; neither may be silently coerced.
    assert result["Sub-Task"].tolist() == ["a"]
    assert result["SubTask"].tolist() == ["b"]


def test_bucketing_and_grouping_replace_their_source_column():
    frame = pd.DataFrame({"amt": [10.0, 500.0, 5000.0], "payer": ["aetna", "zzz", None]})
    bucketed = apply_bucketing(frame, {"amt": {"cuts": [100, 1000], "labels": ["l", "m", "h"]}})
    assert "amt" not in bucketed.columns
    assert bucketed["amt_Bucket"].tolist() == ["l", "m", "h"]

    grouped = apply_grouping(bucketed, {"payer": {
        "kept_values": ["aetna"], "others_label": "other", "null_label": "na",
    }})
    assert "payer" not in grouped.columns
    assert grouped["payer_Grouped"].tolist() == ["aetna", "other", "na"]


def test_feature_selection_is_rename_aware():
    assert translate_feature_selection(
        ["amt", "payer", "age"], {"amt": {}}, {"payer": {}}
    ) == ["amt_Bucket", "payer_Grouped", "age"]


def test_transforms_are_fit_on_train_rows_only():
    """A value that appears only in the test split must not shape the fitted state."""
    frame = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 1000.0],
        "cat": ["a", "a", "b", "b", "only_in_test"],
        "NonVoiceFlag": [0, 1, 0, 1, 1],
    })
    cfg = {
        "outlier_capping": [{"col": "num", "enabled": True}],
        "imputation": [{"col": "num", "strategy": "Median", "enabled": True}],
        "encoding": [{"col": "cat", "method": "Label", "enabled": True}],
        "scaling": [], "log_transform": [],
    }
    _, _, X_test, _, _, _, _, fitted = fit_transform_by_indices(
        frame, cfg, train_idx=[0, 1, 2, 3], test_idx=[4]
    )
    assert "only_in_test" not in fitted["label_encoders"]["cat"]["classes"]
    assert X_test["cat"].iloc[0] == -1, "an unseen category must encode to -1, as at scoring time"
    assert fitted["outlier_bounds"]["num"]["upper"] < 1000.0


def test_modelling_frame_matches_the_champion_feature_set(champion_export):
    """The rebuilt frame must produce exactly the champion's fitted features."""
    import pickle
    import zipfile

    configs_raw = {
        "column_map": champion_export["configs"]["column_map.json"],
        "column_config": champion_export["configs"]["column_config.json"],
        "dtype_config": champion_export["configs"]["dtype_config.json"],
        "derived_config": champion_export["configs"]["derived_config.json"],
        "bucket_config": champion_export["configs"]["bucket_config.json"],
        "grouping_config": champion_export["configs"]["grouping_config.json"],
        "feature_selection": champion_export["configs"]["feature_selection.json"],
        "features_config": champion_export["configs"]["features_config.json"],
    }
    configs = NovaConfigs(configs_raw, date_column="updateddatetimegmt")
    frame = build_modelling_frame(champion_export["labelled"], configs)
    assert "NonVoiceFlag" in frame.columns
    for expected in ("insurancebalance", "agedays", "facilityname",
                     "dosage_days", "amountbilled_Bucket", "payername_Grouped"):
        assert expected in frame.columns, f"{expected} missing from the modelling frame"


# ── Labelling ────────────────────────────────────────────────────────────────

def test_subtask_mapping_reproduces_the_reference_rules():
    frame = pd.DataFrame({
        "SubTask": ["Voice A", "NV A", "KW A", "KW A", "Drop A"],
        "ARComments": ["", "", "we called them", "portal update", ""],
    })
    mappings = [
        {"name": "Voice A", "flag": "Voice"},
        {"name": "NV A", "flag": "Non-Voice"},
        {"name": "KW A", "flag": "Keyword"},
        {"name": "Drop A", "flag": "Ignore"},
    ]
    labelled, stats = labeling.apply_subtask_mapping(frame, mappings, ["called"])
    assert labelled["NonVoiceFlag"].tolist() == [0, 1, 0, 1]
    assert stats["ignored_count"] == 1
    assert stats["voice_count"] == 2 and stats["non_voice_count"] == 2


def test_unmapped_subtask_is_refused_by_default():
    frame = pd.DataFrame({"SubTask": ["Known", "Brand New"], "ARComments": ["", ""]})
    with pytest.raises(ValueError, match="no approved mapping"):
        labeling.apply_subtask_mapping(frame, [{"name": "Known", "flag": "Voice"}], [])


def test_unmapped_subtask_default_requires_explicit_opt_in():
    frame = pd.DataFrame({"SubTask": ["Known", "Brand New"], "ARComments": ["", ""]})
    labelled, stats = labeling.apply_subtask_mapping(
        frame, [{"name": "Known", "flag": "Voice"}], [], allow_unmapped_default=True
    )
    assert labelled["NonVoiceFlag"].tolist() == [0, 1]
    assert stats["unmapped_defaulted_to_non_voice"] is True


def test_suggestions_follow_the_reference_rule_order():
    assert labeling.suggest_flag("Non Voice Task", []) == "Non-Voice"
    assert labeling.suggest_flag("Voice Task", []) == "Voice"
    assert labeling.suggest_flag("Day to Night Transfer", ["Voice"]) == "Ignore"
    assert labeling.suggest_flag("Anything", ["Non Workable"]) == "Ignore"
    assert labeling.suggest_flag("Anything", []) == "Keyword"


# ── Split ────────────────────────────────────────────────────────────────────

def test_temporal_split_has_no_future_leakage():
    rows = 1000
    dates = pd.Series(pd.date_range("2025-01-01", periods=rows, freq="h")).sample(
        frac=1.0, random_state=3
    ).reset_index(drop=True)
    y = pd.Series(np.resize([0, 1], rows))
    train_idx, val_idx, test_idx = temporal_split_indices(dates, y, test_size=0.15)
    described = describe_split(dates, train_idx, val_idx, test_idx)
    assert described["no_future_leakage"] is True
    assert dates.iloc[train_idx].max() <= dates.iloc[val_idx].min()
    assert dates.iloc[val_idx].max() <= dates.iloc[test_idx].min()
    assert len(train_idx) + len(val_idx) + len(test_idx) == rows


def test_undated_rows_land_in_train_not_test():
    dates = pd.Series([pd.NaT] * 50 + list(pd.date_range("2025-01-01", periods=450)))
    y = pd.Series(np.resize([0, 1], 500))
    train_idx, _, test_idx = temporal_split_indices(dates, y, test_size=0.2)
    assert dates.iloc[test_idx].notna().all()
    assert dates.iloc[train_idx].isna().sum() == 50


def test_validation_slice_is_skipped_when_too_small():
    dates = pd.Series(pd.date_range("2025-01-01", periods=60))
    y = pd.Series(np.resize([0, 1], 60))
    _, val_idx, _ = temporal_split_indices(dates, y, test_size=0.2)
    assert len(val_idx) == 0, "a validation slice below the minimum must not be carved out"


# ── Weights ──────────────────────────────────────────────────────────────────

def test_weights_are_multiplicative_and_capped():
    frame = pd.DataFrame({
        "flag_a": ["1", "0", "1", "1"],
        "flag_b": ["1", "0", "0", "1"],
        "SubTask": ["common"] * 4,
    })
    dates = pd.Series(pd.to_datetime(["2026-01-01"] * 4))
    strategy = {
        "enabled": True, "cap": 5.0, "historical_base": 1.0, "normalise_mean_to_one": False,
        "components": {
            "human_correction": {"enabled": True, "weight": 3.0, "column": "flag_a", "true_values": ["1"]},
            "verified_error": {"enabled": True, "weight": 2.5, "column": "flag_b", "true_values": ["1"]},
        },
    }
    weights, summary = weighting.compute_weights(frame, strategy, dates=dates)
    # Rows 0 and 3 fire both components: 1 * 3.0 * 2.5 = 7.5 before the cap.
    assert weights[0] == 5.0
    assert weights[1] == 1.0
    assert weights[2] == 3.0
    assert weights[3] == 5.0
    assert summary["distribution"]["capped_rows"] == 2


def test_weighting_disabled_gives_every_row_one():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    weights, summary = weighting.compute_weights(frame, {"enabled": False})
    assert list(weights) == [1.0, 1.0, 1.0]
    assert summary["skipped"][0]["component"] == "all"


def test_missing_weight_column_is_reported_not_silently_ignored():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    strategy = {
        "enabled": True, "cap": 5.0, "components": {
            "human_correction": {"enabled": True, "weight": 3.0, "column": "not_here"},
        },
    }
    _, summary = weighting.compute_weights(frame, strategy)
    assert any(s["component"] == "human_correction" for s in summary["skipped"])


def test_weight_formula_is_recorded_verbatim():
    text = weighting.formula_text({
        "enabled": True, "cap": 5.0, "historical_base": 1.0,
        "components": {"recency": {"enabled": True, "weight": 1.5, "recent_days": 90}},
    })
    assert "1.5" in text and "90 days" in text and "capped at 5.0" in text


# ── Threshold and encoding ───────────────────────────────────────────────────

def test_threshold_sweep_covers_the_reference_grid():
    y = np.resize([0, 1], 400)
    rng = np.random.default_rng(0)
    proba = np.clip(y * 0.4 + rng.normal(0.3, 0.15, 400), 0, 1)
    result = threshold_sweep(y, proba)
    thresholds = [row["t"] for row in result["sweep"]]
    assert thresholds[0] == 0.1 and thresholds[-1] == 0.9
    assert all(criterion in result["best"] for criterion in ("f1", "recall", "precision"))


def test_metrics_use_the_confirmed_target_encoding():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = metrics_at_threshold(y, proba, 0.5)
    assert metrics["confusion_matrix"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert metrics["actual_non_voice"] == 2 and metrics["actual_voice"] == 2
    assert metrics["predicted_non_voice"] == 2


def test_ml_tag_is_the_inverse_of_the_internal_target():
    """A row predicted Non-Voice (proba >= threshold) must not be tagged as Voice."""
    proba = np.array([0.9, 0.1])
    threshold = 0.5
    ml_tag = np.where(proba < threshold, 1, 0)   # 1 = Voice, 0 = Non-Voice
    internal = (proba >= threshold).astype(int)  # NonVoiceFlag: 1 = Non-Voice
    assert ml_tag.tolist() == [0, 1]
    assert internal.tolist() == [1, 0]
    assert (ml_tag != internal).all(), "ml_tag must be inverted relative to NonVoiceFlag"


# ── Promotion gate ───────────────────────────────────────────────────────────

def _gate(**overrides):
    base = {
        "primary_metric": "f1", "min_primary_improvement_pct": 1.0,
        "protected_metrics": [{"metric": "recall", "max_regression_pct": 0.5}],
        "max_historical_primary_regression_pct": 1.0,
        "require_backtest_pass": False, "require_package_validation": False,
        "approved": True,
    }
    base.update(overrides)
    return base


def test_gate_recommends_when_every_rule_passes():
    result = evaluate_gate({"f1": 0.70, "recall": 0.70}, {"f1": 0.75, "recall": 0.71}, _gate())
    assert result["status"] == "RECOMMENDED"


def test_gate_declines_on_insufficient_improvement():
    result = evaluate_gate({"f1": 0.70, "recall": 0.70}, {"f1": 0.7005, "recall": 0.71}, _gate())
    assert result["status"] == "NOT_RECOMMENDED"


def test_gate_declines_on_protected_metric_regression():
    result = evaluate_gate({"f1": 0.70, "recall": 0.70}, {"f1": 0.80, "recall": 0.60}, _gate())
    assert result["status"] == "NOT_RECOMMENDED"


def test_unapproved_gate_blocks_rather_than_declining():
    result = evaluate_gate({"f1": 0.7}, {"f1": 0.9}, _gate(approved=False))
    assert result["status"] == "BLOCKED"
    assert any("not been approved" in b for b in result["blockers"])


def test_data_quality_blocker_outranks_a_passing_comparison():
    result = evaluate_gate(
        {"f1": 0.70, "recall": 0.70}, {"f1": 0.95, "recall": 0.95}, _gate(),
        data_quality_blockers=["Unmapped SubTasks were defaulted."],
    )
    assert result["status"] == "BLOCKED"


# ── Schema drift and column detection ────────────────────────────────────────

def test_drift_applies_the_rename_map_before_comparing():
    """An inventory-named column must not be reported as a missing production column."""
    from nova_model_enhancer.backend.services.data_profiler import drift_report

    profile = {"column_names": ["Amount Billed", "Payer Name", "AgeDays"]}
    column_map = [
        {"inventory": "Amount Billed", "production": "AmountBilled", "include": True},
        {"inventory": "Payer Name", "production": "PayerName", "include": True},
    ]
    naive = drift_report(profile, ["amountbilled", "payername", "agedays"])
    assert naive["missing_column_count"] == 2, "without the map these look missing"

    aware = drift_report(profile, ["amountbilled", "payername", "agedays"], column_map)
    assert aware["missing_column_count"] == 0
    assert aware["new_column_count"] == 0


def test_date_detection_follows_candidate_priority_not_column_order():
    from nova_model_enhancer.backend.routers.readiness import _ordered_candidates
    from nova_model_enhancer.backend.services.data_profiler import DATE_CANDIDATES

    columns = ["AccountID", "DOSFrom", "UpdatedDateTimeGMT", "MLFlagDate"]
    ordered = _ordered_candidates(columns, DATE_CANDIDATES)
    assert ordered[0] == "UpdatedDateTimeGMT", (
        "the split must default to the update timestamp, not the first date-like column"
    )
