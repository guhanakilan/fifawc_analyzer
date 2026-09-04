"""Same and only the champion's features, feature engineering and threshold.

A challenger trained through different preprocessing, or scored at a different
cutoff, is not comparable to the champion — the difference in metrics is then
the pipeline rather than the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services.nova_transform import (
    DerivedColumnError,
    NovaConfigs,
    build_modelling_frame,
)
from backend.services.pipeline import PipelineError, choose_threshold


def _configs(**overrides):
    raw = {
        "column_map": {"column_map": [
            {"inventory": "Amount Billed", "production": "AmountBilled", "include": True},
            {"inventory": "DOSFrom", "production": "DOSFrom", "include": True},
            {"inventory": "UpdatedDateTimeGMT", "production": "UpdatedDateTimeGMT", "include": True},
        ]},
        "feature_selection": {"selected_columns": ["amountbilled", "dosage_days"]},
        "derived_config": [{
            "output_col": "dosage_days", "col_type": "date_diff",
            "date_col": "updateddatetimegmt", "reference_col": "dosfrom",
        }],
        "features_config": {"imputation": [{"col": "amountbilled", "strategy": "Median",
                                            "enabled": True}]},
    }
    raw.update(overrides)
    return NovaConfigs(raw)


def _frame(with_source=True):
    data = {
        "Amount Billed": [100.0, 200.0, 300.0],
        "NonVoiceFlag": [0, 1, 0],
        "UpdatedDateTimeGMT": pd.to_datetime(["2026-03-01", "2026-03-05", "2026-03-09"]),
    }
    if with_source:
        data["DOSFrom"] = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    return pd.DataFrame(data)


# ── Custom columns ───────────────────────────────────────────────────────────

def test_a_custom_column_is_rebuilt_from_the_champions_definition():
    frame = build_modelling_frame(_frame(), _configs())
    assert "dosage_days" in frame.columns
    assert frame["dosage_days"].tolist() == [59.0, 62.0, 65.0]


def test_a_custom_column_that_cannot_be_rebuilt_is_an_error_not_a_gap():
    """Silently training without a feature the champion has is the failure mode."""
    with pytest.raises(DerivedColumnError) as exc:
        build_modelling_frame(_frame(with_source=False), _configs())
    assert "dosage_days" in str(exc.value)
    assert "would not reproduce the champion's features" in str(exc.value)


def test_an_unsupported_custom_column_type_is_caught():
    """A newer NoVA ML could emit a col_type this version does not implement."""
    configs = _configs(derived_config=[{
        "output_col": "some_new_thing", "col_type": "rolling_window_from_a_future_version",
    }])
    with pytest.raises(DerivedColumnError, match="some_new_thing"):
        build_modelling_frame(_frame(), configs)


# ── Feature set ──────────────────────────────────────────────────────────────

def test_only_the_champions_selected_features_survive():
    frame = build_modelling_frame(_frame(), _configs())
    # DOSFrom and UpdatedDateTimeGMT fed the derived column but are not features.
    assert set(frame.columns) == {"amountbilled", "dosage_days", "NonVoiceFlag"}


# ── Transform configuration ──────────────────────────────────────────────────

def test_a_package_with_no_transform_config_is_refused_rather_than_invented():
    """The enhancer used to fabricate one, which trained a different pipeline."""
    from backend.services.pipeline import prepare_matrices

    configs = _configs(features_config={})
    with pytest.raises(PipelineError) as exc:
        prepare_matrices(
            _frame(), configs, "UpdatedDateTimeGMT",
            {"mode": "temporal", "train_pct": 70, "val_pct": 15, "test_pct": 15},
            {"enabled": False},
        )
    assert "will not substitute its own" in str(exc.value)


# ── Threshold ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("champion_threshold", [0.35, 0.5, 0.62, 0.9])
def test_every_model_is_scored_at_the_champions_threshold(champion_threshold):
    y = np.resize([0, 1], 200)
    rng = np.random.default_rng(4)
    proba = np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1), 0.2), 0.001, 0.999)

    result = choose_threshold(y, proba, y, proba, champion_threshold, "f1")
    assert result["selected_threshold"] == champion_threshold
    assert [c["threshold"] for c in result["candidates"]] == [champion_threshold]
