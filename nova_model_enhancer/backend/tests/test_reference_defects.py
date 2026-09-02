"""Evidence for the reference-application defects in IMPLEMENTATION_GAP_ANALYSIS.md §4.

These are not tests of this application's behaviour — they are executable proof
that the defects are real, run against the reference source itself. They skip
when the reference tree has not been provided, so the suite still passes on a
machine that only has the enhancer.

Point NOVA_ENHANCER_REFERENCE_SCORING at nova-ml's `scoring_client/scoring.py`
to enable them.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nova_model_enhancer.backend.config import reference_scoring_path
from nova_model_enhancer.backend.services.exporter import _load_runtime_module

SHIPPED_RUNTIME = (
    Path(__file__).resolve().parents[1] / "scoring_runtime" / "scoring.py"
)


def _reference_module():
    path = reference_scoring_path()
    if path is None:
        pytest.skip(
            "Reference scoring client not provided. Set NOVA_ENHANCER_REFERENCE_SCORING "
            "to nova-ml's scoring_client/scoring.py to run the defect evidence tests."
        )
    return _load_runtime_module(path, "nova_reference_under_test")


def _shipped_module():
    return _load_runtime_module(SHIPPED_RUNTIME, "nova_shipped_under_test")


@pytest.fixture()
def package_root(tmp_path):
    """A minimal deployment layout in the exact shapes nova-ml writes on disk."""
    root = tmp_path / "placement"
    for folder in ("config", "model", "scoring", "metadata"):
        (root / folder).mkdir(parents=True)

    # Wrapper-object shapes, exactly as file_store.py writes them.
    (root / "config" / "column_map.json").write_text(json.dumps({
        "column_map": [{"inventory": "Amount Billed", "production": "AmountBilled", "include": True}],
        "coverage_threshold": None,
    }))
    (root / "config" / "feature_selection.json").write_text(json.dumps({
        "selected_columns": ["amountbilled", "agedays"],
    }))
    (root / "config" / "dtype_config.json").write_text(json.dumps({}))
    (root / "config" / "derived_config.json").write_text(json.dumps([]))
    (root / "config" / "bucket_config.json").write_text(json.dumps({}))
    (root / "config" / "grouping_config.json").write_text(json.dumps({}))
    (root / "scoring" / "threshold_config.json").write_text(json.dumps({"JOB_x_lr": 0.5}))
    (root / "metadata" / "training_results.json").write_text(json.dumps({"best_model": "JOB_x_lr"}))

    from sklearn.linear_model import LogisticRegression
    import joblib

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["amountbilled", "agedays"])
    y = (X["amountbilled"] > 0).astype(int)
    estimator = LogisticRegression().fit(X.values, y.values)
    joblib.dump(estimator, root / "model" / "JOB_x_lr.pkl")

    with (root / "model" / "fitted_transforms.pkl").open("wb") as handle:
        pickle.dump({"feature_names": ["amountbilled", "agedays"]}, handle)

    return root


@pytest.fixture()
def inventory():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "Amount Billed": rng.normal(size=40),
        "AgeDays": rng.normal(size=40),
    })


def test_d1_reference_client_cannot_read_a_real_column_map(package_root, inventory):
    """nova-ml writes column_map.json wrapped; its own client iterates it as a list."""
    reference = _reference_module()
    with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
        reference.NovaMLPipeline(resource_root=package_root).run(inventory.copy())


def test_shipped_runtime_reads_the_same_files(package_root, inventory):
    shipped = _shipped_module()
    scored = shipped.NovaMLPipeline(resource_root=package_root).run(inventory.copy())
    assert len(scored) == len(inventory)
    assert list(scored.columns)[:2] == list(inventory.columns)
    assert scored["NovaProbability"].nunique() > 1, (
        "a working loader must produce varying probabilities, not a constant"
    )


def test_d2_wrapped_feature_selection_silently_zeroes_every_feature(package_root, inventory):
    """The failure is silent: a constant score, not an exception."""
    shipped = _shipped_module()
    pipeline = shipped.NovaMLPipeline(resource_root=package_root)

    # Reproduce the reference behaviour: read the wrapper object as if it were a list.
    raw = json.loads((package_root / "config" / "feature_selection.json").read_text())
    pipeline.feature_selection = list(raw)          # -> ["selected_columns"]
    _, X = pipeline._prepare(inventory.copy())
    assert (X.values == 0).all(), "the mis-read selection should zero the whole matrix"

    pipeline.feature_selection = raw["selected_columns"]
    _, X_ok = pipeline._prepare(inventory.copy())
    assert not (X_ok.values == 0).all()


def test_d4_unnormalised_one_hot_names_zero_every_one_hot_feature(package_root, inventory):
    """One-hot names are stored cased; the reference normalises only the dict key."""
    from nova_model_enhancer.backend.services.nova_transform import norm_col

    stored = {"payername_Grouped": ["payername_Grouped_aetna", "payername_Grouped_bcbs"]}

    reference_style = {norm_col(k): v for k, v in stored.items()}
    shipped_style = {norm_col(k): [norm_col(c) for c in v] for k, v in stored.items()}

    # At score time the dummies are generated from the lowercased column.
    generated = ["payername_grouped_aetna", "payername_grouped_bcbs"]

    assert not set(reference_style["payername_grouped"]) & set(generated), (
        "reference-style names cannot match the dummies it generates — every "
        "one-hot column is therefore filled with zero"
    )
    assert set(shipped_style["payername_grouped"]) == set(generated)


def test_d3_train_and_score_transform_orders_differ():
    """Imputation lands before capping/log at score time and after them at fit time."""
    import inspect

    from nova_model_enhancer.backend.services import nova_transform

    fit_source = inspect.getsource(nova_transform.fit_transform_by_indices)
    fit_order = [
        step for step in ("outlier", "log_cols", "imputation", "encoding")
        if step in fit_source
    ]
    assert fit_order.index("outlier") < fit_order.index("imputation"), (
        "fitting must keep the reference's own order so the artifact stays interchangeable"
    )

    apply_source = inspect.getsource(nova_transform.apply_fitted_transforms)
    assert apply_source.index("apply_imputation") < apply_source.index("apply_outlier_bounds"), (
        "inference must use the deployed loader's order, or every reported number "
        "disagrees with what the package actually produces"
    )
