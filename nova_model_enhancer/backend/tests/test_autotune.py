"""The escalating retrain loop.

Two properties matter more than any other: the loop must never read the test
split, and it must be able to fail. A loop that always reaches its target is
either lucky or lying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services import autotune


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    logit = 1.2 * X.f0 - 0.9 * X.f1 + 0.5 * X.f2
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X.iloc[:2200], y.iloc[:2200], X.iloc[2200:], y.iloc[2200:]


def _run(data, **overrides):
    X_train, y_train, X_val, y_val = data
    kwargs = dict(
        model_type="lgb",
        champion_params={"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
        X_train=X_train, y_train=y_train, w_train=None,
        X_val=X_val, y_val=y_val,
        max_rounds=4, time_budget_seconds=120, patience=2,
    )
    kwargs.update(overrides)
    return autotune.run(**kwargs)


# ── The ladder ───────────────────────────────────────────────────────────────

def test_the_ladder_grows_from_the_champions_own_values():
    rounds = autotune.build_rounds(
        "lgb", {"n_estimators": 150, "num_leaves": 31, "learning_rate": 0.08}, 4, None
    )
    assert rounds[0]["n_estimators"] == 150, "round one must be the champion's own setting"
    trees = [r["n_estimators"] for r in rounds]
    leaves = [r["num_leaves"] for r in rounds]
    rates = [r["learning_rate"] for r in rounds]
    assert trees == sorted(trees) and trees[-1] > trees[0], "capacity must grow"
    assert leaves == sorted(leaves), "branches must grow"
    assert rates[-1] < rates[0], "learning rate must fall as capacity rises"


def test_parameters_are_clamped_to_sane_bounds():
    """A multiplier must not be able to produce a nonsense model."""
    rounds = autotune.build_rounds("lgb", {"n_estimators": 100000, "num_leaves": 99999}, 3, None)
    assert all(r["n_estimators"] <= autotune.BOUNDS["n_estimators"][1] for r in rounds)
    assert all(r["num_leaves"] <= autotune.BOUNDS["num_leaves"][1] for r in rounds)
    assert all(isinstance(r["n_estimators"], int) for r in rounds)


def test_class_balance_is_applied_only_when_asked():
    assert "class_weight" not in autotune.build_rounds("lgb", {}, 2, None)[0]
    assert autotune.build_rounds("lgb", {}, 2, True)[0]["class_weight"] == "balanced"
    assert autotune.build_rounds("lgb", {}, 2, False)[0]["class_weight"] is None


def test_an_unknown_family_is_refused_rather_than_guessed():
    with pytest.raises(autotune.AutotuneError, match="No escalation ladder"):
        autotune.build_rounds("some_new_model", {}, 3, None)


# ── Stopping ─────────────────────────────────────────────────────────────────

def test_an_unreachable_target_fails_honestly(data):
    """The loop must be able to say no. Reporting success here would be a lie."""
    result = _run(data, target_value=0.999, max_rounds=4)
    assert result["target_reached"] is False
    assert result["stop_reason"] in ("no_improvement", "rounds_exhausted", "gain_within_noise")
    assert result["best_score"] < 0.999


def test_a_reachable_target_stops_as_soon_as_it_is_met(data):
    result = _run(data, target_value=0.30)
    assert result["target_reached"] is True
    assert result["stop_reason"] == "target_reached"
    assert result["rounds_run"] < result["rounds_planned"], "it must not keep going"


def test_no_improvement_stops_the_loop(data):
    result = _run(data, target_value=None, patience=1, max_rounds=6, min_gain=0.5)
    assert result["stop_reason"] == "no_improvement"


def test_the_round_cap_is_respected(data):
    result = _run(data, target_value=0.999, max_rounds=2)
    assert result["rounds_planned"] == 2
    assert result["rounds_run"] <= 2


def test_a_spent_time_budget_stops_the_loop(data):
    result = _run(data, target_value=0.999, max_rounds=8, time_budget_seconds=0.0)
    assert result["stop_reason"] == "time_budget"


def test_cancellation_is_honoured(data):
    result = _run(data, target_value=0.999, should_cancel=lambda: True)
    assert result["stop_reason"] == "cancelled"


# ── The property that makes the number trustworthy ───────────────────────────

def test_the_loop_never_sees_the_test_split(data):
    """Scored on validation only; the test split is read once, afterwards.

    Iterating against test would reach any target and the reported figure would
    measure the search rather than the model.
    """
    X_train, y_train, X_val, y_val = data
    seen = []

    class Tattling(pd.DataFrame):
        """A frame that records every read of its values."""

        @property
        def _constructor(self):
            return Tattling

        @property
        def values(self):
            seen.append("read")
            return super().values

    result = autotune.run(
        model_type="lgb", champion_params={"n_estimators": 60},
        X_train=X_train, y_train=y_train, w_train=None,
        X_val=Tattling(X_val), y_val=y_val,
        target_value=None, max_rounds=2, time_budget_seconds=60,
    )
    assert result["scored_on"] == "validation"
    assert seen, "validation must actually be scored"
    assert "honest estimate" in result["note"]


def test_every_round_is_recorded_with_its_score(data):
    result = _run(data, target_value=None, max_rounds=3, patience=5)
    scored = [h for h in result["history"] if "score" in h]
    assert len(scored) == 3
    assert all(h["metric"] == "f1" for h in scored)
    assert all("params" in h for h in scored)


def test_a_failing_round_does_not_sink_the_loop(data, monkeypatch):
    from backend.services import trainer

    real = trainer.make_estimator
    calls = {"n": 0}

    def flaky(model_type, params, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated fit failure")
        return real(model_type, params, **kwargs)

    monkeypatch.setattr(trainer, "make_estimator", flaky)
    result = _run(data, target_value=None, max_rounds=3, patience=5)
    assert any("error" in h for h in result["history"])
    assert result["best_params"], "the loop must still return its best usable model"


# ── The result has to survive the trip to the comparison screen ──────────────

def test_the_escalation_result_reaches_the_comparison_payload(tmp_path):
    """It was dropped once: build_comparison copies a fixed set of keys.

    The loop's outcome is not decoration — whether the target was reached, and
    that the search never read the test split, are needed to judge the metrics
    beside it. A silent drop between layers is invisible without this.
    """
    import json

    import numpy as np

    from backend.services.comparison import build_comparison

    run_dir = tmp_path / "runs" / "RUN_T"
    run_dir.mkdir(parents=True)
    snapshot_dir = tmp_path / "snaps"
    snapshot_dir.mkdir()

    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    np.save(run_dir / "y_test.npy", y)
    np.save(run_dir / "test_indices.npy", np.arange(300))
    np.save(run_dir / "proba_test_champion.npy", rng.random(300))
    np.save(run_dir / "proba_test_lgb_escalated.npy", rng.random(300))

    autotune_block = {
        "stop_reason": "no_improvement", "target_reached": False,
        "best_score": 0.71, "best_round": 2, "rounds_run": 3, "rounds_planned": 5,
        "target_metric": "f1", "target_value": 0.95, "scored_on": "validation",
        "elapsed_seconds": 12.3, "best_params": {"n_estimators": 300},
        "note": "honest estimate",
    }
    (run_dir / "run_results.json").write_text(json.dumps({
        "run_id": "RUN_T", "snapshot_id": "SNAP_T",
        "champion": {"model_id": "champ", "threshold": 0.5},
        "challengers": {"lgb_escalated": {
            "candidate_id": "lgb_escalated", "model_type": "lgb",
            "selected_threshold": 0.5, "autotune": autotune_block,
        }},
    }))

    frame = pd.DataFrame({
        "UpdatedDateTimeGMT": pd.date_range("2026-01-01", periods=300, freq="h"),
        "NonVoiceFlag": y,
    })
    frame.to_parquet(snapshot_dir / "SNAP_T.parquet", index=False)
    (snapshot_dir / "SNAP_T.manifest.json").write_text(json.dumps({
        "snapshot_id": "SNAP_T", "date_column": "UpdatedDateTimeGMT",
        "row_counts": {"final": 300}, "snapshot_sha256": "x",
    }))

    result = build_comparison(run_dir, snapshot_dir, "SNAP_T", {"primary_metric": "f1"})
    carried = result["candidates"]["lgb_escalated"]["autotune"]
    assert carried is not None, "the escalation result was dropped between layers"
    assert carried["target_reached"] is False
    assert carried["stop_reason"] == "no_improvement"
    assert carried["scored_on"] == "validation"
