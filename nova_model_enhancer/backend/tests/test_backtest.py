"""Item 7 — the multi-model backtest and the opt-in default.

Sharing preprocessing across model types must not change any model's result:
the whole point is that the transform depends on the split, not the estimator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.services import trainer
from backend.services.nova_transform import TARGET


@pytest.fixture
def frame():
    n = 900
    rng = pd.Series(range(n))
    return pd.DataFrame({
        "AgeDays": (rng * 7 % 300).astype(float),
        "Amount": (rng * 13 % 5000).astype(float),
        "Payer": ["Aetna", "BCBS", "United"] * (n // 3),
        TARGET: [(i // 7) % 2 for i in range(n)],
    })


@pytest.fixture
def dates():
    return pd.Series(pd.date_range("2025-01-01", periods=900, freq="8h"))


FEATURES = {"numeric": ["AgeDays", "Amount"], "categorical": ["Payer"], "target": TARGET}


def test_sharing_transforms_does_not_change_a_models_result(frame, dates):
    """The refactor must be a pure speed change, not a behaviour change."""
    single = trainer.rolling_backtest(frame, FEATURES, dates, "lr", n_windows=4, seed=42)
    many = trainer.rolling_backtest_many(
        frame, FEATURES, dates, ["lr", "rf"], n_windows=4, seed=42
    )["lr"]
    assert many["summary"] == single["summary"]
    assert many["results"] == single["results"]


def test_every_requested_model_type_is_returned(frame, dates):
    out = trainer.rolling_backtest_many(frame, FEATURES, dates, ["lr", "rf"], n_windows=3, seed=1)
    assert set(out) == {"lr", "rf"}
    assert all(out[m]["completed"] for m in out)


def test_one_failing_model_does_not_lose_the_others(frame, dates, monkeypatch):
    """The backtest is diagnostic; a single bad estimator must not sink it."""
    real = trainer.make_estimator

    def flaky(model_type, *args, **kwargs):
        if model_type == "rf":
            raise RuntimeError("simulated estimator failure")
        return real(model_type, *args, **kwargs)

    monkeypatch.setattr(trainer, "make_estimator", flaky)
    out = trainer.rolling_backtest_many(frame, FEATURES, dates, ["lr", "rf"], n_windows=3, seed=1)
    assert out["lr"]["completed"] is True
    assert out["rf"]["completed"] is False
    assert "simulated estimator failure" in out["rf"]["error"]


def test_an_unsupported_model_type_is_rejected_up_front(frame, dates):
    with pytest.raises(ValueError, match="Unsupported"):
        trainer.rolling_backtest_many(frame, FEATURES, dates, ["nonsense"], n_windows=3)


def test_too_few_windows_is_rejected(frame, dates):
    with pytest.raises(ValueError, match="at least 3"):
        trainer.rolling_backtest_many(frame, FEATURES, dates, ["lr"], n_windows=2)


def test_window_cap_is_four(dates):
    """Eight windows spent most of the backtest for resolution nobody read."""
    assert trainer.MAX_BACKTEST_WINDOWS == 4
    long_span = pd.Series(pd.date_range("2020-01-01", periods=100, freq="30D"))
    assert trainer.auto_backtest_windows(long_span) == 4


def test_backtest_is_off_by_default():
    from backend.schemas import TrainingRequest

    assert TrainingRequest().run_backtest is False


def test_candidates_are_sequential_by_default():
    """Measured slower when parallel on 4 cores, so the default stays 1."""
    from backend.schemas import TrainingRequest

    assert TrainingRequest().max_parallel_candidates == 1


# ── The gate must not read a skipped backtest as a pass ──────────────────────

def test_a_skipped_backtest_is_reported_not_silently_passed():
    """Before this, backtest_ok=None fell through every check and said nothing.

    With the backtest opt-in and off by default, that would have let every run
    look as though stability had been confirmed when it was never measured.
    """
    from backend.services.evaluator import PROPOSED_GATE, evaluate_gate

    gate = {**PROPOSED_GATE, "require_backtest_pass": True}
    champion = {"f1": 0.70, "precision": 0.70, "recall": 0.70, "auc": 0.78}
    challenger = {"f1": 0.75, "precision": 0.74, "recall": 0.76, "auc": 0.81}

    result = evaluate_gate(
        champion, challenger, gate, backtest_ok=None, backtest_status="not_run"
    )
    stability = [r for r in result["rules"] if "backtest" in r["rule"]]
    assert stability, "the gate said nothing at all about stability"
    assert stability[0]["passed"] is None
    assert "not assessed" in stability[0]["detail"].lower()


def test_a_failed_backtest_still_blocks():
    from backend.services.evaluator import PROPOSED_GATE, evaluate_gate

    gate = {**PROPOSED_GATE, "require_backtest_pass": True}
    result = evaluate_gate(
        {"f1": 0.70}, {"f1": 0.75}, gate, backtest_ok=False, backtest_status="failed"
    )
    assert any("backtest" in b.lower() for b in result["blockers"])
