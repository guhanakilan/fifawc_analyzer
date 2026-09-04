"""Item 8 — significance, operating points, cost and disagreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services import evaluator, guidance


@pytest.fixture
def truth():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 600)
    return y


def _proba_for(y, skill, seed=1):
    """Probabilities with a tunable amount of real signal, and genuine overlap.

    The two class distributions must overlap: a helper that separates them
    cleanly produces a perfect classifier, whose confidence interval is
    correctly zero-width — which then looks like a bug in the interval code
    rather than in the fixture.
    """
    rng = np.random.default_rng(seed)
    centre = 0.5 + 0.45 * skill * (2 * np.asarray(y) - 1)
    return np.clip(rng.normal(centre, 0.30), 0.001, 0.999)


# ── McNemar ──────────────────────────────────────────────────────────────────

def test_identical_models_report_no_discordance(truth):
    p = _proba_for(truth, 0.8)
    result = evaluator.mcnemar(truth, p, p, 0.5, 0.5)
    assert result["discordant"] == 0
    assert result["significant"] is False
    assert "identical" in result["interpretation"]


def test_a_clearly_better_model_is_flagged_significant(truth):
    weak = _proba_for(truth, 0.15, seed=2)
    strong = _proba_for(truth, 0.95, seed=3)
    result = evaluator.mcnemar(truth, weak, strong, 0.5, 0.5)
    assert result["significant"] is True
    assert result["challenger_only_correct"] > result["champion_only_correct"]


def test_two_similar_models_are_not_called_significant(truth):
    a = _proba_for(truth, 0.6, seed=4)
    b = _proba_for(truth, 0.6, seed=5)
    result = evaluator.mcnemar(truth, a, b, 0.5, 0.5)
    assert result["significant"] is False
    assert "chance" in result["interpretation"]


# ── Bootstrap ────────────────────────────────────────────────────────────────

def test_confidence_interval_brackets_its_point_estimate(truth):
    p = _proba_for(truth, 0.7)
    interval = evaluator.bootstrap_interval(truth, p, 0.5, resamples=200)
    assert interval["low"] <= interval["point"] <= interval["high"]
    assert interval["width"] > 0


def test_more_rows_give_a_narrower_interval():
    """The whole point of the interval is that small samples are less certain."""
    rng = np.random.default_rng(11)
    small_y = rng.integers(0, 2, 120)
    large_y = rng.integers(0, 2, 4000)
    small = evaluator.bootstrap_interval(
        small_y, _proba_for(small_y, 0.7), 0.5, resamples=300, seed=3
    )
    large = evaluator.bootstrap_interval(
        large_y, _proba_for(large_y, 0.7), 0.5, resamples=300, seed=3
    )
    assert large["width"] < small["width"]


# ── Operating points ─────────────────────────────────────────────────────────

def test_operating_points_meet_the_targets_they_claim(truth):
    p = _proba_for(truth, 0.8)
    points = evaluator.operating_points(truth, p)
    assert points["available"] is True
    if points["precision_at_recall"]:
        assert points["precision_at_recall"]["recall"] >= 0.80 - 1e-9
    if points["recall_at_precision"]:
        assert points["recall_at_precision"]["precision"] >= 0.80 - 1e-9


def test_a_single_class_test_set_is_reported_not_crashed():
    y = np.zeros(50, dtype=int)
    result = evaluator.operating_points(y, np.linspace(0, 1, 50))
    assert result["available"] is False
    assert "single class" in result["reason"]


def test_top_decile_lift_exceeds_one_for_a_model_with_signal(truth):
    points = evaluator.operating_points(truth, _proba_for(truth, 0.9))
    assert points["top_decile"]["lift"] > 1.0


# ── Cost ─────────────────────────────────────────────────────────────────────

def test_cost_weights_a_missed_voice_more_heavily():
    # 0 = Voice, 1 = Non-Voice. One missed Voice, one wasted Voice.
    y = np.array([0, 1])
    proba = np.array([0.9, 0.1])  # predicts Non-Voice for the Voice row and vice versa
    result = evaluator.cost_weighted(y, proba, 0.5, cost_ratio=3.0)
    assert result["missed_voice"] == 1
    assert result["wasted_voice"] == 1
    assert result["total_cost"] == 4.0  # 1*3 + 1*1


def test_a_perfect_model_costs_nothing():
    y = np.array([0, 0, 1, 1])
    result = evaluator.cost_weighted(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert result["total_cost"] == 0.0


def test_cost_ratio_of_one_treats_both_errors_alike():
    y = np.array([0, 1])
    result = evaluator.cost_weighted(y, np.array([0.9, 0.1]), 0.5, cost_ratio=1.0)
    assert result["total_cost"] == 2.0


# ── Disagreement ─────────────────────────────────────────────────────────────

def test_agreement_is_reported_as_zero_rows(truth):
    p = _proba_for(truth, 0.8)
    result = evaluator.disagreement(truth, p, p, 0.5, 0.5)
    assert result["rows"] == 0


def test_segment_breakdown_finds_where_the_challenger_loses():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    champion = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])   # perfect
    challenger = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])  # wrong on first two
    segments = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    result = evaluator.disagreement(y, champion, challenger, 0.5, 0.5, segments)
    segment_a = next(s for s in result["by_segment"] if s["segment"] == "A")
    assert segment_a["net"] < 0, "a segment the challenger loses must show a negative net"


# ── Guidance ─────────────────────────────────────────────────────────────────

def test_guidance_flags_an_improvement_that_is_not_significant():
    comparison = {
        "leading_candidate": "c1",
        "candidates": {"c1": {
            "test_metrics": {"f1": 0.75},
            "significance": {"p_value": 0.44, "significant": False,
                             "interpretation": "within chance"},
            "confidence_interval": {"metric": "f1", "point": 0.75, "low": 0.70, "high": 0.80},
        }},
        "champion": {"test_metrics": {"f1": 0.74}},
        "backtest_status": "passed",
    }
    titles = [s["title"] for s in guidance.suggest(comparison, {})]
    assert any("not statistically significant" in t for t in titles)
    assert any("confidence interval" in t for t in titles)


def test_guidance_never_recommends_promotion():
    """Promotion is a human decision behind the gate; nudging toward it is not ours."""
    comparison = {
        "leading_candidate": "c1",
        "candidates": {"c1": {
            "test_metrics": {"f1": 0.95, "precision": 0.95, "recall": 0.95},
            "significance": {"p_value": 0.001, "significant": True, "interpretation": "real"},
            "confidence_interval": {"metric": "f1", "point": 0.95, "low": 0.93, "high": 0.97},
        }},
        "champion": {"test_metrics": {"f1": 0.60}},
        "backtest_status": "passed",
    }
    text = " ".join(
        f"{s['title']} {s['why']} {s['action']}" for s in guidance.suggest(comparison, {})
    ).lower()
    assert "promote" not in text and "promotion" not in text


def test_guidance_reports_a_missing_backtest():
    comparison = {
        "leading_candidate": "c1",
        "candidates": {"c1": {"test_metrics": {"f1": 0.8}, "significance": {},
                              "confidence_interval": {}}},
        "champion": {"test_metrics": {"f1": 0.7}},
        "backtest_status": "not_run",
    }
    assert any(
        "not assessed" in s["title"].lower() for s in guidance.suggest(comparison, {})
    )


def test_guidance_spots_a_tuning_range_that_pinned():
    from backend.services.guidance import _tuning_at_bounds

    stuck = _tuning_at_bounds(
        {"learning_rate": 0.3, "num_leaves": 60},
        {"learning_rate": [0.01, 0.3], "num_leaves": [20, 120]},
    )
    assert any("learning_rate" in s for s in stuck)
    assert not any("num_leaves" in s for s in stuck)


def test_guidance_flags_a_challenger_that_costs_more_despite_better_metrics():
    """The case the headline table hides: higher F1, more of the expensive error."""
    comparison = {
        "leading_candidate": "c1",
        "candidates": {"c1": {
            "test_metrics": {"f1": 0.78},
            "significance": {},
            "confidence_interval": {},
            "cost": {"total_cost": 660.0, "cost_ratio": 3.0, "missed_voice": 187},
        }},
        "champion": {
            "test_metrics": {"f1": 0.75},
            "cost": {"total_cost": 618.0, "cost_ratio": 3.0, "missed_voice": 161},
        },
        "backtest_status": "passed",
    }
    suggestions = guidance.suggest(comparison, {})
    match = [s for s in suggestions if "costs more" in s["title"]]
    assert match, [s["title"] for s in suggestions]
    assert match[0]["priority"] == "high"
