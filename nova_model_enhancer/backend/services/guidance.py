"""Concrete next actions, read from a finished run.

Not a score and not a verdict — a short list of things that would plausibly
improve the next run, each tied to the evidence that suggests it. Deterministic
and local, like the readiness rules.

The one thing this never does is recommend promoting a model. Promotion stays a
human decision behind the gate, and a tool that nudges toward it would be
quietly doing the thing the brief forbids.
"""

from __future__ import annotations

from typing import Any


def _tuning_at_bounds(best_params: dict | None, search_spaces: dict | None) -> list[str]:
    """Parameters whose tuned value landed on the edge of their search range.

    A parameter at its bound usually means the range was too narrow, not that
    the optimum happens to sit exactly there.
    """
    if not best_params or not search_spaces:
        return []
    stuck = []
    for name, space in search_spaces.items():
        if name not in best_params or not isinstance(space, (list, tuple)) or len(space) < 2:
            continue
        value = best_params[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        low, high = space[0], space[1]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            continue
        span = float(high) - float(low)
        if span <= 0:
            continue
        # Within 2% of either end.
        if abs(float(value) - float(low)) <= 0.02 * span:
            stuck.append(f"{name} (at its lower bound {low})")
        elif abs(float(value) - float(high)) <= 0.02 * span:
            stuck.append(f"{name} (at its upper bound {high})")
    return stuck


def suggest(comparison: dict, run: dict | None = None) -> list[dict]:
    """Ranked suggestions for the next run. Empty is a valid answer."""
    run = run or {}
    suggestions: list[dict] = []
    leader_id = comparison.get("leading_candidate")
    if not leader_id:
        return suggestions

    leader = (comparison.get("candidates") or {}).get(leader_id) or {}
    champion = comparison.get("champion") or {}
    champion_metrics = champion.get("test_metrics") or {}
    leader_metrics = leader.get("test_metrics") or {}

    def add(priority: str, title: str, why: str, action: str) -> None:
        suggestions.append({"priority": priority, "title": title, "why": why, "action": action})

    # 1. Is the improvement inside the noise?
    significance = leader.get("significance") or {}
    interval = leader.get("confidence_interval") or {}
    if significance.get("p_value") is not None and not significance.get("significant"):
        add(
            "high",
            "The difference from the champion is not statistically significant",
            significance.get("interpretation", ""),
            "Gather more labelled data, or widen the training window, before treating this "
            "as an improvement. On this evidence the two models are equivalent.",
        )
    if interval.get("low") is not None and champion_metrics.get(interval.get("metric")) is not None:
        champion_value = float(champion_metrics[interval["metric"]])
        if interval["low"] <= champion_value <= interval["high"]:
            add(
                "high",
                f"The champion's {interval['metric']} sits inside the challenger's confidence interval",
                f"The challenger scores {interval['point']} with a 95% interval of "
                f"[{interval['low']}, {interval['high']}], which contains the champion's "
                f"{round(champion_value, 4)}.",
                "The apparent gain may be sampling noise. More test rows would narrow this.",
            )

    # 2. Stability was never measured.
    if comparison.get("backtest_status") == "not_run":
        add(
            "medium",
            "Stability over time was not assessed",
            "The rolling backtest was not run, so there is no evidence the improvement holds "
            "across time windows rather than on this one split.",
            "Enable the backtest in Stage 5 and re-run before promoting.",
        )
    else:
        backtest = (run.get("backtest") or {}).get(leader.get("model_type")) or {}
        summary = (backtest.get("summary") or {}).get("f1") or {}
        spread = summary.get("std")
        improvement = None
        if leader_metrics.get("f1") is not None and champion_metrics.get("f1") is not None:
            improvement = float(leader_metrics["f1"]) - float(champion_metrics["f1"])
        if spread is not None and improvement is not None and improvement <= float(spread):
            add(
                "high",
                "The gain is smaller than the backtest's own variation",
                f"F1 improves by {improvement:.4f}, but the backtest's F1 varies by "
                f"{spread:.4f} across windows.",
                "Treat this as within noise. A larger or more recent dataset is more likely "
                "to help than further tuning.",
            )

    # 3. Better on the headline metric but more expensive in practice.
    leader_cost = (leader.get("cost") or {})
    champion_cost = (champion.get("cost") or {})
    if leader_cost.get("total_cost") is not None and champion_cost.get("total_cost") is not None:
        delta = float(leader_cost["total_cost"]) - float(champion_cost["total_cost"])
        if delta > 0:
            add(
                "high",
                "The challenger costs more than the champion despite its metrics",
                f"Weighted cost {leader_cost['total_cost']:.0f} against the champion's "
                f"{champion_cost['total_cost']:.0f} at a {leader_cost.get('cost_ratio')}:1 ratio — "
                f"{leader_cost.get('missed_voice')} missed Voice against "
                f"{champion_cost.get('missed_voice')}. A higher F1 can still mean more of the "
                "error that costs most.",
                "Lower the decision threshold to trade precision for recall on Voice, or "
                "enable class-balance weighting, then re-run and compare cost again.",
            )

    # 4. Recall shortfall with imbalance -> class weighting.
    if (
        leader_metrics.get("recall") is not None
        and leader_metrics.get("precision") is not None
        and leader_metrics["recall"] < leader_metrics["precision"] - 0.05
    ):
        add(
            "medium",
            "Recall is the weaker half of this model",
            f"Precision is {leader_metrics['precision']} against recall "
            f"{leader_metrics['recall']} — the model is missing positives rather than "
            "over-calling them.",
            "Enable class-balance weighting in Stage 4, or lower the decision threshold, "
            "depending on which error costs you more.",
        )

    # 5. Tuning ranges that pinned.
    record = ((run.get("challengers") or {}).get(leader_id)) or {}
    stuck = _tuning_at_bounds(record.get("best_params"), record.get("search_spaces"))
    if stuck:
        add(
            "medium",
            "Tuning stopped at the edge of its search range",
            "These parameters landed on their bounds: " + ", ".join(stuck) + ". That usually "
            "means the range was too narrow rather than that the optimum sits exactly there.",
            "Widen those ranges and re-run the tuning.",
        )

    # 6. Calibration.
    brier_leader = leader_metrics.get("brier_score")
    brier_champion = champion_metrics.get("brier_score")
    if brier_leader is not None and brier_champion is not None and brier_leader > brier_champion:
        add(
            "low",
            "Probabilities are less well calibrated than the champion's",
            f"Brier score {brier_leader} against the champion's {brier_champion} — lower is "
            "better. The ranking may still be good while the probabilities themselves drift.",
            "Enable probability calibration for this candidate if the score is used as a "
            "number rather than only as a ranking.",
        )

    # 7. A segment where the challenger is worse.
    losing = [
        s for s in ((leader.get("disagreement") or {}).get("by_segment") or [])
        if s.get("net", 0) < 0
    ]
    if losing:
        worst = min(losing, key=lambda s: s["net"])
        add(
            "medium",
            f"The challenger is worse on '{worst['segment']}'",
            f"On that segment the champion wins {worst['champion_wins']} of "
            f"{worst['disagreements']} disagreements against the challenger's "
            f"{worst['challenger_wins']}.",
            "Check whether that segment matters operationally before promoting; a model "
            "better overall can still be worse where it counts.",
        )

    order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: order.get(s["priority"], 3))
    return suggestions
