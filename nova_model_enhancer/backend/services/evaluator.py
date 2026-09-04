"""Metrics, threshold optimisation, segment analysis and promotion gates.

Nothing here picks a "best" model on its own. The primary metric, the protected
metrics and every tolerance are supplied by the caller from a persisted, approved
gate configuration.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Threshold sweep grid. The reference swept from 0.10; this application floors
# it at 0.50 by decision — below a coin flip the model is calling Voice on rows
# it believes are Non-Voice, which is not an operating point anyone wants to
# land on by accident. 0.50 to 0.90 inclusive, in steps of 0.05.
MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_STEP = 0.50, 0.90, 0.05
SWEEP_START, SWEEP_STOP, SWEEP_STEP = MIN_THRESHOLD, MAX_THRESHOLD + 0.001, THRESHOLD_STEP


def threshold_grid() -> list[float]:
    """Every selectable threshold, as the UI and the backend both see it."""
    import numpy as _np

    steps = int(round((MAX_THRESHOLD - MIN_THRESHOLD) / THRESHOLD_STEP)) + 1
    return [round(float(v), 2) for v in _np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, steps)]


def clamp_threshold(value: float) -> float:
    """Snap a challenger threshold onto the grid, refusing anything outside it."""
    numeric = float(value)
    if numeric < MIN_THRESHOLD or numeric > MAX_THRESHOLD:
        raise ValueError(
            f"Threshold {numeric:g} is outside the allowed range "
            f"{MIN_THRESHOLD:g}–{MAX_THRESHOLD:g}."
        )
    grid = threshold_grid()
    return min(grid, key=lambda candidate: abs(candidate - numeric))
W_RECALL, W_F1, W_PRECISION, W_ACCURACY, W_SPECIFICITY = 0.40, 0.30, 0.15, 0.10, 0.05

PRIMARY_METRIC_CHOICES = ("f1", "recall", "precision", "auc", "pr_auc", "balanced_accuracy", "weighted_composite")

# Proposed only. Stage 6 requires these to be reviewed and saved by a named
# approver before any recommendation is issued.
# Every field GateConfig accepts must appear here. A field missing from the
# proposal renders as an empty control while the backend quietly applies its own
# default, so the user is shown one rule and given another.
# `test_proposed_gate_covers_every_gate_field` keeps the two in step.
PROPOSED_GATE = {
    "primary_metric": "f1",
    "min_primary_improvement_pct": 1.0,
    "protected_metrics": [
        {"metric": "recall", "max_regression_pct": 0.5},
    ],
    "max_historical_primary_regression_pct": 1.0,
    "require_backtest_pass": True,
    "require_package_validation": True,
    "segment_column": "SubTask",
    "min_segment_rows": 100,
    "approved": False,
}


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr_auc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except ValueError:
        brier = float("nan")

    return {
        "threshold": round(float(threshold), 4),
        "rows": int(total),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round((recall + specificity) / 2.0, 4),
        "auc": None if np.isnan(auc) else round(auc, 4),
        "pr_auc": None if np.isnan(pr_auc) else round(pr_auc, 4),
        "brier_score": None if np.isnan(brier) else round(brier, 4),
        "weighted_composite": round(
            W_RECALL * recall + W_F1 * f1 + W_PRECISION * precision
            + W_ACCURACY * accuracy + W_SPECIFICITY * specificity, 4
        ),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "predicted_non_voice": int((y_pred == 1).sum()),
        "predicted_voice": int((y_pred == 0).sum()),
        "actual_non_voice": int((y_true == 1).sum()),
        "actual_voice": int((y_true == 0).sum()),
    }


def threshold_sweep(y_true, y_proba) -> dict:
    """Full sweep plus the argmax for every candidate criterion.

    The sweep is computed on the validation slice; the chosen threshold is then
    reported on the test slice, so the test set never tunes the threshold.
    """
    rows = []
    for t in np.arange(SWEEP_START, SWEEP_STOP, SWEEP_STEP):
        m = metrics_at_threshold(y_true, y_proba, round(float(t), 2))
        rows.append({
            "t": m["threshold"], "f1": m["f1"], "precision": m["precision"],
            "recall": m["recall"], "specificity": m["specificity"],
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "weighted_composite": m["weighted_composite"],
        })
    best: dict[str, Any] = {}
    for criterion in ("f1", "recall", "precision", "balanced_accuracy", "weighted_composite"):
        top = max(rows, key=lambda r: r[criterion])
        best[criterion] = {"threshold": top["t"], "score": top[criterion]}
    return {"sweep": rows, "best": best}


def calibration_curve_points(y_true, y_proba, bins: int = 10) -> list[dict]:
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=bins, strategy="quantile")
        return [
            {"predicted": round(float(p), 4), "actual": round(float(t), 4)}
            for p, t in zip(prob_pred, prob_true)
        ]
    except Exception:
        return []


def measure_latency(predict_fn, X, repeats: int = 3) -> dict:
    """Inference latency on the benchmark rows, best of `repeats`."""
    rows = max(len(X), 1)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(X)
        best = min(best, time.perf_counter() - start)
    return {
        "total_seconds": round(best, 4),
        "rows": int(rows),
        "microseconds_per_row": round(best / rows * 1_000_000, 2),
    }


def period_breakdown(y_true, y_proba, dates: pd.Series, threshold: float, min_rows: int = 100) -> list[dict]:
    """Metrics by calendar month. Periods below `min_rows` are reported as skipped."""
    if dates is None or dates.notna().sum() == 0:
        return []
    frame = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "p": np.asarray(y_proba, dtype=float),
        "period": pd.to_datetime(dates, errors="coerce").dt.to_period("M"),
    }).dropna(subset=["period"])

    out = []
    for period, group in frame.groupby("period", observed=True):
        if len(group) < min_rows or group["y"].nunique() < 2:
            out.append({
                "period": str(period), "rows": int(len(group)),
                "skipped": "sample too small or single-class",
            })
            continue
        m = metrics_at_threshold(group["y"], group["p"], threshold)
        out.append({
            "period": str(period), "rows": m["rows"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"], "auc": m["auc"],
        })
    return sorted(out, key=lambda r: r["period"])


def segment_breakdown(
    y_true, y_proba, segments: pd.Series, threshold: float,
    min_rows: int = 100, max_segments: int = 15,
) -> list[dict]:
    """Metrics by an approved segment column (typically SubTask)."""
    if segments is None:
        return []
    frame = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "p": np.asarray(y_proba, dtype=float),
        "segment": segments.astype(str).values,
    })
    counts = frame["segment"].value_counts()
    out = []
    for name in counts.index[:max_segments]:
        group = frame[frame["segment"] == name]
        if len(group) < min_rows or group["y"].nunique() < 2:
            out.append({
                "segment": name, "rows": int(len(group)),
                "skipped": "sample too small or single-class",
            })
            continue
        m = metrics_at_threshold(group["y"], group["p"], threshold)
        out.append({
            "segment": name, "rows": m["rows"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
        })
    return out


# ── Promotion gate ───────────────────────────────────────────────────────────

def _pct_delta(challenger: float | None, champion: float | None) -> float | None:
    if challenger is None or champion is None:
        return None
    if champion == 0:
        return None if challenger == 0 else float("inf")
    return (challenger - champion) / abs(champion) * 100.0


def evaluate_gate(champion_metrics: dict, challenger_metrics: dict, gate: dict,
                  historical: dict | None = None, backtest_ok: bool | None = None,
                  backtest_status: str = "unknown",
                  package_ok: bool | None = None,
                  data_quality_blockers: list[str] | None = None) -> dict:
    """Apply the approved gate. Returns RECOMMENDED / NOT_RECOMMENDED / BLOCKED.

    BLOCKED means something outside metric comparison failed (unapproved gate,
    failed package validation, a data-quality blocker). It is never downgraded
    to NOT_RECOMMENDED, because the comparison itself cannot be trusted.
    """
    rules: list[dict] = []
    blockers: list[str] = []

    if not gate.get("approved"):
        blockers.append("Promotion gate has not been approved. Review and save it in Stage 6.")

    for blocker in (data_quality_blockers or []):
        blockers.append(blocker)

    if gate.get("require_package_validation") and package_ok is False:
        blockers.append("Export package validation has not passed.")
    if gate.get("require_backtest_pass"):
        if backtest_ok is False:
            blockers.append("Rolling backtest did not complete successfully.")
        elif backtest_status == "not_run":
            # The backtest is opt-in. Its absence is not a pass: without it there
            # is no evidence the improvement holds over time, and saying nothing
            # would let a one-off result read as a stable one.
            rules.append({
                "rule": "rolling backtest confirms stability over time",
                "champion": None,
                "challenger": None,
                "passed": None,
                "detail": (
                    "Not assessed — the backtest was not run for this run. Enable it in "
                    "Stage 5 to check the improvement holds across time windows."
                ),
            })

    primary = gate.get("primary_metric", "f1")
    min_improvement = float(gate.get("min_primary_improvement_pct", 0.0))
    delta = _pct_delta(challenger_metrics.get(primary), champion_metrics.get(primary))
    rules.append({
        "rule": f"primary metric '{primary}' improves by at least {min_improvement}%",
        "champion": champion_metrics.get(primary),
        "challenger": challenger_metrics.get(primary),
        "delta_pct": None if delta is None else round(delta, 3),
        "passed": bool(delta is not None and delta >= min_improvement),
    })

    for protected in gate.get("protected_metrics", []):
        metric = protected.get("metric")
        tolerance = float(protected.get("max_regression_pct", 0.0))
        pdelta = _pct_delta(challenger_metrics.get(metric), champion_metrics.get(metric))
        rules.append({
            "rule": f"protected metric '{metric}' declines by no more than {tolerance}%",
            "champion": champion_metrics.get(metric),
            "challenger": challenger_metrics.get(metric),
            "delta_pct": None if pdelta is None else round(pdelta, 3),
            "passed": bool(pdelta is not None and pdelta >= -tolerance),
        })

    hist_tolerance = gate.get("max_historical_primary_regression_pct")
    if hist_tolerance is not None and historical:
        hdelta = _pct_delta(
            (historical.get("challenger") or {}).get(primary),
            (historical.get("champion") or {}).get(primary),
        )
        rules.append({
            "rule": f"historical '{primary}' declines by no more than {hist_tolerance}%",
            "champion": (historical.get("champion") or {}).get(primary),
            "challenger": (historical.get("challenger") or {}).get(primary),
            "delta_pct": None if hdelta is None else round(hdelta, 3),
            "passed": bool(hdelta is not None and hdelta >= -float(hist_tolerance)),
        })

    if blockers:
        status = "BLOCKED"
    elif all(r["passed"] for r in rules):
        status = "RECOMMENDED"
    else:
        status = "NOT_RECOMMENDED"

    return {
        "status": status,
        "primary_metric": primary,
        "rules": rules,
        "blockers": blockers,
        "gate": gate,
    }


# ── Item 8: significance, operating points, cost and disagreement ────────────
#
# Everything below is computed from the saved test-set probabilities and labels,
# so it costs one pass over an array rather than a refit.

# A missed Voice (predicted Non-Voice, actually needed a call) stalls the claim;
# a wasted Voice burns an agent's time. Approved as 3:1. Editable per gate.
DEFAULT_COST_RATIO = 3.0


def mcnemar(y_true, proba_a, proba_b, threshold_a: float, threshold_b: float) -> dict:
    """Is the difference between two models real, or could it be noise?

    Compares the two models on the *same* rows and counts only where they
    disagree, which is the question a difference in headline F1 cannot answer:
    +1% on 778 rows may be a handful of flips.
    """
    y = np.asarray(y_true).astype(int)
    pred_a = (np.asarray(proba_a) >= threshold_a).astype(int)
    pred_b = (np.asarray(proba_b) >= threshold_b).astype(int)

    a_right = pred_a == y
    b_right = pred_b == y
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "champion_only_correct": 0, "challenger_only_correct": 0,
            "discordant": 0, "statistic": None, "p_value": None,
            "significant": False,
            "interpretation": "The two models make identical predictions on every test row.",
        }

    # Exact binomial test; the chi-square approximation is unreliable when the
    # discordant count is small, which is exactly when this matters most.
    from scipy.stats import binomtest

    result = binomtest(min(only_a, only_b), n=discordant, p=0.5)
    p_value = float(result.pvalue)
    significant = p_value < 0.05
    better = "challenger" if only_b > only_a else "champion"

    if significant:
        interpretation = (
            f"The {better} is better on significantly more rows than the other "
            f"(p = {p_value:.4f}). This difference is unlikely to be chance."
        )
    else:
        interpretation = (
            f"The models disagree on {discordant} rows, split {only_a}/{only_b} — "
            f"p = {p_value:.3f}, so this difference is within what chance would produce. "
            "Treat the two as equivalent on this evidence."
        )

    return {
        "champion_only_correct": only_a,
        "challenger_only_correct": only_b,
        "discordant": discordant,
        "statistic": float(min(only_a, only_b)),
        "p_value": round(p_value, 6),
        "significant": significant,
        "interpretation": interpretation,
    }


def bootstrap_interval(
    y_true, proba, threshold: float, metric: str = "f1",
    resamples: int = 400, seed: int = 42, confidence: float = 0.95,
) -> dict:
    """Confidence interval for one metric, by resampling the test rows.

    A point estimate on a few hundred rows carries more uncertainty than its
    four decimal places suggest; this says how much.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    if n == 0:
        return {"metric": metric, "point": None, "low": None, "high": None, "resamples": 0}

    scorers = {
        "f1": lambda yt, pr, pb: f1_score(yt, pr, zero_division=0),
        "precision": lambda yt, pr, pb: precision_score(yt, pr, zero_division=0),
        "recall": lambda yt, pr, pb: recall_score(yt, pr, zero_division=0),
        "auc": lambda yt, pr, pb: roc_auc_score(yt, pb) if len(np.unique(yt)) > 1 else np.nan,
    }
    score = scorers.get(metric, scorers["f1"])

    rng = np.random.default_rng(seed)
    point = float(score(y, (p >= threshold).astype(int), p))
    samples = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if len(np.unique(ys)) < 2:
            continue
        value = score(ys, (ps >= threshold).astype(int), ps)
        if not np.isnan(value):
            samples.append(float(value))

    if not samples:
        return {"metric": metric, "point": round(point, 4), "low": None, "high": None,
                "resamples": 0}

    tail = (1.0 - confidence) / 2.0
    low = float(np.quantile(samples, tail))
    high = float(np.quantile(samples, 1.0 - tail))
    return {
        "metric": metric,
        "point": round(point, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        "width": round(high - low, 4),
        "resamples": len(samples),
        "confidence": confidence,
    }


def operating_points(y_true, proba, targets: dict | None = None) -> dict:
    """Where the model sits when framed the way it is actually used.

    "If I must catch 80% of Voice, what precision do I keep?" is the operational
    question; a single F1 hides it.
    """
    from sklearn.metrics import precision_recall_curve

    limits = {"recall_target": 0.80, "precision_target": 0.80, **(targets or {})}
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if len(np.unique(y)) < 2:
        return {"available": False, "reason": "The test set contains a single class."}

    precision, recall, thresholds = precision_recall_curve(y, p)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    def at_least_recall(target):
        mask = recall >= target
        if not mask.any():
            return None
        best = int(np.argmax(np.where(mask, precision, -1)))
        return {
            "threshold": round(float(thresholds[best]), 4),
            "precision": round(float(precision[best]), 4),
            "recall": round(float(recall[best]), 4),
        }

    def at_least_precision(target):
        mask = precision >= target
        if not mask.any():
            return None
        best = int(np.argmax(np.where(mask, recall, -1)))
        return {
            "threshold": round(float(thresholds[best]), 4),
            "precision": round(float(precision[best]), 4),
            "recall": round(float(recall[best]), 4),
        }

    # Lift in the top decile: how much richer the highest-scoring 10% is than
    # the base rate. Prioritisation value, not accuracy.
    order = np.argsort(-p)
    decile = max(1, len(y) // 10)
    base_rate = float(y.mean())
    top_rate = float(y[order[:decile]].mean())
    lift = round(top_rate / base_rate, 3) if base_rate > 0 else None

    return {
        "available": True,
        "precision_at_recall": at_least_recall(limits["recall_target"]),
        "recall_at_precision": at_least_precision(limits["precision_target"]),
        "recall_target": limits["recall_target"],
        "precision_target": limits["precision_target"],
        "top_decile": {
            "rows": decile,
            "positive_rate": round(top_rate, 4),
            "base_rate": round(base_rate, 4),
            "lift": lift,
        },
    }


def cost_weighted(y_true, proba, threshold: float, cost_ratio: float = DEFAULT_COST_RATIO) -> dict:
    """Total cost when the two error types are not equally expensive.

    Internally 0 = Voice and 1 = Non-Voice. A *missed Voice* is predicting
    Non-Voice for a row that actually needed a call — the account goes unworked
    and the claim stalls. A *wasted Voice* is the reverse: an agent calls an
    account that did not need it.
    """
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)

    missed_voice = int(np.sum((y == 0) & (pred == 1)))
    wasted_voice = int(np.sum((y == 1) & (pred == 0)))
    total = float(missed_voice * cost_ratio + wasted_voice)
    rows = len(y)

    return {
        "cost_ratio": cost_ratio,
        "missed_voice": missed_voice,
        "wasted_voice": wasted_voice,
        "total_cost": round(total, 2),
        "cost_per_1000_rows": round(1000.0 * total / rows, 2) if rows else None,
        "note": (
            f"A missed Voice is counted as {cost_ratio:g}x a wasted call. "
            "Lower is better."
        ),
    }


def disagreement(
    y_true, champion_proba, challenger_proba, champion_threshold: float,
    challenger_threshold: float, segments: pd.Series | None = None, top_n: int = 8,
) -> dict:
    """Where the two models differ, in which direction, and on what kind of row.

    Turns "2% better overall" into something a person can check against their
    own knowledge of the work.
    """
    y = np.asarray(y_true).astype(int)
    champ = (np.asarray(champion_proba) >= champion_threshold).astype(int)
    chall = (np.asarray(challenger_proba) >= challenger_threshold).astype(int)

    differ = champ != chall
    rows = int(differ.sum())
    if rows == 0:
        return {"rows": 0, "pct_of_test": 0.0, "note": "The two models agree on every test row."}

    challenger_right = int(np.sum(differ & (chall == y)))
    champion_right = int(np.sum(differ & (champ == y)))

    breakdown = None
    if segments is not None and len(segments) == len(y):
        frame = pd.DataFrame({
            "segment": np.asarray(segments).astype(str),
            "differ": differ,
            "challenger_right": differ & (chall == y),
            "champion_right": differ & (champ == y),
        })
        grouped = frame.groupby("segment").agg(
            rows=("differ", "size"),
            disagreements=("differ", "sum"),
            challenger_wins=("challenger_right", "sum"),
            champion_wins=("champion_right", "sum"),
        ).reset_index()
        grouped = grouped[grouped["disagreements"] > 0]
        grouped["net"] = grouped["challenger_wins"] - grouped["champion_wins"]
        grouped = grouped.sort_values("disagreements", ascending=False).head(top_n)
        breakdown = [
            {
                "segment": r["segment"],
                "rows": int(r["rows"]),
                "disagreements": int(r["disagreements"]),
                "challenger_wins": int(r["challenger_wins"]),
                "champion_wins": int(r["champion_wins"]),
                "net": int(r["net"]),
            }
            for _, r in grouped.iterrows()
        ]

    return {
        "rows": rows,
        "pct_of_test": round(100.0 * rows / len(y), 2),
        "challenger_correct": challenger_right,
        "champion_correct": champion_right,
        "net_to_challenger": challenger_right - champion_right,
        "by_segment": breakdown,
        "note": (
            f"The models differ on {rows} of {len(y)} test rows. Of those, the challenger is "
            f"right on {challenger_right} and the champion on {champion_right}."
        ),
    }
