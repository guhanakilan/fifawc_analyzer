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

# Same sweep grid and composite weights as `routers/evaluation.py`.
SWEEP_START, SWEEP_STOP, SWEEP_STEP = 0.10, 0.92, 0.05
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
    if gate.get("require_backtest_pass") and backtest_ok is False:
        blockers.append("Rolling backtest did not complete successfully.")

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
