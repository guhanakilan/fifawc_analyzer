"""Sample-weight strategy: configuration, preview and application.

Every weight component is multiplicative on top of a 1.0 base, and the product
is capped. Both properties are what stop a row from being counted many times
over when several components happen to fire together.

No weighting is applied unless the strategy was explicitly approved and
persisted — the defaults below are labelled PROPOSED and never auto-approved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Values proposed in the project brief. They are pre-filled in the UI, marked
# PROPOSED, and take effect only once a named approver saves the strategy.
PROPOSED_DEFAULTS = {
    "enabled": False,
    "cap": 5.0,
    "components": {
        "recency": {
            "enabled": False, "weight": 1.5, "recent_days": 90,
            "label": "Recent rows (within N days of the newest row)",
        },
        "human_correction": {
            "enabled": False, "weight": 3.0, "column": None, "true_values": ["1", "true", "yes", "y"],
            "label": "Human-corrected rows (flag column)",
        },
        "verified_error": {
            "enabled": False, "weight": 2.5, "column": None, "true_values": ["1", "true", "yes", "y"],
            "label": "Rows the previous champion got wrong (verified)",
        },
        "rare_subtask": {
            "enabled": False, "weight": 2.0, "max_share_pct": 1.0,
            "label": "Rare approved SubTasks (share below threshold)",
        },
        "class_balance": {
            "enabled": False, "mode": "balanced",
            "label": "Class balance (sklearn 'balanced' weighting)",
        },
    },
    "historical_base": 1.0,
    "normalise_mean_to_one": True,
}


class WeightError(ValueError):
    pass


def _truthy(series: pd.Series, true_values: list[str]) -> pd.Series:
    allowed = {str(v).strip().lower() for v in (true_values or [])}
    return series.fillna("").astype(str).str.strip().str.lower().isin(allowed)


def compute_weights(
    df: pd.DataFrame, strategy: dict, dates: pd.Series | None = None,
    y: pd.Series | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (weights, breakdown). Weights are 1.0 everywhere when disabled."""
    n = len(df)
    base = float(strategy.get("historical_base", 1.0) or 1.0)
    weights = np.full(n, base, dtype=float)
    breakdown: dict = {"base": base, "applied": [], "skipped": []}

    if not strategy.get("enabled"):
        breakdown["skipped"].append({"component": "all", "reason": "Weighting disabled"})
        return weights, _summarise(weights, breakdown, y, capped_rows=0)

    components = strategy.get("components", {})
    cap = float(strategy.get("cap", 5.0) or 5.0)
    if cap <= 0:
        raise WeightError("Weight cap must be greater than zero.")

    # ── Recency ──────────────────────────────────────────────────────────────
    rec = components.get("recency", {})
    if rec.get("enabled"):
        if dates is None or dates.notna().sum() == 0:
            breakdown["skipped"].append({"component": "recency", "reason": "No usable date column"})
        else:
            newest = dates.max()
            days = int(rec.get("recent_days", 90) or 90)
            mask = (newest - dates).dt.days.fillna(10**6) <= days
            weights = np.where(mask.values, weights * float(rec.get("weight", 1.5)), weights)
            breakdown["applied"].append({
                "component": "recency", "multiplier": float(rec.get("weight", 1.5)),
                "rows": int(mask.sum()), "detail": f"Rows within {days} days of {newest}",
            })

    # ── Flag-column components ───────────────────────────────────────────────
    for key in ("human_correction", "verified_error"):
        cfg = components.get(key, {})
        if not cfg.get("enabled"):
            continue
        column = cfg.get("column")
        if not column or column not in df.columns:
            breakdown["skipped"].append({
                "component": key,
                "reason": f"Column {column or '(unset)'} is not present in the dataset",
            })
            continue
        mask = _truthy(df[column], cfg.get("true_values", []))
        weights = np.where(mask.values, weights * float(cfg.get("weight", 1.0)), weights)
        breakdown["applied"].append({
            "component": key, "multiplier": float(cfg.get("weight", 1.0)),
            "rows": int(mask.sum()), "detail": f"Column '{column}' truthy",
        })

    # ── Rare approved SubTask ────────────────────────────────────────────────
    rare = components.get("rare_subtask", {})
    if rare.get("enabled"):
        from .labeling import find_column
        subtask_col = find_column(df, "SubTask")
        if subtask_col is None:
            breakdown["skipped"].append({"component": "rare_subtask", "reason": "No SubTask column"})
        else:
            share = df[subtask_col].astype(str).map(
                df[subtask_col].astype(str).value_counts(normalize=True)
            ) * 100.0
            threshold = float(rare.get("max_share_pct", 1.0) or 1.0)
            mask = share < threshold
            weights = np.where(mask.values, weights * float(rare.get("weight", 2.0)), weights)
            breakdown["applied"].append({
                "component": "rare_subtask", "multiplier": float(rare.get("weight", 2.0)),
                "rows": int(mask.sum()), "detail": f"SubTask share below {threshold}%",
            })

    # ── Class balance ────────────────────────────────────────────────────────
    bal = components.get("class_balance", {})
    if bal.get("enabled"):
        if y is None or y.nunique() < 2:
            breakdown["skipped"].append({"component": "class_balance", "reason": "Both classes required"})
        else:
            from sklearn.utils.class_weight import compute_sample_weight
            balance = compute_sample_weight("balanced", y.values)
            weights = weights * balance
            breakdown["applied"].append({
                "component": "class_balance", "multiplier": None, "rows": n,
                "detail": "sklearn compute_sample_weight('balanced')",
            })

    capped_rows = int((weights > cap).sum())
    weights = np.clip(weights, 0.0, cap)
    breakdown["cap"] = cap
    breakdown["capped_rows"] = capped_rows

    if strategy.get("normalise_mean_to_one", True):
        mean_w = float(weights.mean())
        if mean_w > 0:
            weights = weights / mean_w
            breakdown["normalised_mean_to_one"] = True

    return weights, _summarise(weights, breakdown, y, capped_rows)


def _summarise(weights: np.ndarray, breakdown: dict, y: pd.Series | None, capped_rows: int) -> dict:
    summary = dict(breakdown)
    summary["distribution"] = {
        "rows": int(len(weights)),
        "min": round(float(weights.min()), 4) if len(weights) else 0.0,
        "mean": round(float(weights.mean()), 4) if len(weights) else 0.0,
        "median": round(float(np.median(weights)), 4) if len(weights) else 0.0,
        "max": round(float(weights.max()), 4) if len(weights) else 0.0,
        "std": round(float(weights.std()), 4) if len(weights) else 0.0,
        "rows_above_1": int((weights > 1.0000001).sum()),
        "capped_rows": capped_rows,
    }
    if len(weights):
        edges = np.linspace(float(weights.min()), float(weights.max()) or 1.0, 11)
        counts, _ = np.histogram(weights, bins=edges)
        summary["histogram"] = [
            {"from": round(float(edges[i]), 3), "to": round(float(edges[i + 1]), 3), "rows": int(counts[i])}
            for i in range(len(counts))
        ]
    if y is not None and len(y) == len(weights):
        effective = {}
        for cls in sorted(y.unique()):
            mask = (y == cls).values
            effective[str(int(cls))] = {
                "rows": int(mask.sum()),
                "raw_share_pct": round(float(mask.sum()) / max(len(y), 1) * 100, 2),
                "weighted_share_pct": round(
                    float(weights[mask].sum()) / max(float(weights.sum()), 1e-9) * 100, 2
                ),
            }
        summary["effective_class_balance"] = effective
    return summary


def formula_text(strategy: dict) -> str:
    """A single, auditable line describing exactly what was applied."""
    if not strategy.get("enabled"):
        return "weight = 1.0 for every row (weighting disabled)"
    parts = [f"base {strategy.get('historical_base', 1.0)}"]
    for key, cfg in (strategy.get("components") or {}).items():
        if not cfg.get("enabled"):
            continue
        if key == "class_balance":
            parts.append("x sklearn balanced class weight")
        elif key == "recency":
            parts.append(f"x {cfg.get('weight')} if within {cfg.get('recent_days')} days of newest row")
        elif key == "rare_subtask":
            parts.append(f"x {cfg.get('weight')} if SubTask share < {cfg.get('max_share_pct')}%")
        else:
            parts.append(f"x {cfg.get('weight')} if '{cfg.get('column')}' is truthy")
    text = " ".join(parts) + f", capped at {strategy.get('cap', 5.0)}"
    if strategy.get("normalise_mean_to_one", True):
        text += ", then normalised so mean weight = 1.0"
    return text
