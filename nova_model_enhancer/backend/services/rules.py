"""The readiness rules engine: recommendations and intervention checks.

Every output here is derived from the uploaded data and the champion package by
deterministic rules, and carries the evidence that produced it. Nothing is
applied automatically — a recommendation is a starting point a person accepts
or overrides, and the decision endpoints still require an explicit choice.

Optional local models (see `local_models.py`) can sharpen the SubTask
suggestions. They are never required: when absent, the rules alone answer, and
the report says which layer produced each suggestion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import labeling
from .data_profiler import is_model_output
from .nova_transform import norm_col
from .rules_config import merge_rules

DATE_NAME_TOKENS = ("updateddatetime", "createddate", "postdate", "dos", "date", "time")


def _confidence(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _suggestion(
    field: str, value: Any, rule: str, why: str, confidence: float, evidence: dict | None = None
) -> dict:
    return {
        "field": field,
        "value": value,
        "rule": rule,
        "why": why,
        "confidence": _confidence(confidence),
        "confidence_score": round(float(confidence), 3),
        "evidence": evidence or {},
        "source": "rules",
    }


def derive_labelled_frame(df: pd.DataFrame, configs, subtask_review: dict) -> tuple:
    """The data with the champion's NonVoiceFlag, or (None, reason).

    Uses the canonical `apply_subtask_mapping`, which is the only thing that
    handles the champion's own flag vocabulary correctly: the mappings store
    "Voice"/"Non-Voice"/"Keyword"/"Ignore" as strings, "Keyword" resolves
    against ARComments, and "Ignore" drops rows. Mapping those names straight
    to numbers yields no labels at all and silently disables every check that
    depends on them.

    The whole frame is returned rather than a bare label column because dropping
    Ignore rows re-indexes it — a label Series pulled out on its own no longer
    lines up with the dates in the original frame.
    """
    if subtask_review.get("unmapped") or not subtask_review.get("subtask_column"):
        return None, "SubTasks are unmapped, so labels cannot be derived yet"
    try:
        labelled, _stats = labeling.apply_subtask_mapping(
            df, configs.subtask_mappings, configs.subtask_keywords,
            allow_unmapped_default=False,
        )
    except ValueError as exc:
        return None, str(exc)
    if "NonVoiceFlag" not in labelled:
        return None, "the mapping produced no NonVoiceFlag column"
    return labelled, ""


# ── Individual recommendations ───────────────────────────────────────────────

def recommend_date_column(df: pd.DataFrame, candidates: list[str], config: dict) -> dict | None:
    """Rank date candidates by parse quality, span and agreement with row order."""
    if not config.get("enabled", True) or not candidates:
        return None

    scored = []
    for column in candidates:
        parsed = pd.to_datetime(df[column], errors="coerce")
        total = len(parsed)
        if not total:
            continue
        unparseable = int(parsed.isna().sum())
        unparseable_pct = 100.0 * unparseable / total
        valid = parsed.dropna()
        if valid.empty:
            continue
        span_days = float((valid.max() - valid.min()).days)

        # A column that rises with row order is the one the export was sorted by,
        # which is the clock the pipeline actually ran on.
        order = valid.reset_index(drop=True)
        monotonic = float(order.is_monotonic_increasing)

        recognised = float(any(t in norm_col(column) for t in DATE_NAME_TOKENS)) \
            if config.get("prefer_recognised_names", True) else 0.0

        score = (
            recognised * 0.30
            + max(0.0, 1.0 - unparseable_pct / 100.0) * 0.35
            + min(1.0, span_days / 365.0) * 0.20
            + monotonic * 0.15
        )
        scored.append({
            "column": column, "score": round(score, 4),
            "unparseable_pct": round(unparseable_pct, 3),
            "span_days": span_days, "monotonic": bool(monotonic),
            "recognised_name": bool(recognised),
        })

    if not scored:
        return None
    scored.sort(key=lambda s: -s["score"])
    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    # Two near-identical candidates is exactly when a wrong pick goes unnoticed.
    close = bool(runner_up and (best["score"] - runner_up["score"]) < 0.08)
    confidence = 0.85 if not close else 0.5
    if best["unparseable_pct"] > config.get("max_unparseable_pct", 1.0):
        confidence -= 0.2

    why = (
        f"{best['column']} parses {100 - best['unparseable_pct']:.1f}% of rows and spans "
        f"{best['span_days']:.0f} days"
    )
    if close:
        why += f". {runner_up['column']} scores almost the same — worth a look before accepting"

    return _suggestion(
        "date_column", best["column"], "date_column.ranked", why, max(0.05, confidence),
        {"ranked": scored[:5], "close_call": close},
    )


def recommend_label_source(
    df: pd.DataFrame, subtask_review: dict, target_candidates: list[str], config: dict
) -> dict | None:
    """Prefer deriving from SubTask when the champion's mappings really cover it."""
    if not config.get("enabled", True):
        return None

    known = subtask_review.get("known") or []
    unmapped = subtask_review.get("unmapped") or []
    mapped_rows = sum(int(k.get("rows", 0)) for k in known)
    unmapped_rows = sum(int(u.get("rows", 0)) for u in unmapped)
    total = mapped_rows + unmapped_rows
    coverage = 100.0 * mapped_rows / total if total else 0.0
    minimum = config.get("min_subtask_coverage_pct", 95.0)

    verified = [c for c in target_candidates if not is_model_output(c)]
    binary_verified = []
    for column in verified:
        values = df[column].dropna().unique()
        if 0 < len(values) <= 2:
            binary_verified.append({"column": column, "distinct_values": [str(v) for v in values]})

    if coverage >= minimum:
        return _suggestion(
            "target_mode", "derive_from_subtask", "label_source.subtask_coverage",
            f"The champion's SubTask mappings cover {coverage:.1f}% of rows "
            f"({len(unmapped)} SubTask(s) unmapped)",
            0.9 if not unmapped else 0.6,
            {"coverage_pct": round(coverage, 2), "unmapped_subtasks": len(unmapped),
             "unmapped_rows": unmapped_rows},
        )

    if binary_verified:
        choice = binary_verified[0]
        return _suggestion(
            "target_mode", "existing_column", "label_source.verified_binary_column",
            f"SubTask mappings cover only {coverage:.1f}% of rows, but {choice['column']} "
            f"is a two-valued verified column",
            0.55,
            {"coverage_pct": round(coverage, 2), "candidates": binary_verified},
        )

    return _suggestion(
        "target_mode", None, "label_source.insufficient",
        f"SubTask mappings cover only {coverage:.1f}% of rows and no two-valued verified "
        f"label column was found — this one needs a person",
        0.1,
        {"coverage_pct": round(coverage, 2)},
    )


def recommend_dedup_key(df: pd.DataFrame, date_column: str | None, config: dict) -> dict | None:
    """Score columns on how much they look like a per-account business key."""
    if not config.get("enabled", True) or df.empty:
        return None

    tokens = config.get("name_tokens", [])
    minimum = config.get("min_uniqueness_pct", 90.0)
    rows = len(df)
    scored = []
    for column in df.columns:
        name = norm_col(str(column))
        name_hit = any(token in name for token in tokens)
        try:
            distinct = int(df[column].nunique(dropna=True))
        except TypeError:
            continue
        if not distinct:
            continue
        uniqueness = 100.0 * distinct / rows

        # A key whose duplicates carry different dates is a repeated account
        # touched over time, which is exactly what should collapse to one row.
        duplicates_differ_by_date = None
        if date_column is not None and date_column in df.columns and distinct < rows:
            duplicated = df[df.duplicated(subset=[column], keep=False)]
            if not duplicated.empty:
                spread = duplicated.groupby(column)[date_column].nunique(dropna=True)
                duplicates_differ_by_date = bool((spread > 1).any())

        score = (0.5 if name_hit else 0.0) + min(1.0, uniqueness / 100.0) * 0.35
        if duplicates_differ_by_date:
            score += 0.15
        scored.append({
            "column": str(column), "uniqueness_pct": round(uniqueness, 2),
            "name_match": name_hit, "duplicates_differ_by_date": duplicates_differ_by_date,
            "score": round(score, 4),
        })

    if not scored:
        return None
    scored.sort(key=lambda s: -s["score"])
    best = scored[0]
    if not best["name_match"] and best["uniqueness_pct"] < minimum:
        return _suggestion(
            "dedup_keys", [], "dedup_key.none_convincing",
            "No column looks like a business key — full-row deduplication is the safer default",
            0.3, {"ranked": scored[:5]},
        )

    # Two equally key-shaped columns is a real ambiguity, not a ranking to break
    # arbitrarily: AccountID and PatientAcctNo both look like keys and dedup on
    # the wrong one silently collapses different accounts.
    runner_up = scored[1] if len(scored) > 1 else None
    tied = bool(runner_up and (best["score"] - runner_up["score"]) < 0.02)

    why = (
        f"{best['column']} is {best['uniqueness_pct']:.1f}% unique"
        + (" and matches a known key naming pattern" if best["name_match"] else "")
    )
    confidence = 0.75 if best["name_match"] else 0.45
    if tied:
        why += f", but {runner_up['column']} scores identically — pick the one your source system keys on"
        confidence = min(confidence, 0.4)

    return _suggestion(
        "dedup_keys", [best["column"]], "dedup_key.ranked", why, confidence,
        {"ranked": scored[:5], "tied_with": runner_up["column"] if tied else None},
    )


def recommend_historical_window(
    df: pd.DataFrame, date_column: str | None, labels: pd.Series | None, config: dict
) -> dict | None:
    """Widen back through history while the monthly class balance stays stable."""
    if not config.get("enabled", True) or date_column is None or labels is None:
        return None
    if date_column not in df.columns:
        return None

    parsed = pd.to_datetime(df[date_column], errors="coerce")
    frame = pd.DataFrame({"date": parsed, "label": labels}).dropna()
    if frame.empty:
        return None

    frame["month"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby("month")["label"].agg(["mean", "count"])
    if len(monthly) < config.get("min_months", 3):
        return _suggestion(
            "historical_window_days", None, "historical_window.too_short",
            f"Only {len(monthly)} month(s) of data — use all of it",
            0.7, {"months": len(monthly)},
        )

    monthly = monthly.sort_index()
    recent = monthly.tail(3)["mean"].mean() * 100.0
    tolerance = config.get("stability_tolerance_pct", 2.0)

    # Walk backwards from the newest month while the rate stays near recent.
    stable_months = 0
    for _, row in monthly.iloc[::-1].iterrows():
        if abs(row["mean"] * 100.0 - recent) <= tolerance:
            stable_months += 1
        else:
            break

    newest = frame["date"].max()
    oldest = frame["date"].min()
    full_span_days = int((newest - oldest).days)

    if stable_months >= len(monthly):
        return _suggestion(
            "historical_window_days", None, "historical_window.stable_throughout",
            f"The Non-Voice rate stays within {tolerance:.0f} points of its recent level "
            f"across all {len(monthly)} months — use the full history",
            0.8, {"months": len(monthly), "recent_rate_pct": round(recent, 2),
                  "full_span_days": full_span_days},
        )

    cutoff = monthly.index[-stable_months].to_timestamp() if stable_months else newest
    days = max(30, int((newest - cutoff).days))
    return _suggestion(
        "historical_window_days", days, "historical_window.balance_shift",
        f"The Non-Voice rate moves more than {tolerance:.0f} points beyond "
        f"{stable_months} month(s) back — older data describes a different population",
        0.6,
        {"stable_months": stable_months, "recent_rate_pct": round(recent, 2),
         "monthly_rates_pct": [round(v * 100.0, 2) for v in monthly["mean"].tolist()],
         "full_span_days": full_span_days},
    )


# ── Intervention evaluation ──────────────────────────────────────────────────

def evaluate_interventions(rules: dict, facts: dict) -> list[dict]:
    """Apply the intervention rules to measured facts.

    `facts` carries only measured numbers; the rule decides what they mean, so a
    threshold change never needs a code change.
    """
    findings = []
    for rule in rules.get("interventions", []):
        if rule.get("action") == "off":
            continue
        rule_id = rule["id"]
        threshold = rule.get("threshold")
        measured: Any = None
        triggered = False

        if rule_id == "new_subtask":
            measured = facts.get("unmapped_subtasks", 0)
            triggered = measured > 0
        elif rule_id == "missing_required_column":
            measured = facts.get("missing_required", [])
            triggered = bool(measured)
        elif rule_id == "model_output_as_label":
            measured = facts.get("label_is_model_output", False)
            triggered = bool(measured)
        elif rule_id == "too_few_rows":
            measured = facts.get("row_count", 0)
            triggered = threshold is not None and measured < threshold
        elif rule_id == "class_balance_shift":
            measured = facts.get("class_balance_shift_pts")
            triggered = measured is not None and threshold is not None and measured > threshold
        elif rule_id == "duplicate_rows":
            measured = facts.get("duplicate_pct")
            triggered = measured is not None and threshold is not None and measured > threshold
        elif rule_id == "unparseable_dates":
            measured = facts.get("unparseable_date_pct")
            triggered = measured is not None and threshold is not None and measured > threshold
        elif rule_id == "new_grouped_level":
            measured = facts.get("new_grouped_level_pct")
            triggered = measured is not None and threshold is not None and measured > threshold

        if triggered:
            findings.append({
                "id": rule_id,
                "action": rule["action"],
                "label": rule["label"],
                "why": rule["why"],
                "threshold": threshold,
                "unit": rule.get("unit"),
                "measured": measured,
            })

    order = {"block": 0, "warn": 1}
    findings.sort(key=lambda f: order.get(f["action"], 2))
    return findings


def blocking(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["action"] == "block"]


__all__ = [
    "derive_labelled_frame",
    "merge_rules",
    "recommend_date_column",
    "recommend_label_source",
    "recommend_dedup_key",
    "recommend_historical_window",
    "evaluate_interventions",
    "blocking",
]
