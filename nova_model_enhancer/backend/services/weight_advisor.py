"""Rule-based weighting proposals, read from the frozen snapshot.

The previous behaviour offered one static default set to every placement. These
rules look at the data that was actually frozen — its balance, its span, its
recency profile, which flag columns exist, how rare the rarest SubTask is — and
propose a strategy with a stated reason for each component.

Client dimension: a NoVA export's manifest carries only `placement_id`; there is
no client field anywhere in the package. FacilityName is treated as the client
dimension and PayerName as a secondary one when those columns are present, both
overridable, and the rules fall back to placement plus measured characteristics
when neither exists. That choice is configuration, not an assumption baked into
the logic.

Nothing here is applied. A named approver still has to save a strategy, and a
proposal that recommends no weighting at all is a legitimate outcome.
"""

from __future__ import annotations

import re
from copy import deepcopy

import pandas as pd

from .nova_transform import norm_col
from .weighting import PROPOSED_DEFAULTS

# Which column plays which role, when present. Overridable per job.
DEFAULT_DIMENSIONS = {
    "client": ["facilityname", "facility", "clientname", "client"],
    "secondary": ["payername", "payer", "insurancename"],
}

DEFAULT_THRESHOLDS = {
    # Beyond this majority share, an unweighted fit will chase the majority class.
    "imbalance_majority_pct": 70.0,
    # Recent rows worth up-weighting only if there are enough of them to matter.
    "recent_days": 90,
    "recent_share_min_pct": 20.0,
    # A balance that has moved this far across the span means the population changed.
    "balance_drift_pts": 10.0,
    # Below either of these there is too little signal to justify weighting at all.
    "min_rows_for_weighting": 2000,
    "min_span_days_for_weighting": 90,
    # A SubTask below this share is rare enough to be drowned out.
    "rare_subtask_max_share_pct": 1.0,
}

FLAG_TOKENS = {
    "human_correction": ("correct", "verified", "override"),
    "verified_error": ("error", "misclass", "wrong"),
}


def _key(name) -> str:
    """Column name reduced to letters and digits only.

    norm_col keeps internal spaces ("Payer Name" -> "payer name"), so a lookup
    keyed on "payername" silently never matches an inventory-named upload. The
    snapshot holds whatever names were uploaded, which may be either form.
    """
    return re.sub(r"[^a-z0-9]", "", norm_col(str(name)))


def _find(columns, candidates) -> str | None:
    lookup = {_key(c): str(c) for c in columns}
    for candidate in candidates:
        hit = lookup.get(_key(candidate))
        if hit is not None:
            return hit
    return None


def _find_flag(columns, tokens) -> str | None:
    for column in columns:
        name = _key(column)
        if any(token in name for token in tokens):
            return str(column)
    return None


def measure(df: pd.DataFrame, dates: pd.Series, y: pd.Series, thresholds: dict) -> dict:
    """The facts the rules key off. Measured once, reported alongside the proposal."""
    rows = int(len(df))
    valid_dates = dates.dropna()
    span_days = int((valid_dates.max() - valid_dates.min()).days) if len(valid_dates) > 1 else 0

    counts = y.value_counts()
    majority_pct = 100.0 * float(counts.max()) / rows if rows and len(counts) else 0.0

    recent_share = None
    if len(valid_dates) and span_days:
        cutoff = valid_dates.max() - pd.Timedelta(days=int(thresholds["recent_days"]))
        recent_share = round(100.0 * float((valid_dates >= cutoff).mean()), 2)

    # Has the class balance drifted across the span?
    balance_drift = None
    if len(valid_dates) > 1 and span_days >= 60:
        frame = pd.DataFrame({"date": dates, "y": y}).dropna()
        if not frame.empty:
            halves = frame.sort_values("date")
            midpoint = len(halves) // 2
            first = 100.0 * float(halves.iloc[:midpoint]["y"].mean())
            second = 100.0 * float(halves.iloc[midpoint:]["y"].mean())
            balance_drift = round(abs(second - first), 2)

    rarest_subtask_pct = None
    subtask_col = _find(df.columns, ["subtask"])
    if subtask_col is not None and rows:
        shares = df[subtask_col].astype(str).value_counts(normalize=True) * 100.0
        if len(shares):
            rarest_subtask_pct = round(float(shares.min()), 3)

    return {
        "rows": rows,
        "span_days": span_days,
        "majority_class_pct": round(majority_pct, 2),
        "recent_share_pct": recent_share,
        "balance_drift_pts": balance_drift,
        "rarest_subtask_pct": rarest_subtask_pct,
        "subtask_column": subtask_col,
        "correction_column": _find_flag(df.columns, FLAG_TOKENS["human_correction"]),
        "error_column": _find_flag(df.columns, FLAG_TOKENS["verified_error"]),
        "client_column": _find(df.columns, DEFAULT_DIMENSIONS["client"]),
        "secondary_column": _find(df.columns, DEFAULT_DIMENSIONS["secondary"]),
    }


def propose(facts: dict, thresholds: dict | None = None) -> dict:
    """A weighting strategy with a reason per component, derived from the facts."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    strategy = deepcopy(PROPOSED_DEFAULTS)
    reasons: list[dict] = []

    def enable(component: str, reason: str, **settings) -> None:
        strategy["components"][component]["enabled"] = True
        strategy["components"][component].update(settings)
        strategy["enabled"] = True
        reasons.append({"component": component, "enabled": True, "why": reason})

    def skip(component: str, reason: str) -> None:
        reasons.append({"component": component, "enabled": False, "why": reason})

    # Too little data to weight at all. Weighting a small sample mostly amplifies
    # its noise, so this rule overrides every other one.
    too_few = facts["rows"] < limits["min_rows_for_weighting"]
    too_short = 0 < facts["span_days"] < limits["min_span_days_for_weighting"]
    if too_few or too_short:
        detail = []
        if too_few:
            detail.append(f"{facts['rows']} rows is below {limits['min_rows_for_weighting']}")
        if too_short:
            detail.append(
                f"{facts['span_days']} days is below {limits['min_span_days_for_weighting']}"
            )
        return {
            "strategy": deepcopy(PROPOSED_DEFAULTS),
            "recommend_weighting": False,
            "headline": "No weighting recommended",
            "why": (
                "There is not enough data for weighting to be justified — "
                + " and ".join(detail)
                + ". Weighting a small or short sample mostly amplifies its noise."
            ),
            "reasons": reasons,
            "facts": facts,
        }

    if facts["majority_class_pct"] > limits["imbalance_majority_pct"]:
        enable(
            "class_balance",
            f"The majority class holds {facts['majority_class_pct']:.1f}% of rows, above "
            f"{limits['imbalance_majority_pct']:.0f}%. Unweighted, the fit chases it.",
        )
    else:
        skip(
            "class_balance",
            f"Classes are near even ({facts['majority_class_pct']:.1f}% majority) — "
            "balancing would change little.",
        )

    drift = facts.get("balance_drift_pts")
    if drift is not None and drift > limits["balance_drift_pts"]:
        enable(
            "recency",
            f"The class balance moved {drift:.1f} points between the older and newer halves "
            f"of the span, so recent rows describe today's population better.",
            weight=2.0, recent_days=int(limits["recent_days"]),
        )
    elif (facts.get("recent_share_pct") or 0) >= limits["recent_share_min_pct"]:
        enable(
            "recency",
            f"{facts['recent_share_pct']:.1f}% of rows fall in the last "
            f"{limits['recent_days']} days — enough recent data to lean on.",
            weight=1.5, recent_days=int(limits["recent_days"]),
        )
    else:
        skip("recency", "Recent rows are neither numerous enough nor different enough to up-weight.")

    if facts.get("correction_column"):
        enable(
            "human_correction",
            f"'{facts['correction_column']}' looks like a human-correction flag, and a "
            "verified correction is stronger evidence than an untouched row.",
            column=facts["correction_column"], weight=3.0,
        )
    else:
        skip("human_correction", "No human-correction flag column is present.")

    if facts.get("error_column"):
        enable(
            "verified_error",
            f"'{facts['error_column']}' looks like a verified-error flag — rows the previous "
            "champion got wrong are the ones worth learning from.",
            column=facts["error_column"], weight=2.5,
        )
    else:
        skip("verified_error", "No verified-error flag column is present.")

    rarest = facts.get("rarest_subtask_pct")
    if rarest is not None and rarest < limits["rare_subtask_max_share_pct"]:
        enable(
            "rare_subtask",
            f"The rarest SubTask is {rarest:.2f}% of rows, below "
            f"{limits['rare_subtask_max_share_pct']:.0f}% — it would otherwise be drowned out.",
            weight=2.0, max_share_pct=float(limits["rare_subtask_max_share_pct"]),
        )
    else:
        skip("rare_subtask", "No SubTask is rare enough to need up-weighting.")

    active = [r["component"] for r in reasons if r["enabled"]]
    return {
        "strategy": strategy,
        "recommend_weighting": bool(active),
        "headline": (
            f"{len(active)} component(s) proposed" if active else "No weighting recommended"
        ),
        "why": (
            "Proposed from this snapshot's own characteristics."
            if active
            else "Nothing in this snapshot argues for weighting; an unweighted fit is the "
                 "cleaner comparison against the champion."
        ),
        "reasons": reasons,
        "facts": facts,
    }
