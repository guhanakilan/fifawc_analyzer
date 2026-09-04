"""Default rule configuration for readiness recommendations and interventions.

These are *starting values*, not policy. Every threshold here is overridable
per job through the rules API, and the effective set is recorded with the
decisions it influenced so a past run can be explained.

Nothing in this module reaches a network or calls a model. Recommendations are
computed from the uploaded data and the champion package alone, and each one
carries the rule that fired and the evidence for it, so a person can agree or
disagree on the facts rather than on a black box.
"""

from __future__ import annotations

from copy import deepcopy

# ── Intervention rules ───────────────────────────────────────────────────────
#
# `block` stops the stage until a person decides. `warn` is visible but passable.
# Every rule states, in `why`, what goes wrong if it is ignored.

DEFAULT_INTERVENTIONS: list[dict] = [
    {
        "id": "new_subtask",
        "action": "block",
        "label": "A SubTask appears that the champion has never seen",
        "why": (
            "An unmapped SubTask has no Voice/Non-Voice meaning. Letting it fall to a "
            "default silently invents labels for those rows."
        ),
        "threshold": None,
    },
    {
        "id": "missing_required_column",
        "action": "block",
        "label": "A column the model requires is absent",
        "why": (
            "The champion's feature vector cannot be rebuilt, so the comparison would "
            "not be measuring the same thing."
        ),
        "threshold": None,
    },
    {
        "id": "model_output_as_label",
        "action": "block",
        "label": "The chosen label column is written by a scoring run",
        "why": (
            "Training on a model's own predictions recycles them as ground truth and "
            "compounds its existing errors."
        ),
        "threshold": None,
    },
    {
        "id": "too_few_rows",
        "action": "block",
        "label": "Fewer rows than the minimum",
        "why": (
            "Below this the test split is too small for a difference between models to "
            "mean anything."
        ),
        "threshold": 500,
        "unit": "rows",
    },
    {
        "id": "class_balance_shift",
        "action": "warn",
        "label": "Class balance has moved away from the champion's",
        "why": (
            "The champion was tuned at its own Voice rate. A large shift means the "
            "comparison is across two different populations."
        ),
        "threshold": 10.0,
        "unit": "percentage points",
    },
    {
        "id": "duplicate_rows",
        "action": "warn",
        "label": "Duplicate rows",
        "why": (
            "Duplicates leak between train and test and inflate every score, most of "
            "all when they straddle the split."
        ),
        "threshold": 5.0,
        "unit": "% of rows",
    },
    {
        "id": "unparseable_dates",
        "action": "warn",
        "label": "Rows whose date cannot be parsed",
        "why": (
            "A temporal split silently drops or misplaces these, so the test set stops "
            "being the newest data."
        ),
        "threshold": 1.0,
        "unit": "% of rows",
    },
    {
        "id": "new_grouped_level",
        "action": "warn",
        "label": "A new level in a column the champion groups",
        "why": (
            "The fitted grouping has no bucket for it, so it lands in the fallback and "
            "its rows are scored on less information."
        ),
        "threshold": 2.0,
        "unit": "% of rows",
    },
]

# ── Recommendation rules ─────────────────────────────────────────────────────

DEFAULT_RECOMMENDATIONS: dict = {
    "date_column": {
        "enabled": True,
        "prefer_recognised_names": True,
        "max_unparseable_pct": 1.0,
        "min_span_days": 30,
    },
    "label_source": {
        "enabled": True,
        # Deriving from SubTask needs the champion's mappings to cover nearly
        # everything, or the derived label is mostly guesswork.
        "min_subtask_coverage_pct": 95.0,
    },
    "dedup_key": {
        "enabled": True,
        "min_uniqueness_pct": 90.0,
        "name_tokens": ["accountid", "patientacct", "acctno", "claimid", "invoice"],
    },
    "historical_window": {
        "enabled": True,
        # Widen back through history while the monthly Non-Voice rate stays
        # close to the recent mean; stop where the population changed.
        "stability_tolerance_pct": 2.0,
        "min_months": 3,
    },
    "subtask_mapping": {
        "enabled": True,
        # Optional layers, both off unless the model is actually installed.
        "use_embeddings": True,
        "embedding_min_similarity": 0.55,
        "use_llm": False,
    },
}

DEFAULT_RULES: dict = {
    "version": 1,
    "interventions": DEFAULT_INTERVENTIONS,
    "recommendations": DEFAULT_RECOMMENDATIONS,
}


def default_rules() -> dict:
    """A deep copy, so a caller mutating its rules cannot alter the defaults."""
    return deepcopy(DEFAULT_RULES)


def merge_rules(overrides: dict | None) -> dict:
    """Overlay saved overrides on the defaults.

    Interventions are matched by id so a stored set from an older version does
    not drop a rule added since; unknown ids are ignored rather than resurrected.
    """
    rules = default_rules()
    if not overrides:
        return rules

    by_id = {rule["id"]: rule for rule in rules["interventions"]}
    for saved in overrides.get("interventions") or []:
        rule = by_id.get(saved.get("id"))
        if rule is None:
            continue
        if saved.get("action") in ("block", "warn", "off"):
            rule["action"] = saved["action"]
        if "threshold" in saved and rule.get("threshold") is not None:
            try:
                rule["threshold"] = float(saved["threshold"])
            except (TypeError, ValueError):
                pass

    for key, saved in (overrides.get("recommendations") or {}).items():
        if key in rules["recommendations"] and isinstance(saved, dict):
            rules["recommendations"][key].update(saved)
    return rules
