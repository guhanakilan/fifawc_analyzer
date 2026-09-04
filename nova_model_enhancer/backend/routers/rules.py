"""Stage 3 recommendations, intervention checks and the rule configuration.

Recommendations are advisory. Nothing here writes a decision — the readiness
endpoints still require an explicit choice, and every suggestion carries the
rule that produced it and the evidence behind it so it can be argued with.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..database import get_decision, get_decisions, get_training_assets, record_audit, set_decision
from ..services import labeling, local_models, rules as rules_engine
from ..services.champion import load_configs
from ..services.data_profiler import column_lineage, is_model_output
from ..services.nova_transform import norm_col
from ..services.rules_config import default_rules
from .packages import require_job
from .readiness import _job_paths, _load_combined, _ordered_candidates

router = APIRouter(prefix="/api/rules", tags=["3 · Recommendations"])

RULES_KEY = "readiness_rules"


class RulesUpdate(BaseModel):
    interventions: list[dict] = Field(default_factory=list)
    recommendations: dict = Field(default_factory=dict)
    actor: str | None = None


def effective_rules(job_id: str) -> dict:
    """The rule set in force for this job: defaults with any saved overrides."""
    saved = get_decision(job_id, RULES_KEY)
    return rules_engine.merge_rules((saved or {}).get("value"))


@router.get("/{job_id}")
def get_rules(job_id: str):
    require_job(job_id)
    return {
        "rules": effective_rules(job_id),
        "defaults": default_rules(),
        "local_models": local_models.status_report(),
    }


@router.post("/{job_id}")
def save_rules(job_id: str, request: RulesUpdate):
    """Store rule overrides for this job. Thresholds are policy, not code."""
    require_job(job_id)
    merged = rules_engine.merge_rules(request.model_dump(exclude={"actor"}))
    set_decision(job_id, RULES_KEY, merged, request.actor or "local-user")
    record_audit(
        job_id, request.actor or "local-user", "rules.updated",
        "Readiness rule thresholds changed",
        {"interventions": merged["interventions"]},
    )
    return {"rules": merged}


@router.get("/{job_id}/recommendations")
def recommendations(job_id: str):
    """Rule-based suggestions for every Stage 3 decision, with their evidence."""
    require_job(job_id)
    assets = get_training_assets(job_id)
    if not assets:
        raise HTTPException(
            status_code=409, detail="Upload training data before asking for recommendations."
        )

    paths = _job_paths(job_id)
    configs = load_configs(paths["extract_dir"])
    df = _load_combined(job_id)
    columns = [str(c) for c in df.columns]

    rules = effective_rules(job_id)
    recommendation_config = rules["recommendations"]

    from ..services.data_profiler import DATE_CANDIDATES, TARGET_CANDIDATES

    date_candidates = _ordered_candidates(columns, DATE_CANDIDATES)
    target_candidates = _ordered_candidates(columns, TARGET_CANDIDATES)
    subtask_review = labeling.subtask_inventory(df, configs.subtask_mappings)

    saved = get_decisions(job_id)
    chosen_date = (saved.get("readiness_decisions", {}).get("value", {}) or {}).get("date_column")
    date_column = chosen_date or (date_candidates[0] if date_candidates else None)

    suggestions = []
    for suggestion in (
        rules_engine.recommend_date_column(df, date_candidates, recommendation_config["date_column"]),
        rules_engine.recommend_label_source(
            df, subtask_review, target_candidates, recommendation_config["label_source"]
        ),
        rules_engine.recommend_dedup_key(df, date_column, recommendation_config["dedup_key"]),
    ):
        if suggestion:
            suggestions.append(suggestion)

    # The historical window needs labels, which only exist once SubTask mappings
    # resolve. Skipped rather than guessed at when they do not.
    labelled, _ = rules_engine.derive_labelled_frame(df, configs, subtask_review)
    window = rules_engine.recommend_historical_window(
        labelled if labelled is not None else df,
        date_column,
        labelled["NonVoiceFlag"] if labelled is not None else None,
        recommendation_config["historical_window"],
    )
    if window:
        suggestions.append(window)

    # Optional embedding layer for unmapped SubTasks.
    subtask_config = recommendation_config["subtask_mapping"]
    embedding_matches: dict = {}
    if subtask_config.get("enabled", True) and subtask_config.get("use_embeddings", True):
        embedding_matches = local_models.match_subtasks(
            [u["subtask"] for u in subtask_review.get("unmapped", [])],
            [k["subtask"] for k in subtask_review.get("known", [])],
            float(subtask_config.get("embedding_min_similarity", 0.55)),
        )
    known_flags = {k["subtask"]: k.get("flag") for k in subtask_review.get("known", [])}
    subtask_suggestions = []
    for entry in subtask_review.get("unmapped", []):
        name = entry["subtask"]
        match = embedding_matches.get(name)
        if match and known_flags.get(match["nearest"]) is not None:
            subtask_suggestions.append({
                "subtask": name, "rows": entry["rows"],
                "suggested_flag": known_flags[match["nearest"]],
                "source": "embeddings",
                "why": f"Closest known SubTask is {match['nearest']} "
                       f"(similarity {match['similarity']})",
                "similarity": match["similarity"],
            })
        else:
            subtask_suggestions.append({
                "subtask": name, "rows": entry["rows"],
                "suggested_flag": entry.get("suggested_flag"),
                "source": "rules",
                "why": "Matched by the champion's own keyword rules",
            })

    return {
        "suggestions": suggestions,
        "subtask_suggestions": subtask_suggestions,
        "local_models": local_models.status_report(),
        "note": (
            "Every suggestion is advisory. Nothing here is applied until you choose it "
            "in the decisions form below."
        ),
    }


@router.get("/{job_id}/interventions")
def interventions(job_id: str):
    """Which rules fire on this data, and whether each blocks or warns."""
    require_job(job_id)
    assets = get_training_assets(job_id)
    if not assets:
        raise HTTPException(
            status_code=409, detail="Upload training data before checking interventions."
        )

    paths = _job_paths(job_id)
    configs = load_configs(paths["extract_dir"])
    df = _load_combined(job_id)
    columns = [str(c) for c in df.columns]
    rules = effective_rules(job_id)

    subtask_review = labeling.subtask_inventory(df, configs.subtask_mappings)
    fitted = get_decision(job_id, "champion_feature_names")
    lineage = column_lineage(
        configs, columns, (fitted or {}).get("value", {}).get("feature_names")
    )

    saved = (get_decisions(job_id).get("readiness_decisions", {}) or {}).get("value") or {}
    target_column = saved.get("target_column")

    # Before a decision is saved, check the column that would be recommended:
    # a date check that reports nothing until after the choice is made is a
    # check that never catches anything.
    from ..services.data_profiler import DATE_CANDIDATES

    date_column = saved.get("date_column")
    if not date_column:
        detected = _ordered_candidates(columns, DATE_CANDIDATES)
        date_column = detected[0] if detected else None

    unparseable_pct = None
    if date_column and date_column in df.columns and len(df):
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        unparseable_pct = round(100.0 * float(parsed.isna().sum()) / len(df), 3)

    # Champion's own Voice rate, for the balance comparison.
    champion_voice_rate = (configs.training_results or {}).get("voice_rate")
    balance_shift = None
    observed_voice_rate = None
    labelled, _ = rules_engine.derive_labelled_frame(df, configs, subtask_review)
    if labelled is not None and len(labelled):
        observed_voice_rate = round(
            100.0 * float((labelled["NonVoiceFlag"] == labeling.VOICE).mean()), 3
        )
        if champion_voice_rate is not None:
            balance_shift = round(
                abs(observed_voice_rate - float(champion_voice_rate)), 3
            )

    # A level the champion groups but has never seen before.
    new_level_pct = None
    if configs.grouping_config and len(df):
        worst = 0.0
        for column, spec in configs.grouping_config.items():
            match = next((c for c in df.columns if norm_col(str(c)) == norm_col(str(column))), None)
            if match is None or not isinstance(spec, dict):
                continue
            known = {norm_col(str(k)) for k in (spec.get("mapping") or spec.get("groups") or {})}
            if not known:
                continue
            unseen = df[match].astype(str).map(lambda v: norm_col(v) not in known)
            worst = max(worst, 100.0 * float(unseen.mean()))
        new_level_pct = round(worst, 3)

    facts = {
        "row_count": int(len(df)),
        "unmapped_subtasks": len(subtask_review.get("unmapped", [])),
        "missing_required": lineage["missing_required"],
        "label_is_model_output": bool(target_column and is_model_output(target_column)),
        "duplicate_pct": round(100.0 * float(df.duplicated().mean()), 3) if len(df) else 0.0,
        "unparseable_date_pct": unparseable_pct,
        "date_column_checked": date_column,
        "class_balance_shift_pts": balance_shift,
        "observed_voice_rate_pct": observed_voice_rate,
        "champion_voice_rate_pct": champion_voice_rate,
        "new_grouped_level_pct": new_level_pct,
    }
    findings = rules_engine.evaluate_interventions(rules, facts)
    return {
        "findings": findings,
        "blocking": rules_engine.blocking(findings),
        "facts": facts,
        "clear": not findings,
    }
