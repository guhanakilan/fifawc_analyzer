"""Stage 4 — sample weight strategy: preview, then explicit approval."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..database import get_decision, record_audit, set_decision, update_job
from ..schemas import WeightApprovalRequest, WeightPreviewRequest
from ..services import snapshot as snapshot_service, weight_advisor, weighting
from ..services.nova_transform import TARGET
from .packages import require_job
from .readiness import _job_paths

router = APIRouter(prefix="/api/weights", tags=["4 · Weight strategy"])


def _snapshot_frame(job_id: str):
    stored = get_decision(job_id, "active_snapshot")
    if stored is None:
        raise HTTPException(status_code=409, detail="Build a dataset snapshot before configuring weights.")
    paths = _job_paths(job_id)
    snapshot_id = stored["value"]
    df = snapshot_service.load_snapshot(paths["snapshot_dir"], snapshot_id)
    manifest = snapshot_service.load_manifest(paths["snapshot_dir"], snapshot_id)
    return df, manifest


@router.get("/{job_id}/options")
def options(job_id: str):
    """The proposed defaults and the columns each component could bind to."""
    require_job(job_id)
    df, manifest = _snapshot_frame(job_id)
    from ..services.nova_transform import norm_col

    flag_columns = [
        c for c in df.columns
        if any(token in norm_col(c) for token in ("correct", "verified", "override", "error", "misclass"))
    ]
    # Proposal derived from this snapshot rather than a single static default.
    dates = pd.to_datetime(df[manifest["date_column"]], errors="coerce")
    facts = weight_advisor.measure(
        df, dates, df[TARGET].astype(int), weight_advisor.DEFAULT_THRESHOLDS
    )
    advice = weight_advisor.propose(facts)

    saved = get_decision(job_id, "weight_strategy")
    return {
        "advice": advice,
        "dimensions": {
            "client": facts.get("client_column"),
            "secondary": facts.get("secondary_column"),
            "note": (
                "A NoVA export's manifest carries only placement_id — there is no client "
                "field in the package. FacilityName is treated as the client dimension and "
                "PayerName as a secondary one when present; the rules fall back to placement "
                "and measured data characteristics when neither exists."
            ),
        },
        "proposed_defaults": weighting.PROPOSED_DEFAULTS,
        "proposed_note": (
            "These values come from the project brief and are proposals only. "
            "Nothing is applied until a named approver saves a strategy."
        ),
        "snapshot_rows": manifest["row_counts"]["final"],
        "date_column": manifest["date_column"],
        "date_range": manifest["date_range"],
        "class_distribution": manifest["target"]["distribution"],
        "candidate_flag_columns": flag_columns,
        "has_subtask_column": any(norm_col(c) == "subtask" for c in df.columns),
        "approved_strategy": saved["value"] if saved else None,
        "approved_by": saved["approver"] if saved else None,
    }


@router.post("/{job_id}/preview")
def preview(job_id: str, request: WeightPreviewRequest):
    """Compute the weights without saving them, so the effect is visible first."""
    require_job(job_id)
    df, manifest = _snapshot_frame(job_id)
    dates = pd.to_datetime(df[manifest["date_column"]], errors="coerce")
    y = df[TARGET].astype(int)
    try:
        _, summary = weighting.compute_weights(df, request.strategy, dates=dates, y=y)
    except weighting.WeightError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "summary": summary,
        "formula": weighting.formula_text(request.strategy),
        "approved": False,
        "note": "Preview only — this strategy has not been saved.",
    }


@router.post("/{job_id}/approve")
def approve(job_id: str, request: WeightApprovalRequest):
    """Persist the exact approved formula together with the approver's identity."""
    require_job(job_id)
    df, manifest = _snapshot_frame(job_id)
    dates = pd.to_datetime(df[manifest["date_column"]], errors="coerce")
    y = df[TARGET].astype(int)
    try:
        _, summary = weighting.compute_weights(df, request.strategy, dates=dates, y=y)
    except weighting.WeightError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = {
        "strategy": request.strategy,
        "formula": weighting.formula_text(request.strategy),
        "summary": summary,
        "notes": request.notes,
        "snapshot_id": manifest["snapshot_id"],
    }
    set_decision(job_id, "weight_strategy", payload, approver=request.approver)
    update_job(job_id, status="WEIGHTS_APPROVED", current_stage="WEIGHT_STRATEGY")
    record_audit(job_id, request.approver, "weights.approved", payload["formula"], summary["distribution"])
    return {"approved": True, "approver": request.approver, **payload}


@router.get("/{job_id}")
def approved(job_id: str):
    require_job(job_id)
    saved = get_decision(job_id, "weight_strategy")
    if saved is None:
        raise HTTPException(status_code=404, detail="No weight strategy has been approved for this job.")
    return {"approver": saved["approver"], "updated_at": saved["updated_at"], **saved["value"]}
