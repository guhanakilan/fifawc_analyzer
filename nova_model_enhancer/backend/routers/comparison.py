"""Stage 6 — champion comparison, promotion gate and typed approval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..database import get_decision, record_audit, set_decision, update_job, utc_now
from ..schemas import GateApprovalRequest, PromotionApprovalRequest
from ..services import comparison as comparison_service
from ..services import evaluator
from ..services.safety import assert_safe_id
from .packages import require_job
from .readiness import _job_paths

router = APIRouter(prefix="/api/comparison", tags=["6 · Comparison & approval"])


def _gate_for(job_id: str) -> dict:
    saved = get_decision(job_id, "promotion_gate")
    if saved:
        return {**saved["value"], "approved": True, "approver": saved["approver"]}
    return {**evaluator.PROPOSED_GATE, "approved": False}


@router.get("/{job_id}/gate")
def gate(job_id: str):
    require_job(job_id)
    saved = get_decision(job_id, "promotion_gate")
    return {
        "gate": _gate_for(job_id),
        "proposed": evaluator.PROPOSED_GATE,
        "proposed_note": (
            "The proposed thresholds come from the project brief and are a demonstration "
            "default. Until a named approver saves a gate, every comparison is reported as "
            "BLOCKED and no promotion is possible."
        ),
        "primary_metric_choices": list(evaluator.PRIMARY_METRIC_CHOICES),
        "approved_by": saved["approver"] if saved else None,
        "approved_at": saved["updated_at"] if saved else None,
    }


@router.post("/{job_id}/gate")
def save_gate(job_id: str, request: GateApprovalRequest):
    require_job(job_id)
    payload = request.gate.model_dump()
    set_decision(job_id, "promotion_gate", payload, approver=request.approver)
    record_audit(
        job_id, request.approver, "gate.approved",
        f"primary={payload['primary_metric']}, min improvement {payload['min_primary_improvement_pct']}%",
        payload,
    )
    return {"approved": True, "gate": {**payload, "approved": True, "approver": request.approver}}


@router.get("/{job_id}/runs/{run_id}")
def compare(job_id: str, run_id: str):
    """Champion and challengers on identical rows, judged by the approved gate."""
    require_job(job_id)
    assert_safe_id(run_id)
    paths = _job_paths(job_id)
    snapshot = get_decision(job_id, "active_snapshot")
    if snapshot is None:
        raise HTTPException(status_code=409, detail="This job has no active snapshot.")

    gate_config = _gate_for(job_id)
    try:
        result = comparison_service.build_comparison(
            paths["runs_dir"] / run_id, paths["snapshot_dir"], snapshot["value"], gate_config,
            segment_column=gate_config.get("segment_column"),
            min_segment_rows=int(gate_config.get("min_segment_rows", 100)),
        )
    except comparison_service.ComparisonError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    approval = get_decision(job_id, f"approval_{run_id}")
    result["approval"] = approval["value"] if approval else None
    result["gate_approved"] = bool(gate_config.get("approved"))
    return result


class ThresholdRequest(BaseModel):
    candidate_id: str
    threshold: float


@router.get("/thresholds")
def thresholds():
    """The selectable threshold grid, so the UI cannot offer a value the API rejects."""
    return {
        "grid": evaluator.threshold_grid(),
        "min": evaluator.MIN_THRESHOLD,
        "max": evaluator.MAX_THRESHOLD,
        "step": evaluator.THRESHOLD_STEP,
        "note": (
            f"Thresholds run from {evaluator.MIN_THRESHOLD:g} to "
            f"{evaluator.MAX_THRESHOLD:g} in steps of {evaluator.THRESHOLD_STEP:g}. Below "
            "0.5 the model would be calling Voice on rows it believes are Non-Voice."
        ),
    }


@router.post("/{job_id}/runs/{run_id}/threshold")
def rescore_at_threshold(job_id: str, run_id: str, request: ThresholdRequest):
    """Metrics for one candidate at a chosen threshold, without retraining.

    Read-only: the saved probabilities are re-thresholded, nothing is persisted
    and the run's own selected threshold is untouched.
    """
    require_job(job_id)
    assert_safe_id(run_id)
    assert_safe_id(request.candidate_id)

    try:
        threshold = evaluator.clamp_threshold(request.threshold)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    run_dir = _job_paths(job_id)["runs_dir"] / run_id
    y_path = run_dir / "y_test.npy"
    proba_path = run_dir / f"proba_test_{request.candidate_id}.npy"
    if not y_path.exists() or not proba_path.exists():
        raise HTTPException(
            status_code=404, detail="Predictions for that candidate are not available."
        )

    import numpy as np

    y_test = np.load(y_path)
    proba = np.load(proba_path)
    gate_config = _gate_for(job_id)
    cost_ratio = float(gate_config.get("cost_ratio") or evaluator.DEFAULT_COST_RATIO)

    return {
        "candidate_id": request.candidate_id,
        "threshold": threshold,
        "requested_threshold": request.threshold,
        "snapped": threshold != float(request.threshold),
        "test_metrics": evaluator.metrics_at_threshold(y_test, proba, threshold),
        "cost": evaluator.cost_weighted(y_test, proba, threshold, cost_ratio),
    }


@router.post("/{job_id}/approve")
def approve(job_id: str, request: PromotionApprovalRequest):
    """Record a typed promotion decision. Nothing is promoted without one."""
    require_job(job_id)
    assert_safe_id(request.run_id)
    if request.typed_confirmation.strip() != request.candidate_id:
        raise HTTPException(
            status_code=422,
            detail=f"Type the candidate id exactly ({request.candidate_id}) to confirm this decision.",
        )

    gate_config = _gate_for(job_id)
    if request.decision == "APPROVED" and not gate_config.get("approved"):
        raise HTTPException(
            status_code=409,
            detail="Approve the promotion gate before approving a model for promotion.",
        )

    paths = _job_paths(job_id)
    snapshot = get_decision(job_id, "active_snapshot")
    if snapshot is None:
        raise HTTPException(status_code=409, detail="This job has no active snapshot.")
    result = comparison_service.build_comparison(
        paths["runs_dir"] / request.run_id, paths["snapshot_dir"], snapshot["value"], gate_config,
        segment_column=gate_config.get("segment_column"),
        min_segment_rows=int(gate_config.get("min_segment_rows", 100)),
    )
    candidate = (result.get("candidates") or {}).get(request.candidate_id)
    if candidate is None or "skipped" in candidate:
        raise HTTPException(status_code=404, detail="That candidate was not trained in this run.")

    gate_result = (result.get("gate_results") or {}).get(request.candidate_id, {})
    if request.decision == "APPROVED" and gate_result.get("status") == "BLOCKED":
        raise HTTPException(
            status_code=409,
            detail="This candidate is BLOCKED: " + "; ".join(gate_result.get("blockers", [])),
        )

    record = {
        "decision": request.decision,
        "run_id": request.run_id,
        "selected_candidate_id": request.candidate_id,
        "selected_model_type": candidate.get("model_type"),
        "selected_threshold": candidate.get("selected_threshold"),
        "approver": request.approver,
        "approved_at": utc_now(),
        "notes": request.notes,
        "gate_result_at_approval": gate_result,
        "champion_metrics_at_approval": (result.get("champion") or {}).get("test_metrics"),
        "challenger_metrics_at_approval": candidate.get("test_metrics"),
        "snapshot_id": snapshot["value"],
        "override_of_recommendation": gate_result.get("status") == "NOT_RECOMMENDED"
                                      and request.decision == "APPROVED",
    }
    set_decision(job_id, f"approval_{request.run_id}", record, approver=request.approver)
    update_job(
        job_id,
        status="PROMOTION_APPROVED" if request.decision == "APPROVED" else "PROMOTION_REJECTED",
        current_stage="COMPARISON",
    )
    record_audit(
        job_id, request.approver,
        "promotion.approved" if request.decision == "APPROVED" else "promotion.rejected",
        f"{request.candidate_id} in run {request.run_id} — gate {gate_result.get('status')}",
        record,
    )
    return record
