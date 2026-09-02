"""Audit trail — every consequential action, in one place."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..database import list_audit
from .packages import require_job

router = APIRouter(prefix="/api/audit", tags=["Audit"])


@router.get("/{job_id}")
def job_audit(job_id: str, limit: int = Query(300, ge=1, le=2000)):
    require_job(job_id)
    return {"events": list_audit(job_id, limit=limit)}


@router.get("")
def all_audit(limit: int = Query(300, ge=1, le=2000)):
    return {"events": list_audit(None, limit=limit)}
