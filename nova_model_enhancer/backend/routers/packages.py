"""Stage 1 — champion package intake and compatibility validation."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import MAX_ZIP_BYTES, job_dir
from ..database import (
    create_job,
    get_decision,
    get_job,
    get_training_assets,
    list_exports,
    list_jobs,
    job_overview,
    record_audit,
    update_job,
)
from ..schemas import CompatibilityRequest
from ..services.champion import ChampionLoadError, load_champion
from ..services.package_validator import (
    PackageValidationError,
    sha256_file,
    validate_and_extract,
)
from ..services.safety import safe_filename

router = APIRouter(prefix="/api/packages", tags=["1 · Champion package"])


def require_job(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Retraining job not found.")
    return job


@router.post("/upload")
def upload_package(file: UploadFile = File(...)):
    """Store, validate and extract a completed NoVA run export.

    The uploaded ZIP is kept byte-for-byte alongside its extraction, so the
    champion can always be re-derived from the original artifact.
    """
    filename = safe_filename(file.filename or "", fallback="package.zip")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="Upload a .zip NoVA export package.")

    job_id = f"RETRAIN_{uuid.uuid4().hex[:12].upper()}"
    directory = job_dir(job_id)
    upload_dir = directory / "champion" / "uploaded"
    extract_dir = directory / "champion" / "extracted"
    upload_dir.mkdir(parents=True, exist_ok=False)
    zip_path = upload_dir / "source_nova_export.zip"

    size = 0
    try:
        with zip_path.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ZIP_BYTES:
                    raise PackageValidationError(
                        f"ZIP exceeds the {MAX_ZIP_BYTES // (1024 * 1024)} MB upload limit."
                    )
                target.write(chunk)

        validation = validate_and_extract(zip_path, extract_dir)
        if not validation["valid"]:
            shutil.rmtree(extract_dir, ignore_errors=True)

        metadata = validation["metadata"]
        record = {
            "job_id": job_id,
            "placement_id": metadata.get("placement_id"),
            "source_run_id": metadata.get("run_id"),
            "source_model_id": metadata.get("model_id"),
            "original_filename": filename,
            "package_sha256": sha256_file(zip_path),
            "status": "PACKAGE_READY" if validation["valid"] else "PACKAGE_INVALID",
            "current_stage": "CHAMPION_PACKAGE",
            "validation": validation,
        }
        create_job(record)
        record_audit(
            job_id, "local-user", "package.upload",
            f"{filename} ({size} bytes) — {'accepted' if validation['valid'] else 'rejected'}",
            {"sha256": record["package_sha256"], "blocking_failures": validation["blocking_failures"]},
        )
        return {**record, "uploaded_bytes": size}
    except PackageValidationError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        file.file.close()


@router.get("/jobs")
def jobs():
    return {"jobs": list_jobs()}


@router.get("/overview")
def overview():
    """Home-screen listing: every job with its run, dataset and export counts."""
    return {"jobs": job_overview()}


@router.get("/jobs/{job_id}")
def job(job_id: str):
    return require_job(job_id)


@router.post("/jobs/{job_id}/compatibility")
def compatibility_check(job_id: str, request: CompatibilityRequest):
    """Load the champion estimator in an explicit, acknowledged trust step.

    This is the first and only place the application unpickles anything from the
    uploaded package.
    """
    job = require_job(job_id)
    if not job["validation"].get("valid"):
        raise HTTPException(status_code=409, detail="This package failed validation and cannot be loaded.")
    if not request.trust_local_package:
        raise HTTPException(
            status_code=400,
            detail=(
                "Loading a model file executes code contained in the uploaded package. "
                "Confirm you trust this local file before continuing."
            ),
        )

    extract_dir = job_dir(job_id) / "champion" / "extracted"
    try:
        champion = load_champion(extract_dir)
    except ChampionLoadError as exc:
        record_audit(job_id, request.actor or "local-user", "package.compatibility.failed", str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = {
        "model_id": champion.model_id,
        "model_class": type(champion.estimator).__name__,
        "threshold": champion.threshold,
        "feature_count": len(champion.feature_names),
        "feature_names_preview": champion.feature_names[:25],
        "fitted_state": {
            "imputation_values": len(champion.fitted.get("imputation_vals") or {}),
            "outlier_bounds": len(champion.fitted.get("outlier_bounds") or {}),
            "label_encoders": len(champion.fitted.get("label_encoders") or {}),
            "frequency_maps": len(champion.fitted.get("freq_maps") or {}),
            "one_hot_columns": len(champion.fitted.get("onehot_cols") or {}),
            "scalers": len(champion.fitted.get("scalers") or {}),
            "log_columns": len(champion.fitted.get("log_cols") or []),
        },
        "supports_predict_proba": hasattr(champion.estimator, "predict_proba"),
        "subtask_mappings": len(champion.configs.subtask_mappings),
        "subtask_keywords": len(champion.configs.subtask_keywords),
        "feature_selection_count": len(champion.configs.feature_selection),
        "has_features_config": bool(champion.configs.features_config),
    }
    update_job(job_id, status="CHAMPION_VERIFIED")
    record_audit(
        job_id, request.actor or "local-user", "package.compatibility.passed",
        f"{result['model_class']} loaded with {result['feature_count']} features",
        result,
    )
    return result


@router.get("/jobs/{job_id}/progress")
def job_progress(job_id: str):
    """What has actually been completed for this job, read from durable state.

    The UI derives which stages are reachable from this rather than from what it
    happens to have in memory, so a browser refresh does not re-lock stages the
    user already finished.
    """
    job = require_job(job_id)
    assets = get_training_assets(job_id)
    snapshot = get_decision(job_id, "active_snapshot")
    weights = get_decision(job_id, "weight_strategy")
    ml_tag = get_decision(job_id, "ml_tag")

    runs_dir = job_dir(job_id) / "runs"
    runs = (
        sorted((p.parent.name for p in runs_dir.glob("*/run_results.json")), reverse=True)
        if runs_dir.is_dir() else []
    )
    approved_run = next(
        (
            run_id for run_id in runs
            if (get_decision(job_id, f"approval_{run_id}") or {}).get("value", {}).get("decision")
            == "APPROVED"
        ),
        None,
    )

    return {
        "job_id": job_id,
        "package_valid": bool(job["validation"].get("valid")),
        "data_uploaded": bool(assets),
        "snapshot_id": snapshot["value"] if snapshot else None,
        "weights_approved": weights is not None,
        "run_id": runs[0] if runs else None,
        "run_count": len(runs),
        "approved_run_id": approved_run,
        "ml_tag_approved": ml_tag is not None,
        "export_count": len(list_exports(job_id)),
    }
