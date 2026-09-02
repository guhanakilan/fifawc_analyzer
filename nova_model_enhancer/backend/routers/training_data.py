"""Stage 2 — labelled training data intake and profiling."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import MAX_DATA_BYTES, PROFILE_CHUNK_ROWS, job_dir
from ..database import (
    create_training_asset,
    delete_training_asset,
    get_training_asset,
    get_training_assets,
    record_audit,
    update_job,
)
from ..services.champion import load_configs
from ..services.data_profiler import (
    SUPPORTED_DATA_SUFFIXES,
    DataReadError,
    drift_report,
    profile_dataset,
    sha256_file,
)
from ..services.safety import safe_filename
from .packages import require_job

router = APIRouter(prefix="/api/training-data", tags=["2 · Training data"])

ROLES = {"historical", "new", "combined"}


@router.post("/{job_id}/upload")
def upload_training_data(
    job_id: str,
    file: UploadFile = File(...),
    role: str = Form("combined"),
):
    """Accept one labelled dataset. Multiple uploads accumulate for the job.

    `role` distinguishes a combined historical+new file from separate historical
    and incremental uploads; all of them are concatenated at snapshot time.
    """
    job = require_job(job_id)
    if not job["validation"].get("valid"):
        raise HTTPException(status_code=409, detail="Upload a valid champion package first.")
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(ROLES)}.")

    filename = safe_filename(file.filename or "", fallback="dataset")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DATA_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Upload Parquet, CSV, XLSX or XLS. Received {suffix or 'a file with no extension'}.",
        )

    asset_id = f"DATA_{uuid.uuid4().hex[:12].upper()}"
    data_dir = job_dir(job_id) / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    stored_path = data_dir / f"{asset_id}{suffix}"

    size = 0
    try:
        with stored_path.open("wb") as target:
            while chunk := file.file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_DATA_BYTES:
                    raise DataReadError(
                        f"Training data exceeds the {MAX_DATA_BYTES // (1024 ** 3)} GB limit."
                    )
                target.write(chunk)
        if size == 0:
            raise DataReadError("The uploaded file is empty.")

        summary = profile_dataset(stored_path, chunk_rows=PROFILE_CHUNK_ROWS)
        if summary["rows"] == 0:
            raise DataReadError("The uploaded file contains no data rows.")

        configs = load_configs(job_dir(job_id) / "champion" / "extracted")
        expected = configs.column_config or configs.feature_selection
        summary["schema_drift"] = (
            drift_report(summary, expected, configs.column_map) if expected else None
        )

        record = {
            "asset_id": asset_id, "job_id": job_id, "role": role,
            "original_filename": filename, "stored_path": str(stored_path),
            "file_type": suffix.lstrip("."), "sha256": sha256_file(stored_path),
            "rows_count": summary["rows"], "columns_count": summary["columns"],
            "summary": summary,
        }
        create_training_asset(record)
        update_job(job_id, status="DATA_UPLOADED", current_stage="TRAINING_DATA")
        record_audit(
            job_id, "local-user", "training-data.upload",
            f"{filename} as '{role}' — {summary['rows']} rows, {summary['columns']} columns",
            {"asset_id": asset_id, "sha256": record["sha256"]},
        )
        return {**record, "uploaded_bytes": size}
    except DataReadError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read the training data: {exc}") from exc
    finally:
        file.file.close()


@router.get("/{job_id}")
def training_data(job_id: str):
    require_job(job_id)
    assets = get_training_assets(job_id)
    return {
        "assets": assets,
        "total_rows": sum(a["rows_count"] for a in assets),
        "roles": sorted({a["role"] for a in assets}),
    }


@router.delete("/{job_id}/{asset_id}")
def remove_training_data(job_id: str, asset_id: str):
    """Remove one uploaded dataset. Only this asset's own file is deleted."""
    require_job(job_id)
    asset = get_training_asset(asset_id)
    if asset is None or asset["job_id"] != job_id:
        raise HTTPException(status_code=404, detail="Training data asset not found for this job.")
    path = Path(asset["stored_path"])
    expected_root = (job_dir(job_id) / "datasets").resolve()
    if path.resolve().parent != expected_root:
        raise HTTPException(status_code=409, detail="Asset path is outside this job's dataset folder.")
    path.unlink(missing_ok=True)
    delete_training_asset(asset_id)
    record_audit(job_id, "local-user", "training-data.remove", asset["original_filename"],
                 {"asset_id": asset_id})
    return {"removed": asset_id}
