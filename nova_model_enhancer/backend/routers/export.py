"""Stage 7 — ml_tag approval, package build, smoke test and download."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import MAX_INVENTORY_BYTES, job_dir, reference_scoring_path
from ..database import (
    create_export,
    get_decision,
    get_export,
    list_audit,
    list_exports,
    next_export_version,
    record_audit,
    set_decision,
    update_job,
)
from ..schemas import ExportRequestBody, MlTagApprovalRequest
from ..services import comparison as comparison_service
from ..services import exporter, report as report_service, snapshot as snapshot_service
from ..services.data_profiler import DataReadError, read_dataset
from ..services.safety import assert_safe_id, safe_filename, scrub
from .comparison import _gate_for
from .packages import require_job
from .readiness import _job_paths

router = APIRouter(prefix="/api/export", tags=["7 · Export & validation"])



@router.get("/{job_id}/ml-tag")
def ml_tag(job_id: str):
    """The blocked business decision: what `ml_tag` actually means."""
    require_job(job_id)
    saved = get_decision(job_id, "ml_tag")
    return {
        "approved_config": saved["value"] if saved else None,
        "approved_by": saved["approver"] if saved else None,
        "blocked": saved is None,
        "decision_required": (
            "The reference scoring client appends VoiceNonVoiceFlag with 1 = Voice and "
            "0 = Non-Voice, inverted from the internal NonVoiceFlag target (0 = Voice, "
            "1 = Non-Voice). No source in the supplied material states the ml_tag "
            "convention, so it must be confirmed by an authorised approver. Export is "
            "blocked until it is."
        ),
        "candidate_conventions": [
            {
                "label": "Match VoiceNonVoiceFlag (inverted from the training target)",
                "column_name": "ml_tag", "voice_value": 1, "non_voice_value": 0,
                "note": "Consistent with what the reference loader already appends.",
            },
            {
                "label": "Match the internal NonVoiceFlag target",
                "column_name": "ml_tag", "voice_value": 0, "non_voice_value": 1,
                "note": "Consistent with training, inverted relative to the reference loader.",
            },
        ],
    }


@router.post("/{job_id}/ml-tag")
def approve_ml_tag(job_id: str, request: MlTagApprovalRequest):
    require_job(job_id)
    if str(request.voice_value) == str(request.non_voice_value):
        raise HTTPException(status_code=422, detail="Voice and Non-Voice must map to different values.")
    payload = {
        "column_name": request.column_name,
        "voice_value": request.voice_value,
        "non_voice_value": request.non_voice_value,
        "approved": True,
        "approver": request.approver,
        "notes": request.notes,
    }
    set_decision(job_id, "ml_tag", payload, approver=request.approver)
    record_audit(
        job_id, request.approver, "ml_tag.approved",
        f"{request.column_name}: {request.voice_value} = Voice, {request.non_voice_value} = Non-Voice",
        payload,
    )
    return payload


@router.post("/{job_id}/inventory-sample")
def upload_inventory_sample(job_id: str, file: UploadFile = File(...)):
    """A de-identified inventory sample used to prove scoring compatibility."""
    require_job(job_id)
    filename = safe_filename(file.filename or "", fallback="inventory")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".parquet", ".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=422, detail="Upload the sample as Parquet, CSV or Excel.")

    directory = job_dir(job_id) / "inventory"
    directory.mkdir(parents=True, exist_ok=True)
    stored_path = directory / f"sample{suffix}"
    size = 0
    try:
        with stored_path.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_INVENTORY_BYTES:
                    raise DataReadError("The inventory sample exceeds the 512 MB limit.")
                target.write(chunk)
        frame = read_dataset(stored_path)
        if frame.empty:
            raise DataReadError("The inventory sample contains no rows.")
    except DataReadError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=scrub(exc)) from exc
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read the sample: {scrub(exc)}") from exc
    finally:
        file.file.close()

    set_decision(job_id, "inventory_sample", {
        "path": str(stored_path), "filename": filename,
        "rows": int(len(frame)), "columns": [str(c) for c in frame.columns],
    })
    record_audit(job_id, "local-user", "export.inventory_sample",
                 f"{filename}: {len(frame)} rows, {frame.shape[1]} columns")
    return {"rows": int(len(frame)), "columns": int(frame.shape[1]),
            "column_names": [str(c) for c in frame.columns][:60], "filename": filename}


def _inventory_for(job_id: str, snapshot_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use the uploaded sample when present; otherwise derive one from the snapshot.

    A derived sample keeps the check meaningful without an uploaded file, and its
    origin is recorded in the validation report so nobody mistakes it for a real
    inventory extract.
    """
    stored = get_decision(job_id, "inventory_sample")
    if stored:
        path = Path(stored["value"]["path"])
        if path.exists():
            return read_dataset(path), f"uploaded sample '{stored['value']['filename']}'"
    sample = snapshot_df.head(min(500, len(snapshot_df))).copy()
    drop = [c for c in ("NonVoiceFlag", "__source_role__") if c in sample.columns]
    return sample.drop(columns=drop).reset_index(drop=True), "derived from the dataset snapshot"


def _enhancer_predictions(
    extract_dir: Path, run_dir: Path, candidate: dict, inventory: pd.DataFrame,
    date_column: str | None,
) -> tuple[np.ndarray | None, str]:
    """Score the inventory sample in-process with the promoted model.

    The package smoke test compares its own output against this, so agreement is
    evidence that the ZIP reproduces the model the comparison was based on — not
    just that the ZIP is internally consistent.
    """
    import pickle

    import joblib

    from ..services.champion import load_configs, predict_proba
    from ..services.nova_transform import (
        TARGET, apply_fitted_transforms, build_modelling_frame,
    )

    model_rel = candidate.get("model_path")
    fitted_path = run_dir / "models" / "fitted_transforms.pkl"
    if not model_rel or not (run_dir / model_rel).exists() or not fitted_path.exists():
        return None, "unavailable — the promoted model or its fitted state is missing"
    try:
        estimator = joblib.load(run_dir / model_rel)
        with fitted_path.open("rb") as handle:
            fitted = pickle.load(handle)
        configs = load_configs(extract_dir, date_column=date_column)
        frame = build_modelling_frame(inventory.copy(), configs)
        frame = frame.drop(columns=[c for c in (TARGET, date_column) if c and c in frame.columns])
        X = apply_fitted_transforms(frame, fitted)
        return predict_proba(estimator, X), "computed in-process from the promoted challenger"
    except Exception as exc:  # never block the export on the extra evidence
        return None, f"unavailable — {type(exc).__name__}: {exc}"


@router.post("/{job_id}/build")
def build(job_id: str, request: ExportRequestBody):
    """Assemble the package, then prove it by loading and scoring through it."""
    job = require_job(job_id)
    assert_safe_id(request.run_id)
    paths = _job_paths(job_id)

    approval = get_decision(job_id, f"approval_{request.run_id}")
    if approval is None or approval["value"].get("decision") != "APPROVED":
        raise HTTPException(
            status_code=409,
            detail="A typed promotion approval for this run is required before export.",
        )
    if approval["value"].get("selected_candidate_id") != request.candidate_id:
        raise HTTPException(
            status_code=409,
            detail=f"The approved candidate is {approval['value'].get('selected_candidate_id')}, "
                   f"not {request.candidate_id}.",
        )

    ml_tag_decision = get_decision(job_id, "ml_tag")
    if ml_tag_decision is None:
        raise HTTPException(
            status_code=409,
            detail="The ml_tag encoding has not been approved. Export is blocked until an "
                   "authorised approver confirms what ml_tag means.",
        )

    snapshot = get_decision(job_id, "active_snapshot")
    run_dir = paths["runs_dir"] / request.run_id
    run_path = run_dir / "run_results.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    run_record = json.loads(run_path.read_text(encoding="utf-8"))

    gate_config = _gate_for(job_id)
    comparison = comparison_service.build_comparison(
        run_dir, paths["snapshot_dir"], snapshot["value"], gate_config,
        segment_column=gate_config.get("segment_column"),
        min_segment_rows=int(gate_config.get("min_segment_rows", 100)),
    )
    snapshot_manifest = snapshot_service.load_manifest(paths["snapshot_dir"], snapshot["value"])
    audit = list_audit(job_id, limit=500)

    version = next_export_version(job_id)
    model_id = f"{job_id}_V{version:03d}_{request.candidate_id}"
    candidate = (run_record.get("challengers") or {}).get(request.candidate_id) or {}
    threshold = float(candidate.get("selected_threshold", 0.5))

    # ── Report first, so it can be bundled ──────────────────────────────────
    report_path = run_dir / f"retraining_report_V{version:03d}.xlsx"
    report_service.build_report(report_path, {
        "job_id": job_id, "placement_id": job.get("placement_id"), "version": version,
        "exported_at": approval["value"].get("approved_at"),
        "run": run_record, "comparison": comparison, "approval": approval["value"],
        "snapshot_manifest": snapshot_manifest, "validation": {"status": "pending"},
        "audit": audit,
    })

    export_request = exporter.ExportRequest(
        job_id=job_id, version=version, placement_id=job.get("placement_id") or "UNKNOWN",
        candidate_id=request.candidate_id, model_id=model_id, threshold=threshold,
        ml_tag_config=ml_tag_decision["value"], approval=approval["value"],
        run_record=run_record, comparison=comparison,
        snapshot_manifest=snapshot_manifest, audit=audit,
    )
    previous = list_exports(job_id)

    try:
        built = exporter.build_package(
            request=export_request, extract_dir=paths["extract_dir"], run_dir=run_dir,
            exports_dir=paths["exports_dir"], report_path=report_path, previous_exports=previous,
        )
    except exporter.ExportError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    snapshot_df = snapshot_service.load_snapshot(paths["snapshot_dir"], snapshot["value"])
    inventory, inventory_origin = _inventory_for(job_id, snapshot_df)

    expected_proba, expectation_note = _enhancer_predictions(
        paths["extract_dir"], run_dir, candidate, inventory,
        snapshot_manifest.get("date_column"),
    )

    validation = exporter.smoke_test_package(
        built["temp_path"], inventory, expected_proba, ml_tag_decision["value"],
        reference_scoring_path=reference_scoring_path(),
    )
    validation["inventory_source"] = inventory_origin
    validation["expected_prediction_source"] = expectation_note
    validation["package_version"] = version

    if validation["status"] != "passed":
        built["temp_path"].unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        record_audit(job_id, request.actor or "local-user", "export.validation_failed",
                     "; ".join(validation.get("failed_checks", [])), validation)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Package validation failed — the ZIP was discarded rather than published.",
                "validation": validation,
            },
        )

    # Rebuild the report now that validation has a real result.
    report_service.build_report(report_path, {
        "job_id": job_id, "placement_id": job.get("placement_id"), "version": version,
        "exported_at": approval["value"].get("approved_at"),
        "run": run_record, "comparison": comparison, "approval": approval["value"],
        "snapshot_manifest": snapshot_manifest, "validation": validation, "audit": audit,
    })

    finalised = exporter.finalise_package(built, validation)
    export_id = f"EXP_{uuid.uuid4().hex[:10].upper()}"
    create_export({
        "export_id": export_id, "job_id": job_id, "version": version, "model_id": model_id,
        "zip_path": finalised["zip_path"], "zip_sha256": finalised["zip_sha256"],
        "approval": approval["value"], "report": validation,
    })
    update_job(job_id, status="EXPORTED", current_stage="EXPORT")
    record_audit(
        job_id, request.actor or "local-user", "export.built",
        f"{finalised['zip_name']} (v{version}) sha256 {finalised['zip_sha256'][:16]}…",
        {"export_id": export_id, "validation_status": validation["status"]},
    )
    return {
        "export_id": export_id, "version": version, "model_id": model_id,
        "threshold": threshold, "validation": validation, **finalised,
    }


@router.get("/{job_id}/exports")
def exports(job_id: str):
    require_job(job_id)
    records = list_exports(job_id)
    for record in records:
        record["exists"] = Path(record["zip_path"]).exists()
        record["zip_name"] = Path(record["zip_path"]).name
    return {
        "exports": records,
        "rollback_note": (
            "Every version is retained. To roll back, download the previous version and "
            "unzip it over the placement folder. The enhancer never deploys automatically."
        ),
    }


@router.get("/download/{export_id}")
def download(export_id: str):
    record = get_export(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    path = Path(record["zip_path"])
    expected_root = (job_dir(record["job_id"]) / "exports").resolve()
    if path.resolve().parent != expected_root or not path.exists():
        raise HTTPException(status_code=404, detail="Export file is no longer available.")
    record_audit(record["job_id"], "local-user", "export.downloaded", path.name)
    return FileResponse(path, media_type="application/zip", filename=path.name)
