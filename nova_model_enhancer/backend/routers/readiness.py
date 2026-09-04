"""Stage 3 — data readiness, mapping decisions and the immutable snapshot."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import job_dir
from ..database import (
    get_decision,
    get_decisions,
    get_training_assets,
    record_audit,
    set_decision,
    update_job,
)
from ..schemas import ReadinessDecisions
from ..services import labeling
from ..services import rules as rules_engine
from ..services import snapshot as snapshot_service
from ..services.champion import load_configs
from ..services.data_profiler import (
    DATE_CANDIDATES,
    MODEL_OUTPUT_COLUMNS,
    TARGET_CANDIDATES,
    column_lineage,
    drift_report,
    is_model_output,
    match_column,
    read_dataset,
)
from ..services.nova_transform import norm_col
from ..services.package_validator import sha256_bytes
from .packages import require_job

router = APIRouter(prefix="/api/readiness", tags=["3 · Readiness & snapshot"])

_SAMPLE_ROWS_FOR_SUBTASK_REVIEW = 400_000


def _job_paths(job_id: str) -> dict[str, Path]:
    directory = job_dir(job_id)
    return {
        "extract_dir": directory / "champion" / "extracted",
        "snapshot_dir": directory / "snapshots",
        "runs_dir": directory / "runs",
        "exports_dir": directory / "exports",
    }


def _load_combined(job_id: str) -> pd.DataFrame:
    assets = get_training_assets(job_id)
    if not assets:
        raise HTTPException(status_code=409, detail="Upload training data before reviewing readiness.")
    frames = []
    for asset in assets:
        frame = read_dataset(Path(asset["stored_path"]))
        frame["__source_role__"] = asset["role"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _ordered_candidates(columns: list[str], candidates: tuple | list) -> list[str]:
    """Columns that match a candidate list, ranked by the candidate's own order."""
    ordered: list[str] = []
    for candidate in candidates:
        hit = match_column(columns, candidate)
        if hit is not None and hit not in ordered:
            ordered.append(hit)
    return ordered


@router.get("/{job_id}/review")
def review(job_id: str):
    """Everything Stage 3 needs a human to look at before a snapshot exists.

    No decision is inferred here: detected values are offered as candidates and
    the SubTask review is returned unresolved when anything is unmapped.
    """
    require_job(job_id)
    paths = _job_paths(job_id)
    assets = get_training_assets(job_id)
    if not assets:
        raise HTTPException(status_code=409, detail="Upload training data before reviewing readiness.")

    configs = load_configs(paths["extract_dir"])
    df = _load_combined(job_id)

    columns = [str(c) for c in df.columns]
    # Ordered by candidate priority, not by the order the columns happen to sit
    # in: the first entry becomes the pre-selected default, and picking (say)
    # DOSFrom over UpdatedDateTimeGMT would silently split the data on the wrong
    # clock.
    date_candidates = _ordered_candidates(columns, DATE_CANDIDATES)
    target_candidates = _ordered_candidates(columns, TARGET_CANDIDATES)

    sample = df if len(df) <= _SAMPLE_ROWS_FOR_SUBTASK_REVIEW else df.sample(
        _SAMPLE_ROWS_FOR_SUBTASK_REVIEW, random_state=42
    )
    subtasks = labeling.subtask_inventory(sample, configs.subtask_mappings)

    expected = configs.column_config or configs.feature_selection
    drift = (
        drift_report({"column_names": columns}, expected, configs.column_map)
        if expected else {"missing_columns": [], "missing_column_count": 0}
    )
    # Layered view: where each column left the pipeline. The fitted layer is
    # only available once the trust-gated compatibility step has recorded it,
    # so its absence degrades the report rather than blocking it.
    fitted_decision = get_decision(job_id, "champion_feature_names")
    lineage = column_lineage(
        configs,
        columns,
        (fitted_decision or {}).get("value", {}).get("feature_names"),
    )

    # The real span of each date candidate, so the window pickers open on the
    # data that exists rather than on today's date.
    date_spans = {}
    for candidate in date_candidates:
        parsed = pd.to_datetime(df[candidate], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            continue
        date_spans[candidate] = {
            "from": valid.min().date().isoformat(),
            "to": valid.max().date().isoformat(),
            "days": int((valid.max() - valid.min()).days),
            "unparseable": int(parsed.isna().sum()),
        }

    duplicate_full_rows = int(df.duplicated().sum())
    key_candidates = [
        c for c in columns
        if any(token in norm_col(c) for token in ("accountid", "patientacct", "claimid", "invoice", "acctno"))
    ]

    saved = get_decisions(job_id)
    return {
        "columns": columns,
        "row_count": int(len(df)),
        "date_candidates": date_candidates,
        "target_candidates": target_candidates,
        "detected_date_column": date_candidates[0] if date_candidates else None,
        "detected_target_column": next(
            (c for c in target_candidates if not is_model_output(c)), None
        ),
        "model_output_columns_present": [c for c in target_candidates if is_model_output(c)],
        "model_output_warning": (
            "A column listed here is written by a NoVA scoring run, not verified by a person. "
            "Training on it would feed the model its own predictions as ground truth. Choosing "
            "one requires an explicit acknowledgement."
        ),
        "champion_target_encoding": {
            "column": "NonVoiceFlag", "voice": labeling.VOICE, "non_voice": labeling.NON_VOICE,
            "source": "reference routers/flag.py::run_flag — confirmed, not assumed",
        },
        "subtask_review": subtasks,
        "requires_subtask_decision": bool(subtasks.get("unmapped")),
        "schema_drift": {
            "expected_columns": len(expected) if expected else 0,
            "missing_columns": drift["missing_columns"],
            "missing_column_count": drift["missing_column_count"],
        },
        "column_lineage": lineage,
        "date_spans": date_spans,
        "duplicates": {
            "full_row_duplicates": duplicate_full_rows,
            "key_column_candidates": key_candidates,
            "note": "A business deduplication key is never inferred. Choose one or confirm full-row dedup.",
        },
        "flag_columns_for_weighting": [
            c for c in columns
            if any(token in norm_col(c) for token in ("correct", "verified", "override", "error", "misclass"))
        ],
        "saved_decisions": saved.get("readiness", {}).get("value"),
        "champion_has_subtask_mappings": bool(configs.subtask_mappings),
    }


@router.post("/{job_id}/decisions")
def save_decisions(job_id: str, decisions: ReadinessDecisions):
    """Persist the approved readiness decisions. Required before any snapshot."""
    require_job(job_id)
    payload = decisions.model_dump()
    if decisions.target_mode == "existing" and not decisions.target_column:
        raise HTTPException(status_code=422, detail="Select the label column to use.")
    if (
        decisions.target_mode == "existing"
        and decisions.target_column
        and is_model_output(decisions.target_column)
        and not decisions.acknowledge_model_output_target
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{decisions.target_column}' is a column a NoVA scoring run writes, not one a "
                "person verified. Training on it would recycle the model's own predictions as "
                "ground truth. Choose a verified label column, derive labels from SubTask "
                "mappings, or explicitly acknowledge this if the column has since been "
                "human-verified."
            ),
        )
    if decisions.dedup_mode == "key_columns" and not decisions.dedup_keys:
        raise HTTPException(status_code=422, detail="Select at least one deduplication key column.")
    set_decision(job_id, "readiness", payload, approver=decisions.approver)
    record_audit(
        job_id, decisions.approver, "readiness.decisions",
        f"target={decisions.target_mode}, dedup={decisions.dedup_mode}, date={decisions.date_column}",
        payload,
    )
    return {"saved": True, "decisions": payload}


class WindowPreviewRequest(BaseModel):
    date_column: str
    date_from: str | None = None
    date_to: str | None = None


@router.post("/{job_id}/window-preview")
def window_preview(job_id: str, request: WindowPreviewRequest):
    """Rows and class balance for a proposed training window, before committing.

    Read-only: nothing is saved and no snapshot is built. This exists so the
    consequence of narrowing the window is visible while choosing it, rather
    than discovered after the dataset is frozen.
    """
    require_job(job_id)
    if not get_training_assets(job_id):
        raise HTTPException(status_code=409, detail="Upload training data first.")

    df = _load_combined(job_id)
    if request.date_column not in df.columns:
        raise HTTPException(
            status_code=422, detail="That date column is not present in the uploaded data."
        )

    configs = load_configs(_job_paths(job_id)["extract_dir"])
    dates = pd.to_datetime(df[request.date_column], errors="coerce")

    inside = pd.Series(True, index=dates.index)
    try:
        if request.date_from:
            inside &= dates >= pd.to_datetime(request.date_from)
        if request.date_to:
            upper = pd.to_datetime(request.date_to)
            if upper == upper.normalize():
                upper = upper + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            inside &= dates <= upper
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="That window could not be read as dates.") from exc

    # Unparseable dates are kept, matching what the snapshot builder does.
    keep = dates.isna() | inside.fillna(False)
    selected = df[keep]

    subtask_review = labeling.subtask_inventory(selected, configs.subtask_mappings)
    labelled, reason = rules_engine.derive_labelled_frame(selected, configs, subtask_review)
    balance = None
    if labelled is not None and len(labelled):
        balance = {
            "voice_pct": round(100.0 * float((labelled["NonVoiceFlag"] == labeling.VOICE).mean()), 2),
            "non_voice_pct": round(
                100.0 * float((labelled["NonVoiceFlag"] == labeling.NON_VOICE).mean()), 2
            ),
            "labelled_rows": int(len(labelled)),
        }

    valid = dates[keep].dropna()
    return {
        "rows_selected": int(len(selected)),
        "rows_total": int(len(df)),
        "rows_excluded": int(len(df) - len(selected)),
        "actual_from": valid.min().date().isoformat() if len(valid) else None,
        "actual_to": valid.max().date().isoformat() if len(valid) else None,
        "class_balance": balance,
        "balance_unavailable_reason": None if balance else reason,
    }


@router.post("/{job_id}/snapshot")
def build_snapshot(job_id: str):
    """Freeze the dataset. Runs synchronously so its failure is immediate and visible."""
    require_job(job_id)
    stored = get_decision(job_id, "readiness")
    if stored is None:
        raise HTTPException(status_code=409, detail="Save the readiness decisions before building a snapshot.")

    value = stored["value"]
    paths = _job_paths(job_id)
    assets = get_training_assets(job_id)
    if not assets:
        raise HTTPException(status_code=409, detail="No training data is attached to this job.")

    configs = load_configs(paths["extract_dir"])
    encoding = value.get("target_encoding") or {}
    decisions = snapshot_service.SnapshotDecisions(
        date_column=value["date_column"],
        target_mode=value["target_mode"],
        target_column=value.get("target_column"),
        target_encoding=encoding,
        dedup_mode=value["dedup_mode"],
        dedup_keys=value.get("dedup_keys", []),
        subtask_mappings=value.get("subtask_mappings") or configs.subtask_mappings,
        subtask_keywords=value.get("subtask_keywords") or configs.subtask_keywords,
        allow_unmapped_default=bool(value.get("allow_unmapped_default")),
        historical_window_days=value.get("historical_window_days"),
        date_from=value.get("date_from"),
        date_to=value.get("date_to"),
        approver=value["approver"],
    )

    snapshot_id = f"SNAP_{uuid.uuid4().hex[:10].upper()}"
    fingerprints = {
        name: sha256_bytes(path.read_bytes())[:16]
        for name, path in sorted(
            (p.name, p) for p in paths["extract_dir"].rglob("*.json")
        )
    }

    try:
        manifest = snapshot_service.build_snapshot(
            [(a["role"], Path(a["stored_path"])) for a in assets],
            decisions, paths["snapshot_dir"], snapshot_id, fingerprints,
        )
    except snapshot_service.SnapshotError as exc:
        record_audit(job_id, decisions.approver, "snapshot.failed", str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    set_decision(job_id, "active_snapshot", snapshot_id, approver=decisions.approver)
    update_job(job_id, status="SNAPSHOT_READY", current_stage="READINESS")
    record_audit(
        job_id, decisions.approver, "snapshot.created",
        f"{snapshot_id}: {manifest['row_counts']['final']} rows, sha256 {manifest['snapshot_sha256'][:16]}…",
        {"snapshot_id": snapshot_id, "row_counts": manifest["row_counts"]},
    )
    return manifest


@router.get("/{job_id}/snapshot")
def active_snapshot(job_id: str):
    require_job(job_id)
    stored = get_decision(job_id, "active_snapshot")
    if stored is None:
        raise HTTPException(status_code=404, detail="No snapshot has been built for this job yet.")
    paths = _job_paths(job_id)
    return snapshot_service.load_manifest(paths["snapshot_dir"], stored["value"])
