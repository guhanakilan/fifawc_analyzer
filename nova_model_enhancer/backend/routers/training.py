"""Stage 5 — background challenger training and tuning."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import job_dir
from ..database import (
    get_decision,
    list_background_jobs,
    get_background_job,
    record_audit,
    request_cancel,
    update_job,
)
from ..schemas import TrainingRequest
from ..services import pipeline, tasks, trainer
from ..services.champion import load_configs
from .packages import require_job
from .readiness import _job_paths

router = APIRouter(prefix="/api/training", tags=["5 · Retrain & tune"])


@router.get("/{job_id}/options")
def options(job_id: str):
    """What can actually be trained here, and with which champion parameters."""
    job = require_job(job_id)
    paths = _job_paths(job_id)
    configs = load_configs(paths["extract_dir"])
    champion_id = configs.champion_model_id
    champion_result = (configs.training_results.get("results") or {}).get(champion_id, {})
    available = trainer.available_model_types()
    champion_family = champion_result.get("model_type")

    plan = trainer.build_candidate_plan(
        champion_family=champion_family,
        champion_params=champion_result.get("best_params"),
        available=available,
    )
    weight = get_decision(job_id, "weight_strategy")
    snapshot = get_decision(job_id, "active_snapshot")
    return {
        "available_model_types": available,
        "unavailable_note": {
            key: f"{key} cannot be trained: its library is not installed in this environment."
            for key, ok in available.items() if not ok
        },
        "champion_model_id": champion_id,
        "champion_family": champion_family,
        "champion_params": champion_result.get("best_params"),
        "champion_threshold": configs.champion_threshold,
        "default_candidate_plan": plan,
        "default_search_spaces": trainer.DEFAULT_SEARCH_SPACES,
        "weight_strategy_approved": weight is not None,
        "snapshot_ready": snapshot is not None,
        "proposed_split": {"mode": "temporal", "train_pct": 70, "val_pct": 15, "test_pct": 15},
        "split_note": "70/15/15 is the proposal in the project brief, not an approved rule.",
    }


@router.post("/{job_id}/start")
def start(job_id: str, request: TrainingRequest):
    """Queue a retraining run. Returns the task id immediately."""
    require_job(job_id)
    snapshot = get_decision(job_id, "active_snapshot")
    if snapshot is None:
        raise HTTPException(status_code=409, detail="Build a dataset snapshot before training.")
    weight = get_decision(job_id, "weight_strategy")
    if weight is None:
        raise HTTPException(
            status_code=409,
            detail="Approve a sample-weight strategy in Stage 4 before training "
                   "(an explicit 'no weighting' strategy is a valid approval).",
        )

    running = [
        t for t in list_background_jobs(job_id, kind="training")
        if t["status"] in ("queued", "running")
    ]
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Training task {running[0]['task_id']} is already running for this job.",
        )

    paths = _job_paths(job_id)
    configs = load_configs(paths["extract_dir"])
    champion_id = configs.champion_model_id
    champion_result = (configs.training_results.get("results") or {}).get(champion_id, {})

    run_id = f"RUN_{uuid.uuid4().hex[:10].upper()}"
    settings = {
        "run_id": run_id,
        "snapshot_id": snapshot["value"],
        "split": request.split.model_dump(),
        "weight_strategy": weight["value"]["strategy"],
        "n_trials": request.n_trials,
        "timeout_seconds": request.timeout_seconds,
        "n_jobs": request.n_jobs,
        "seed": request.seed,
        "second_family": request.second_family,
        "include_baseline": request.include_baseline,
        "run_backtest": request.run_backtest,
        "backtest_windows": request.backtest_windows,
        "threshold_criterion": request.threshold_criterion,
        "champion_params": champion_result.get("best_params"),
    }
    run_paths = {
        "run_dir": str(paths["runs_dir"] / run_id),
        "extract_dir": str(paths["extract_dir"]),
        "snapshot_dir": str(paths["snapshot_dir"]),
    }

    def worker(context):
        return pipeline.run_retraining(context, job_id, settings, run_paths)

    task_id = tasks.start_task(job_id, "training", {"run_id": run_id, **request.model_dump()}, worker)
    update_job(job_id, status="TRAINING_RUNNING", current_stage="RETRAIN_TUNE")
    record_audit(
        job_id, request.actor or "local-user", "training.started",
        f"run {run_id}, seed {request.seed}, split {request.split.mode}",
        {"task_id": task_id, "settings": {k: v for k, v in settings.items() if k != "champion_params"}},
    )
    return {"task_id": task_id, "run_id": run_id, "status": "queued"}


@router.get("/{job_id}/tasks")
def job_tasks(job_id: str):
    require_job(job_id)
    return {"tasks": list_background_jobs(job_id)}


@router.get("/tasks/{task_id}")
def task_status(task_id: str):
    """Live task state, read from the database so a refresh recovers it."""
    task = get_background_job(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@router.get("/tasks/{task_id}/log")
def task_log(task_id: str, tail: int = 300):
    if get_background_job(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"task_id": task_id, "lines": tasks.read_log(task_id, tail_lines=tail)}


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    task = get_background_job(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if not request_cancel(task_id):
        raise HTTPException(
            status_code=409,
            detail=f"Task is {task['status']} and can no longer be cancelled.",
        )
    record_audit(task["job_id"], "local-user", "training.cancel_requested", task_id)
    return {"cancelled": True, "task_id": task_id,
            "note": "The worker stops at its next checkpoint; artifacts already written are kept."}


@router.get("/{job_id}/runs")
def runs(job_id: str):
    """Completed runs, newest first, read from the artifacts on disk."""
    require_job(job_id)
    runs_dir = _job_paths(job_id)["runs_dir"]
    out = []
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*/run_results.json"), reverse=True):
            record = json.loads(path.read_text(encoding="utf-8"))
            trained = [k for k, v in (record.get("challengers") or {}).items() if "skipped" not in v]
            out.append({
                "run_id": record.get("run_id"),
                "created_at": record.get("created_at"),
                "snapshot_id": record.get("snapshot_id"),
                "candidates_trained": trained,
                "champion_model_id": (record.get("champion") or {}).get("model_id"),
                "feature_count": record.get("feature_count"),
                "split_mode": (record.get("split") or {}).get("mode"),
            })
    return {"runs": out}


@router.get("/{job_id}/runs/{run_id}")
def run_detail(job_id: str, run_id: str):
    require_job(job_id)
    from ..services.safety import assert_safe_id

    assert_safe_id(run_id)
    path = _job_paths(job_id)["runs_dir"] / run_id / "run_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    return json.loads(path.read_text(encoding="utf-8"))
