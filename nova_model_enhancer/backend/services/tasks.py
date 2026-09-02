"""Background task runner with SQLite-persisted state.

State lives in the database, not in process memory, so the UI recovers after a
browser refresh and reports honestly after a backend restart (tasks that were
running when the process died are marked `interrupted`, never left spinning).
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable

from .. import database
from ..config import job_dir

_THREADS: dict[str, threading.Thread] = {}


def new_task_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12].upper()}"


class TaskContext:
    """Handed to the worker: progress, logging and cancellation."""

    def __init__(self, task_id: str, log_path: Path):
        self.task_id = task_id
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        from ..database import utc_now
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()}  {message}\n")

    def progress(self, fraction: float, message: str) -> None:
        database.update_background_job(
            self.task_id, status="running",
            progress=max(0.0, min(1.0, float(fraction))), message=message,
        )
        self.log(message)

    def cancelled(self) -> bool:
        return database.is_cancelled(self.task_id)


def start_task(job_id: str, kind: str, request: dict, worker: Callable[[TaskContext], dict]) -> str:
    """Register and start a task. Returns its id immediately."""
    task_id = new_task_id(kind.upper())
    log_path = job_dir(job_id) / "logs" / f"{task_id}.log"
    database.create_background_job(task_id, job_id, kind, request, str(log_path))
    context = TaskContext(task_id, log_path)
    context.log(f"Task {task_id} ({kind}) queued for job {job_id}")

    def _run() -> None:
        try:
            database.update_background_job(task_id, status="running", progress=0.0, message="Starting")
            result = worker(context)
            if context.cancelled():
                database.update_background_job(
                    task_id, status="cancelled", progress=1.0,
                    message="Cancelled by user request",
                )
                context.log("Task cancelled by user request")
                return
            database.update_background_job(
                task_id, status="complete", progress=1.0,
                message="Complete", result=result,
            )
            context.log("Task complete")
        except Exception as exc:  # noqa: BLE001 — the message is the user-facing error
            if context.cancelled():
                database.update_background_job(
                    task_id, status="cancelled", progress=1.0,
                    message="Cancelled by user request",
                )
                context.log(f"Task cancelled during: {exc}")
                return
            detail = traceback.format_exc()
            context.log(f"Task failed: {exc}\n{detail}")
            database.update_background_job(
                task_id, status="failed", progress=1.0,
                message=str(exc)[:400], error=str(exc)[:2000],
            )

    thread = threading.Thread(target=_run, name=f"task-{task_id}", daemon=True)
    _THREADS[task_id] = thread
    thread.start()
    return task_id


def read_log(task_id: str, tail_lines: int = 400) -> list[str]:
    record = database.get_background_job(task_id)
    if not record or not record.get("log_path"):
        return []
    path = Path(record["log_path"])
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-tail_lines:]
