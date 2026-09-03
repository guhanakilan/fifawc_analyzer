"""SQLite metadata store with explicit, versioned migrations.

Large objects (packages, datasets, snapshots, models, exports) live on the
filesystem under the job directory. This database holds only metadata, status
and the audit trail.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import database_path, ensure_workspace

_WRITE_LOCK = threading.RLock()

SCHEMA_VERSION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextmanager
def connection():
    ensure_workspace()
    db = sqlite3.connect(database_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with _WRITE_LOCK:
            yield db
            db.commit()
    finally:
        db.close()


# ── Migrations ───────────────────────────────────────────────────────────────
# Each entry is (version, [statements]). Migrations are additive and idempotent
# where SQLite allows it; an existing workspace is never dropped or rewritten.

_MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        """
        CREATE TABLE IF NOT EXISTS retraining_jobs (
            job_id            TEXT PRIMARY KEY,
            placement_id      INTEGER,
            source_run_id     TEXT,
            source_model_id   TEXT,
            original_filename TEXT NOT NULL,
            package_sha256    TEXT NOT NULL,
            status            TEXT NOT NULL,
            current_stage     TEXT NOT NULL,
            validation_json   TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS training_assets (
            asset_id          TEXT PRIMARY KEY,
            job_id            TEXT NOT NULL,
            role              TEXT NOT NULL DEFAULT 'combined',
            original_filename TEXT NOT NULL,
            stored_path       TEXT NOT NULL,
            file_type         TEXT NOT NULL,
            sha256            TEXT NOT NULL DEFAULT '',
            rows_count        INTEGER NOT NULL,
            columns_count     INTEGER NOT NULL,
            summary_json      TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES retraining_jobs(job_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id     TEXT,
            actor      TEXT NOT NULL,
            action     TEXT NOT NULL,
            detail     TEXT NOT NULL DEFAULT '',
            payload    TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_audit_job ON audit_events(job_id, event_id)",
    ]),
    (2, [
        """
        CREATE TABLE IF NOT EXISTS job_decisions (
            job_id     TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            approver   TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (job_id, key),
            FOREIGN KEY(job_id) REFERENCES retraining_jobs(job_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS background_jobs (
            task_id      TEXT PRIMARY KEY,
            job_id       TEXT NOT NULL,
            kind         TEXT NOT NULL,
            status       TEXT NOT NULL,
            progress     REAL NOT NULL DEFAULT 0.0,
            message      TEXT NOT NULL DEFAULT '',
            request_json TEXT NOT NULL DEFAULT '{}',
            result_json  TEXT,
            error        TEXT,
            log_path     TEXT NOT NULL DEFAULT '',
            cancelled    INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES retraining_jobs(job_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_bg_job ON background_jobs(job_id, created_at)",
    ]),
    (3, [
        """
        CREATE TABLE IF NOT EXISTS exports (
            export_id     TEXT PRIMARY KEY,
            job_id        TEXT NOT NULL,
            version       INTEGER NOT NULL,
            model_id      TEXT NOT NULL,
            zip_path      TEXT NOT NULL,
            zip_sha256    TEXT NOT NULL,
            approval_json TEXT NOT NULL DEFAULT '{}',
            report_json   TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES retraining_jobs(job_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_exports_job ON exports(job_id, version)",
    ]),
]


def initialize_database() -> None:
    ensure_workspace()
    with connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = db.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for version, statements in _MIGRATIONS:
            if version <= current:
                continue
            for statement in statements:
                db.execute(statement)
            db.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def schema_version() -> int:
    with connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = db.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


# ── Jobs ─────────────────────────────────────────────────────────────────────

def create_job(record: dict) -> None:
    now = utc_now()
    with connection() as db:
        db.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, placement_id, source_run_id, source_model_id,
                original_filename, package_sha256, status, current_stage,
                validation_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["job_id"], record.get("placement_id"),
                record.get("source_run_id"), record.get("source_model_id"),
                record["original_filename"], record["package_sha256"],
                record["status"], record["current_stage"],
                json.dumps(record["validation"]), now, now,
            ),
        )


def _job_row_to_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["validation"] = json.loads(item.pop("validation_json"))
    return item


def get_job(job_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM retraining_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _job_row_to_dict(row) if row else None


def list_jobs(limit: int = 100) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM retraining_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_job_row_to_dict(r) for r in rows]


def job_overview(limit: int = 100) -> list[dict]:
    """Every job with the counts the home screen needs, in one pass.

    Read-only aggregation over the tables that already persist the work, so the
    home screen reflects real durable state rather than anything a browser kept.
    """
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM retraining_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        jobs = [_job_row_to_dict(r) for r in rows]
        if not jobs:
            return []
        ids = [j["job_id"] for j in jobs]
        marks = ",".join("?" * len(ids))

        tasks: dict[str, dict] = {j: {"total": 0, "complete": 0, "interrupted": 0, "running": 0}
                                  for j in ids}
        for row in db.execute(
            f"SELECT job_id, status, COUNT(*) AS n FROM background_jobs "
            f"WHERE job_id IN ({marks}) AND kind = 'training' GROUP BY job_id, status", ids
        ):
            bucket = tasks[row["job_id"]]
            bucket["total"] += row["n"]
            if row["status"] == "complete":
                bucket["complete"] += row["n"]
            elif row["status"] == "interrupted":
                bucket["interrupted"] += row["n"]
            elif row["status"] in ("queued", "running"):
                bucket["running"] += row["n"]

        exports = {j: 0 for j in ids}
        for row in db.execute(
            f"SELECT job_id, COUNT(*) AS n FROM exports WHERE job_id IN ({marks}) GROUP BY job_id", ids
        ):
            exports[row["job_id"]] = row["n"]

        datasets = {j: 0 for j in ids}
        for row in db.execute(
            f"SELECT job_id, COUNT(*) AS n FROM training_assets WHERE job_id IN ({marks}) GROUP BY job_id",
            ids,
        ):
            datasets[row["job_id"]] = row["n"]

    for job in jobs:
        counts = tasks[job["job_id"]]
        job["runs"] = counts
        job["export_count"] = exports[job["job_id"]]
        job["dataset_count"] = datasets[job["job_id"]]
        job["resumable"] = counts["interrupted"] > 0
    return jobs


def update_job(job_id: str, *, status: str | None = None, current_stage: str | None = None) -> None:
    sets, params = ["updated_at = ?"], [utc_now()]
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if current_stage is not None:
        sets.append("current_stage = ?")
        params.append(current_stage)
    params.append(job_id)
    with connection() as db:
        db.execute(f"UPDATE retraining_jobs SET {', '.join(sets)} WHERE job_id = ?", params)


# ── Training assets ──────────────────────────────────────────────────────────

def create_training_asset(record: dict) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO training_assets (
                asset_id, job_id, role, original_filename, stored_path, file_type,
                sha256, rows_count, columns_count, summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["asset_id"], record["job_id"], record.get("role", "combined"),
                record["original_filename"], record["stored_path"], record["file_type"],
                record.get("sha256", ""), record["rows_count"], record["columns_count"],
                json.dumps(record["summary"]), utc_now(),
            ),
        )


def get_training_assets(job_id: str) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM training_assets WHERE job_id = ? ORDER BY created_at ASC", (job_id,)
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["summary"] = json.loads(item.pop("summary_json"))
        out.append(item)
    return out


def get_training_asset(asset_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM training_assets WHERE asset_id = ?", (asset_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["summary"] = json.loads(item.pop("summary_json"))
    return item


def delete_training_asset(asset_id: str) -> None:
    with connection() as db:
        db.execute("DELETE FROM training_assets WHERE asset_id = ?", (asset_id,))


# ── Decisions (persisted human choices) ──────────────────────────────────────

def set_decision(job_id: str, key: str, value: Any, approver: str = "") -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO job_decisions (job_id, key, value, approver, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id, key) DO UPDATE SET
                value = excluded.value, approver = excluded.approver,
                updated_at = excluded.updated_at
            """,
            (job_id, key, json.dumps(value), approver, utc_now()),
        )


def get_decision(job_id: str, key: str) -> dict | None:
    with connection() as db:
        row = db.execute(
            "SELECT * FROM job_decisions WHERE job_id = ? AND key = ?", (job_id, key)
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["value"] = json.loads(item["value"])
    return item


def get_decisions(job_id: str) -> dict[str, dict]:
    with connection() as db:
        rows = db.execute("SELECT * FROM job_decisions WHERE job_id = ?", (job_id,)).fetchall()
    out = {}
    for row in rows:
        item = dict(row)
        item["value"] = json.loads(item["value"])
        out[item["key"]] = item
    return out


# ── Background tasks ─────────────────────────────────────────────────────────

def create_background_job(task_id: str, job_id: str, kind: str, request: dict, log_path: str) -> None:
    now = utc_now()
    with connection() as db:
        db.execute(
            """
            INSERT INTO background_jobs (
                task_id, job_id, kind, status, progress, message,
                request_json, log_path, created_at, updated_at
            ) VALUES (?, ?, ?, 'queued', 0.0, 'Queued', ?, ?, ?, ?)
            """,
            (task_id, job_id, kind, json.dumps(request), log_path, now, now),
        )


def update_background_job(
    task_id: str, *, status: str | None = None, progress: float | None = None,
    message: str | None = None, result: dict | None = None, error: str | None = None,
) -> None:
    sets, params = ["updated_at = ?"], [utc_now()]
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if progress is not None:
        sets.append("progress = ?")
        params.append(float(progress))
    if message is not None:
        sets.append("message = ?")
        params.append(message)
    if result is not None:
        sets.append("result_json = ?")
        params.append(json.dumps(result))
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    params.append(task_id)
    with connection() as db:
        db.execute(f"UPDATE background_jobs SET {', '.join(sets)} WHERE task_id = ?", params)


def request_cancel(task_id: str) -> bool:
    with connection() as db:
        cur = db.execute(
            "UPDATE background_jobs SET cancelled = 1, updated_at = ?, message = ? "
            "WHERE task_id = ? AND status IN ('queued','running')",
            (utc_now(), "Cancellation requested", task_id),
        )
    return cur.rowcount > 0


def is_cancelled(task_id: str) -> bool:
    with connection() as db:
        row = db.execute("SELECT cancelled FROM background_jobs WHERE task_id = ?", (task_id,)).fetchone()
    return bool(row and row["cancelled"])


def _bg_row(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["request"] = json.loads(item.pop("request_json") or "{}")
    raw_result = item.pop("result_json", None)
    item["result"] = json.loads(raw_result) if raw_result else None
    item["cancelled"] = bool(item["cancelled"])
    return item


def get_background_job(task_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM background_jobs WHERE task_id = ?", (task_id,)).fetchone()
    return _bg_row(row) if row else None


def list_background_jobs(job_id: str, kind: str | None = None) -> list[dict]:
    query = "SELECT * FROM background_jobs WHERE job_id = ?"
    params: list[Any] = [job_id]
    if kind:
        query += " AND kind = ?"
        params.append(kind)
    query += " ORDER BY created_at DESC"
    with connection() as db:
        rows = db.execute(query, params).fetchall()
    return [_bg_row(r) for r in rows]


def recover_orphaned_background_jobs() -> int:
    """Mark tasks left running by a backend restart as interrupted.

    The process that owned them is gone, so their in-memory thread is too. The
    UI must show that honestly rather than spinning on a task nobody is running.
    """
    with connection() as db:
        cur = db.execute(
            "UPDATE background_jobs SET status = 'interrupted', "
            "message = 'Backend restarted before this task finished.', updated_at = ? "
            "WHERE status IN ('queued','running')",
            (utc_now(),),
        )
    return cur.rowcount


# ── Exports ──────────────────────────────────────────────────────────────────

def create_export(record: dict) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO exports (
                export_id, job_id, version, model_id, zip_path, zip_sha256,
                approval_json, report_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["export_id"], record["job_id"], record["version"], record["model_id"],
                record["zip_path"], record["zip_sha256"],
                json.dumps(record.get("approval", {})), json.dumps(record.get("report", {})),
                utc_now(),
            ),
        )


def list_exports(job_id: str) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM exports WHERE job_id = ? ORDER BY version DESC", (job_id,)
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["approval"] = json.loads(item.pop("approval_json"))
        item["report"] = json.loads(item.pop("report_json"))
        out.append(item)
    return out


def get_export(export_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM exports WHERE export_id = ?", (export_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["approval"] = json.loads(item.pop("approval_json"))
    item["report"] = json.loads(item.pop("report_json"))
    return item


def next_export_version(job_id: str) -> int:
    with connection() as db:
        row = db.execute(
            "SELECT MAX(version) AS v FROM exports WHERE job_id = ?", (job_id,)
        ).fetchone()
    return int(row["v"] or 0) + 1


# ── Audit ────────────────────────────────────────────────────────────────────

def record_audit(job_id: str | None, actor: str, action: str, detail: str = "", payload: dict | None = None) -> None:
    with connection() as db:
        db.execute(
            "INSERT INTO audit_events (job_id, actor, action, detail, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, actor or "local-user", action, detail, json.dumps(payload or {}), utc_now()),
        )


def list_audit(job_id: str | None = None, limit: int = 500) -> list[dict]:
    query = "SELECT * FROM audit_events"
    params: list[Any] = []
    if job_id:
        query += " WHERE job_id = ?"
        params.append(job_id)
    query += " ORDER BY event_id DESC LIMIT ?"
    params.append(limit)
    with connection() as db:
        rows = db.execute(query, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        out.append(item)
    return out
