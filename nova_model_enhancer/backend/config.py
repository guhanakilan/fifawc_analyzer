"""Application paths, limits and tunables.

Everything the enhancer writes lives under WORKSPACE_ROOT. Nothing outside it is
ever created, moved or deleted by the application.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

# The workspace can be relocated (e.g. to a larger drive) without code changes.
# Resolved on every access rather than bound at import time: a value captured at
# import would silently ignore an environment set up afterwards, which is exactly
# how a test run ends up writing into the application directory.
DEFAULT_WORKSPACE = APP_ROOT / "workspace"


def workspace_root() -> Path:
    return Path(os.environ.get("NOVA_ENHANCER_WORKSPACE") or DEFAULT_WORKSPACE).resolve()


def jobs_root() -> Path:
    return workspace_root() / "jobs"


def database_path() -> Path:
    return workspace_root() / "enhancer.sqlite3"

# ── Upload limits ────────────────────────────────────────────────────────────
MAX_ZIP_BYTES = 500 * 1024 * 1024            # champion package
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024  # zip-bomb ceiling
MAX_FILES = 300                               # archive member ceiling
MAX_DATA_BYTES = 2 * 1024 * 1024 * 1024       # training data
MAX_INVENTORY_BYTES = 512 * 1024 * 1024       # scoring compatibility sample

# Rows read per chunk when profiling large delimited files.
PROFILE_CHUNK_ROWS = 200_000

# ── Training defaults (mirrors the reference stack) ──────────────────────────
SUPPORTED_MODEL_TYPES = ("xgb", "lgb", "rf", "gb", "lr")
DEFAULT_SEED = 42

# ── CORS ────────────────────────────────────────────────────────────────────
DEV_ORIGINS = [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

APP_VERSION = "1.0.0"
PIPELINE_VERSION = 2   # matches nova-ml pipeline_version.json (bucket/grouping included)


def ensure_workspace() -> None:
    workspace_root().mkdir(parents=True, exist_ok=True)
    jobs_root().mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    """Resolve a job directory from an *id we generated*, never a client path."""
    from .services.safety import assert_safe_id

    assert_safe_id(job_id)
    return jobs_root() / job_id
