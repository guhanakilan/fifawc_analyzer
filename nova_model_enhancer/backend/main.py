"""NoVA Model Enhancer API.

Localhost-first. Open-source dependencies only; no external service is contacted
at any point in the workflow.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import APP_VERSION, DEV_ORIGINS, ensure_workspace, workspace_root
from .database import initialize_database, recover_orphaned_background_jobs, schema_version
from .routers.audit import router as audit_router
from .routers.comparison import router as comparison_router
from .routers.export import router as export_router
from .routers.packages import router as packages_router
from .routers.readiness import router as readiness_router
from .routers.rules import router as rules_router
from .routers.training import router as training_router
from .routers.training_data import router as training_data_router
from .routers.weights import router as weights_router
from .schemas import HealthResponse
from .services.trainer import available_model_types

logging.basicConfig(
    level=os.environ.get("NOVA_ENHANCER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("nova_enhancer")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_workspace()
    initialize_database()
    orphaned = recover_orphaned_background_jobs()
    if orphaned:
        logger.warning(
            "%d background task(s) were running when the backend last stopped; "
            "they are marked 'interrupted'.", orphaned,
        )
    logger.info("Workspace: %s", workspace_root())
    yield


app = FastAPI(
    title="NoVA Model Enhancer API",
    version=APP_VERSION,
    description=(
        "Champion-to-challenger retraining for a completed NoVA ML run. "
        "Promotion is always a recommendation followed by manual approval."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

for router in (
    packages_router, training_data_router, readiness_router, weights_router,
    rules_router, training_router, comparison_router, export_router, audit_router,
):
    app.include_router(router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """A real health check: the workspace is probed and the schema is read."""
    writable = False
    detail = ""
    try:
        ensure_workspace()
        handle, path = tempfile.mkstemp(dir=str(workspace_root()))
        os.close(handle)
        os.unlink(path)
        writable = True
    except Exception as exc:
        detail = f"Workspace is not writable: {exc}"

    version = 0
    try:
        version = schema_version()
    except Exception as exc:
        detail = (detail + " " if detail else "") + f"Database unavailable: {exc}"

    families = available_model_types()
    unavailable = [k for k, ok in families.items() if not ok]
    if unavailable and not detail:
        detail = f"Model families unavailable in this environment: {', '.join(unavailable)}"

    ok = writable and version > 0
    return HealthResponse(
        status="ok" if ok else "degraded",
        application="NoVA Model Enhancer",
        version=APP_VERSION,
        schema_version=version,
        workspace_writable=writable,
        model_families_available=families,
        detail=detail,
    )
