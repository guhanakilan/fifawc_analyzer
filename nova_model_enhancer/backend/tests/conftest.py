"""Test bootstrap.

The workspace environment variable is set at *import* time, before any
application module can be imported, so a test run can never touch the
application's own workspace directory.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WORKSPACE = tempfile.TemporaryDirectory(prefix="nova_enhancer_tests_")
os.environ["NOVA_ENHANCER_WORKSPACE"] = _WORKSPACE.name


@pytest.fixture(scope="session")
def _isolated_workspace():
    yield Path(_WORKSPACE.name)


def pytest_sessionfinish(session, exitstatus):
    _WORKSPACE.cleanup()


@pytest.fixture(scope="session")
def champion_export(_isolated_workspace):
    from nova_model_enhancer.backend.tests import fixtures
    path = _isolated_workspace / "fixtures" / "plc984_export.zip"
    return fixtures.write_champion_export(path, rows=3000)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from nova_model_enhancer.backend.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def client_module():
    """Module-scoped client so a single journey can drive every stage once."""
    from fastapi.testclient import TestClient

    from nova_model_enhancer.backend.main import app
    with TestClient(app) as test_client:
        yield test_client
