"""Item 10 — one operator identity in place of four name boxes.

The audit trail is the thing that must not change: this alters who has to type
and how often, not what gets recorded against a decision.
"""

from __future__ import annotations

import io

import pytest
from backend.tests import fixtures


@pytest.fixture
def job(client, champion_export):
    with open(champion_export["zip_path"], "rb") as fh:
        return client.post(
            "/api/packages/upload", files={"file": ("p.zip", fh, "application/zip")}
        ).json()["job_id"]


def test_operator_starts_empty_and_persists_once_set(client, job):
    assert client.get(f"/api/packages/jobs/{job}/operator").json()["operator"] == ""

    saved = client.post(f"/api/packages/jobs/{job}/operator", json={"name": "  Guhan A  "})
    assert saved.status_code == 200
    assert saved.json()["operator"] == "Guhan A", "the name must be trimmed"

    assert client.get(f"/api/packages/jobs/{job}/operator").json()["operator"] == "Guhan A"


def test_an_empty_name_is_refused(client, job):
    for blank in ["", "   "]:
        response = client.post(f"/api/packages/jobs/{job}/operator", json={"name": blank})
        assert response.status_code == 422


def test_setting_the_operator_is_audited(client, job):
    client.post(f"/api/packages/jobs/{job}/operator", json={"name": "Guhan A"})
    events = client.get(f"/api/audit/{job}").json()["events"]
    assert any(e["action"] == "operator.set" and e["actor"] == "Guhan A" for e in events)


def test_each_decision_still_records_its_own_approver(client, job):
    """Merging the input must not merge the record.

    Every decision keeps its own approver and timestamp; the shared identity
    only supplies the value, so an audit still shows who approved what and when.
    """
    client.post(f"/api/packages/jobs/{job}/operator", json={"name": "Guhan A"})

    df = fixtures.make_labelled_dataset(rows=800)
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    client.post(
        f"/api/training-data/{job}/upload",
        files={"file": ("d.parquet", buffer, "application/octet-stream")},
        data={"role": "combined"},
    )
    client.post(f"/api/readiness/{job}/decisions", json={
        "date_column": "UpdatedDateTimeGMT",
        "target_mode": "derive_from_subtask",
        "dedup_mode": "none",
        "dedup_keys": [],
        "subtask_mappings": [{"name": n, "flag": f} for n, f in fixtures.SUBTASKS.items()],
        "subtask_keywords": fixtures.KEYWORDS,
        "approver": "Guhan A",
    })

    events = client.get(f"/api/audit/{job}").json()["events"]
    readiness = [e for e in events if e["action"] == "readiness.decisions"]
    assert readiness, "the readiness decision must still be audited in its own right"
    assert readiness[0]["actor"] == "Guhan A"
    assert readiness[0]["created_at"]
