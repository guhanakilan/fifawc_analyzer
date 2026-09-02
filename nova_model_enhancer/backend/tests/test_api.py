"""API behaviour: real health, stage gating and refusal to fake success."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from nova_model_enhancer.backend.tests import fixtures


def test_health_is_computed_not_hardcoded(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert body["workspace_writable"] is True
    assert body["schema_version"] >= 3
    assert set(body["model_families_available"]) == {"rf", "gb", "lr", "xgb", "lgb"}


def test_unknown_job_is_a_404_everywhere(client):
    for path in (
        "/api/packages/jobs/RETRAIN_NOPE",
        "/api/training-data/RETRAIN_NOPE",
        "/api/readiness/RETRAIN_NOPE/review",
        "/api/audit/RETRAIN_NOPE",
    ):
        assert client.get(path).status_code == 404, path


def test_non_zip_upload_is_refused(client):
    response = client.post(
        "/api/packages/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert response.status_code == 422
    assert ".zip" in response.json()["detail"]


def _upload_invalid_package(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model/only_a_model.pkl", b"placeholder")
    buffer.seek(0)
    return client.post(
        "/api/packages/upload",
        files={"file": ("bad.zip", buffer, "application/zip")},
    )


def test_invalid_package_is_never_reported_as_ready(client):
    response = _upload_invalid_package(client)
    assert response.status_code == 200
    body = response.json()
    assert body["validation"]["valid"] is False
    assert body["status"] == "PACKAGE_INVALID"
    assert body["validation"]["blocking_failures"]
    statuses = {c["status"] for c in body["validation"]["checks"] if c["blocking"]}
    assert "failed" in statuses


def test_invalid_package_cannot_accept_training_data(client):
    job_id = _upload_invalid_package(client).json()["job_id"]
    buffer = io.BytesIO()
    pd.DataFrame({"a": [1]}).to_parquet(buffer, index=False)
    buffer.seek(0)
    response = client.post(
        f"/api/training-data/{job_id}/upload",
        files={"file": ("d.parquet", buffer, "application/octet-stream")},
        data={"role": "combined"},
    )
    assert response.status_code == 409


def test_model_load_requires_explicit_trust(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]

    refused = client.post(
        f"/api/packages/jobs/{job_id}/compatibility",
        json={"trust_local_package": False},
    )
    assert refused.status_code == 400
    assert "trust" in refused.json()["detail"].lower()

    accepted = client.post(
        f"/api/packages/jobs/{job_id}/compatibility",
        json={"trust_local_package": True, "actor": "tester"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["feature_count"] > 0
    return job_id


def test_stages_are_gated_in_order(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]

    # No data yet -> readiness refuses rather than showing an empty "ready" screen.
    assert client.get(f"/api/readiness/{job_id}/review").status_code == 409
    # No snapshot yet -> weights refuse.
    assert client.get(f"/api/weights/{job_id}/options").status_code == 409
    # No snapshot or weights -> training refuses.
    started = client.post(f"/api/training/{job_id}/start", json={})
    assert started.status_code == 409
    assert "snapshot" in started.json()["detail"].lower()


def test_snapshot_requires_saved_decisions(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    buffer = io.BytesIO()
    champion_export["raw"].to_parquet(buffer, index=False)
    buffer.seek(0)
    client.post(
        f"/api/training-data/{job_id}/upload",
        files={"file": ("d.parquet", buffer, "application/octet-stream")},
        data={"role": "combined"},
    )
    response = client.post(f"/api/readiness/{job_id}/snapshot")
    assert response.status_code == 409
    assert "decisions" in response.json()["detail"].lower()


def test_decisions_require_an_approver(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    response = client.post(f"/api/readiness/{job_id}/decisions", json={
        "date_column": "UpdatedDateTimeGMT", "target_mode": "derive_from_subtask",
        "dedup_mode": "full_row", "approver": "   ",
    })
    assert response.status_code == 422


def test_gate_is_unapproved_until_someone_saves_it(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    body = client.get(f"/api/comparison/{job_id}/gate").json()
    assert body["gate"]["approved"] is False
    assert body["approved_by"] is None
    assert "BLOCKED" in body["proposed_note"]


def test_ml_tag_is_blocked_and_states_the_required_decision(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    body = client.get(f"/api/export/{job_id}/ml-tag").json()
    assert body["blocked"] is True
    assert body["approved_config"] is None
    assert len(body["candidate_conventions"]) == 2


def test_ml_tag_rejects_an_ambiguous_encoding(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    response = client.post(f"/api/export/{job_id}/ml-tag", json={
        "column_name": "ml_tag", "voice_value": 1, "non_voice_value": 1, "approver": "tester",
    })
    assert response.status_code == 422


def test_export_requires_a_promotion_approval(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    response = client.post(f"/api/export/{job_id}/build",
                           json={"run_id": "RUN_NOPE", "candidate_id": "lgb_tuned"})
    assert response.status_code == 409
    assert "approval" in response.json()["detail"].lower()


def test_download_of_an_unknown_export_is_a_404(client):
    assert client.get("/api/export/download/EXP_NOPE").status_code == 404


def test_run_id_path_traversal_is_refused(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job_id = client.post(
            "/api/packages/upload",
            files={"file": ("plc984.zip", handle, "application/zip")},
        ).json()["job_id"]
    response = client.get(f"/api/training/{job_id}/runs/..%2F..%2Fetc")
    assert response.status_code in (400, 404, 422, 500)
    assert "etc" not in str(response.content).lower() or response.status_code != 200
