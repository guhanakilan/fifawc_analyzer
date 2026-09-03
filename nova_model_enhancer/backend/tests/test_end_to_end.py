"""One synthetic run through all seven stages, asserting the real guarantees."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nova_model_enhancer.backend.tests import fixtures

APPROVER = "qa.approver"


def _wait_for_task(client, task_id: str, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/api/training/tasks/{task_id}").json()
        if task["status"] in ("complete", "failed", "cancelled", "interrupted"):
            return task
        time.sleep(1.0)
    raise AssertionError(f"Task {task_id} did not finish within {timeout}s")


@pytest.fixture(scope="module")
def journey(champion_export, client_module):
    """Drive the whole workflow once; every test then asserts against its output."""
    client = client_module
    zip_path: Path = champion_export["zip_path"]

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    with zip_path.open("rb") as handle:
        response = client.post(
            "/api/packages/upload",
            files={"file": ("plc984_export.zip", handle, "application/zip")},
        )
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["validation"]["valid"] is True
    job_id = job["job_id"]

    compat = client.post(
        f"/api/packages/jobs/{job_id}/compatibility",
        json={"trust_local_package": True, "actor": APPROVER},
    )
    assert compat.status_code == 200, compat.text

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    raw = champion_export["raw"]
    buffer = io.BytesIO()
    raw.to_parquet(buffer, index=False)
    buffer.seek(0)
    upload = client.post(
        f"/api/training-data/{job_id}/upload",
        files={"file": ("plc984_labelled.parquet", buffer, "application/octet-stream")},
        data={"role": "combined"},
    )
    assert upload.status_code == 200, upload.text

    # ── Stage 3 ─────────────────────────────────────────────────────────────
    review = client.get(f"/api/readiness/{job_id}/review").json()
    assert review["requires_subtask_decision"] is False

    # The lineage report must reach the API, and a clean upload must not be
    # accused of missing a column it was never supposed to carry.
    lineage = review["column_lineage"]
    assert lineage["missing_required"] == [], lineage["missing_required"]
    assert lineage["layers"]["selected"] > 0
    assert all(
        not (c["derived"] and not c["present_in_upload"] and c["required"])
        for c in lineage["columns"]
    ), "a derived column was reported as a required upload"

    decisions = {
        "date_column": "UpdatedDateTimeGMT",
        "target_mode": "derive_from_subtask",
        "dedup_mode": "key_columns",
        "dedup_keys": ["AccountID"],
        "subtask_mappings": [{"name": n, "flag": f} for n, f in fixtures.SUBTASKS.items()],
        "subtask_keywords": fixtures.KEYWORDS,
        "allow_unmapped_default": False,
        "approver": APPROVER,
    }
    saved = client.post(f"/api/readiness/{job_id}/decisions", json=decisions)
    assert saved.status_code == 200, saved.text
    snapshot = client.post(f"/api/readiness/{job_id}/snapshot")
    assert snapshot.status_code == 200, snapshot.text
    manifest = snapshot.json()

    # ── Stage 4 ─────────────────────────────────────────────────────────────
    strategy = {
        "enabled": True, "cap": 5.0, "historical_base": 1.0, "normalise_mean_to_one": True,
        "components": {
            "recency": {"enabled": True, "weight": 1.5, "recent_days": 90},
            "human_correction": {"enabled": True, "weight": 3.0, "column": "HumanCorrected",
                                 "true_values": ["1", "true", "yes"]},
            "verified_error": {"enabled": True, "weight": 2.5, "column": "PreviousModelError",
                               "true_values": ["1", "true", "yes"]},
            "rare_subtask": {"enabled": True, "weight": 2.0, "max_share_pct": 1.0},
            "class_balance": {"enabled": False},
        },
    }
    preview = client.post(f"/api/weights/{job_id}/preview", json={"strategy": strategy})
    assert preview.status_code == 200, preview.text
    approve = client.post(
        f"/api/weights/{job_id}/approve",
        json={"strategy": strategy, "approver": APPROVER, "notes": "synthetic run"},
    )
    assert approve.status_code == 200, approve.text

    # ── Stage 5 ─────────────────────────────────────────────────────────────
    start = client.post(f"/api/training/{job_id}/start", json={
        "split": {"mode": "temporal", "train_pct": 70, "val_pct": 15, "test_pct": 15},
        "n_trials": 5, "n_jobs": 1, "seed": 42, "include_baseline": True,
        "run_backtest": True, "backtest_windows": 3, "actor": APPROVER,
    })
    assert start.status_code == 200, start.text
    task = _wait_for_task(client, start.json()["task_id"])
    assert task["status"] == "complete", task.get("error") or task["message"]
    run_id = start.json()["run_id"]

    # ── Stage 6 ─────────────────────────────────────────────────────────────
    gate = {
        "primary_metric": "f1", "min_primary_improvement_pct": -100.0,
        "protected_metrics": [{"metric": "recall", "max_regression_pct": 100.0}],
        "max_historical_primary_regression_pct": 100.0,
        "require_backtest_pass": False, "require_package_validation": False,
        "segment_column": "SubTask", "min_segment_rows": 50,
    }
    saved_gate = client.post(f"/api/comparison/{job_id}/gate",
                             json={"gate": gate, "approver": APPROVER})
    assert saved_gate.status_code == 200, saved_gate.text
    comparison = client.get(f"/api/comparison/{job_id}/runs/{run_id}")
    assert comparison.status_code == 200, comparison.text
    comparison = comparison.json()

    leader = comparison["leading_candidate"]
    promotion = client.post(f"/api/comparison/{job_id}/approve", json={
        "run_id": run_id, "candidate_id": leader, "decision": "APPROVED",
        "approver": APPROVER, "typed_confirmation": leader, "notes": "synthetic acceptance run",
    })
    assert promotion.status_code == 200, promotion.text

    # ── Stage 7 ─────────────────────────────────────────────────────────────
    inventory = fixtures.make_inventory_sample(rows=250)
    inv_buffer = io.BytesIO()
    inventory.to_parquet(inv_buffer, index=False)
    inv_buffer.seek(0)
    sample = client.post(
        f"/api/export/{job_id}/inventory-sample",
        files={"file": ("inventory_sample.parquet", inv_buffer, "application/octet-stream")},
    )
    assert sample.status_code == 200, sample.text

    blocked = client.post(f"/api/export/{job_id}/build",
                          json={"run_id": run_id, "candidate_id": leader})
    assert blocked.status_code == 409, "Export must be blocked until ml_tag is approved"

    tag = client.post(f"/api/export/{job_id}/ml-tag", json={
        "column_name": "ml_tag", "voice_value": 1, "non_voice_value": 0,
        "approver": APPROVER, "notes": "matches VoiceNonVoiceFlag",
    })
    assert tag.status_code == 200, tag.text

    built = client.post(f"/api/export/{job_id}/build",
                        json={"run_id": run_id, "candidate_id": leader, "actor": APPROVER})
    assert built.status_code == 200, built.text

    return {
        "client": client, "job_id": job_id, "run_id": run_id, "leader": leader,
        "manifest": manifest, "comparison": comparison, "export": built.json(),
        "inventory": inventory,
    }


# ── Stage assertions ─────────────────────────────────────────────────────────

def test_snapshot_is_frozen_and_labelled(journey):
    manifest = journey["manifest"]
    assert manifest["target"]["encoding"] == {"voice": 0, "non_voice": 1}
    assert manifest["row_counts"]["final"] > 0
    assert manifest["snapshot_sha256"]
    assert manifest["exclusions"]["deduplication_mode"] == "key_columns"
    assert manifest["label_stats"]["ignored_count"] > 0, "Ignore-mapped SubTasks must be dropped"


def test_split_has_no_future_leakage(journey):
    split = journey["comparison"]["split"]
    assert split["mode"] == "temporal"
    assert split["no_future_leakage"] is True
    assert split["train"]["rows"] > split["test"]["rows"]


def test_champion_and_challengers_share_the_benchmark(journey):
    comparison = journey["comparison"]
    benchmark_rows = comparison["benchmark"]["rows"]
    assert comparison["champion"]["test_metrics"]["rows"] == benchmark_rows
    trained = [c for c in comparison["candidates"].values() if "skipped" not in c]
    assert trained, "at least one challenger must train"
    for candidate in trained:
        assert candidate["test_metrics"]["rows"] == benchmark_rows


def test_gate_is_reported_explicitly(journey):
    gate_result = journey["comparison"]["gate_result"]
    assert gate_result["status"] in ("RECOMMENDED", "NOT_RECOMMENDED", "BLOCKED")
    assert gate_result["rules"], "the gate must show the rules it applied"


def test_export_contains_the_real_layout(journey):
    export = journey["export"]
    with zipfile.ZipFile(export["zip_path"]) as archive:
        names = set(archive.namelist())
    model_id = export["model_id"]
    for required in (
        f"model/{model_id}.pkl", "model/fitted_transforms.pkl",
        "config/feature_selection.json", "config/features_config.json",
        "scoring/threshold_config.json", "scoring/ml_tag_config.json",
        "metadata/training_results.json", "metadata/manifest.json",
        "metadata/validation_report.json", "metadata/dataset_manifest.json",
        "metadata/rollback_manifest.json", "pipeline/scoring.py",
        "reports/retraining_report.xlsx", "README.txt",
    ):
        assert required in names, f"{required} missing from the export"


def test_package_validation_passed_and_preserves_the_inventory(journey):
    validation = journey["export"]["validation"]
    assert validation["status"] == "passed", validation.get("failed_checks")
    by_key = {c["key"]: c for c in validation["checks"]}
    assert by_key["ml_tag_rows"]["status"] == "passed"
    assert by_key["ml_tag_columns"]["status"] == "passed"
    assert by_key["ml_tag_single"]["status"] == "passed"
    assert by_key["ml_tag_no_leak"]["status"] == "passed"
    assert by_key["ml_tag_inversion"]["status"] == "passed"
    assert by_key["agreement"]["status"] == "passed", by_key["agreement"]["detail"]


def test_previous_versions_are_retained(journey):
    client, job_id = journey["client"], journey["job_id"]
    exports = client.get(f"/api/export/{job_id}/exports").json()["exports"]
    assert exports and all(e["exists"] for e in exports)


def test_champion_package_is_untouched(journey, champion_export):
    """The uploaded ZIP must be byte-identical after the whole workflow."""
    from nova_model_enhancer.backend.services.package_validator import sha256_file

    client, job_id = journey["client"], journey["job_id"]
    job = client.get(f"/api/packages/jobs/{job_id}").json()
    assert job["package_sha256"] == sha256_file(champion_export["zip_path"])


def test_audit_trail_records_every_decision(journey):
    client, job_id = journey["client"], journey["job_id"]
    actions = {e["action"] for e in client.get(f"/api/audit/{job_id}").json()["events"]}
    for expected in (
        "package.upload", "package.compatibility.passed", "training-data.upload",
        "readiness.decisions", "snapshot.created", "weights.approved",
        "training.started", "gate.approved", "promotion.approved",
        "ml_tag.approved", "export.built",
    ):
        assert expected in actions, f"{expected} missing from the audit trail"
