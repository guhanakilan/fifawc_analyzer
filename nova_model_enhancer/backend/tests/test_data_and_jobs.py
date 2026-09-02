"""Data intake, snapshot immutability, background-job lifecycle and determinism."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nova_model_enhancer.backend import database
from nova_model_enhancer.backend.services import snapshot as snapshot_service
from nova_model_enhancer.backend.services import tasks, trainer
from nova_model_enhancer.backend.services.data_profiler import (
    DataReadError,
    drift_report,
    iter_chunks,
    profile_dataset,
    read_dataset,
)
from nova_model_enhancer.backend.services.safety import UnsafeIdentifier, assert_safe_id, safe_filename


def _frame(rows: int = 500) -> pd.DataFrame:
    return pd.DataFrame({
        "AccountID": [f"A{i}" for i in range(rows)],
        "SubTask": np.resize(["Voice A", "NV A"], rows),
        "ARComments": ["called payer"] * rows,
        "Amount": np.arange(rows, dtype=float),
        "UpdatedDateTimeGMT": pd.date_range("2025-01-01", periods=rows, freq="h"),
    })


# ── File formats ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".xlsx"])
def test_every_supported_format_round_trips(tmp_path, suffix):
    frame = _frame(200)
    path = tmp_path / f"data{suffix}"
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False)
    profile = profile_dataset(path)
    assert profile["rows"] == 200
    assert profile["date_column_detected"] == "UpdatedDateTimeGMT"
    assert profile["subtask_column_detected"] == "SubTask"


def test_large_csv_is_profiled_in_chunks(tmp_path):
    """A chunked read must produce the same totals as a single read."""
    frame = _frame(5000)
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    chunks = list(iter_chunks(path, chunk_rows=500))
    assert len(chunks) == 10, "the reader must actually chunk rather than slurping the file"
    profile = profile_dataset(path, chunk_rows=500)
    assert profile["rows"] == 5000
    assert profile["min_date"] and profile["max_date"]


def test_duplicate_rows_are_counted_across_chunk_boundaries(tmp_path):
    frame = pd.concat([_frame(100), _frame(100)], ignore_index=True)
    path = tmp_path / "dupes.csv"
    frame.to_csv(path, index=False)
    assert profile_dataset(path, chunk_rows=50)["duplicate_rows"] == 100


def test_malformed_csv_reports_a_usable_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6,7,8\n", encoding="utf-8")
    with pytest.raises(DataReadError, match="malformed"):
        read_dataset(path)


def test_non_utf8_csv_is_decoded_rather_than_crashing(tmp_path):
    path = tmp_path / "cp1252.csv"
    path.write_bytes("name,value\nCaf\xe9,1\n".encode("cp1252"))
    frame = read_dataset(path)
    assert len(frame) == 1


def test_unsupported_extension_is_refused(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(DataReadError, match="Supported formats"):
        read_dataset(path)


def test_schema_drift_reports_both_directions():
    profile = {"column_names": ["AmountBilled", "NewColumn"]}
    drift = drift_report(profile, ["amountbilled", "insurancebalance"])
    assert drift["missing_columns"] == ["insurancebalance"]
    assert drift["new_columns"] == ["newcolumn"]


# ── Snapshot ─────────────────────────────────────────────────────────────────

def _decisions(**overrides) -> snapshot_service.SnapshotDecisions:
    base = dict(
        date_column="UpdatedDateTimeGMT", target_mode="derive_from_subtask",
        target_column=None, target_encoding={}, dedup_mode="full_row", dedup_keys=[],
        subtask_mappings=[{"name": "Voice A", "flag": "Voice"}, {"name": "NV A", "flag": "Non-Voice"}],
        subtask_keywords=["called"], allow_unmapped_default=False,
        historical_window_days=None, approver="tester",
    )
    base.update(overrides)
    return snapshot_service.SnapshotDecisions(**base)


def test_snapshot_is_written_once_and_hashed(tmp_path):
    path = tmp_path / "data.parquet"
    _frame(400).to_parquet(path, index=False)
    manifest = snapshot_service.build_snapshot(
        [("combined", path)], _decisions(), tmp_path / "snaps", "SNAP_TEST", {"cfg": "abc"}
    )
    parquet = tmp_path / "snaps" / "SNAP_TEST.parquet"
    assert parquet.exists()
    assert manifest["snapshot_sha256"]
    assert manifest["target"]["encoding"] == {"voice": 0, "non_voice": 1}
    assert manifest["row_counts"]["final"] == 400

    # Reloading returns identical content — the snapshot is inert.
    reloaded = snapshot_service.load_snapshot(tmp_path / "snaps", "SNAP_TEST")
    assert len(reloaded) == 400
    from nova_model_enhancer.backend.services.data_profiler import sha256_file
    assert sha256_file(parquet) == manifest["snapshot_sha256"]


def test_snapshot_deduplicates_by_the_approved_key(tmp_path):
    frame = pd.concat([_frame(100), _frame(100)], ignore_index=True)
    path = tmp_path / "dupes.parquet"
    frame.to_parquet(path, index=False)
    manifest = snapshot_service.build_snapshot(
        [("combined", path)], _decisions(dedup_mode="key_columns", dedup_keys=["AccountID"]),
        tmp_path / "snaps", "SNAP_KEY", {},
    )
    assert manifest["row_counts"]["final"] == 100
    assert manifest["exclusions"]["duplicate_rows_removed"] == 100


def test_snapshot_refuses_an_unknown_deduplication_key(tmp_path):
    path = tmp_path / "d.parquet"
    _frame(50).to_parquet(path, index=False)
    with pytest.raises(snapshot_service.SnapshotError, match="not present"):
        snapshot_service.build_snapshot(
            [("combined", path)], _decisions(dedup_mode="key_columns", dedup_keys=["NoSuchColumn"]),
            tmp_path / "snaps", "SNAP_BAD", {},
        )


def test_snapshot_refuses_a_single_class_dataset(tmp_path):
    frame = _frame(100)
    frame["SubTask"] = "Voice A"
    path = tmp_path / "onecls.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(snapshot_service.SnapshotError, match="only one class"):
        snapshot_service.build_snapshot(
            [("combined", path)], _decisions(), tmp_path / "snaps", "SNAP_ONE", {}
        )


def test_snapshot_refuses_labels_outside_the_approved_encoding(tmp_path):
    frame = _frame(100)
    frame["Label"] = np.resize(["0", "1", "maybe"], 100)
    path = tmp_path / "labels.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(snapshot_service.SnapshotError, match="outside the approved encoding"):
        snapshot_service.build_snapshot(
            [("combined", path)],
            _decisions(target_mode="existing", target_column="Label",
                       target_encoding={"voice_values": ["0"], "non_voice_values": ["1"]}),
            tmp_path / "snaps", "SNAP_ENC", {},
        )


def test_snapshot_honours_the_historical_window(tmp_path):
    path = tmp_path / "win.parquet"
    _frame(1000).to_parquet(path, index=False)   # 1000 hours ≈ 41 days
    manifest = snapshot_service.build_snapshot(
        [("combined", path)], _decisions(historical_window_days=10),
        tmp_path / "snaps", "SNAP_WIN", {},
    )
    assert manifest["exclusions"]["rows_outside_window"] > 0
    assert manifest["row_counts"]["final"] < 1000


def test_atomic_write_leaves_no_partial_file(tmp_path):
    destination = tmp_path / "nested" / "out.parquet"
    snapshot_service.atomic_write_parquet(_frame(10), destination)
    assert destination.exists()
    assert not list(tmp_path.rglob("*.tmp"))


# ── Background jobs ──────────────────────────────────────────────────────────

def test_task_state_survives_the_client_and_reports_completion(_isolated_workspace):
    database.initialize_database()
    database.create_job({
        "job_id": "RETRAIN_TASKTEST", "placement_id": 984, "source_run_id": None,
        "source_model_id": None, "original_filename": "x.zip", "package_sha256": "abc",
        "status": "PACKAGE_READY", "current_stage": "CHAMPION_PACKAGE", "validation": {},
    })

    def worker(context):
        context.progress(0.5, "halfway")
        return {"answer": 42}

    task_id = tasks.start_task("RETRAIN_TASKTEST", "training", {}, worker)
    for _ in range(100):
        record = database.get_background_job(task_id)
        if record["status"] == "complete":
            break
        time.sleep(0.05)
    assert record["status"] == "complete"
    assert record["result"] == {"answer": 42}
    assert any("halfway" in line for line in tasks.read_log(task_id))


def test_cancellation_is_observed_by_the_worker(_isolated_workspace):
    database.initialize_database()
    database.create_job({
        "job_id": "RETRAIN_CANCELTEST", "placement_id": 984, "source_run_id": None,
        "source_model_id": None, "original_filename": "x.zip", "package_sha256": "abc",
        "status": "PACKAGE_READY", "current_stage": "CHAMPION_PACKAGE", "validation": {},
    })

    started = {"value": False}

    def worker(context):
        started["value"] = True
        for _ in range(200):
            if context.cancelled():
                raise trainer.TrainingCancelled("cancelled")
            time.sleep(0.02)
        return {"finished": True}

    task_id = tasks.start_task("RETRAIN_CANCELTEST", "training", {}, worker)
    for _ in range(100):
        if started["value"]:
            break
        time.sleep(0.02)
    assert database.request_cancel(task_id) is True
    for _ in range(200):
        record = database.get_background_job(task_id)
        if record["status"] in ("cancelled", "complete", "failed"):
            break
        time.sleep(0.05)
    assert record["status"] == "cancelled"


def test_a_backend_restart_marks_running_tasks_interrupted(_isolated_workspace):
    database.initialize_database()
    database.create_job({
        "job_id": "RETRAIN_ORPHAN", "placement_id": 984, "source_run_id": None,
        "source_model_id": None, "original_filename": "x.zip", "package_sha256": "abc",
        "status": "PACKAGE_READY", "current_stage": "CHAMPION_PACKAGE", "validation": {},
    })
    database.create_background_job("TASK_ORPHAN", "RETRAIN_ORPHAN", "training", {}, "")
    database.update_background_job("TASK_ORPHAN", status="running", progress=0.4)
    assert database.recover_orphaned_background_jobs() >= 1
    record = database.get_background_job("TASK_ORPHAN")
    assert record["status"] == "interrupted"
    assert "restarted" in record["message"]


def test_failed_worker_reports_its_reason(_isolated_workspace):
    database.initialize_database()
    database.create_job({
        "job_id": "RETRAIN_FAILTEST", "placement_id": 984, "source_run_id": None,
        "source_model_id": None, "original_filename": "x.zip", "package_sha256": "abc",
        "status": "PACKAGE_READY", "current_stage": "CHAMPION_PACKAGE", "validation": {},
    })

    def worker(context):
        raise ValueError("the snapshot has only one class")

    task_id = tasks.start_task("RETRAIN_FAILTEST", "training", {}, worker)
    for _ in range(100):
        record = database.get_background_job(task_id)
        if record["status"] == "failed":
            break
        time.sleep(0.05)
    assert record["status"] == "failed"
    assert "only one class" in record["error"]


# ── Determinism ──────────────────────────────────────────────────────────────

def test_training_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + rng.normal(0, 0.4, 400) > 0).astype(int))

    def _run():
        return trainer.train_candidate(
            model_type="rf", mode="fixed",
            X_train=X.iloc[:300], y_train=y.iloc[:300], w_train=None,
            X_val=None, y_val=None, X_test=X.iloc[300:], y_test=y.iloc[300:],
            feature_names=list(X.columns),
            fixed_params={"n_estimators": 40, "max_depth": 4}, n_jobs=1, seed=42,
        )

    first, second = _run(), _run()
    assert np.array_equal(first["_proba_test"], second["_proba_test"])
    assert first["cv_mean"] == second["cv_mean"]


# ── Path safety ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "job id", "x" * 100, "job;rm -rf"])
def test_unsafe_identifiers_are_refused(bad):
    with pytest.raises(UnsafeIdentifier):
        assert_safe_id(bad)


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"),
    ("C:\\Users\\x\\data.parquet", "C__Users_x_data.parquet"),
    ("CON.txt", "upload"),
    ("", "upload"),
])
def test_filenames_are_reduced_to_a_safe_basename(raw, expected):
    assert safe_filename(raw) == expected
