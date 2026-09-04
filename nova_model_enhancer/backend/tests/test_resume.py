"""Item 1 — checkpointing and resume after an interrupted training run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.services.pipeline import (
    _json_safe,
    load_candidate_checkpoints,
    write_candidate_checkpoint,
)


def _finished_candidate(run_dir, candidate_id="lgb_tuned"):
    """Write the artifacts a completed candidate leaves behind."""
    models = run_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / f"{candidate_id}.pkl").write_bytes(b"estimator")
    np.save(run_dir / f"proba_test_{candidate_id}.npy", np.array([0.1, 0.9]))
    return {
        "candidate_id": candidate_id,
        "label": "LightGBM (tuned)",
        "model_type": "lgb",
        "model_path": f"models/{candidate_id}.pkl",
        "selected_threshold": 0.55,
        "test_metrics": {"f1": np.float64(0.71), "support": np.int64(778)},
    }


def test_checkpoint_round_trips_without_stringifying_numbers(tmp_path):
    """A numpy float must come back as a float, not the string '0.71'."""
    result = _finished_candidate(tmp_path)
    write_candidate_checkpoint(tmp_path, "lgb_tuned", result)

    recovered = load_candidate_checkpoints(tmp_path)
    assert set(recovered) == {"lgb_tuned"}
    metrics = recovered["lgb_tuned"]["test_metrics"]
    assert metrics["f1"] == pytest.approx(0.71)
    assert isinstance(metrics["f1"], float)
    assert isinstance(metrics["support"], int)


def test_checkpoint_is_ignored_when_its_model_file_is_gone(tmp_path):
    """A checkpoint must never be trusted past the artifacts it points at."""
    result = _finished_candidate(tmp_path)
    write_candidate_checkpoint(tmp_path, "lgb_tuned", result)
    (tmp_path / "models" / "lgb_tuned.pkl").unlink()

    assert load_candidate_checkpoints(tmp_path) == {}


def test_checkpoint_is_ignored_when_probabilities_are_gone(tmp_path):
    result = _finished_candidate(tmp_path)
    write_candidate_checkpoint(tmp_path, "lgb_tuned", result)
    (tmp_path / "proba_test_lgb_tuned.npy").unlink()

    assert load_candidate_checkpoints(tmp_path) == {}


def test_corrupt_checkpoint_does_not_break_recovery(tmp_path):
    """A half-written JSON file is skipped, not raised."""
    good = _finished_candidate(tmp_path, "lgb_tuned")
    write_candidate_checkpoint(tmp_path, "lgb_tuned", good)
    (tmp_path / "checkpoints" / "xgb_tuned.json").write_text("{ truncated")

    assert set(load_candidate_checkpoints(tmp_path)) == {"lgb_tuned"}


def test_no_checkpoints_directory_is_not_an_error(tmp_path):
    assert load_candidate_checkpoints(tmp_path) == {}


def test_json_safe_handles_nested_numpy():
    payload = {
        "arr": np.array([1, 2]),
        "nested": [{"v": np.float32(0.5)}],
        "flag": np.bool_(True),
    }
    assert _json_safe(payload) == {"arr": [1, 2], "nested": [{"v": 0.5}], "flag": True}
    json.dumps(_json_safe(payload))  # must be serialisable without a default=
