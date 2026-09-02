"""Champion ZIP intake defences and artifact classification."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from nova_model_enhancer.backend.services.package_validator import (
    BLOCKING_ARTIFACTS,
    WARNING_ARTIFACTS,
    PackageValidationError,
    validate_and_extract,
)


def _minimal_members(prefix: str = "") -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for name in BLOCKING_ARTIFACTS:
        folder = "model" if name.endswith(".pkl") else (
            "scoring" if name.startswith("threshold") else
            "metadata" if name.startswith("training") else "config"
        )
        payload = (
            json.dumps({"best_model": "JOB_x_lgb"}).encode()
            if name == "training_results.json"
            else json.dumps({}).encode() if name.endswith(".json") else b"placeholder"
        )
        members[f"{prefix}{folder}/{name}"] = payload
    members[f"{prefix}model/JOB_x_lgb.pkl"] = b"placeholder"
    return members


def _write(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_valid_minimal_package(tmp_path):
    result = validate_and_extract(
        _write(tmp_path / "ok.zip", _minimal_members()), tmp_path / "out"
    )
    assert result["valid"] is True
    assert result["metadata"]["model_id"] == "JOB_x_lgb"
    assert (tmp_path / "out" / "model" / "JOB_x_lgb.pkl").exists()


def test_wrapper_directory_is_unwrapped(tmp_path):
    result = validate_and_extract(
        _write(tmp_path / "wrapped.zip", _minimal_members("PLC_984_export/")), tmp_path / "out"
    )
    assert result["valid"] is True
    assert result["wrapper_directory"] == "PLC_984_export"
    assert (tmp_path / "out" / "model" / "fitted_transforms.pkl").exists()


def test_parent_path_traversal_is_rejected(tmp_path):
    package = _write(tmp_path / "bad.zip", {"../escape.json": b"{}"})
    with pytest.raises(PackageValidationError, match="parent path"):
        validate_and_extract(package, tmp_path / "out")


def test_absolute_path_is_rejected(tmp_path):
    package = _write(tmp_path / "abs.zip", {"/etc/passwd.json": b"{}"})
    with pytest.raises(PackageValidationError, match="absolute or parent path"):
        validate_and_extract(package, tmp_path / "out")


def test_windows_drive_path_is_rejected(tmp_path):
    package = _write(tmp_path / "drive.zip", {"C:/windows/system.json": b"{}"})
    with pytest.raises(PackageValidationError, match="absolute or parent path"):
        validate_and_extract(package, tmp_path / "out")


def test_zip_bomb_ratio_is_rejected(tmp_path):
    package = tmp_path / "bomb.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("config/huge.json", b"0" * (80 * 1024 * 1024))
    with pytest.raises(PackageValidationError, match="ZIP bomb"):
        validate_and_extract(package, tmp_path / "out")


def test_file_count_limit(tmp_path, monkeypatch):
    import nova_model_enhancer.backend.services.package_validator as validator

    monkeypatch.setattr(validator, "MAX_FILES", 5)
    members = {f"config/file_{i}.json": b"{}" for i in range(10)}
    with pytest.raises(PackageValidationError, match="more than the 5 allowed"):
        validator.validate_and_extract(_write(tmp_path / "many.zip", members), tmp_path / "out")


def test_unsupported_file_type_is_rejected(tmp_path):
    members = _minimal_members()
    members["config/install.exe"] = b"MZ"
    with pytest.raises(PackageValidationError, match="Unsupported file type"):
        validate_and_extract(_write(tmp_path / "exe.zip", members), tmp_path / "out")


def test_duplicate_logical_artifact_blocks(tmp_path):
    members = _minimal_members()
    members["backup/feature_selection.json"] = b"{}"
    result = validate_and_extract(_write(tmp_path / "dupe.zip", members), tmp_path / "out")
    assert result["valid"] is False
    assert "Duplicate artifacts" in result["blocking_failures"]


def test_missing_core_artifact_blocks(tmp_path):
    members = _minimal_members()
    del members["model/fitted_transforms.pkl"]
    result = validate_and_extract(_write(tmp_path / "missing.zip", members), tmp_path / "out")
    assert result["valid"] is False
    assert "fitted_transforms.pkl" in result["missing_blocking_files"]
    assert not (tmp_path / "out").exists(), "an invalid package must not be extracted"


def test_missing_supporting_artifact_is_a_warning_only(tmp_path):
    result = validate_and_extract(_write(tmp_path / "warn.zip", _minimal_members()), tmp_path / "out")
    assert result["valid"] is True
    supporting = next(c for c in result["checks"] if c["key"] == "supporting")
    assert supporting["status"] == "warning"
    assert set(result["missing_supporting_files"]) == set(WARNING_ARTIFACTS)


def test_corrupt_json_blocks(tmp_path):
    members = _minimal_members()
    members["config/feature_selection.json"] = b"{not json"
    result = validate_and_extract(_write(tmp_path / "corrupt.zip", members), tmp_path / "out")
    assert result["valid"] is False
    json_check = next(c for c in result["checks"] if c["key"] == "json")
    assert json_check["status"] == "failed"


def test_multiple_model_files_block(tmp_path):
    members = _minimal_members()
    members["model/JOB_x_rf.pkl"] = b"placeholder"
    result = validate_and_extract(_write(tmp_path / "two.zip", members), tmp_path / "out")
    assert result["valid"] is False
    model_check = next(c for c in result["checks"] if c["key"] == "model")
    assert "2 candidate model files" in model_check["detail"]


def test_no_pickle_is_opened_during_intake(tmp_path, monkeypatch):
    """Intake must never unpickle. A poisoned .pkl would raise if it were loaded."""
    import pickle

    def _explode(*args, **kwargs):
        raise AssertionError("intake must not unpickle anything")

    monkeypatch.setattr(pickle, "load", _explode)
    monkeypatch.setattr(pickle, "loads", _explode)
    result = validate_and_extract(_write(tmp_path / "safe.zip", _minimal_members()), tmp_path / "out")
    assert result["valid"] is True


def test_not_a_zip(tmp_path):
    path = tmp_path / "notzip.zip"
    path.write_bytes(b"this is not a zip file")
    with pytest.raises(PackageValidationError, match="not a readable ZIP"):
        validate_and_extract(path, tmp_path / "out")
