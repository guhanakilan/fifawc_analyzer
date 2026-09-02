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


# ── Offline inspector ────────────────────────────────────────────────────────

def _run_inspector(zip_path):
    """Invoke tools/inspect_package.py the way a user would, in a subprocess."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "tools" / "inspect_package.py"
    return subprocess.run(
        [sys.executable, str(script), str(zip_path)],
        capture_output=True, text=True, timeout=120,
    )


def test_inspector_accepts_a_valid_package(tmp_path):
    result = _run_inspector(_write(tmp_path / "ok.zip", _minimal_members()))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACCEPTED: True" in result.stdout
    assert "JOB_x_lgb" in result.stdout


def test_inspector_explains_a_hard_rejection_without_stack_trace(tmp_path):
    members = _minimal_members()
    members["reports/curve.png"] = b"\x89PNG"
    result = _run_inspector(_write(tmp_path / "png.zip", members))
    assert result.returncode == 1
    assert "REJECTED OUTRIGHT" in result.stdout
    assert "Unsupported file type" in result.stdout
    # It must name the offending extension and the allow-list, so the user can act.
    assert ".png" in result.stdout
    assert "Currently allowed" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_inspector_names_a_missing_blocking_artifact(tmp_path):
    members = _minimal_members()
    del members["model/fitted_transforms.pkl"]
    result = _run_inspector(_write(tmp_path / "missing.zip", members))
    assert result.returncode == 1
    assert "[MISSING] fitted_transforms.pkl" in result.stdout
    assert "ACCEPTED: False" in result.stdout


def test_inspector_path_imports_only_the_standard_library():
    """It must work on a bare interpreter, before setup.bat has built the venv.

    Asserted structurally rather than by sandboxing an interpreter: every module
    the inspector pulls in is parsed, and any third-party top-level import fails
    the test.
    """
    import ast
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    modules = [
        root / "tools" / "inspect_package.py",
        root / "backend" / "services" / "package_validator.py",
        root / "backend" / "services" / "safety.py",
        root / "backend" / "config.py",
    ]

    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import inside this application.
                if node.level:
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top and top not in stdlib and top != "nova_model_enhancer":
                    offenders.append(f"{module.name}: {name}")

    assert not offenders, (
        "the intake path must not need the third-party stack, but imports: "
        + ", ".join(offenders)
    )
