"""Champion ZIP intake: defence, artifact resolution and classification.

Blocker vs warning is decided by real downstream need, verified against the
reference code, not by the file simply appearing in the export list:

  * blocking  — the retrain/score path cannot run without it
  * warning   — degrades fidelity or auditability but has a defined fallback
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..config import MAX_EXTRACTED_BYTES, MAX_FILES, MAX_ZIP_BYTES

# Logical artifact -> why it matters. Resolved case-insensitively by basename so
# a wrapper directory or a flat archive still matches.
BLOCKING_ARTIFACTS = {
    "fitted_transforms.pkl": "Fitted preprocessing state — model.pkl is unusable without it.",
    "feature_selection.json": "Feature list; fixes which columns enter the model.",
    "features_config.json": "Transform configuration required to refit preprocessing.",
    "threshold_config.json": "Decision threshold per model.",
    "training_results.json": "Identifies the champion model and its source metrics.",
}
WARNING_ARTIFACTS = {
    "column_map.json": "Inventory→production rename map; without it column names must already match.",
    "subtask_mappings.json": "SubTask→Voice/Non-Voice rules; without it labels cannot be re-derived.",
    "column_config.json": "Stage 03 column filter; without it every column is carried forward.",
    "dtype_config.json": "Dtype overrides; without them types are inferred.",
    "derived_config.json": "Derived column definitions.",
    "bucket_config.json": "Numeric bucket cut-points.",
    "grouping_config.json": "Categorical grouping rules.",
    "model_selection_config.json": "Champion hyperparameters — needed to rerun the champion family as-is.",
    "pipeline_version.json": "Transform-chain version marker.",
    "manifest.json": "Placement/run identity of the source export.",
}
OPTIONAL_ARTIFACTS = {"production.parquet", "README.txt", "scoring.py", "retraining_report.xlsx"}

ALLOWED_SUFFIXES = {".json", ".pkl", ".txt", ".md", ".py", ".parquet", ".xlsx", ".csv"}

# Canonical destination folder for each logical artifact.
_FOLDER_FOR = {
    **{name: "config" for name in (
        "column_map.json", "subtask_mappings.json", "column_config.json", "dtype_config.json",
        "derived_config.json", "bucket_config.json", "grouping_config.json",
        "feature_selection.json", "features_config.json",
    )},
    "fitted_transforms.pkl": "model",
    "threshold_config.json": "scoring",
    "training_results.json": "metadata",
    "model_selection_config.json": "metadata",
    "pipeline_version.json": "metadata",
    "manifest.json": "metadata",
}

MODEL_SUFFIX = ".pkl"


@dataclass
class Check:
    key: str
    label: str
    status: str      # passed | warning | failed
    detail: str
    blocking: bool


class PackageValidationError(ValueError):
    """Rejects the upload outright — nothing is persisted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    path = PurePosixPath(normalized)
    return not path.is_absolute() and ".." not in path.parts


def _strip_wrapper(names: list[str]) -> str:
    """Return the single common top-level directory, or '' when there isn't one.

    NoVA exports are sometimes re-zipped by a mail client or a file explorer,
    which wraps everything in one folder. That wrapper is not a difference in
    the package and must not be treated as one.
    """
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    roots = {n for n in names if "/" not in n}
    if len(tops) == 1 and not roots:
        return next(iter(tops)) + "/"
    return ""


def validate_and_extract(zip_path: Path, extraction_dir: Path) -> dict[str, Any]:
    """Validate the archive, then extract it into a canonical folder layout.

    Extraction happens only when no blocking check failed, so an unusable
    package never leaves files behind.
    """
    checks: list[Check] = []

    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise PackageValidationError(
            f"ZIP is larger than the {MAX_ZIP_BYTES // (1024 * 1024)} MB upload limit."
        )
    if not zipfile.is_zipfile(zip_path):
        raise PackageValidationError("Uploaded file is not a readable ZIP archive.")

    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise PackageValidationError(f"ZIP is corrupt — first bad member: {bad}")

        members = [m for m in archive.infolist() if not m.is_dir()]
        if not members:
            raise PackageValidationError("ZIP contains no files.")
        if len(members) > MAX_FILES:
            raise PackageValidationError(
                f"ZIP contains {len(members)} files, more than the {MAX_FILES} allowed."
            )

        unsafe = [m.filename for m in members if not _is_safe_member(m.filename)]
        if unsafe:
            raise PackageValidationError(
                f"Rejected: archive member uses an absolute or parent path ({unsafe[0]!r})."
            )

        declared_total = sum(m.file_size for m in members)
        if declared_total > MAX_EXTRACTED_BYTES:
            raise PackageValidationError(
                "Extracted package would exceed the 2 GB limit (possible ZIP bomb)."
            )
        compressed_total = sum(m.compress_size for m in members) or 1
        ratio = declared_total / compressed_total
        if ratio > 500 and declared_total > 50 * 1024 * 1024:
            raise PackageValidationError(
                f"Rejected: compression ratio {ratio:.0f}:1 on {declared_total // (1024*1024)} MB "
                "is characteristic of a ZIP bomb."
            )

        names = [m.filename.replace("\\", "/") for m in members]
        wrapper = _strip_wrapper(names)
        rel_names = [n[len(wrapper):] if wrapper else n for n in names]

        unsupported = sorted({
            n for n in rel_names if Path(n).suffix.lower() not in ALLOWED_SUFFIXES
        })
        if unsupported:
            raise PackageValidationError(
                f"Unsupported file type in package: {unsupported[0]!r}. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}."
            )

        checks.append(Check("zip", "Archive integrity",
                            "passed", f"{len(members)} files, {declared_total // 1024} KB uncompressed", True))
        checks.append(Check("paths", "Safe archive paths", "passed",
                            "No absolute paths, parent traversal or drive letters", True))
        checks.append(Check(
            "wrapper", "Archive layout", "passed",
            f"Wrapper directory '{wrapper.rstrip('/')}' detected and unwrapped" if wrapper
            else "Standard top-level layout", False,
        ))

        # ── Resolve logical artifacts by basename ────────────────────────────
        by_basename: dict[str, list[str]] = {}
        for original, relative in zip(names, rel_names):
            by_basename.setdefault(Path(relative).name, []).append(original)

        duplicates = {
            base: paths for base, paths in by_basename.items()
            if len(paths) > 1 and base in (BLOCKING_ARTIFACTS | WARNING_ARTIFACTS)
        }
        checks.append(Check(
            "duplicates", "Duplicate artifacts",
            "passed" if not duplicates else "failed",
            "No logical artifact appears twice" if not duplicates
            else "Ambiguous duplicates: " + "; ".join(
                f"{b} ({len(p)}x)" for b, p in sorted(duplicates.items())
            ),
            True,
        ))

        missing_blocking = sorted(k for k in BLOCKING_ARTIFACTS if k not in by_basename)
        missing_warning = sorted(k for k in WARNING_ARTIFACTS if k not in by_basename)
        checks.append(Check(
            "core", "Core artifacts",
            "passed" if not missing_blocking else "failed",
            "All artifacts required for retraining and scoring are present" if not missing_blocking
            else "Missing: " + ", ".join(missing_blocking),
            True,
        ))
        checks.append(Check(
            "supporting", "Supporting configuration",
            "passed" if not missing_warning else "warning",
            "All supporting configuration present" if not missing_warning
            else "Missing (non-blocking): " + ", ".join(missing_warning),
            False,
        ))

        # ── Champion model file ──────────────────────────────────────────────
        model_candidates = sorted({
            rel for rel in rel_names
            if rel.lower().endswith(MODEL_SUFFIX) and Path(rel).name != "fitted_transforms.pkl"
        })
        if len(model_candidates) == 1:
            model_status, model_detail = "passed", model_candidates[0]
        elif not model_candidates:
            model_status, model_detail = "failed", "No champion estimator .pkl found."
        else:
            model_status = "failed"
            model_detail = (
                f"{len(model_candidates)} candidate model files found "
                f"({', '.join(Path(c).name for c in model_candidates[:4])}). "
                "Exactly one champion estimator is required."
            )
        checks.append(Check("model", "Champion estimator", model_status, model_detail, True))

        # ── JSON parse (no unpickling at intake) ─────────────────────────────
        parsed: dict[str, Any] = {}
        json_errors: list[str] = []
        for original, relative in zip(names, rel_names):
            if not relative.lower().endswith(".json"):
                continue
            try:
                parsed[Path(relative).name] = json.loads(
                    archive.read(original).decode("utf-8-sig")
                )
            except Exception as exc:
                json_errors.append(f"{relative}: {exc}")
        checks.append(Check(
            "json", "JSON configuration",
            "passed" if not json_errors else "failed",
            f"{len(parsed)} JSON files parsed" if not json_errors
            else "Unreadable: " + "; ".join(json_errors[:3]),
            True,
        ))
        checks.append(Check(
            "pickle", "Deferred model load", "passed",
            "No .pkl was opened during intake. The estimator is loaded only in the "
            "compatibility check, which requires explicit local-trust confirmation.",
            False,
        ))

        blocking_failures = [c for c in checks if c.blocking and c.status == "failed"]
        valid = not blocking_failures

        extracted_files: list[str] = []
        if valid:
            extraction_dir.mkdir(parents=True, exist_ok=True)
            root = extraction_dir.resolve()
            for member, relative in zip(members, rel_names):
                base = Path(relative).name
                folder = _FOLDER_FOR.get(base)
                if folder is None:
                    # Keep the package's own structure for anything not in the
                    # canonical map, minus the wrapper directory.
                    destination = (root / relative).resolve()
                elif base == "fitted_transforms.pkl":
                    destination = (root / "model" / base).resolve()
                else:
                    destination = (root / folder / base).resolve()
                if base.lower().endswith(MODEL_SUFFIX) and base != "fitted_transforms.pkl":
                    destination = (root / "model" / base).resolve()
                if root != destination and root not in destination.parents:
                    raise PackageValidationError("Extraction path escaped the workspace.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                extracted_files.append(str(destination.relative_to(root)).replace("\\", "/"))

    # ── Metadata read out of the parsed JSON ────────────────────────────────
    training = parsed.get("training_results.json") or {}
    manifest = parsed.get("manifest.json") or {}
    thresholds = parsed.get("threshold_config.json") or {}
    features_cfg = parsed.get("features_config.json") or {}
    fs_raw = parsed.get("feature_selection.json")
    feature_list = (
        fs_raw.get("selected_columns", []) if isinstance(fs_raw, dict) else (fs_raw or [])
    )

    model_id = training.get("best_model") or manifest.get("model_id")
    if not model_id and model_candidates:
        model_id = Path(model_candidates[0]).stem
    model_result = (training.get("results") or {}).get(model_id, {}) if model_id else {}

    threshold = None
    if isinstance(thresholds, dict) and model_id in thresholds:
        try:
            threshold = float(thresholds[model_id])
        except (TypeError, ValueError):
            threshold = None

    return {
        "valid": valid,
        "checks": [asdict(c) for c in checks],
        "blocking_failures": [c.label for c in blocking_failures],
        "missing_blocking_files": missing_blocking,
        "missing_supporting_files": missing_warning,
        "model_files": model_candidates,
        "extracted_files": sorted(extracted_files),
        "wrapper_directory": wrapper.rstrip("/") or None,
        "metadata": {
            "placement_id": manifest.get("placement_id"),
            "run_id": manifest.get("run_id") or training.get("job_id"),
            "model_id": model_id,
            "model_family": model_result.get("model_type"),
            "threshold": threshold,
            "feature_count": len(feature_list) or None,
            "training_from": training.get("training_from"),
            "training_to": training.get("training_to"),
            "split_mode": training.get("split_mode") or (features_cfg.get("split") or {}).get("mode"),
            "source_metrics": model_result.get("test_metrics") or {},
            "source_cv_mean": model_result.get("cv_mean") or {},
            "voice_rate": training.get("voice_rate"),
            "exported_at": manifest.get("exported_at"),
            "file_count": len(members),
            "extracted_bytes": declared_total,
        },
    }
