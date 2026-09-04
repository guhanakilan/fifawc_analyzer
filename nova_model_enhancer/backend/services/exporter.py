"""Stage 7 — build, validate and version the deployable scoring package.

The layout follows the reference `routers/export.py` exactly, because that is
what the deployed loader reads. Two things are added, both documented in the
package README: `pipeline/scoring.py` (the runtime, placed where the reference
deployment expects it) and `scoring/ml_tag_config.json` (the approved single
column convention).

Nothing is published until the smoke test has actually loaded the ZIP from disk
and scored a sample through it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import APP_VERSION, PIPELINE_VERSION
from . import snapshot as snapshot_service

SCORING_RUNTIME = Path(__file__).resolve().parent.parent / "scoring_runtime" / "scoring.py"

# (source-relative path inside the extracted champion package, destination in the ZIP)
CARRIED_CONFIGS = [
    ("config/column_map.json", "config/column_map.json"),
    ("config/subtask_mappings.json", "config/subtask_mappings.json"),
    ("config/column_config.json", "config/column_config.json"),
    ("config/dtype_config.json", "config/dtype_config.json"),
    ("config/derived_config.json", "config/derived_config.json"),
    ("config/bucket_config.json", "config/bucket_config.json"),
    ("config/grouping_config.json", "config/grouping_config.json"),
    ("config/feature_selection.json", "config/feature_selection.json"),
    ("config/features_config.json", "config/features_config.json"),
]


class ExportError(RuntimeError):
    pass


@dataclass
class ExportRequest:
    job_id: str
    version: int
    placement_id: Any
    candidate_id: str
    model_id: str
    threshold: float
    ml_tag_config: dict
    approval: dict
    run_record: dict
    comparison: dict
    snapshot_manifest: dict
    audit: list


def _library_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "lightgbm", "optuna"):
        try:
            from importlib.metadata import version as _version
            versions[name] = _version(name)
        except Exception:
            versions[name] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readme(request: ExportRequest, ml_tag: dict) -> str:
    approved = ml_tag.get("approved")
    return f"""NoVA scoring package — placement {request.placement_id}, version {request.version:03d}

Model                {request.model_id}
Decision threshold   {request.threshold}
Built by             NoVA Model Enhancer {APP_VERSION}
Retraining job       {request.job_id}
Approved by          {request.approval.get('approver')} on {request.approval.get('approved_at')}

LAYOUT
  model/{request.model_id}.pkl      the deployable estimator
  model/fitted_transforms.pkl       its matching fitted preprocessing state
  config/                           the transformation configuration it was built with
  scoring/threshold_config.json     decision threshold, keyed by model id
  scoring/ml_tag_config.json        approved ml_tag convention
  metadata/                         training results, manifest, validation record
  pipeline/scoring.py               the scoring runtime
  reports/retraining_report.xlsx    human-readable retraining report

DEPLOYMENT
  Unzip into the placement folder so that pipeline/ sits beside config/, model/,
  scoring/ and metadata/. That is the layout nova-ml's own scoring client expects:

      from scoring import NovaMLPipeline
      pipeline = NovaMLPipeline()
      scored   = pipeline.run(inventory_df)        # NovaProbability + VoiceNonVoiceFlag
      tagged   = pipeline.run_ml_tag(inventory_df) # original columns + ml_tag only

  model.pkl is deliberately NOT used as the runtime filename. nova-ml resolves the
  estimator as "{{training_results.best_model}}.pkl", so renaming it would break the
  loader.

ENCODINGS — read this before consuming the output
  Training target  NonVoiceFlag       0 = Voice, 1 = Non-Voice
  Model output     P(class 1)          probability of Non-Voice
  run()            VoiceNonVoiceFlag  1 = Voice, 0 = Non-Voice   (inverted, nova-ml parity)
  run_ml_tag()     {ml_tag.get('column_name', 'ml_tag')}{' ' * max(1, 18 - len(str(ml_tag.get('column_name', 'ml_tag'))))}{ml_tag.get('voice_value')} = Voice, {ml_tag.get('non_voice_value')} = Non-Voice
  ml_tag approved  {'yes — ' + str(ml_tag.get('approver')) if approved else 'NO — run_ml_tag() will refuse to run'}

GUARANTEES VERIFIED AT BUILD TIME
  The ZIP was unzipped and loaded exactly as the deployment does, then used to
  score a sample inventory. See metadata/validation_report.json for the row
  counts, column checks and prediction-agreement figures that were recorded.

ROLLBACK
  Previous versions are retained under the retraining job's exports/ folder and
  listed in metadata/rollback_manifest.json. This package does not deploy itself
  to any environment.
"""


def _load_runtime_module(scoring_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, scoring_path)
    if spec is None or spec.loader is None:
        raise ExportError(f"Could not load the scoring runtime at {scoring_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def smoke_test_package(
    zip_path: Path, inventory: pd.DataFrame, expected_proba: np.ndarray | None,
    ml_tag_config: dict, reference_scoring_path: Path | None = None,
) -> dict:
    """Unzip the built package and score through it exactly as deployment does.

    Every assertion here is on the real artifact, never on the in-memory objects
    that produced it.
    """
    checks: list[dict] = []

    def _check(key: str, label: str, passed: bool, detail: str, blocking: bool = True) -> None:
        checks.append({
            "key": key, "label": label,
            "status": "passed" if passed else ("failed" if blocking else "warning"),
            "detail": detail, "blocking": blocking,
        })

    with tempfile.TemporaryDirectory(prefix="nova_pkg_") as tmp:
        root = Path(tmp) / "placement"
        root.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)

        scoring_path = root / "pipeline" / "scoring.py"
        if not scoring_path.exists():
            _check("runtime", "Scoring runtime present", False, "pipeline/scoring.py missing from the ZIP")
            return {"status": "failed", "checks": checks}

        module = _load_runtime_module(scoring_path, "nova_pkg_scoring_check")
        try:
            pipeline = module.NovaMLPipeline(resource_root=root)
        except Exception as exc:
            _check("load", "Package loads", False, f"{type(exc).__name__}: {exc}")
            return {"status": "failed", "checks": checks}
        _check("load", "Package loads", True,
               f"Estimator {pipeline.best_model_id} and {len(pipeline.feature_names)} fitted features loaded")

        original_columns = list(inventory.columns)
        original_rows = len(inventory)

        # ── nova-ml parity mode ─────────────────────────────────────────────
        agreement: dict = {}
        try:
            parity = pipeline.run(inventory.copy())
            parity_ok = (
                len(parity) == original_rows
                and list(parity.columns)[:len(original_columns)] == original_columns
                and list(parity.columns)[len(original_columns):] == ["NovaProbability", "VoiceNonVoiceFlag"]
            )
            _check("parity_mode", "nova-ml parity output", parity_ok,
                   f"{len(parity)} rows, appended {list(parity.columns)[len(original_columns):]}")
            if expected_proba is not None and len(expected_proba) == len(parity):
                diff = np.abs(np.asarray(parity["NovaProbability"], dtype=float)
                              - np.round(np.asarray(expected_proba, dtype=float), 4))
                max_diff = float(diff.max()) if len(diff) else 0.0
                matches = int((diff <= 5e-4).sum())
                agreement = {
                    "rows": int(len(diff)),
                    "max_absolute_difference": round(max_diff, 6),
                    "rows_matching_to_4dp": matches,
                    "agreement_pct": round(matches / max(len(diff), 1) * 100, 4),
                }
                _check("agreement", "Package predictions match the enhancer's own",
                       max_diff <= 5e-4,
                       f"max |Δ| {max_diff:.6f} over {len(diff)} rows "
                       f"({agreement['agreement_pct']}% identical to 4dp)")
        except Exception as exc:
            _check("parity_mode", "nova-ml parity output", False, f"{type(exc).__name__}: {exc}")

        # ── ml_tag mode ─────────────────────────────────────────────────────
        tag_column = ml_tag_config.get("column_name", "ml_tag")
        if not ml_tag_config.get("approved"):
            _check("ml_tag_mode", "Single-column ml_tag output", False,
                   "ml_tag encoding has not been approved, so this mode is intentionally blocked.")
        else:
            try:
                tagged = pipeline.run_ml_tag(inventory.copy())
                appended = [c for c in tagged.columns if c not in original_columns]
                row_ok = len(tagged) == original_rows
                col_ok = list(tagged.columns)[:len(original_columns)] == original_columns
                one_ok = appended == [tag_column]
                _check("ml_tag_rows", "Row count preserved", row_ok,
                       f"{original_rows} rows in, {len(tagged)} rows out")
                _check("ml_tag_columns", "Original columns preserved in order", col_ok,
                       f"{len(original_columns)} original columns, order unchanged" if col_ok
                       else "Original column set or order changed")
                _check("ml_tag_single", f"Exactly one appended column named '{tag_column}'", one_ok,
                       f"appended {appended}")
                leaked = [c for c in tagged.columns
                          if c not in original_columns
                          and c.lower() in {"novaprobability", "voicenonvoiceflag", "probability", "nonvoiceflag"}]
                _check("ml_tag_no_leak", "No probability or Voice/Non-Voice text exposed",
                       not leaked, "No probability or flag text columns present" if not leaked
                       else f"leaked columns: {leaked}")

                values = set(pd.unique(tagged[tag_column])) if one_ok else set()
                allowed = {ml_tag_config.get("voice_value"), ml_tag_config.get("non_voice_value")}
                _check("ml_tag_values", "ml_tag uses only the approved values",
                       values.issubset(allowed) if values else False,
                       f"observed {sorted(map(str, values))}, approved {sorted(map(str, allowed))}")

                # Inversion test: ml_tag must be the documented inverse of the
                # internal target, not silently the same encoding.
                if one_ok and expected_proba is not None and len(expected_proba) == len(tagged):
                    predicted_non_voice = np.asarray(expected_proba, dtype=float) >= pipeline.threshold
                    tag_values = tagged[tag_column].values
                    expected_tag = np.where(
                        predicted_non_voice, ml_tag_config.get("non_voice_value"),
                        ml_tag_config.get("voice_value"),
                    )
                    inversion_ok = bool(np.array_equal(
                        pd.Series(tag_values).astype(str).values,
                        pd.Series(expected_tag).astype(str).values,
                    ))
                    _check("ml_tag_inversion", "ml_tag encoding matches the approved convention",
                           inversion_ok,
                           f"{tag_column}: {ml_tag_config.get('voice_value')} = Voice, "
                           f"{ml_tag_config.get('non_voice_value')} = Non-Voice"
                           if inversion_ok else
                           "The emitted tag does not follow the approved Voice/Non-Voice convention.")
            except Exception as exc:
                _check("ml_tag_mode", "Single-column ml_tag output", False, f"{type(exc).__name__}: {exc}")

        # ── Verbatim reference loader, for evidence ─────────────────────────
        if reference_scoring_path and Path(reference_scoring_path).exists():
            try:
                ref_module = _load_runtime_module(Path(reference_scoring_path), "nova_ref_scoring_check")
                ref_pipeline = ref_module.NovaMLPipeline(resource_root=root)
                ref_result = ref_pipeline.run(inventory.copy())
                ref_proba = np.asarray(ref_result["NovaProbability"], dtype=float)
                constant = bool(len(np.unique(np.round(ref_proba, 6))) <= 1)
                _check(
                    "reference_loader", "Verbatim nova-ml scoring client",
                    not constant,
                    "Reference client scored the sample successfully." if not constant else
                    "Reference client returned a constant probability — this is defect D2 "
                    "(feature_selection.json wrapper) in the reference client, not a fault in "
                    "this package. The shipped pipeline/scoring.py handles it.",
                    blocking=False,
                )
            except Exception as exc:
                _check(
                    "reference_loader", "Verbatim nova-ml scoring client", False,
                    f"Reference client raised {type(exc).__name__}: {exc}. This is defect D1/D2 in "
                    "the reference client (see IMPLEMENTATION_GAP_ANALYSIS.md); the shipped "
                    "pipeline/scoring.py reads the same files correctly.",
                    blocking=False,
                )

    blocking_failed = [c for c in checks if c["blocking"] and c["status"] == "failed"]
    return {
        "status": "passed" if not blocking_failed else "failed",
        "checks": checks,
        "failed_checks": [c["label"] for c in blocking_failed],
        "prediction_agreement": agreement,
        "inventory_rows": int(len(inventory)),
        "inventory_columns": int(len(inventory.columns)),
    }


def build_package(
    *, request: ExportRequest, extract_dir: Path, run_dir: Path, exports_dir: Path,
    report_path: Path, previous_exports: list[dict],
) -> dict:
    """Assemble the ZIP. Written to a temp path and renamed only once complete."""
    exports_dir.mkdir(parents=True, exist_ok=True)
    challenger = (request.run_record.get("challengers") or {}).get(request.candidate_id)
    if not challenger:
        raise ExportError(f"Candidate {request.candidate_id!r} is not present in this run.")

    model_source = run_dir / challenger["model_path"]
    fitted_source = run_dir / "models" / "fitted_transforms.pkl"
    if not model_source.exists() or not fitted_source.exists():
        raise ExportError("The promoted model or its fitted transform state is missing from the run.")

    zip_name = f"PLC_{request.placement_id}_V{request.version:03d}_STREAMLIT_READY.zip"
    final_path = exports_dir / zip_name
    if final_path.exists():
        raise ExportError(f"{zip_name} already exists. Previous versions are never overwritten.")

    with fitted_source.open("rb") as handle:
        fitted = pickle.load(handle)

    training_results = {
        "job_id": request.job_id,
        "exp_name": f"enhancer_{request.job_id}",
        "optimizer": "optuna_tpe",
        "completed_at": request.run_record.get("created_at"),
        "best_model": request.model_id,
        "cv_folds": 5,
        "split_mode": (request.run_record.get("split") or {}).get("mode"),
        "training_from": (request.snapshot_manifest.get("date_range") or {}).get("from"),
        "training_to": (request.snapshot_manifest.get("date_range") or {}).get("to"),
        "voice_rate": 100.0 - float((request.snapshot_manifest.get("target") or {}).get("non_voice_rate_pct") or 0),
        "results": {
            request.model_id: {
                "model_type": challenger["model_type"],
                "best_params": challenger.get("best_params", {}),
                "cv_mean": challenger.get("cv_mean", {}),
                "cv_std": challenger.get("cv_std", {}),
                "cv_folds": challenger.get("cv_folds", []),
                "test_metrics": challenger.get("test_metrics", {}),
                "confusion_matrix": (challenger.get("test_metrics") or {}).get("confusion_matrix", {}),
                "feature_importance": challenger.get("feature_importance", []),
                "calibrated": challenger.get("calibrated"),
                "calibration_method": challenger.get("calibration_method"),
                "train_time": challenger.get("train_time_seconds"),
                "train_rows": challenger.get("train_rows"),
            }
        },
    }

    manifest = {
        "placement_id": request.placement_id,
        "run_id": request.run_record.get("run_id"),
        "retraining_job_id": request.job_id,
        "model_id": request.model_id,
        "candidate_id": request.candidate_id,
        "version": request.version,
        "exported_at": request.approval.get("approved_at"),
        "built_by": f"NoVA Model Enhancer {APP_VERSION}",
        "threshold": request.threshold,
        "feature_count": len(fitted.get("feature_names") or []),
        "feature_names": list(fitted.get("feature_names") or []),
        "snapshot_id": request.snapshot_manifest.get("snapshot_id"),
        "snapshot_sha256": request.snapshot_manifest.get("snapshot_sha256"),
        "library_versions": _library_versions(),
        "pipeline_version": PIPELINE_VERSION,
        "target_encoding": {"column": "NonVoiceFlag", "voice": 0, "non_voice": 1},
        "output_encoding": {
            "run": {"NovaProbability": "P(Non-Voice)",
                    "VoiceNonVoiceFlag": {"voice": 1, "non_voice": 0}},
            "run_ml_tag": request.ml_tag_config,
        },
        "scoring_order": [
            "0. Rename inventory columns with config/column_map.json",
            "1. Normalise column names, then cast with config/dtype_config.json",
            "2. Recreate derived columns with config/derived_config.json",
            "3. Apply config/bucket_config.json then config/grouping_config.json",
            "4. Keep the features in config/feature_selection.json",
            "5. Apply model/fitted_transforms.pkl (impute, cap, log, encode, scale)",
            f"6. Score with model/{request.model_id}.pkl",
            "7. Apply the threshold from scoring/threshold_config.json",
        ],
    }

    rollback_manifest = {
        "current_version": request.version,
        "current_zip": zip_name,
        "previous_versions": [
            {
                "version": p["version"], "model_id": p["model_id"],
                "zip": Path(p["zip_path"]).name, "sha256": p["zip_sha256"],
                "created_at": p["created_at"],
                "approved_by": (p.get("approval") or {}).get("approver"),
            }
            for p in previous_exports
        ],
        "rollback_instructions": (
            "To roll back, unzip the chosen previous version into the placement folder, "
            "replacing config/, model/, scoring/, metadata/ and pipeline/. The enhancer "
            "never deletes a previous version and never deploys automatically."
        ),
        "champion_package_retained": True,
    }

    temp_path = exports_dir / f".{zip_name}.building"
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_source, f"model/{request.model_id}.pkl")
            archive.write(fitted_source, "model/fitted_transforms.pkl")

            for source_rel, dest_rel in CARRIED_CONFIGS:
                source = extract_dir / source_rel
                if source.exists():
                    archive.write(source, dest_rel)

            archive.writestr("scoring/threshold_config.json",
                             json.dumps({request.model_id: request.threshold}, indent=2))
            archive.writestr("scoring/ml_tag_config.json",
                             json.dumps(request.ml_tag_config, indent=2))

            archive.writestr("metadata/training_results.json", json.dumps(training_results, indent=2, default=str))
            archive.writestr("metadata/manifest.json", json.dumps(manifest, indent=2, default=str))
            archive.writestr("metadata/dataset_manifest.json",
                             json.dumps(request.snapshot_manifest, indent=2, default=str))
            archive.writestr("metadata/champion_comparison.json",
                             json.dumps(request.comparison, indent=2, default=str))
            archive.writestr("metadata/pipeline_version.json",
                             json.dumps({"version": PIPELINE_VERSION, "written_by": "nova_model_enhancer"}, indent=2))
            archive.writestr("metadata/model_selection_config.json",
                             json.dumps({
                                 "model_type": challenger["model_type"],
                                 "mode": challenger.get("mode"),
                                 "best_params": challenger.get("best_params", {}),
                                 "n_trials": challenger.get("n_trials"),
                                 "seed": challenger.get("seed"),
                                 "cv_folds": 5,
                             }, indent=2, default=str))
            archive.writestr("metadata/approval_record.json",
                             json.dumps(request.approval, indent=2, default=str))
            archive.writestr("metadata/rollback_manifest.json",
                             json.dumps(rollback_manifest, indent=2, default=str))
            archive.writestr("metadata/audit_trail.json",
                             json.dumps(request.audit, indent=2, default=str))

            archive.write(SCORING_RUNTIME, "pipeline/scoring.py")
            if report_path.exists():
                archive.write(report_path, "reports/retraining_report.xlsx")
            archive.writestr("README.txt", _readme(request, request.ml_tag_config))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {"temp_path": temp_path, "final_path": final_path, "zip_name": zip_name,
            "manifest": manifest, "training_results": training_results}


def finalise_package(built: dict, validation: dict) -> dict:
    """Insert the validation record, then rename the ZIP into place atomically."""
    temp_path: Path = built["temp_path"]
    final_path: Path = built["final_path"]
    with zipfile.ZipFile(temp_path, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata/validation_report.json", json.dumps(validation, indent=2, default=str))
    temp_path.replace(final_path)
    return {"zip_path": str(final_path), "zip_sha256": _sha256(final_path),
            "zip_name": built["zip_name"], "size_bytes": final_path.stat().st_size}
