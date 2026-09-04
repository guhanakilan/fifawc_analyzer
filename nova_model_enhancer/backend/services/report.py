"""Human-readable retraining report workbook.

Counts, metrics, decisions and approvals only — no raw data rows ever enter
this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _sheet(writer, name: str, frame: pd.DataFrame) -> None:
    # Excel sheet names are capped at 31 characters.
    frame.to_excel(writer, sheet_name=name[:31], index=False)


def build_report(destination: Path, payload: dict) -> Path:
    """Write the retraining report workbook and return its path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    run = payload.get("run", {})
    comparison = payload.get("comparison", {})
    approval = payload.get("approval", {})
    manifest = payload.get("snapshot_manifest", {})
    validation = payload.get("validation", {})

    with pd.ExcelWriter(destination, engine="xlsxwriter") as writer:
        summary_rows = [
            ("Placement", payload.get("placement_id")),
            ("Retraining job", payload.get("job_id")),
            ("Export version", payload.get("version")),
            ("Exported at", payload.get("exported_at")),
            ("Champion model", (run.get("champion") or {}).get("model_id")),
            ("Champion threshold", (run.get("champion") or {}).get("threshold")),
            ("Promoted model", approval.get("selected_candidate_id")),
            ("Promoted threshold", approval.get("selected_threshold")),
            ("Gate status", (comparison.get("gate_result") or {}).get("status")),
            ("Primary metric", (comparison.get("gate_result") or {}).get("primary_metric")),
            ("Approved by", approval.get("approver")),
            ("Approved at", approval.get("approved_at")),
            ("Approval notes", approval.get("notes")),
            ("Snapshot id", manifest.get("snapshot_id")),
            ("Snapshot SHA-256", manifest.get("snapshot_sha256")),
            ("Snapshot rows", (manifest.get("row_counts") or {}).get("final")),
            ("Weight formula", run.get("weight_formula")),
            ("Split mode", (run.get("split") or {}).get("mode")),
            ("Package validation", validation.get("status")),
        ]
        _sheet(writer, "Summary", pd.DataFrame(summary_rows, columns=["Item", "Value"]))

        metric_rows = []
        champion_metrics = (comparison.get("champion") or {}).get("test_metrics") or {}
        for name, metrics in (comparison.get("candidates") or {}).items():
            test = metrics.get("test_metrics") or {}
            metric_rows.append({
                "Model": name,
                "Threshold": metrics.get("selected_threshold"),
                **{k: test.get(k) for k in (
                    "f1", "precision", "recall", "specificity", "accuracy",
                    "auc", "pr_auc", "brier_score", "rows",
                )},
                "Predicted Non-Voice": test.get("predicted_non_voice"),
                "Predicted Voice": test.get("predicted_voice"),
            })
        metric_rows.insert(0, {
            "Model": f"CHAMPION · {(run.get('champion') or {}).get('model_id')}",
            "Threshold": (run.get("champion") or {}).get("threshold"),
            **{k: champion_metrics.get(k) for k in (
                "f1", "precision", "recall", "specificity", "accuracy",
                "auc", "pr_auc", "brier_score", "rows",
            )},
            "Predicted Non-Voice": champion_metrics.get("predicted_non_voice"),
            "Predicted Voice": champion_metrics.get("predicted_voice"),
        })
        _sheet(writer, "Model comparison", pd.DataFrame(metric_rows))

        gate_rules = (comparison.get("gate_result") or {}).get("rules") or []
        if gate_rules:
            _sheet(writer, "Promotion gate", pd.DataFrame(gate_rules))

        periods = comparison.get("period_breakdown") or {}
        period_rows = []
        for model_name, rows in periods.items():
            for row in rows:
                period_rows.append({"Model": model_name, **row})
        if period_rows:
            _sheet(writer, "By period", pd.DataFrame(period_rows))

        segments = comparison.get("segment_breakdown") or {}
        segment_rows = []
        for model_name, rows in segments.items():
            for row in rows:
                segment_rows.append({"Model": model_name, **row})
        if segment_rows:
            _sheet(writer, "By segment", pd.DataFrame(segment_rows))

        backtest_rows = []
        for model_type, result in (run.get("backtest") or {}).items():
            for row in result.get("results", []):
                backtest_rows.append({"Model": model_type, **row})
        if backtest_rows:
            _sheet(writer, "Backtest", pd.DataFrame(backtest_rows))

        weights = run.get("weights") or {}
        weight_rows = [("Formula", run.get("weight_formula"))]
        for key, value in (weights.get("distribution") or {}).items():
            weight_rows.append((f"Distribution · {key}", value))
        for applied in weights.get("applied", []):
            weight_rows.append((f"Applied · {applied.get('component')}",
                                f"x{applied.get('multiplier')} on {applied.get('rows')} rows"))
        for skipped in weights.get("skipped", []):
            weight_rows.append((f"Skipped · {skipped.get('component')}", skipped.get("reason")))
        _sheet(writer, "Weights", pd.DataFrame(weight_rows, columns=["Item", "Value"]))

        dataset_rows = [
            ("Rows loaded", (manifest.get("row_counts") or {}).get("loaded")),
            ("Rows after labelling", (manifest.get("row_counts") or {}).get("after_labelling")),
            ("Rows final", (manifest.get("row_counts") or {}).get("final")),
            ("Duplicates removed", (manifest.get("exclusions") or {}).get("duplicate_rows_removed")),
            ("Deduplication mode", (manifest.get("exclusions") or {}).get("deduplication_mode")),
            ("Deduplication keys", ", ".join((manifest.get("exclusions") or {}).get("deduplication_keys") or [])),
            ("Rows ignored by SubTask mapping",
             (manifest.get("exclusions") or {}).get("rows_ignored_by_subtask_mapping")),
            ("Date range from", (manifest.get("date_range") or {}).get("from")),
            ("Date range to", (manifest.get("date_range") or {}).get("to")),
            ("Non-Voice rate %", (manifest.get("target") or {}).get("non_voice_rate_pct")),
        ]
        for source in manifest.get("sources", []):
            dataset_rows.append((f"Source · {source.get('filename')}",
                                 f"{source.get('rows')} rows, sha256 {source.get('sha256', '')[:16]}…"))
        _sheet(writer, "Dataset", pd.DataFrame(dataset_rows, columns=["Item", "Value"]))

        checks = validation.get("checks") or []
        if checks:
            _sheet(writer, "Package validation", pd.DataFrame(checks))

        audit = payload.get("audit") or []
        if audit:
            _sheet(writer, "Audit trail", pd.DataFrame([
                {"When": a.get("created_at"), "Actor": a.get("actor"),
                 "Action": a.get("action"), "Detail": a.get("detail")}
                for a in audit
            ]))

    return destination
