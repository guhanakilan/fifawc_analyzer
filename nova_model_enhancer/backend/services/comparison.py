"""Stage 6 — assemble the champion/challenger comparison on identical rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluator, labeling
from . import snapshot as snapshot_service
from .nova_transform import TARGET


class ComparisonError(RuntimeError):
    pass


def _load_probabilities(run_dir: Path, name: str) -> np.ndarray | None:
    path = run_dir / f"proba_test_{name}.npy"
    return np.load(path) if path.exists() else None


def build_comparison(
    run_dir: Path, snapshot_dir: Path, snapshot_id: str, gate: dict,
    segment_column: str | None = None, min_segment_rows: int = 100,
) -> dict:
    """Evaluate every trained candidate and the champion on the same test rows.

    Ranking is only ever by the approved primary metric — there is no built-in
    notion of "best".
    """
    run_path = run_dir / "run_results.json"
    if not run_path.exists():
        raise ComparisonError("This run has no results yet.")
    run = json.loads(run_path.read_text(encoding="utf-8"))

    y_test = np.load(run_dir / "y_test.npy")
    test_indices = np.load(run_dir / "test_indices.npy")
    champion_proba = _load_probabilities(run_dir, "champion")
    if champion_proba is None:
        raise ComparisonError("The champion benchmark predictions are missing from this run.")

    df = snapshot_service.load_snapshot(snapshot_dir, snapshot_id)
    manifest = snapshot_service.load_manifest(snapshot_dir, snapshot_id)
    date_column = manifest["date_column"]
    dates = pd.to_datetime(df[date_column], errors="coerce").reset_index(drop=True).iloc[test_indices]

    segments = None
    if segment_column:
        from .data_profiler import match_column
        resolved = match_column(df.columns, segment_column)
        if resolved:
            segments = df[resolved].reset_index(drop=True).iloc[test_indices]

    champion_threshold = float((run.get("champion") or {}).get("threshold", 0.5))
    champion_metrics = evaluator.metrics_at_threshold(y_test, champion_proba, champion_threshold)

    # Historical vs recent halves of the benchmark, split at its own median date.
    historical_mask = recent_mask = None
    if dates.notna().any():
        midpoint = dates.dropna().quantile(0.5)
        historical_mask = (dates <= midpoint).values
        recent_mask = (dates > midpoint).values

    def _slice_metrics(mask, y, proba, threshold):
        if mask is None or mask.sum() < 30 or len(np.unique(y[mask])) < 2:
            return None
        return evaluator.metrics_at_threshold(y[mask], proba[mask], threshold)

    candidates: dict = {}
    period_breakdown: dict = {}
    segment_breakdown: dict = {}
    historical: dict = {"champion": _slice_metrics(historical_mask, y_test, champion_proba, champion_threshold)}
    recent: dict = {"champion": _slice_metrics(recent_mask, y_test, champion_proba, champion_threshold)}

    period_breakdown["champion"] = evaluator.period_breakdown(
        y_test, champion_proba, dates, champion_threshold, min_rows=min_segment_rows
    )
    if segments is not None:
        segment_breakdown["champion"] = evaluator.segment_breakdown(
            y_test, champion_proba, segments, champion_threshold, min_rows=min_segment_rows
        )

    for candidate_id, record in (run.get("challengers") or {}).items():
        if "skipped" in record:
            candidates[candidate_id] = {"skipped": record["skipped"], "model_type": record.get("model_type")}
            continue
        proba = _load_probabilities(run_dir, candidate_id)
        if proba is None:
            candidates[candidate_id] = {"skipped": "Predictions were not persisted for this candidate."}
            continue
        threshold = float(record.get("selected_threshold", 0.5))
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "label": record.get("label", candidate_id),
            "model_type": record.get("model_type"),
            "mode": record.get("mode"),
            "selected_threshold": threshold,
            "threshold_analysis": record.get("threshold_analysis"),
            "test_metrics": record.get("test_metrics") or evaluator.metrics_at_threshold(y_test, proba, threshold),
            "cv_mean": record.get("cv_mean"),
            "cv_std": record.get("cv_std"),
            "calibrated": record.get("calibrated"),
            "calibration_method": record.get("calibration_method"),
            "calibration_curve": record.get("calibration_curve"),
            "latency": record.get("latency"),
            "feature_importance": record.get("feature_importance"),
            "best_params": record.get("best_params"),
            "train_time_seconds": record.get("train_time_seconds"),
        }
        historical[candidate_id] = _slice_metrics(historical_mask, y_test, proba, threshold)
        recent[candidate_id] = _slice_metrics(recent_mask, y_test, proba, threshold)
        period_breakdown[candidate_id] = evaluator.period_breakdown(
            y_test, proba, dates, threshold, min_rows=min_segment_rows
        )
        if segments is not None:
            segment_breakdown[candidate_id] = evaluator.segment_breakdown(
                y_test, proba, segments, threshold, min_rows=min_segment_rows
            )

    trained = {k: v for k, v in candidates.items() if "skipped" not in v}
    primary = gate.get("primary_metric", "f1")
    ranked = sorted(
        trained.items(),
        key=lambda kv: (kv[1]["test_metrics"].get(primary) if kv[1]["test_metrics"].get(primary) is not None else -1),
        reverse=True,
    )

    backtest_ok = None
    backtest = run.get("backtest") or {}
    if backtest:
        backtest_ok = all(r.get("completed") for r in backtest.values())

    data_quality_blockers: list[str] = []
    if (manifest.get("row_counts") or {}).get("final", 0) < 500:
        data_quality_blockers.append(
            f"The snapshot has only {(manifest.get('row_counts') or {}).get('final')} rows — too few "
            "for a trustworthy champion comparison."
        )
    if manifest.get("label_stats", {}).get("unmapped_defaulted_to_non_voice"):
        data_quality_blockers.append(
            "Unmapped SubTasks were defaulted to Non-Voice. Labels nobody approved cannot support promotion."
        )

    gate_results: dict = {}
    for candidate_id, record in trained.items():
        gate_results[candidate_id] = evaluator.evaluate_gate(
            champion_metrics=champion_metrics,
            challenger_metrics=record["test_metrics"],
            gate=gate,
            historical={
                "champion": historical.get("champion"),
                "challenger": historical.get(candidate_id),
            },
            backtest_ok=backtest_ok,
            package_ok=None,
            data_quality_blockers=data_quality_blockers,
        )

    leader = ranked[0][0] if ranked else None
    return {
        "run_id": run.get("run_id"),
        "snapshot_id": snapshot_id,
        "benchmark": {
            "rows": int(len(y_test)),
            "actual_non_voice": int((y_test == labeling.NON_VOICE).sum()),
            "actual_voice": int((y_test == labeling.VOICE).sum()),
            "date_from": str(dates.min()) if dates.notna().any() else None,
            "date_to": str(dates.max()) if dates.notna().any() else None,
            "note": "Champion and every challenger are scored on these identical rows.",
        },
        "champion": {
            "model_id": (run.get("champion") or {}).get("model_id"),
            "model_family": (run.get("champion") or {}).get("model_family"),
            "threshold": champion_threshold,
            "test_metrics": champion_metrics,
            "source_metrics_from_package": (run.get("champion") or {}).get("source_metrics_from_package"),
            "latency": (run.get("champion") or {}).get("latency"),
        },
        "candidates": candidates,
        "ranking": [candidate_id for candidate_id, _ in ranked],
        "leading_candidate": leader,
        "gate_results": gate_results,
        "gate_result": gate_results.get(leader) if leader else None,
        "historical": historical,
        "recent": recent,
        "period_breakdown": period_breakdown,
        "segment_breakdown": segment_breakdown,
        "backtest": backtest,
        "backtest_completed": backtest_ok,
        "data_quality_blockers": data_quality_blockers,
        "split": run.get("split"),
        "weights": run.get("weights"),
        "weight_formula": run.get("weight_formula"),
    }
