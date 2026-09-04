"""The retraining run: split, weight, train challengers, benchmark the champion.

One entry point (`run_retraining`) executes the whole Stage 5 workload inside a
background task and writes every artifact under the job directory. The champion
is never touched: it is loaded read-only and scored through its own fitted state.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import champion as champion_service
from . import evaluator, snapshot as snapshot_service, trainer, weighting
from .nova_transform import (
    TARGET,
    NovaConfigs,
    apply_fitted_transforms,
    build_modelling_frame,
    fit_transform_by_indices,
)
from .splitter import describe_split, random_split_indices, temporal_split_indices


class PipelineError(RuntimeError):
    pass


def _atomic_json(payload: Any, destination: Path) -> None:
    snapshot_service.atomic_write_text(json.dumps(payload, indent=2, default=str), destination)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and arrays to native Python for round-trippable JSON.

    Checkpoints are read back to resume a run, so `default=str` is not good
    enough here: a numpy float written as "0.71" must not come back as a string.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def checkpoint_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoints"


def write_candidate_checkpoint(run_dir: Path, candidate_id: str, result: dict) -> None:
    """Persist one finished candidate so a restart never retrains it.

    Written immediately after the candidate completes, alongside its already
    persisted model .pkl and probability .npy, so the three stay consistent.
    """
    directory = checkpoint_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(_json_safe(result), directory / f"{candidate_id}.json")


def load_candidate_checkpoints(run_dir: Path) -> dict[str, dict]:
    """Read back candidates finished by an earlier attempt at this run.

    A checkpoint counts only when the artifacts it refers to are still present;
    a half-written run directory resumes by retraining, never by trusting a
    checkpoint whose model file has gone.
    """
    directory = checkpoint_dir(run_dir)
    if not directory.is_dir():
        return {}
    recovered: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_id = payload.get("candidate_id")
        if not candidate_id:
            continue
        if "skipped" not in payload:
            model_rel = payload.get("model_path")
            if not model_rel or not (Path(run_dir) / model_rel).exists():
                continue
            if not (Path(run_dir) / f"proba_test_{candidate_id}.npy").exists():
                continue
        recovered[candidate_id] = payload
    return recovered


def _default_features_config(frame: pd.DataFrame, base: dict, date_column: str | None) -> dict:
    """Fill in a transform config when the champion package did not carry one.

    Only used when `features_config.json` is absent — every column gets the
    reference's own default treatment for its dtype, and the choice is recorded
    in the run manifest so it is visible rather than implicit.
    """
    cfg = {k: list(v) for k, v in base.items() if isinstance(v, list)}
    cfg.setdefault("outlier_capping", [])
    cfg.setdefault("log_transform", [])
    cfg.setdefault("imputation", [])
    cfg.setdefault("encoding", [])
    cfg.setdefault("scaling", [])
    configured = {e.get("col") for group in cfg.values() for e in group if isinstance(e, dict)}

    for column in frame.columns:
        if column in (TARGET, date_column) or column in configured:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            cfg["imputation"].append({"col": column, "strategy": "Median", "enabled": True})
        else:
            cfg["encoding"].append({"col": column, "method": "Label", "enabled": True})
    return cfg


def prepare_matrices(
    df: pd.DataFrame, configs: NovaConfigs, date_column: str,
    split_config: dict, weight_strategy: dict,
) -> dict:
    """Build the modelling frame, split it, and compute approved sample weights."""
    frame = build_modelling_frame(df, configs)
    if TARGET not in frame.columns:
        raise PipelineError("The modelling frame lost its NonVoiceFlag target column.")

    date_series = None
    if date_column in frame.columns:
        date_series = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)
        frame = frame.drop(columns=[date_column])
    elif date_column in df.columns:
        date_series = pd.to_datetime(df[date_column], errors="coerce").reset_index(drop=True)

    frame = frame.reset_index(drop=True)
    if date_series is not None and len(date_series) != len(frame):
        raise PipelineError(
            f"Date column length ({len(date_series)}) does not match the modelling frame "
            f"({len(frame)}). The snapshot and its date column are out of step."
        )

    features_config = configs.features_config or {}
    used_default_config = not features_config
    if used_default_config:
        features_config = _default_features_config(frame, {}, date_column)

    y_full = frame[TARGET].fillna(0).astype(int)
    mode = split_config.get("mode", "temporal")
    train_pct = float(split_config.get("train_pct", 70)) / 100.0
    val_pct = float(split_config.get("val_pct", 15)) / 100.0
    test_pct = float(split_config.get("test_pct", 15)) / 100.0
    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 1.0) > 0.01:
        raise PipelineError(
            f"Split percentages must total 100 (received {round(total_pct * 100)})."
        )
    seed = int(split_config.get("seed", 42))

    if mode == "temporal":
        if date_series is None:
            raise PipelineError(
                f"Temporal split requires the approved date column '{date_column}', "
                "which is not present in the snapshot."
            )
        train_idx, val_idx, test_idx = temporal_split_indices(date_series, y_full, test_size=test_pct)
    else:
        train_idx, val_idx, test_idx = random_split_indices(
            y_full, test_size=test_pct, seed=seed, stratify=bool(split_config.get("stratify", True))
        )

    weights_full, weight_summary = weighting.compute_weights(
        df.reset_index(drop=True), weight_strategy, dates=date_series, y=y_full
    )

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, fitted = fit_transform_by_indices(
        frame, features_config, train_idx, test_idx, val_idx
    )

    return {
        "frame": frame,
        "features_config": features_config,
        "used_default_features_config": used_default_config,
        "dates": date_series,
        "indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        "split_description": {
            "mode": mode, "seed": seed,
            "train_pct": round(train_pct * 100), "val_pct": round(val_pct * 100),
            "test_pct": round(test_pct * 100),
            **describe_split(date_series, train_idx, val_idx, test_idx),
        },
        "matrices": {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        },
        "feature_names": feature_names,
        "fitted": fitted,
        "weights": {
            "train": weights_full[train_idx],
            "val": weights_full[val_idx] if len(val_idx) else None,
            "test": weights_full[test_idx],
            "summary": weight_summary,
        },
    }


def benchmark_champion(
    extract_dir: Path, df: pd.DataFrame, indices: dict, date_column: str,
) -> dict:
    """Score the champion on the same rows the challengers are judged on.

    Uses the champion's own fitted transforms and its own threshold, so the
    number reported is what that model would actually have produced.
    """
    champ = champion_service.load_champion(extract_dir, date_column=date_column)
    configs = champ.configs
    configs.date_column = date_column
    frame = build_modelling_frame(df, configs)
    frame = frame.drop(columns=[c for c in (TARGET, date_column) if c in frame.columns])
    X_all = apply_fitted_transforms(frame, champ.fitted)

    y_all = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    proba_all = champion_service.predict_proba(champ.estimator, X_all)

    test_idx = indices["test"]
    val_idx = indices["val"]
    latency = evaluator.measure_latency(
        lambda X: champion_service.predict_proba(champ.estimator, X), X_all.iloc[test_idx]
    )

    return {
        "model_id": champ.model_id,
        "model_family": (configs.training_results.get("results") or {})
                        .get(champ.model_id, {}).get("model_type"),
        "threshold": champ.threshold,
        "feature_count": len(champ.feature_names),
        "proba_all": proba_all,
        "proba_test": proba_all[test_idx],
        "proba_val": proba_all[val_idx] if len(val_idx) else None,
        "y_all": y_all.values,
        "y_test": y_all.values[test_idx],
        "y_val": y_all.values[val_idx] if len(val_idx) else None,
        "latency": latency,
        "source_metrics": (configs.training_results.get("results") or {})
                          .get(champ.model_id, {}).get("test_metrics", {}),
    }


def choose_threshold(
    y_val, proba_val, y_test, proba_test, champion_threshold: float, criterion: str,
) -> dict:
    """Compare the champion threshold, 0.5, and the validation-optimised value.

    The sweep runs on validation rows only. Every candidate is then reported on
    the test rows, which the sweep never saw.
    """
    candidates: dict[str, float] = {
        "champion_threshold": float(champion_threshold),
        "neutral_0.5": 0.5,
    }
    sweep = None
    if proba_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
        sweep = evaluator.threshold_sweep(y_val, proba_val)
        best = sweep["best"].get(criterion) or sweep["best"]["f1"]
        candidates["validation_optimised"] = float(best["threshold"])
    else:
        candidates["validation_optimised"] = float(champion_threshold)

    comparison = []
    for name, value in candidates.items():
        metrics = evaluator.metrics_at_threshold(y_test, proba_test, value)
        comparison.append({"candidate": name, "threshold": value, "test_metrics": metrics})

    selected = max(comparison, key=lambda row: row["test_metrics"].get(criterion) or 0)
    return {
        "criterion": criterion,
        "validation_sweep": sweep,
        "candidates": comparison,
        "selected_threshold": selected["threshold"],
        "selected_candidate": selected["candidate"],
        "selection_note": (
            "Threshold candidates are generated on the validation slice; the value "
            "reported here is each candidate's performance on the held-out test slice."
        ),
    }


def run_retraining(context, job_id: str, settings: dict, paths: dict) -> dict:
    """Full Stage 5 workload. `context` is a tasks.TaskContext."""
    run_dir = Path(paths["run_dir"])
    extract_dir = Path(paths["extract_dir"])
    snapshot_dir = Path(paths["snapshot_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    context.progress(0.02, "Loading immutable dataset snapshot")
    df = snapshot_service.load_snapshot(snapshot_dir, settings["snapshot_id"])
    manifest = snapshot_service.load_manifest(snapshot_dir, settings["snapshot_id"])
    date_column = manifest["date_column"]

    configs = champion_service.load_configs(extract_dir, date_column=date_column)

    context.progress(0.06, "Building modelling frame, split and sample weights")
    prepared = prepare_matrices(
        df, configs, date_column, settings["split"], settings["weight_strategy"]
    )
    matrices = prepared["matrices"]
    indices = prepared["indices"]

    if matrices["X_train"].empty or matrices["X_train"].shape[1] == 0:
        raise PipelineError(
            "The modelling frame produced no usable features. Check the champion's "
            "feature_selection.json against the uploaded data's columns."
        )

    context.progress(0.10, "Scoring the uploaded champion on the same benchmark rows")
    champion_bench = benchmark_champion(extract_dir, df, indices, date_column)

    available = trainer.available_model_types()
    plan = settings.get("candidates") or trainer.build_candidate_plan(
        champion_family=champion_bench.get("model_family"),
        champion_params=settings.get("champion_params"),
        available=available,
        second_family=settings.get("second_family"),
        include_baseline=settings.get("include_baseline", True),
        n_trials=settings.get("n_trials"),
    )
    if not plan:
        raise PipelineError(
            "No candidate models can be trained: none of the supported families are "
            "installed in this environment."
        )

    context.log(f"Candidate plan: {', '.join(c['candidate_id'] for c in plan)}")
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # The fitted preprocessing state is refit for the challengers and shared by
    # all of them: they are trained on identical matrices, so the comparison
    # between challengers isolates the estimator.
    fitted_path = models_dir / "fitted_transforms.pkl"
    with fitted_path.open("wb") as handle:
        pickle.dump(prepared["fitted"], handle)

    # Candidates finished by an earlier attempt at this same run_id. Resuming
    # reuses them verbatim rather than paying for the same fit twice; a fresh
    # run ignores them entirely.
    resumed: dict[str, dict] = (
        load_candidate_checkpoints(run_dir) if settings.get("resume") else {}
    )
    if resumed:
        context.log(
            f"Resuming run {settings['run_id']}: reusing "
            f"{', '.join(sorted(resumed))} from the interrupted attempt"
        )

    challengers: dict[str, dict] = dict(resumed)

    pending = [c for c in plan if c["candidate_id"] not in resumed]
    for candidate in plan:
        if candidate["candidate_id"] in resumed:
            context.log(f"Reusing {candidate['candidate_id']} from the interrupted run")

    # Candidates that cannot run at all are settled before anything is scheduled.
    runnable = []
    for candidate in pending:
        model_type = candidate["model_type"]
        if not available.get(model_type):
            context.log(f"Skipping {candidate['candidate_id']}: {model_type} is not installed")
            challengers[candidate["candidate_id"]] = {
                "candidate_id": candidate["candidate_id"], "model_type": model_type,
                "skipped": f"{model_type} is not installed in this environment",
            }
        else:
            runnable.append(candidate)

    # Candidates are independent fits over identical matrices, so they *can* run
    # concurrently with the cores split between them, as the NoVA workbench does.
    #
    # Measured on 4 cores with 40,000 rows, that is not a win: 1 worker with 4
    # cores took 29.7s, 4 workers with 1 core each 30.0s, and 2 workers with 2
    # cores each 38.6s. The tree models already parallelise internally, so
    # splitting the cores just moves the same work around and adds contention.
    # Sequential is therefore the default. On a machine with many more cores
    # than one model can use, raising max_parallel_candidates may pay off —
    # measure it there rather than assuming, as these numbers show.
    total_cores = os.cpu_count() or 1
    workers = max(1, min(len(runnable), int(settings.get("max_parallel_candidates") or 1)))
    per_model_jobs = max(1, total_cores // workers) if workers > 1 else int(settings.get("n_jobs", -1))
    if runnable:
        context.log(
            f"Training {len(runnable)} candidate(s) across {workers} worker(s), "
            f"{per_model_jobs} core(s) each of {total_cores}"
        )

    completed = 0
    progress_lock = threading.Lock()

    def _train_one(candidate: dict) -> tuple[str, dict]:
        """Fit one candidate and write its artifacts. Runs on a worker thread."""
        nonlocal completed
        model_type = candidate["model_type"]
        label = candidate.get("label", candidate["candidate_id"])

        def _progress(fraction: float, message: str) -> None:
            # Per-candidate fractions are meaningless once several run at once,
            # so only the message is surfaced; the bar tracks completions.
            with progress_lock:
                context.log(f"{candidate['candidate_id']}: {message}")

        try:
            result = trainer.train_candidate(
                model_type=model_type,
                mode=candidate.get("mode", "tuned"),
                X_train=matrices["X_train"], y_train=matrices["y_train"],
                w_train=prepared["weights"]["train"],
                X_val=matrices["X_val"], y_val=matrices["y_val"],
                X_test=matrices["X_test"], y_test=matrices["y_test"],
                feature_names=prepared["feature_names"],
                fixed_params=candidate.get("fixed_params"),
                search_spaces=candidate.get("search_spaces"),
                n_trials=int(candidate.get("n_trials", settings.get("n_trials", 30))),
                timeout_seconds=settings.get("timeout_seconds"),
                n_jobs=per_model_jobs,
                seed=int(settings.get("seed", 42)),
                progress=_progress,
                should_cancel=context.cancelled,
            )
        except trainer.ModelUnavailable as exc:
            return candidate["candidate_id"], {
                "candidate_id": candidate["candidate_id"], "model_type": model_type,
                "skipped": str(exc),
            }

        estimator = result.pop("_estimator")
        result.pop("_raw_estimator", None)
        proba_val = result.pop("_proba_val")
        proba_test = result.pop("_proba_test")

        model_path = models_dir / f"{candidate['candidate_id']}.pkl"
        joblib.dump(estimator, model_path)

        threshold_analysis = choose_threshold(
            matrices["y_val"], proba_val, matrices["y_test"], proba_test,
            champion_bench["threshold"], settings.get("threshold_criterion", "f1"),
        )
        selected_threshold = threshold_analysis["selected_threshold"]
        latency = evaluator.measure_latency(
            lambda X, est=estimator: trainer._predict_proba(est, X), matrices["X_test"]
        )
        test_metrics = evaluator.metrics_at_threshold(
            matrices["y_test"], proba_test, selected_threshold
        )
        result.update({
            "candidate_id": candidate["candidate_id"],
            "label": label,
            "model_path": str(model_path.relative_to(run_dir)),
            "threshold_analysis": threshold_analysis,
            "selected_threshold": selected_threshold,
            "test_metrics": test_metrics,
            "calibration_curve": evaluator.calibration_curve_points(matrices["y_test"], proba_test),
            "latency": latency,
        })
        np.save(run_dir / f"proba_test_{candidate['candidate_id']}.npy", proba_test)
        # Checkpoint last, once the model and probabilities are both on disk, so
        # a checkpoint always implies its artifacts exist.
        write_candidate_checkpoint(run_dir, candidate["candidate_id"], result)

        with progress_lock:
            completed += 1
            context.progress(
                0.14 + 0.72 * (completed / max(len(runnable), 1)),
                f"{completed} of {len(runnable)} candidates finished ({label})",
            )
            context.log(
                f"{candidate['candidate_id']}: test F1 {test_metrics['f1']} at threshold "
                f"{selected_threshold} ({threshold_analysis['selected_candidate']})"
            )
        return candidate["candidate_id"], result

    if context.cancelled():
        raise trainer.TrainingCancelled("Cancelled before training started")

    if runnable:
        context.progress(0.14, f"Training {len(runnable)} candidate(s)")
        if workers == 1:
            for candidate in runnable:
                if context.cancelled():
                    raise trainer.TrainingCancelled("Cancelled before the next candidate")
                candidate_id, result = _train_one(candidate)
                challengers[candidate_id] = result
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_train_one, c): c for c in runnable}
                for future in as_completed(futures):
                    # A cancellation inside any worker surfaces here; the others
                    # observe context.cancelled() and stop at their next check.
                    candidate_id, result = future.result()
                    challengers[candidate_id] = result

    trained = {k: v for k, v in challengers.items() if "skipped" not in v}
    if not trained:
        raise PipelineError("No candidate finished training — see the task log for the reason.")

    # ── Rolling backtest for stability ──────────────────────────────────────
    backtest: dict = {}
    backtest_requested = bool(settings.get("run_backtest", False))
    if backtest_requested and prepared["dates"] is not None:
        context.progress(0.90, "Running rolling-origin backtest")
        windows = int(settings.get("backtest_windows") or trainer.auto_backtest_windows(prepared["dates"]))
        model_types = sorted({v["model_type"] for v in trained.values()})
        try:
            # One pass over the windows for every model type: the preprocessing
            # state depends on the split, not the estimator.
            backtest = trainer.rolling_backtest_many(
                prepared["frame"], prepared["features_config"], prepared["dates"],
                model_types, n_windows=windows, seed=int(settings.get("seed", 42)),
                should_cancel=context.cancelled,
            )
        except trainer.TrainingCancelled:
            raise
        except Exception as exc:  # diagnostic only — never fails the run
            context.log(f"Backtest skipped: {exc}")
            backtest = {
                m: {"model_type": m, "error": str(exc), "completed": False} for m in model_types
            }

    context.progress(0.96, "Writing run artifacts")
    np.save(run_dir / "proba_test_champion.npy", champion_bench["proba_test"])
    np.save(run_dir / "y_test.npy", np.asarray(matrices["y_test"]))
    np.save(run_dir / "test_indices.npy", np.asarray(indices["test"]))

    champion_test_metrics = evaluator.metrics_at_threshold(
        champion_bench["y_test"], champion_bench["proba_test"], champion_bench["threshold"]
    )

    run_record = {
        "run_id": settings["run_id"],
        "snapshot_id": settings["snapshot_id"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            k: v for k, v in settings.items() if k not in ("champion_params",)
        },
        "split": prepared["split_description"],
        "weights": prepared["weights"]["summary"],
        "weight_formula": weighting.formula_text(settings["weight_strategy"]),
        "used_default_features_config": prepared["used_default_features_config"],
        "feature_count": len(prepared["feature_names"]),
        "feature_names": prepared["feature_names"],
        "champion": {
            "model_id": champion_bench["model_id"],
            "model_family": champion_bench["model_family"],
            "threshold": champion_bench["threshold"],
            "feature_count": champion_bench["feature_count"],
            "benchmark_metrics": champion_test_metrics,
            "source_metrics_from_package": champion_bench["source_metrics"],
            "latency": champion_bench["latency"],
        },
        "challengers": challengers,
        "backtest": backtest,
        "backtest_requested": backtest_requested,
        "available_model_types": available,
    }
    _atomic_json(run_record, run_dir / "run_results.json")
    context.progress(1.0, "Retraining complete")
    return {
        "run_id": settings["run_id"],
        "run_dir": str(run_dir),
        "candidates_trained": sorted(trained),
        "candidates_skipped": sorted(k for k, v in challengers.items() if "skipped" in v),
    }
