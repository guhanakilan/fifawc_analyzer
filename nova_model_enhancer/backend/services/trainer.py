"""Challenger training and tuning.

Estimator construction, search spaces, CV structure and calibration all follow
`backend/routers/training.py` in the reference, so a challenger produced here is
structurally the same kind of artifact as the champion it is compared against.

Preprocessing is always refit for a challenger. The champion's fitted imputer,
encoders and scalers are used only to score the champion itself — reusing them
as a challenger's fitted state would leak the champion's training distribution
into the challenger and make the comparison meaningless.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning,
)

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "lr": "Logistic Regression",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}

# Identical to `_DEFAULT_SEARCH_SPACES` in the reference trainer.
DEFAULT_SEARCH_SPACES: dict[str, dict] = {
    "rf": {
        "search_spaces": {
            "n_estimators": {"type": "int", "low": 100, "high": 200, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 10, "high": 50},
        },
        "n_trials": 30,
        "fixed_params": {"class_weight": "balanced"},
    },
    "gb": {
        "search_spaces": {
            "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
            "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 2, "high": 7},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
        "n_trials": 30,
        "fixed_params": {},
    },
    "lr": {
        "search_spaces": {"C": {"type": "float_log", "low": 0.01, "high": 100.0}},
        "n_trials": 20,
        "fixed_params": {"class_weight": "balanced"},
    },
    "xgb": {
        "search_spaces": {
            "n_estimators": {"type": "int", "low": 100, "high": 400, "step": 50},
            "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        },
        "n_trials": 40,
        "fixed_params": {},
    },
    "lgb": {
        "search_spaces": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.2},
            "num_leaves": {"type": "int", "low": 20, "high": 80},
            "min_child_samples": {"type": "int", "low": 10, "high": 100},
        },
        "n_trials": 40,
        "fixed_params": {"class_weight": "balanced"},
    },
}

BASELINE_PARAMS = {"C": 1.0, "class_weight": "balanced"}
HPO_CV_FOLDS = 3
FINAL_CV_FOLDS = 5
ISOTONIC_MIN_VAL_ROWS = 1000


class TrainingCancelled(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    """Raised when a model family's dependency is not installed."""


def available_model_types() -> dict[str, bool]:
    """Which families this installation can actually train right now."""
    available = {"rf": True, "gb": True, "lr": True}
    for key, module in (("xgb", "xgboost"), ("lgb", "lightgbm")):
        try:
            __import__(module)
            available[key] = True
        except ImportError:
            available[key] = False
    return available


def make_estimator(model_id: str, params: dict, n_jobs: int = -1, seed: int = 42):
    """Mirrors `routers/training.py::_make_estimator`."""
    params = dict(params or {})
    if model_id == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if model_id == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params, random_state=seed)
    if model_id == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params, random_state=seed, max_iter=1000, solver="lbfgs")
    if model_id == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelUnavailable("xgboost is not installed in this environment.") from exc
        return XGBClassifier(**params, random_state=seed, eval_metric="logloss",
                             verbosity=0, n_jobs=n_jobs)
    if model_id == "lgb":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ModelUnavailable("lightgbm is not installed in this environment.") from exc
        return LGBMClassifier(**params, random_state=seed, verbose=-1, n_jobs=n_jobs)
    raise ValueError(f"Unknown model type: {model_id!r}")


def effective_sample_weight(model_id: str, y, w_train):
    """GradientBoosting has no class_weight, so balance is folded into weights.

    Multiplicative, so an approved recency/correction weight and class balance
    both apply instead of one silently replacing the other. No-op elsewhere.
    """
    if model_id != "gb":
        return w_train
    from sklearn.utils.class_weight import compute_sample_weight
    balance = compute_sample_weight("balanced", y)
    return balance if w_train is None else balance * w_train


def calibrate(model_id: str, fitted_est, X_val, y_val):
    """Wrap in CalibratedClassifierCV so predict_proba is an honest probability.

    Calibrated on the validation slice the estimator never saw, with the base
    estimator frozen. Falls back to the raw estimator when there is no
    trustworthy validation slice.
    """
    if X_val is None or len(X_val) == 0:
        return fitted_est, False, None
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        method = "isotonic" if len(X_val) >= ISOTONIC_MIN_VAL_ROWS else "sigmoid"
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_est), method=method)
        calibrated.fit(X_val, y_val)
        return calibrated, True, method
    except Exception as exc:
        logger.warning("[%s] calibration failed, using raw probabilities: %s", model_id, exc)
        return fitted_est, False, None


def suggest_params(trial, search_spaces: dict) -> dict:
    params: dict = {}
    for name, spec in search_spaces.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif kind == "categorical":
            choices = [v for v in (spec.get("values") or spec.get("choices") or []) if v is not None]
            if choices:
                params[name] = trial.suggest_categorical(name, choices)
    return params


def _predict_proba(estimator, X) -> np.ndarray:
    values = X.values if hasattr(X, "values") else X
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(values))[:, 1]
    return np.asarray(estimator.predict(values), dtype=float)


def train_candidate(
    *,
    model_type: str,
    mode: str,                       # "tuned" | "fixed"
    X_train, y_train, w_train,
    X_val, y_val,
    X_test, y_test,
    feature_names: list[str],
    fixed_params: dict | None = None,
    search_spaces: dict | None = None,
    n_trials: int = 30,
    timeout_seconds: int | None = None,
    n_jobs: int = -1,
    seed: int = 42,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Train one candidate and return its full result record.

    `mode="fixed"` reproduces a known parameter set without searching — used for
    the champion-family-with-prior-parameters candidate and the simple baseline.
    """
    from sklearn.model_selection import StratifiedKFold

    def _emit(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    def _check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise TrainingCancelled(f"{model_type} cancelled by user request")

    X_train_v = X_train.values if hasattr(X_train, "values") else X_train
    y_train_v = y_train.values if hasattr(y_train, "values") else y_train
    weights = effective_sample_weight(model_type, y_train_v, w_train)

    start = time.time()
    trial_history: list[dict] = []
    optimisation_history: list[dict] = []
    param_importance: dict = {}
    trials_run = 0

    if mode == "fixed":
        best_params = dict(fixed_params or {})
        _emit(0.15, f"{MODEL_LABELS.get(model_type, model_type)} — fitting fixed parameters")
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        spaces = search_spaces or DEFAULT_SEARCH_SPACES.get(model_type, {}).get("search_spaces", {})
        base_params = dict(fixed_params if fixed_params is not None
                           else DEFAULT_SEARCH_SPACES.get(model_type, {}).get("fixed_params", {}))
        best_so_far = [0.0]

        def objective(trial):
            _check_cancelled()
            params = {**base_params, **suggest_params(trial, spaces)}
            cv = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=seed)
            fold_scores = []
            from sklearn.metrics import f1_score
            for fold_index, (tr, va) in enumerate(cv.split(X_train_v, y_train_v)):
                _check_cancelled()
                est = make_estimator(model_type, params, n_jobs=n_jobs, seed=seed)
                est.fit(
                    X_train_v[tr], y_train_v[tr],
                    sample_weight=None if weights is None else weights[tr],
                )
                fold_scores.append(
                    float(f1_score(y_train_v[va], est.predict(X_train_v[va]), zero_division=0))
                )
                trial.report(float(np.mean(fold_scores)), step=fold_index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mean_f1 = float(np.mean(fold_scores))
            is_best = mean_f1 > best_so_far[0]
            if is_best:
                best_so_far[0] = mean_f1
            trial_history.append({
                "trial": trial.number + 1, "params": params,
                "cv_f1": round(mean_f1, 4), "cv_f1_std": round(float(np.std(fold_scores)), 4),
                "is_best": is_best,
            })
            optimisation_history.append({"trial": trial.number + 1, "best_f1": round(best_so_far[0], 4)})
            _emit(
                0.1 + 0.6 * (trial.number + 1) / max(n_trials, 1),
                f"{MODEL_LABELS.get(model_type, model_type)} — trial {trial.number + 1}/{n_trials}, "
                f"best CV F1 {best_so_far[0]:.4f}",
            )
            return mean_f1

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, n_jobs=1,
                       show_progress_bar=False)
        trials_run = len(study.trials)
        best_params = {**base_params, **study.best_params}
        try:
            from optuna.importance import get_param_importances
            param_importance = {k: round(float(v), 4) for k, v in get_param_importances(study).items()}
        except Exception:
            param_importance = {}

    # ── Final CV on the chosen parameters ───────────────────────────────────
    _check_cancelled()
    _emit(0.75, f"{MODEL_LABELS.get(model_type, model_type)} — {FINAL_CV_FOLDS}-fold cross-validation")
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    final_cv = StratifiedKFold(n_splits=FINAL_CV_FOLDS, shuffle=True, random_state=seed)
    fold_results = []
    for index, (tr, va) in enumerate(final_cv.split(X_train_v, y_train_v)):
        _check_cancelled()
        est = make_estimator(model_type, best_params, n_jobs=n_jobs, seed=seed)
        est.fit(X_train_v[tr], y_train_v[tr], sample_weight=None if weights is None else weights[tr])
        pred = est.predict(X_train_v[va])
        proba = _predict_proba(est, X_train_v[va])
        fold_results.append({
            "fold": index + 1,
            "f1": round(float(f1_score(y_train_v[va], pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_train_v[va], pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_train_v[va], pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_train_v[va], proba)), 4),
        })

    def _cv_mean(key: str) -> float:
        return round(sum(f[key] for f in fold_results) / len(fold_results), 4)

    def _cv_std(key: str) -> float:
        mean = _cv_mean(key)
        return round((sum((f[key] - mean) ** 2 for f in fold_results) / len(fold_results)) ** 0.5, 4)

    # ── Fit on the full training slice ──────────────────────────────────────
    _check_cancelled()
    _emit(0.9, f"{MODEL_LABELS.get(model_type, model_type)} — fitting final estimator")
    best_est = make_estimator(model_type, best_params, n_jobs=n_jobs, seed=seed)
    best_est.fit(X_train_v, y_train_v, sample_weight=weights)
    train_time = round(time.time() - start, 2)

    feature_importance: list[dict] = []
    try:
        if hasattr(best_est, "feature_importances_"):
            imps = best_est.feature_importances_
            total = float(imps.sum()) or 1.0
            feature_importance = sorted(
                [{"feature": f, "importance": round(float(i / total), 4)}
                 for f, i in zip(feature_names, imps)],
                key=lambda x: -x["importance"],
            )[:25]
        elif hasattr(best_est, "coef_"):
            coef = best_est.coef_[0] if best_est.coef_.ndim > 1 else best_est.coef_
            abs_coef = np.abs(coef)
            total = float(abs_coef.sum()) or 1.0
            feature_importance = sorted(
                [{"feature": f, "importance": round(float(c / total), 4)}
                 for f, c in zip(feature_names, abs_coef)],
                key=lambda x: -x["importance"],
            )[:25]
    except Exception as exc:
        logger.warning("feature importance failed for %s: %s", model_type, exc)

    final_est, calibrated, calibration_method = calibrate(
        model_type, best_est,
        X_val.values if X_val is not None and hasattr(X_val, "values") else X_val,
        y_val.values if y_val is not None and hasattr(y_val, "values") else y_val,
    )

    proba_val = (
        _predict_proba(final_est, X_val) if X_val is not None and len(X_val) else None
    )
    proba_test = _predict_proba(final_est, X_test)

    return {
        "model_type": model_type,
        "model_label": MODEL_LABELS.get(model_type, model_type),
        "mode": mode,
        "best_params": best_params,
        "n_trials": trials_run,
        "trial_history": trial_history[-200:],
        "optimisation_history": optimisation_history,
        "param_importance": param_importance,
        "cv_folds": fold_results,
        "cv_mean": {k: _cv_mean(k) for k in ("f1", "precision", "recall", "auc")},
        "cv_std": {k: _cv_std(k) for k in ("f1", "precision", "recall", "auc")},
        "train_time_seconds": train_time,
        "train_rows": int(len(y_train_v)),
        "feature_importance": feature_importance,
        "calibrated": calibrated,
        "calibration_method": calibration_method,
        "seed": seed,
        "_estimator": final_est,
        "_raw_estimator": best_est,
        "_proba_val": None if proba_val is None else np.asarray(proba_val),
        "_proba_test": np.asarray(proba_test),
    }


def build_candidate_plan(
    champion_family: str | None,
    champion_params: dict | None,
    available: dict[str, bool],
    second_family: str | None = None,
    include_baseline: bool = True,
    n_trials: int | None = None,
) -> list[dict]:
    """The candidate slate described in the brief, filtered to what can run.

    1. champion family with the champion's own prior parameters (fixed)
    2. champion family with a fresh Optuna search
    3. one alternative family with a fresh Optuna search
    4. logistic-regression baseline
    """
    plan: list[dict] = []
    champion_family = champion_family if champion_family in available and available.get(champion_family) else None

    if champion_family:
        if champion_params:
            plan.append({
                "candidate_id": f"{champion_family}_prior",
                "model_type": champion_family, "mode": "fixed",
                "fixed_params": champion_params,
                "label": f"{MODEL_LABELS.get(champion_family, champion_family)} — champion's prior parameters",
            })
        plan.append({
            "candidate_id": f"{champion_family}_tuned",
            "model_type": champion_family, "mode": "tuned",
            "n_trials": n_trials or DEFAULT_SEARCH_SPACES[champion_family]["n_trials"],
            "label": f"{MODEL_LABELS.get(champion_family, champion_family)} — fresh Optuna search",
        })

    if second_family is None:
        for candidate in ("lgb", "xgb", "rf", "gb"):
            if candidate != champion_family and available.get(candidate):
                second_family = candidate
                break
    if second_family and available.get(second_family) and second_family != champion_family:
        plan.append({
            "candidate_id": f"{second_family}_tuned",
            "model_type": second_family, "mode": "tuned",
            "n_trials": n_trials or DEFAULT_SEARCH_SPACES[second_family]["n_trials"],
            "label": f"{MODEL_LABELS.get(second_family, second_family)} — alternative family",
        })

    if include_baseline and available.get("lr"):
        plan.append({
            "candidate_id": "lr_baseline",
            "model_type": "lr", "mode": "fixed", "fixed_params": dict(BASELINE_PARAMS),
            "label": "Logistic Regression — simple baseline",
        })
    return plan


BACKTEST_FIXED_PARAMS = {
    "lgb": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "class_weight": "balanced"},
    "xgb": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5},
    "rf": {"n_estimators": 150, "max_depth": 8, "class_weight": "balanced"},
    "gb": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4},
    "lr": {"C": 1.0, "class_weight": "balanced"},
}


def rolling_backtest_many(
    df: pd.DataFrame, features_config: dict, dates: pd.Series, model_types: list[str],
    n_windows: int = 4, seed: int = 42, should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Backtest several model types over one set of windows.

    The preprocessing state for a window depends only on the data and the split,
    never on the estimator, so it is fitted once per window and reused by every
    model type. Measured, that saves about 0.6s of an 11s backtest — small, but
    it is pure duplication otherwise.

    Returns {model_type: result}. A model type that fails is recorded with its
    error rather than failing the whole backtest, which is diagnostic only.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from .nova_transform import TARGET, fit_transform_by_indices

    unsupported = [m for m in model_types if m not in BACKTEST_FIXED_PARAMS]
    if unsupported:
        raise ValueError(f"Unsupported backtest model type(s): {unsupported!r}")
    if n_windows < 3:
        raise ValueError("n_windows must be at least 3 (two training windows plus one test window).")

    order = np.argsort(dates.fillna(pd.Timestamp.min).values, kind="stable")
    windows = np.array_split(order, n_windows)
    y_full = df[TARGET].fillna(0).astype(int)

    windows_meta = []
    for index, window in enumerate(windows):
        window_dates = dates.iloc[window]
        windows_meta.append({
            "window": index + 1, "rows": int(len(window)),
            "date_from": str(window_dates.min()) if window_dates.notna().any() else None,
            "date_to": str(window_dates.max()) if window_dates.notna().any() else None,
        })

    results: dict[str, list] = {m: [] for m in model_types}
    errors: dict[str, str] = {}

    for i in range(1, n_windows):
        if should_cancel and should_cancel():
            raise TrainingCancelled("Backtest cancelled by user request")
        train_idx = np.concatenate(windows[:i])
        test_idx = windows[i]

        if y_full.iloc[test_idx].nunique() < 2:
            for model_type in model_types:
                results[model_type].append({
                    "window": i + 1, "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "skipped": "test window contains a single class",
                })
            continue

        # Fitted once for this window, shared by every model type below.
        X_tr, _, X_te, y_tr, _, y_te, _, _ = fit_transform_by_indices(
            df, features_config, train_idx, test_idx, val_idx=None
        )

        for model_type in model_types:
            if model_type in errors:
                continue
            try:
                est = make_estimator(
                    model_type, BACKTEST_FIXED_PARAMS[model_type], n_jobs=-1, seed=seed
                )
                est.fit(X_tr.values, y_tr.values)
                pred = est.predict(X_te.values)
                proba = _predict_proba(est, X_te.values)
                results[model_type].append({
                    "window": i + 1, "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
                    "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
                    "auc": round(float(roc_auc_score(y_te, proba)), 4),
                })
            except TrainingCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — diagnostic, never fails the run
                errors[model_type] = str(exc)

    out = {}
    for model_type in model_types:
        if model_type in errors:
            out[model_type] = {
                "model_type": model_type, "error": errors[model_type], "completed": False,
            }
            continue
        out[model_type] = _summarise_backtest(model_type, n_windows, results[model_type], windows_meta)
    return out


def _summarise_backtest(model_type: str, n_windows: int, results: list, windows_meta: list) -> dict:
    scored = [r for r in results if "skipped" not in r]
    summary = {}
    for key in ("f1", "precision", "recall", "auc"):
        values = [r[key] for r in scored]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }
    return {
        "model_type": model_type, "n_windows": n_windows,
        "windows": windows_meta, "results": results, "summary": summary,
        "completed": True,
    }


def rolling_backtest(
    df: pd.DataFrame, features_config: dict, dates: pd.Series, model_type: str,
    n_windows: int = 4, seed: int = 42, should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Rolling-origin backtest for one model type.

    Thin wrapper over `rolling_backtest_many`; kept because a single model type
    is still the natural unit for a caller that only wants one.
    """
    result = rolling_backtest_many(
        df, features_config, dates, [model_type], n_windows=n_windows,
        seed=seed, should_cancel=should_cancel,
    )[model_type]
    if not result.get("completed", True):
        raise RuntimeError(result["error"])
    return result


# Windows cost roughly a model fit each. Four gives a stability trend without
# doubling the run; the cap was 8, which spent most of the backtest's time for
# resolution nobody was reading.
MAX_BACKTEST_WINDOWS = 4


def auto_backtest_windows(dates: pd.Series) -> int:
    """One window per month of history, floored at 3 and capped at MAX."""
    if dates is None or dates.notna().sum() < 2:
        return 3
    span_days = (dates.max() - dates.min()).days
    return int(min(MAX_BACKTEST_WINDOWS, max(3, round(span_days / 30.0))))
