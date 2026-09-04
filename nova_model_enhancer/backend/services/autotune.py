"""Escalating retrain loop: add capacity until the target is met or the data says no.

Each round gives the model more capacity — more trees, deeper or wider trees —
paired with the regularisation that lets that capacity pay off rather than
simply memorise. The champion's own family is kept throughout, and the feature
set is never touched, so what changes between rounds is the model's settings and
nothing else.

Two rules make the result trustworthy:

  * **Every round is scored on validation.** The test split is not read here at
    all. A loop that iterated against test would reliably reach any target you
    set, and the number it reported would measure how long you searched rather
    than how the model will perform.
  * **The loop can fail.** If the target is not reachable from this data it
    stops and says so, rather than grinding on or quietly lowering the bar.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import trainer

# Multiplied into the champion's own values, round by round. Capacity rises
# while the learning rate falls and regularisation tightens — more trees at the
# same learning rate mostly overfits.
ESCALATION = {
    "lgb": [
        {"n_estimators": 1.0, "num_leaves": 1.0, "learning_rate": 1.0},
        {"n_estimators": 1.5, "num_leaves": 1.25, "learning_rate": 0.8, "min_child_samples": 1.0},
        {"n_estimators": 2.0, "num_leaves": 1.5, "learning_rate": 0.6, "min_child_samples": 1.5},
        {"n_estimators": 3.0, "num_leaves": 2.0, "learning_rate": 0.5, "min_child_samples": 2.0},
        {"n_estimators": 4.0, "num_leaves": 2.5, "learning_rate": 0.4, "min_child_samples": 2.5},
    ],
    "xgb": [
        {"n_estimators": 1.0, "max_depth": 1.0, "learning_rate": 1.0},
        {"n_estimators": 1.5, "max_depth": 1.2, "learning_rate": 0.8, "min_child_weight": 1.0},
        {"n_estimators": 2.0, "max_depth": 1.4, "learning_rate": 0.6, "min_child_weight": 1.5},
        {"n_estimators": 3.0, "max_depth": 1.6, "learning_rate": 0.5, "min_child_weight": 2.0},
        {"n_estimators": 4.0, "max_depth": 1.8, "learning_rate": 0.4, "min_child_weight": 2.5},
    ],
    "rf": [
        {"n_estimators": 1.0, "max_depth": 1.0},
        {"n_estimators": 1.5, "max_depth": 1.25, "min_samples_leaf": 0.8},
        {"n_estimators": 2.0, "max_depth": 1.5, "min_samples_leaf": 0.6},
        {"n_estimators": 3.0, "max_depth": 2.0, "min_samples_leaf": 0.5},
    ],
    "gb": [
        {"n_estimators": 1.0, "max_depth": 1.0, "learning_rate": 1.0},
        {"n_estimators": 1.5, "max_depth": 1.2, "learning_rate": 0.8},
        {"n_estimators": 2.0, "max_depth": 1.4, "learning_rate": 0.6},
        {"n_estimators": 3.0, "max_depth": 1.6, "learning_rate": 0.5},
    ],
    "lr": [
        {"C": 1.0}, {"C": 3.0}, {"C": 10.0}, {"C": 0.3}, {"C": 0.1},
    ],
}

# Sensible floors and ceilings, so a multiplier cannot produce a nonsense model.
BOUNDS = {
    "n_estimators": (50, 2000),
    "num_leaves": (8, 512),
    "max_depth": (2, 16),
    "min_child_samples": (5, 200),
    "min_child_weight": (1, 50),
    "min_samples_leaf": (1, 100),
    "learning_rate": (0.005, 0.5),
    "C": (0.001, 1000.0),
}

INTEGER_PARAMS = {
    "n_estimators", "num_leaves", "max_depth", "min_child_samples",
    "min_child_weight", "min_samples_leaf",
}

DEFAULT_BASE = {
    "lgb": {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
    "xgb": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1},
    "rf": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    "gb": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
    "lr": {"C": 1.0},
}


class AutotuneError(RuntimeError):
    pass


def _clamp(name: str, value: float) -> Any:
    low, high = BOUNDS.get(name, (None, None))
    if low is not None:
        value = max(low, min(high, value))
    return int(round(value)) if name in INTEGER_PARAMS else round(float(value), 6)


def build_rounds(
    model_type: str, champion_params: dict | None, max_rounds: int, balance: bool | None,
) -> list[dict]:
    """The parameter set for each round, grown from the champion's own values."""
    ladder = ESCALATION.get(model_type)
    if ladder is None:
        raise AutotuneError(f"No escalation ladder is defined for model type {model_type!r}.")

    base = {**DEFAULT_BASE.get(model_type, {}), **{
        k: v for k, v in (champion_params or {}).items()
        if k in BOUNDS and isinstance(v, (int, float)) and not isinstance(v, bool)
    }}

    rounds = []
    for index in range(max_rounds):
        # Past the end of the ladder, keep extending its last step.
        step = ladder[min(index, len(ladder) - 1)]
        if index >= len(ladder):
            growth = 1.0 + 0.5 * (index - len(ladder) + 1)
            step = {k: (v * growth if k in ("n_estimators", "num_leaves") else v)
                    for k, v in step.items()}

        params = {}
        for name, multiplier in step.items():
            if name in base:
                params[name] = _clamp(name, base[name] * multiplier)
        # Carry anything the champion set that this ladder does not touch.
        for name, value in base.items():
            params.setdefault(name, value)
        if balance is not None and model_type in ("lgb", "rf", "lr"):
            params["class_weight"] = "balanced" if balance else None
        rounds.append(params)
    return rounds


def run(
    *,
    model_type: str,
    champion_params: dict | None,
    X_train, y_train, w_train,
    X_val, y_val,
    target_metric: str = "f1",
    target_value: float | None = None,
    threshold: float = 0.5,
    max_rounds: int = 8,
    time_budget_seconds: float | None = 300.0,
    patience: int = 3,
    min_gain: float = 0.001,
    noise_std: float | None = None,
    balance: bool | None = None,
    seed: int = 42,
    n_jobs: int = -1,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Escalate until the target is met, progress stalls, or a budget runs out.

    Scored on validation throughout: `X_val`/`y_val` are the only data read here.
    Returns the best parameters found, the full history and why it stopped.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    scorers = {
        "f1": lambda yt, pred, pb: f1_score(yt, pred, zero_division=0),
        "precision": lambda yt, pred, pb: precision_score(yt, pred, zero_division=0),
        "recall": lambda yt, pred, pb: recall_score(yt, pred, zero_division=0),
        "auc": lambda yt, pred, pb: roc_auc_score(yt, pb) if len(np.unique(yt)) > 1 else float("nan"),
    }
    score_fn = scorers.get(target_metric, scorers["f1"])

    rounds = build_rounds(model_type, champion_params, max_rounds, balance)
    started = time.perf_counter()
    history: list[dict] = []
    best: dict | None = None
    since_improvement = 0
    stop_reason = "rounds_exhausted"

    for index, params in enumerate(rounds):
        if should_cancel and should_cancel():
            stop_reason = "cancelled"
            break

        elapsed = time.perf_counter() - started
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            stop_reason = "time_budget"
            break

        if progress:
            progress(index / max(len(rounds), 1),
                     f"Escalation round {index + 1} of {len(rounds)}")

        try:
            estimator = trainer.make_estimator(model_type, params, n_jobs=n_jobs, seed=seed)
            estimator.fit(X_train.values, y_train.values,
                          **({"sample_weight": w_train} if w_train is not None else {}))
            proba = trainer._predict_proba(estimator, X_val.values)
            pred = (proba >= threshold).astype(int)
            score = float(score_fn(y_val, pred, proba))
        except Exception as exc:  # noqa: BLE001 — a bad rung must not sink the loop
            history.append({"round": index + 1, "params": params, "error": str(exc)})
            continue

        improved = best is None or score > best["score"] + min_gain
        entry = {
            "round": index + 1,
            "params": params,
            "score": round(score, 5),
            "metric": target_metric,
            "improved": improved,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        history.append(entry)

        if improved:
            best = {"score": score, "params": params, "round": index + 1}
            since_improvement = 0
        else:
            since_improvement += 1

        if target_value is not None and score >= target_value:
            stop_reason = "target_reached"
            break

        # The gain is smaller than the run-to-run variation, so more searching
        # is chasing randomness rather than signal.
        if (
            noise_std is not None
            and best is not None
            and index > 0
            and (best["score"] - history[0].get("score", best["score"])) < noise_std
            and since_improvement >= 1
        ):
            stop_reason = "gain_within_noise"
            break

        if since_improvement >= patience:
            stop_reason = "no_improvement"
            break

    if best is None:
        # Stopping before any round finished is not a failure — the budget ran
        # out, or the run was cancelled. Report that plainly rather than raising
        # an error implying the rounds themselves went wrong.
        if stop_reason in ("time_budget", "cancelled"):
            return {
                "model_type": model_type,
                "best_params": None,
                "best_score": None,
                "best_round": None,
                "target_metric": target_metric,
                "target_value": target_value,
                "target_reached": False,
                "rounds_run": 0,
                "rounds_planned": len(rounds),
                "stop_reason": stop_reason,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "history": history,
                "scored_on": "validation",
                "note": STOP_REASONS.get(stop_reason, "Stopped before any round completed."),
            }
        raise AutotuneError(
            "No escalation round produced a usable model — see the run log for each "
            "round's error."
        )

    reached = target_value is None or best["score"] >= target_value
    return {
        "model_type": model_type,
        "best_params": best["params"],
        "best_score": round(best["score"], 5),
        "best_round": best["round"],
        "target_metric": target_metric,
        "target_value": target_value,
        "target_reached": bool(reached),
        "rounds_run": len([h for h in history if "score" in h]),
        "rounds_planned": len(rounds),
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "history": history,
        "scored_on": "validation",
        "note": (
            "Every round above is scored on the validation split. The test split was "
            "not read during the search, so the test metrics reported for the winning "
            "model remain an honest estimate."
        ),
    }


STOP_REASONS = {
    "target_reached": "The target was met.",
    "no_improvement": "Stopped: several consecutive rounds failed to improve. The data, "
                      "not the search, is the ceiling here.",
    "time_budget": "Stopped: the time budget ran out before the target was met.",
    "rounds_exhausted": "Stopped: the round limit was reached before the target was met.",
    "gain_within_noise": "Stopped: the improvement is smaller than the run-to-run variation, "
                         "so further searching would be chasing randomness.",
    "cancelled": "Cancelled by user request.",
}
