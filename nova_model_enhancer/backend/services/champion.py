"""Champion package access.

Unpickling executes code. Intake therefore never touches a `.pkl`: it reads
JSON, hashes bytes and records what it found. Loading the estimator and the
fitted transform state happens only inside the explicit compatibility check,
which the user starts after acknowledging a local-trust warning.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .nova_transform import (
    NovaConfigs,
    apply_fitted_transforms,
    build_modelling_frame,
)

CONFIG_FILES = {
    "column_map": "config/column_map.json",
    "column_config": "config/column_config.json",
    "dtype_config": "config/dtype_config.json",
    "derived_config": "config/derived_config.json",
    "bucket_config": "config/bucket_config.json",
    "grouping_config": "config/grouping_config.json",
    "feature_selection": "config/feature_selection.json",
    "features_config": "config/features_config.json",
    "subtask_mappings": "config/subtask_mappings.json",
    "threshold_config": "scoring/threshold_config.json",
    "training_results": "metadata/training_results.json",
    "model_selection_config": "metadata/model_selection_config.json",
    "pipeline_version": "metadata/pipeline_version.json",
    "manifest": "metadata/manifest.json",
}


class ChampionLoadError(RuntimeError):
    pass


def read_configs(extract_dir: Path) -> dict[str, Any]:
    """Read every JSON artifact present. Missing files are simply absent."""
    raw: dict[str, Any] = {}
    for key, relative in CONFIG_FILES.items():
        path = extract_dir / relative
        if not path.exists():
            continue
        try:
            raw[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise ChampionLoadError(f"{relative} is not readable JSON: {exc}") from exc
    return raw


def load_configs(extract_dir: Path, date_column: str | None = None) -> NovaConfigs:
    return NovaConfigs(read_configs(extract_dir), date_column=date_column)


def find_model_file(extract_dir: Path, model_id: str | None) -> Path | None:
    model_dir = extract_dir / "model"
    if not model_dir.is_dir():
        return None
    if model_id:
        candidate = model_dir / f"{model_id}.pkl"
        if candidate.exists():
            return candidate
    candidates = [p for p in sorted(model_dir.glob("*.pkl")) if p.name != "fitted_transforms.pkl"]
    return candidates[0] if len(candidates) == 1 else None


@dataclass
class ChampionArtifacts:
    estimator: Any
    fitted: dict
    configs: NovaConfigs
    model_id: str
    threshold: float
    model_path: Path

    @property
    def feature_names(self) -> list[str]:
        return list(self.fitted.get("feature_names") or [])


def load_champion(extract_dir: Path, date_column: str | None = None) -> ChampionArtifacts:
    """Load estimator + fitted transforms. Only ever called from a trusted step."""
    import joblib

    configs = load_configs(extract_dir, date_column=date_column)
    fitted_path = extract_dir / "model" / "fitted_transforms.pkl"
    if not fitted_path.exists():
        raise ChampionLoadError(
            "model/fitted_transforms.pkl is missing — a model.pkl without its matching "
            "fitted transform state cannot be scored."
        )
    try:
        with fitted_path.open("rb") as handle:
            fitted = pickle.load(handle)
    except Exception as exc:
        raise ChampionLoadError(f"fitted_transforms.pkl could not be unpickled: {exc}") from exc
    if not isinstance(fitted, dict) or not fitted.get("feature_names"):
        raise ChampionLoadError(
            "fitted_transforms.pkl does not contain a 'feature_names' list — the champion's "
            "exact feature order is unknown, so it cannot be scored safely."
        )

    model_id = configs.champion_model_id
    model_path = find_model_file(extract_dir, model_id)
    if model_path is None:
        raise ChampionLoadError(
            f"Champion estimator '{model_id}.pkl' was not found under model/."
            if model_id else
            "Could not resolve which model file is the champion (no best_model in "
            "training_results.json and more than one candidate .pkl)."
        )
    try:
        estimator = joblib.load(model_path)
    except Exception as exc:
        raise ChampionLoadError(f"Champion estimator could not be loaded: {exc}") from exc
    if not hasattr(estimator, "predict"):
        raise ChampionLoadError("Loaded champion object has no predict() method.")

    return ChampionArtifacts(
        estimator=estimator,
        fitted=fitted,
        configs=configs,
        model_id=model_id or model_path.stem,
        threshold=configs.champion_threshold,
        model_path=model_path,
    )


def predict_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """P(class 1) = P(Non-Voice), using the estimator's own interface."""
    values = X.values
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(values))[:, 1]
    return np.asarray(estimator.predict(values), dtype=float)


def score_labelled_frame(
    champion: ChampionArtifacts, df_raw: pd.DataFrame, date_column: str | None = None,
) -> np.ndarray:
    """Score already-labelled rows through the champion's own fitted state.

    The champion's imputer/encoders/scalers are applied as-is and never refitted:
    refitting them would measure a model that never existed.
    """
    configs = champion.configs
    configs.date_column = date_column
    frame = build_modelling_frame(df_raw, configs)
    frame = frame.drop(columns=[c for c in (["NonVoiceFlag", date_column]) if c and c in frame.columns])
    X = apply_fitted_transforms(frame, champion.fitted)
    return predict_proba(champion.estimator, X)


def feature_alignment_report(champion: ChampionArtifacts, df_raw: pd.DataFrame,
                             date_column: str | None = None) -> dict:
    """Compare the champion's expected features against what the data produces."""
    configs = champion.configs
    configs.date_column = date_column
    frame = build_modelling_frame(df_raw, configs)
    frame = frame.drop(columns=[c for c in (["NonVoiceFlag", date_column]) if c and c in frame.columns])
    produced = apply_fitted_transforms(frame, champion.fitted)

    expected = champion.feature_names
    # apply_fitted_transforms reindexes to the fitted names, so compare against the
    # pre-reindex frame to see what the data actually offered.
    available = set(frame.columns)
    onehot_parents = set(champion.fitted.get("onehot_cols") or {})
    onehot_children = {c for cols in (champion.fitted.get("onehot_cols") or {}).values() for c in cols}

    missing = [
        f for f in expected
        if f not in available and f not in onehot_children
    ]
    unexpected = sorted(
        c for c in available
        if c not in expected and c not in onehot_parents
    )
    return {
        "expected_feature_count": len(expected),
        "produced_row_count": int(len(produced)),
        "missing_features": missing[:50],
        "missing_feature_count": len(missing),
        "unused_columns": unexpected[:50],
        "unused_column_count": len(unexpected),
    }
