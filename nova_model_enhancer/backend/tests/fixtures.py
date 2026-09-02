"""Synthetic PLC 984 fixtures reproducing the real NoVA artifact contract.

Nothing here is patient data: every value is generated. The point is that the
generated export has the *shape* of a real nova-ml Stage 12 export — the same
filenames, the same JSON wrapper keys, the same fitted_transforms structure and
the same `{best_model}.pkl` naming — so tests exercise the real contract.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PLACEMENT_ID = 984
RUN_ID = "RUN_20260115_PLC984"
CHAMPION_MODEL_ID = "JOB_20260115_lgb"

PAYERS = ["aetna", "cigna", "bcbs", "united", "medicare", "humana"]
FACILITIES = ["north_clinic", "south_clinic", "east_hospital", "west_hospital"]
SUBTASKS = {
    "Called Insurance": "Voice",
    "IVR Follow Up": "Voice",
    "Portal Status Check": "Non-Voice",
    "Claim Correction": "Non-Voice",
    "Fax Follow Up": "Non-Voice",
    "Mixed Follow Up": "Keyword",
    "Day to Night Transfer": "Ignore",
}
KEYWORDS = ["called", "spoke with", "voicemail", "call ref#"]

# Inventory-side names, mapped to production names via column_map.json.
INVENTORY_RENAMES = {
    "Amount Billed": "AmountBilled",
    "Insurance Balance": "InsuranceBalance",
    "Payer Name": "PayerName",
}


def make_labelled_dataset(rows: int = 6000, seed: int = 7, start: str = "2025-01-01") -> pd.DataFrame:
    """A labelled production-style dataset with a genuine, learnable signal.

    The SubTask a row receives (and therefore its Voice / Non-Voice label) is
    driven by a latent propensity computed from the same columns the champion is
    allowed to see. Without that, no model could beat chance and a champion
    comparison would be meaningless.
    """
    rng = np.random.default_rng(seed)

    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(rng.integers(0, 420, size=rows)), unit="D"
    )
    amount = np.round(rng.gamma(shape=2.2, scale=520, size=rows), 2)
    balance = np.round(amount * rng.uniform(0.05, 0.95, size=rows), 2)
    age_days = rng.integers(1, 400, size=rows)
    payer = rng.choice(PAYERS, size=rows, p=[0.22, 0.18, 0.2, 0.16, 0.14, 0.10])
    facility = rng.choice(FACILITIES, size=rows)
    dos_offset = rng.integers(10, 120, size=rows)

    # Latent propensity to need a phone call: large balances, old accounts and
    # certain payers/facilities push a row towards Voice work.
    payer_effect = pd.Series(payer).map(
        {"aetna": 0.9, "cigna": 0.4, "bcbs": -0.2, "united": -0.5, "medicare": -1.0, "humana": 0.1}
    ).to_numpy(dtype=float)
    facility_effect = pd.Series(facility).map(
        {"north_clinic": 0.5, "south_clinic": -0.3, "east_hospital": 0.2, "west_hospital": -0.4}
    ).to_numpy(dtype=float)
    latent = (
        1.35 * (np.log1p(balance) - np.log1p(balance).mean()) / (np.log1p(balance).std() or 1)
        + 0.85 * (age_days - age_days.mean()) / (age_days.std() or 1)
        + 0.70 * (dos_offset - dos_offset.mean()) / (dos_offset.std() or 1)
        + payer_effect + facility_effect
        + rng.normal(0, 0.75, size=rows)
    )
    voice_probability = 1.0 / (1.0 + np.exp(-latent))

    voice_subtasks = ["Called Insurance", "IVR Follow Up"]
    nonvoice_subtasks = ["Portal Status Check", "Claim Correction", "Fax Follow Up"]
    draw = rng.random(rows)
    subtasks = np.where(
        draw < 0.05, "Mixed Follow Up",
        np.where(
            draw < 0.08, "Day to Night Transfer",
            np.where(
                rng.random(rows) < voice_probability,
                rng.choice(voice_subtasks, size=rows, p=[0.65, 0.35]),
                rng.choice(nonvoice_subtasks, size=rows, p=[0.4, 0.35, 0.25]),
            ),
        ),
    )

    # Keyword-routed rows resolve through ARComments, which follows the same latent.
    voice_words = rng.choice(
        ["called payer for status", "spoke with rep, claim pending", "left voicemail for adjuster"],
        size=rows,
    )
    other_words = rng.choice(
        ["portal shows claim in process", "submitted corrected claim", "faxed medical records"],
        size=rows,
    )
    comments = np.where(rng.random(rows) < voice_probability, voice_words, other_words)

    frame = pd.DataFrame({
        "AccountID": [f"ACC{100000 + i}" for i in range(rows)],
        "PatientAcctNo": [f"PA{500000 + i}" for i in range(rows)],
        "Amount Billed": amount,
        "Insurance Balance": balance,
        "AgeDays": age_days,
        "Payer Name": payer,
        "FacilityName": facility,
        "SubTask": subtasks,
        "Task": np.where(
            pd.Series(subtasks).isin(voice_subtasks).values,
            "Voice Follow Up", "Non Voice Follow Up",
        ),
        "ARComments": comments,
        "DOSFrom": dates - pd.to_timedelta(dos_offset, unit="D"),
        "UpdatedDateTimeGMT": dates,
        "HumanCorrected": rng.choice(["0", "1"], size=rows, p=[0.94, 0.06]),
        "PreviousModelError": rng.choice(["0", "1"], size=rows, p=[0.91, 0.09]),
    })
    return frame


def apply_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Label with the reference's own SubTask/keyword rules."""
    from ..services.labeling import apply_subtask_mapping

    mappings = [{"name": name, "flag": flag} for name, flag in SUBTASKS.items()]
    labelled, _ = apply_subtask_mapping(frame, mappings, KEYWORDS)
    return labelled


def build_configs() -> dict:
    """Config artifacts in the exact on-disk shapes nova-ml writes."""
    column_map = [
        {"inventory": inv, "production": prod, "include": True}
        for inv, prod in INVENTORY_RENAMES.items()
    ]
    column_map += [
        {"inventory": name, "production": name, "include": True}
        for name in ("AgeDays", "FacilityName", "SubTask", "Task", "ARComments",
                     "DOSFrom", "UpdatedDateTimeGMT", "AccountID", "PatientAcctNo")
    ]
    matched = ["amountbilled", "insurancebalance", "agedays", "payername",
               "facilityname", "subtask", "arcomments", "dosfrom", "updateddatetimegmt"]

    return {
        "column_map.json": {"column_map": column_map, "coverage_threshold": None},
        "column_config.json": {"matched_columns": matched},
        "dtype_config.json": {
            "amountbilled": {"dtype": "float64", "fallback": 0},
            "insurancebalance": {"dtype": "float64", "fallback": 0},
            "agedays": {"dtype": "int64", "fallback": 0},
            "payername": {"dtype": "object", "fallback": "unknown"},
            "facilityname": {"dtype": "object", "fallback": "unknown"},
        },
        "derived_config.json": [
            {
                "output_col": "dosage_days", "col_type": "date_diff",
                "date_col": "updateddatetimegmt", "reference_col": "dosfrom",
                "fallback_value": 0, "use_run_date_for_scoring": True,
            },
        ],
        "bucket_config.json": {
            "amountbilled": {"cuts": [400, 900, 1800], "labels": ["low", "mid", "high", "very_high"]},
        },
        "grouping_config.json": {
            "payername": {
                "kept_values": ["aetna", "cigna", "bcbs", "united"],
                "others_label": "other", "null_label": "na",
            },
        },
        "feature_selection.json": {
            "selected_columns": [
                "amountbilled", "insurancebalance", "agedays", "payername",
                "facilityname", "dosage_days",
            ],
        },
        "features_config.json": {
            "outlier_capping": [
                {"col": "insurancebalance", "enabled": True},
                {"col": "dosage_days", "enabled": True},
            ],
            "log_transform": [{"col": "insurancebalance", "enabled": True}],
            "imputation": [
                {"col": "insurancebalance", "strategy": "Median", "enabled": True},
                {"col": "agedays", "strategy": "Median", "enabled": True},
                {"col": "dosage_days", "strategy": "Zero", "enabled": True},
            ],
            "encoding": [
                {"col": "amountbilled_Bucket", "method": "Label", "enabled": True},
                {"col": "payername_Grouped", "method": "One-Hot", "enabled": True},
                {"col": "facilityname", "method": "Frequency", "enabled": True},
            ],
            "scaling": [{"col": "agedays", "scaler": "Standard", "enabled": True}],
            "split": {"mode": "temporal", "train_pct": 80, "stratify": True, "seed": 42},
            "temporal_weight": {"enabled": False},
        },
        "subtask_mappings.json": {
            "mappings": [{"name": name, "flag": flag} for name, flag in SUBTASKS.items()],
            "keywords": KEYWORDS,
        },
    }


def train_reference_champion(labelled: pd.DataFrame, configs: dict, seed: int = 42):
    """Train a champion exactly as nova-ml's own pipeline would.

    Uses this application's reference-parity transform chain, which is the point:
    if the chain were wrong, the champion would not be reproducible.
    """
    from ..services.nova_transform import NovaConfigs, build_modelling_frame, fit_transform_by_indices
    from ..services.splitter import temporal_split_indices
    from ..services.trainer import make_estimator

    raw = {
        "column_map": configs["column_map.json"],
        "column_config": configs["column_config.json"],
        "dtype_config": configs["dtype_config.json"],
        "derived_config": configs["derived_config.json"],
        "bucket_config": configs["bucket_config.json"],
        "grouping_config": configs["grouping_config.json"],
        "feature_selection": configs["feature_selection.json"],
        "features_config": configs["features_config.json"],
        "subtask_mappings": configs["subtask_mappings.json"],
    }
    nova_configs = NovaConfigs(raw, date_column="updateddatetimegmt")
    frame = build_modelling_frame(labelled, nova_configs)
    dates = pd.to_datetime(frame["updateddatetimegmt"], errors="coerce").reset_index(drop=True)
    frame = frame.drop(columns=["updateddatetimegmt"]).reset_index(drop=True)

    y_full = frame["NonVoiceFlag"].astype(int)
    train_idx, val_idx, test_idx = temporal_split_indices(dates, y_full, test_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, fitted = fit_transform_by_indices(
        frame, configs["features_config.json"], train_idx, test_idx, val_idx
    )

    params = {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 31, "class_weight": "balanced"}
    estimator = make_estimator("lgb", params, n_jobs=1, seed=seed)
    estimator.fit(X_train.values, y_train.values)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    proba = estimator.predict_proba(X_test.values)[:, 1]
    predictions = (proba >= 0.5).astype(int)
    test_metrics = {
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float((predictions == y_test.values).mean()), 4),
        "test_rows": int(len(y_test)),
    }
    training_results = {
        "job_id": "JOB_20260115",
        "exp_name": "plc984_baseline",
        "optimizer": "optuna_tpe",
        "best_model": CHAMPION_MODEL_ID,
        "cv_folds": 5,
        "completed_at": "2026-01-15T09:14:00",
        "split_mode": "temporal",
        "training_from": str(dates.min()),
        "training_to": str(dates.max()),
        "voice_rate": round(float((y_full == 0).mean() * 100), 1),
        "results": {
            CHAMPION_MODEL_ID: {
                "model_type": "lgb",
                "best_params": params,
                "cv_mean": {"f1": test_metrics["f1"], "precision": test_metrics["precision"],
                            "recall": test_metrics["recall"], "auc": test_metrics["auc"]},
                "cv_std": {"f1": 0.01, "precision": 0.01, "recall": 0.01, "auc": 0.005},
                "test_metrics": test_metrics,
                "train_rows": int(len(y_train)),
            }
        },
    }
    return estimator, fitted, training_results, feature_names


def write_champion_export(destination: Path, seed: int = 42, rows: int = 6000) -> dict:
    """Produce a complete synthetic NoVA export ZIP plus its source data."""
    import joblib

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = make_labelled_dataset(rows=rows, seed=seed)
    labelled = apply_labels(frame)
    configs = build_configs()
    estimator, fitted, training_results, feature_names = train_reference_champion(
        labelled, configs, seed=seed
    )

    staging = destination.parent / f".{destination.stem}_staging"
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    (staging / "config").mkdir(parents=True)
    (staging / "model").mkdir(parents=True)
    (staging / "scoring").mkdir(parents=True)
    (staging / "metadata").mkdir(parents=True)

    for name, payload in configs.items():
        (staging / "config" / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    joblib.dump(estimator, staging / "model" / f"{CHAMPION_MODEL_ID}.pkl")
    with (staging / "model" / "fitted_transforms.pkl").open("wb") as handle:
        pickle.dump(fitted, handle)

    (staging / "scoring" / "threshold_config.json").write_text(
        json.dumps({CHAMPION_MODEL_ID: 0.5}, indent=2), encoding="utf-8"
    )
    (staging / "metadata" / "training_results.json").write_text(
        json.dumps(training_results, indent=2), encoding="utf-8"
    )
    (staging / "metadata" / "model_selection_config.json").write_text(
        json.dumps({"cv_folds": 5, "optimizer": "optuna_tpe",
                    "best_params": training_results["results"][CHAMPION_MODEL_ID]["best_params"]},
                   indent=2),
        encoding="utf-8",
    )
    (staging / "metadata" / "pipeline_version.json").write_text(
        json.dumps({"version": 2, "written_at": "stage_08"}, indent=2), encoding="utf-8"
    )
    (staging / "metadata" / "manifest.json").write_text(
        json.dumps({
            "placement_id": PLACEMENT_ID, "run_id": RUN_ID, "model_id": CHAMPION_MODEL_ID,
            "exported_at": "2026-01-15T09:20:00Z",
            "metrics": training_results["results"][CHAMPION_MODEL_ID]["test_metrics"],
        }, indent=2),
        encoding="utf-8",
    )

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(staging)).replace("\\", "/"))

    import shutil
    shutil.rmtree(staging)

    return {
        "zip_path": destination,
        "labelled": labelled,
        "raw": frame,
        "configs": configs,
        "training_results": training_results,
        "feature_names": feature_names,
    }


def make_inventory_sample(rows: int = 300, seed: int = 99) -> pd.DataFrame:
    """An unlabelled inventory extract, in inventory-side column names."""
    frame = make_labelled_dataset(rows=rows, seed=seed, start="2026-02-01")
    return frame.drop(columns=["SubTask", "Task", "ARComments", "HumanCorrected",
                               "PreviousModelError"]).reset_index(drop=True)
