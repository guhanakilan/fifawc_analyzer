"""Reference-parity NoVA transformation chain.

Every function here reproduces behaviour read out of the reference application.
Where the reference has two code paths for the same idea (the training path in
`data/feature_pipeline.py` + `routers/eda.py`, and the scoring path in
`scoring_client/scoring.py`), the difference is preserved exactly rather than
harmonised, because the champion's fitted state was produced under the training
path and is consumed under the scoring path.

Known difference, deliberately preserved:
  * training bucket/grouping suffixes are ``_Bucket`` / ``_Grouped`` (capitalised)
  * scoring bucket/grouping suffixes are ``_bucket`` / ``_grouped`` (lowercase)
Both resolve to the same identity because the scoring loader normalises
``feature_names`` with :func:`norm_col` before use.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET = "NonVoiceFlag"


# ── Column identity ──────────────────────────────────────────────────────────

def norm_col(name: Any) -> str:
    """strip -> collapse whitespace -> drop special chars -> lowercase.

    Byte-identical to the reference `_norm_col` in `routers/eda.py`,
    `routers/scoring.py`, `data/file_store.py` and `scoring_client/scoring.py`.
    Any divergence silently re-identifies columns.
    """
    s = re.sub(r"\s+", " ", str(name).strip())
    s = re.sub(r"[^\w\s]", "", s).strip()
    return s.lower()


def dedupe_columns(cols: Iterable[str]) -> list[str]:
    """Append _2, _3, ... to repeated names so every lookup stays a Series.

    Matches `scoring_client/scoring.py::_dedupe_columns`. A genuine collision
    (e.g. "Sub-Task" and "SubTask" both normalising to "subtask") must resolve to
    the same subtask/subtask_2 identity at score time as it did at train time.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        n = seen.get(c, 0)
        out.append(c if n == 0 else f"{c}_{n + 1}")
        seen[c] = n + 1
    return out


def normalise_frame_columns(df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
    """Normalise every column name, preserving `NonVoiceFlag`'s exact casing.

    Mirrors `routers/eda.py::_load_eda_df`, which excludes the target from
    normalisation so downstream code can address it by its documented name.
    """
    df = df.copy()
    df.columns = dedupe_columns(
        [c if (keep_target and c == TARGET) else norm_col(c) for c in df.columns]
    )
    return df


# ── Tolerant config readers ──────────────────────────────────────────────────
# nova-ml writes several configs wrapped in a single-key object while its own
# scoring client reads them as bare collections (defects D1/D2 in
# IMPLEMENTATION_GAP_ANALYSIS.md). Read both shapes; never guess between them.

def unwrap_list(value: Any, key: str) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        inner = value.get(key)
        if isinstance(inner, list):
            return inner
    return []


def unwrap_dict(value: Any, key: str | None = None) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        if key and isinstance(value.get(key), dict):
            return value[key]
        return value
    return {}


# ── Column rename (inventory -> production) ─────────────────────────────────

def build_rename_map(column_map: list, df_columns: Iterable[str]) -> dict[str, str]:
    """Actual-column -> normalised-production-name rename dict.

    Matches `scoring_client/scoring.py::_build_rename_map`.
    """
    norm_to_prod = {
        norm_col(r["inventory"]): norm_col(r["production"])
        for r in column_map
        if isinstance(r, dict)
        and r.get("include", True)
        and r.get("inventory") and r.get("production")
        and norm_col(r["inventory"]) != norm_col(r["production"])
    }
    rename: dict[str, str] = {}
    for col in df_columns:
        prod = norm_to_prod.get(norm_col(col))
        if prod and prod != col:
            rename[col] = prod
    return rename


# ── Dtype coercion ───────────────────────────────────────────────────────────

def apply_dtype_config(df: pd.DataFrame, dtype_config: dict) -> pd.DataFrame:
    """Reimplementation of `file_store.apply_dtype_config` / scoring's copy.

    Colliding normalised names are refused rather than resolved to an arbitrary
    physical column — guessing there applies a dtype override to the wrong data.
    """
    if not dtype_config:
        return df
    df = df.copy()

    norm_map: dict[str, str] = {}
    collided: set[str] = set()
    for c in df.columns:
        key = norm_col(c)
        if key in norm_map and norm_map[key] != c:
            collided.add(key)
        else:
            norm_map[key] = c

    for col, spec in dtype_config.items():
        if not isinstance(spec, dict):
            continue
        key = norm_col(col)
        actual = col if col in df.columns else (norm_map.get(key) if key not in collided else None)
        if actual is None:
            continue
        col = actual
        target_dtype = spec.get("dtype", "")
        fallback = spec.get("fallback")
        if not target_dtype:
            continue
        try:
            if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.StringDtype):
                s = df[col].astype(str).str.strip()
                df[col] = s.where(s.ne("") & s.ne("nan") & s.ne("None"), other=pd.NA)

            if target_dtype in ("float64", "float32"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if fallback is not None:
                    df[col] = df[col].fillna(float(fallback))
                df[col] = df[col].astype(target_dtype)
            elif target_dtype in ("int64", "int32"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                fb = float(fallback) if fallback is not None else 0.0
                df[col] = df[col].fillna(fb).astype(target_dtype)
            elif target_dtype == "bool":
                df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
            elif target_dtype in ("object", "string"):
                df[col] = df[col].astype(object)
                mask = df[col].notna()
                if mask.any():
                    s = df[col][mask].astype(str).str.lower()
                    s = s.str.replace(r"(?:[^\w\s]|_)+", "", regex=True).str.strip()
                    s = s.str.replace(" ", "_", regex=False)
                    df.loc[mask, col] = s.where(s.ne("") & s.ne("nan") & s.ne("none"), other=pd.NA)
                if fallback:
                    df[col] = df[col].fillna(str(fallback))
            elif "datetime" in target_dtype:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(target_dtype)
        except Exception:
            if fallback is not None:
                try:
                    df[col] = fallback
                except Exception:
                    pass
    return df


# ── Derived columns ──────────────────────────────────────────────────────────

def _eval_condition(df: pd.DataFrame, col: str, op: str, value) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if op == "is_null":
        return series.isna()
    if op == "is_not_null":
        return series.notna()
    if is_numeric:
        try:
            val = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return pd.Series([False] * len(df), index=df.index)
        if op == ">":  return series > val
        if op == ">=": return series >= val
        if op == "<":  return series < val
        if op == "<=": return series <= val
        if op == "==": return series == val
        if op == "!=": return series != val
    else:
        str_s = series.fillna("").astype(str)
        val = str(value) if value is not None else ""
        if op == "==":          return str_s == val
        if op == "!=":          return str_s != val
        if op == "contains":    return str_s.str.contains(val, na=False)
        if op == "starts_with": return str_s.str.startswith(val, na=False)
    return pd.Series([False] * len(df), index=df.index)


def apply_derived_config(df: pd.DataFrame, derived: list) -> pd.DataFrame:
    """Recreate date_diff / date_part / condition columns.

    Mirrors `routers/custom_cols._apply_derived_config` and the scoring client's
    copy, including its case-insensitive resolution of source column names
    against an already-normalised frame.
    """
    if not derived:
        return df
    df = df.copy()
    col_lower = {c.lower(): c for c in df.columns}

    def _res(name: str) -> str:
        if not name:
            return name
        return col_lower.get(str(name).lower(), name)

    for col_def in derived:
        if not isinstance(col_def, dict):
            continue
        out_col = col_def.get("output_col", "")
        col_type = col_def.get("col_type", "condition")
        fallback_val = col_def.get("fallback_value") or None
        if not out_col:
            continue

        if col_type == "date_diff":
            date_col = _res(col_def.get("date_col", ""))
            ref_col = _res(col_def.get("reference_col", ""))
            if not date_col or not ref_col or date_col not in df.columns or ref_col not in df.columns:
                if fallback_val is not None:
                    df[out_col] = fallback_val
                continue
            try:
                d1 = pd.to_datetime(df[date_col], errors="coerce", format="mixed").dt.normalize()
                d2 = pd.to_datetime(df[ref_col], errors="coerce", format="mixed").dt.normalize()
                diff = (d1 - d2).dt.total_seconds() / 86400
                if fallback_val is not None:
                    diff = diff.fillna(float(fallback_val))
                df[out_col] = diff.round(2)
            except Exception:
                if fallback_val is not None:
                    df[out_col] = fallback_val
            continue

        if col_type == "date_part":
            src_col = _res(col_def.get("date_part_col", ""))
            part_type = col_def.get("date_part_type", "month")
            if src_col not in df.columns:
                if fallback_val is not None:
                    df[out_col] = fallback_val
                continue
            try:
                dt = pd.to_datetime(df[src_col], errors="coerce", format="mixed")
                if part_type == "month":
                    result = dt.dt.strftime("%b")
                elif part_type == "year":
                    result = dt.dt.year.astype("Int64").astype(str)
                elif part_type == "quarter":
                    result = "Q" + dt.dt.quarter.astype("Int64").astype(str)
                elif part_type == "month_year":
                    result = dt.dt.strftime("%b-%y")
                elif part_type == "quarter_year":
                    result = ("Q" + dt.dt.quarter.astype("Int64").astype(str)
                              + "-" + dt.dt.strftime("%y"))
                else:
                    result = dt.dt.strftime("%b")
                if fallback_val is not None:
                    result = result.fillna(fallback_val)
                df[out_col] = result
            except Exception:
                if fallback_val is not None:
                    df[out_col] = fallback_val
            continue

        branches = col_def.get("branches", [])
        result = pd.Series([None] * len(df), index=df.index, dtype=object)
        for branch in branches:
            if branch.get("is_else"):
                continue
            try:
                mask = _eval_condition(df, _res(branch["source_col"]), branch["op"], branch.get("value"))
                unset = result.isna()
                result[unset & mask] = branch["result"]
            except Exception:
                continue
        for branch in branches:
            if branch.get("is_else"):
                result[result.isna()] = branch["result"]
                break
        if fallback_val is not None:
            result[result.isna()] = fallback_val
        df[out_col] = result

    return df


# ── Bucketing / grouping ─────────────────────────────────────────────────────

def apply_bucketing(df: pd.DataFrame, bucket_config: dict, suffix: str = "_Bucket") -> pd.DataFrame:
    """col -> col{suffix} (string), original dropped. Mirrors `_load_base_df`."""
    if not bucket_config:
        return df
    df = df.copy()
    for col, cfg in bucket_config.items():
        if col not in df.columns:
            continue
        try:
            cuts = sorted(float(c) for c in cfg.get("cuts", []))
            labels = cfg.get("labels", [])
            if cuts and labels:
                bins = [-np.inf] + cuts + [np.inf]
                df[f"{col}{suffix}"] = pd.cut(
                    pd.to_numeric(df[col], errors="coerce"),
                    bins=bins, labels=labels, include_lowest=True,
                ).astype(str)
        except Exception:
            pass
        df = df.drop(columns=[col])
    return df


def apply_grouping(df: pd.DataFrame, grouping_config: dict, suffix: str = "_Grouped") -> pd.DataFrame:
    """col -> col{suffix} (string), original dropped. Mirrors `_load_base_df`."""
    if not grouping_config:
        return df
    df = df.copy()
    for col, cfg in grouping_config.items():
        if col not in df.columns:
            continue
        kept = set(cfg.get("kept_values", []))
        others_label = cfg.get("others_label", "Other")
        null_label = cfg.get("null_label", "NA")
        source = df[col]
        grouped = source.astype(str).where(source.astype(str).isin(kept), other=others_label)
        grouped = grouped.where(source.notna(), other=null_label)
        df[f"{col}{suffix}"] = grouped
        df = df.drop(columns=[col])
    return df


def translate_feature_selection(
    selected: list, bucket_config: dict, grouping_config: dict,
    bucket_suffix: str = "_Bucket", grouping_suffix: str = "_Grouped",
) -> list[str]:
    """Map original names in feature_selection.json to their bucketed/grouped form."""
    rename = {orig: f"{orig}{bucket_suffix}" for orig in bucket_config}
    rename.update({orig: f"{orig}{grouping_suffix}" for orig in grouping_config})
    return [rename.get(c, c) for c in selected]


# ── The full modelling frame (training path) ─────────────────────────────────

def build_modelling_frame(df: pd.DataFrame, configs: "NovaConfigs") -> pd.DataFrame:
    """Reproduce nova-ml's training-time frame from a labelled DataFrame.

    Chain, in the reference's order:
      1. inventory -> production rename + column normalisation (target keeps casing)
      2. Stage 03 column filter (`column_config.json`)
      3. Stage 03 dtype overrides (`dtype_config.json`)
      4. Stage 04 derived columns (`derived_config.json`)
      5. Stage 07 bucketing then grouping
      6. Stage 06 rename-aware feature selection, target always retained
    """
    rename = build_rename_map(configs.column_map, df.columns)
    if rename:
        df = df.rename(columns=rename)
    df = normalise_frame_columns(df)

    if configs.column_config:
        keep = [c for c in configs.column_config if c in df.columns]
        if TARGET in df.columns and TARGET not in keep:
            keep.append(TARGET)
        # Retain the date column even when the column filter omits it — the
        # temporal split and recency weighting both need it, and it is dropped
        # again before the feature matrix is built.
        if configs.date_column and configs.date_column in df.columns and configs.date_column not in keep:
            keep.append(configs.date_column)
        df = df[keep]

    df = apply_dtype_config(df, configs.dtype_config)
    df = apply_derived_config(df, configs.derived_config)
    df = apply_bucketing(df, configs.bucket_config, "_Bucket")
    df = apply_grouping(df, configs.grouping_config, "_Grouped")

    if configs.feature_selection:
        translated = set(translate_feature_selection(
            configs.feature_selection, configs.bucket_config, configs.grouping_config
        ))
        keep = [c for c in df.columns if c == TARGET or c in translated
                or c == configs.date_column]
        df = df[keep]
    return df


# ── features_config.json fit / apply ─────────────────────────────────────────

def fit_outlier_bounds(df_train: pd.DataFrame, cfg_list: list) -> dict:
    bounds: dict = {}
    for entry in cfg_list:
        if not entry.get("enabled", False):
            continue
        col = entry.get("col")
        if not col or col not in df_train.columns:
            continue
        arr = pd.to_numeric(df_train[col], errors="coerce").dropna().astype(float)
        if len(arr) < 4:
            continue
        q1, q3 = float(arr.quantile(0.25)), float(arr.quantile(0.75))
        iqr = q3 - q1
        bounds[col] = {"lower": round(q1 - 1.5 * iqr, 6), "upper": round(q3 + 1.5 * iqr, 6)}
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    for col, b in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=b["lower"], upper=b["upper"])
    return df


def fit_log_cols(df_train: pd.DataFrame, cfg_list: list) -> list:
    return [e["col"] for e in cfg_list
            if e.get("enabled", False) and e.get("col") in df_train.columns]


def apply_log_cols(df: pd.DataFrame, log_cols: list) -> pd.DataFrame:
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
    return df


def fit_imputation(df_train: pd.DataFrame, cfg_list: list) -> dict:
    vals: dict = {}
    for entry in cfg_list:
        if not entry.get("enabled", False):
            continue
        col, strategy = entry.get("col"), entry.get("strategy", "Median")
        if not col or col not in df_train.columns:
            continue
        if strategy == "Median":
            fill_val = pd.to_numeric(df_train[col], errors="coerce").median()
        elif strategy == "Mean":
            fill_val = pd.to_numeric(df_train[col], errors="coerce").mean()
        elif strategy == "Zero":
            fill_val = 0
        elif strategy == "Mode":
            mode_s = df_train[col].mode()
            fill_val = mode_s.iloc[0] if len(mode_s) > 0 else np.nan
        elif strategy == "Missing":
            fill_val = "Missing"
        else:
            continue
        vals[col] = (
            fill_val if isinstance(fill_val, str)
            else float(fill_val) if fill_val is not None and fill_val == fill_val
            else None
        )
    return vals


def apply_imputation(df: pd.DataFrame, vals: dict) -> pd.DataFrame:
    for col, fill_val in vals.items():
        if col not in df.columns or fill_val is None:
            continue
        if isinstance(fill_val, (int, float)) and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(fill_val)
    return df


def fit_encoding(df_train: pd.DataFrame, cfg_list: list) -> tuple[dict, dict, dict]:
    label_encoders, freq_maps, onehot_cols = {}, {}, {}
    for entry in cfg_list:
        if not entry.get("enabled", False):
            continue
        col, method = entry.get("col"), entry.get("method", "Label")
        if not col or col not in df_train.columns:
            continue
        if method == "Label":
            classes = sorted(df_train[col].astype(str).fillna("NA").unique().tolist())
            label_encoders[col] = {"classes": classes}
        elif method == "Frequency":
            freq = df_train[col].astype(str).value_counts(normalize=True).to_dict()
            freq_maps[col] = {str(k): round(float(v), 8) for k, v in freq.items()}
        elif method == "One-Hot":
            dummies = pd.get_dummies(df_train[col].astype(str).fillna("NA"), prefix=col)
            onehot_cols[col] = list(dummies.columns)
    return label_encoders, freq_maps, onehot_cols


def apply_encoding(df: pd.DataFrame, label_encoders: dict, freq_maps: dict, onehot_cols: dict) -> pd.DataFrame:
    """Unseen categories: label -> -1, frequency -> 0.0, one-hot -> all-zero row.

    Identical to both the reference training and scoring paths, so internal
    validation metrics remain an honest proxy for real scoring behaviour.
    """
    for col, enc in label_encoders.items():
        if col in df.columns:
            class_map = {c: i for i, c in enumerate(enc.get("classes", []))}
            df[col] = df[col].astype(str).map(class_map).fillna(-1).astype(int)
    for col, freq_map in freq_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map({str(k): v for k, v in freq_map.items()}).fillna(0.0)
    for col, oh_cols in onehot_cols.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str).fillna("NA"), prefix=col)
            df = df.drop(columns=[col])
            for oh_col in oh_cols:
                if oh_col not in dummies.columns:
                    dummies[oh_col] = 0
            df = pd.concat([df, dummies[oh_cols]], axis=1)
    return df


def _to_xy(d: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in d.columns if c != target]
    X = d[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = d[target].fillna(0).astype(int)
    return X, y


def fit_transform_by_indices(
    df: pd.DataFrame, cfg: dict, train_idx, test_idx, val_idx=None,
) -> tuple:
    """Fit every features_config transform on train rows only; apply to val/test.

    Mirrors `feature_pipeline._fit_transform_by_indices` step for step so a
    challenger's fitted state is interchangeable with a nova-ml champion's.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, fitted).
    """
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy() if val_idx is not None and len(val_idx) else None
    df_test = df.iloc[test_idx].copy()

    outlier_bounds = fit_outlier_bounds(df_train, cfg.get("outlier_capping", []))
    df_train = apply_outlier_bounds(df_train, outlier_bounds)
    df_test = apply_outlier_bounds(df_test, outlier_bounds)
    if df_val is not None:
        df_val = apply_outlier_bounds(df_val, outlier_bounds)

    log_cols = fit_log_cols(df_train, cfg.get("log_transform", []))
    df_train = apply_log_cols(df_train, log_cols)
    df_test = apply_log_cols(df_test, log_cols)
    if df_val is not None:
        df_val = apply_log_cols(df_val, log_cols)

    imputation_vals = fit_imputation(df_train, cfg.get("imputation", []))
    df_train = apply_imputation(df_train, imputation_vals)
    df_test = apply_imputation(df_test, imputation_vals)
    if df_val is not None:
        df_val = apply_imputation(df_val, imputation_vals)

    label_encoders, freq_maps, onehot_cols = fit_encoding(df_train, cfg.get("encoding", []))
    df_train = apply_encoding(df_train, label_encoders, freq_maps, onehot_cols)
    df_test = apply_encoding(df_test, label_encoders, freq_maps, onehot_cols)
    if df_val is not None:
        df_val = apply_encoding(df_val, label_encoders, freq_maps, onehot_cols)

    X_train, y_train = _to_xy(df_train)
    X_test, y_test = _to_xy(df_test)
    X_val, y_val = (_to_xy(df_val) if df_val is not None else (None, None))

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    if X_val is not None:
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)

    scalers: dict = {}
    for entry in cfg.get("scaling", []):
        if not entry.get("enabled", False):
            continue
        col, scaler_type = entry.get("col"), entry.get("scaler", "Standard")
        if not col or col not in X_train.columns:
            continue
        if scaler_type == "Standard":
            sc = StandardScaler()
        elif scaler_type == "MinMax":
            sc = MinMaxScaler()
        else:
            continue
        X_train[[col]] = sc.fit_transform(X_train[[col]])
        X_test[[col]] = sc.transform(X_test[[col]])
        if X_val is not None:
            X_val[[col]] = sc.transform(X_val[[col]])
        scalers[col] = sc

    feature_names = list(X_train.columns)
    fitted = {
        "outlier_bounds": outlier_bounds,
        "imputation_vals": imputation_vals,
        "label_encoders": label_encoders,
        "freq_maps": freq_maps,
        "onehot_cols": onehot_cols,
        "scalers": scalers,
        "log_cols": log_cols,
        "feature_names": feature_names,
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, fitted


def apply_fitted_transforms(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Apply an already-fitted state to a frame — never refits anything.

    Used to score rows through a *specific* model's own fitted state, which is
    the only way a champion comparison is honest.

    Step order is the SCORING order (impute -> cap -> log -> encode -> scale),
    not the fitting order (cap -> log -> impute -> encode -> scale). The
    reference application genuinely differs between the two — see defect D3 in
    IMPLEMENTATION_GAP_ANALYSIS.md — and the deployed loader uses the scoring
    order. Following the fitting order here would make every number this
    application reports disagree with what the deployed package actually
    produces, which the Stage 7 smoke test would then flag as a package fault.
    """
    df = df.copy()
    df = apply_imputation(df, fitted.get("imputation_vals") or {})
    df = apply_outlier_bounds(df, fitted.get("outlier_bounds") or {})
    df = apply_log_cols(df, fitted.get("log_cols") or [])
    df = apply_encoding(
        df,
        fitted.get("label_encoders") or {},
        fitted.get("freq_maps") or {},
        fitted.get("onehot_cols") or {},
    )
    for col, scaler in (fitted.get("scalers") or {}).items():
        if col in df.columns:
            try:
                df[[col]] = scaler.transform(df[[col]])
            except Exception:
                pass
    feature_names = fitted.get("feature_names") or []
    X = df.reindex(columns=feature_names, fill_value=0)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


# ── Config container ─────────────────────────────────────────────────────────

class NovaConfigs:
    """The champion package's configuration set, shape-tolerantly parsed."""

    def __init__(self, raw: dict, date_column: str | None = None):
        self.raw = raw
        self.column_map = unwrap_list(raw.get("column_map"), "column_map")
        self.column_config = unwrap_list(raw.get("column_config"), "matched_columns")
        self.feature_selection = unwrap_list(raw.get("feature_selection"), "selected_columns")
        self.dtype_config = unwrap_dict(raw.get("dtype_config"))
        self.derived_config = unwrap_list(raw.get("derived_config"), "derived_config")
        self.bucket_config = unwrap_dict(raw.get("bucket_config"))
        self.grouping_config = unwrap_dict(raw.get("grouping_config"))
        self.features_config = unwrap_dict(raw.get("features_config"))
        self.threshold_config = unwrap_dict(raw.get("threshold_config"))
        self.training_results = unwrap_dict(raw.get("training_results"))
        subtask_raw = raw.get("subtask_mappings") or {}
        self.subtask_mappings = unwrap_list(subtask_raw, "mappings")
        self.subtask_keywords = (
            subtask_raw.get("keywords", []) if isinstance(subtask_raw, dict) else []
        )
        self.date_column = date_column

    @property
    def champion_model_id(self) -> str | None:
        best = self.training_results.get("best_model")
        if best:
            return best
        results = self.training_results.get("results") or {}
        if results:
            return max(
                results.items(),
                key=lambda kv: (kv[1].get("test_metrics", {}) or {}).get("auc", 0),
            )[0]
        return None

    @property
    def champion_threshold(self) -> float:
        model_id = self.champion_model_id
        if model_id and model_id in self.threshold_config:
            try:
                return float(self.threshold_config[model_id])
            except (TypeError, ValueError):
                pass
        return 0.5
