"""scoring.py — NoVA scoring runtime shipped inside an enhancer export.

This is the reference nova-ml scoring client (`scoring_client/scoring.py`) with
two documented changes and nothing else. Every transformation step below is
byte-for-byte the reference logic, so a package produced by the NoVA Model
Enhancer scores identically to one produced by nova-ml's own Stage 12 export.

Change 1 — shape-tolerant config reading.
    nova-ml writes `column_map.json` as {"column_map": [...]} and
    `feature_selection.json` as {"selected_columns": [...]}, but the reference
    client reads both as bare lists. Against a real export that makes the
    rename map raise AttributeError and makes feature selection silently drop
    every feature, so the model scores a constant. `_read_json` here unwraps the
    documented wrapper key when it is present and otherwise behaves exactly as
    before.

Change 2 — normalised one-hot column names.
    `fitted_transforms.pkl` stores one-hot column names as they were generated at
    training time ("payername_Grouped_aetna"). The reference client normalises the
    dict key but not those names, so they never match the lowercase dummies it
    generates at scoring time and every one-hot feature is silently zeroed. Both
    are normalised here.

Change 3 — an additional `ml_tag` output mode.
    `run()` is unchanged and appends NovaProbability + VoiceNonVoiceFlag.
    `run_ml_tag()` returns the original inventory columns, in their original
    order, plus exactly one appended column named `ml_tag`, and exposes no
    probability. Its encoding is read from `scoring/ml_tag_config.json`, which
    the enhancer writes only after a named approver confirms the convention.

Deployment layout (identical to nova-ml):
    <Placement>/config/    column_map.json, dtype_config.json, derived_config.json,
                           bucket_config.json, grouping_config.json,
                           feature_selection.json
    <Placement>/model/     {model_id}.pkl, fitted_transforms.pkl
    <Placement>/scoring/   threshold_config.json, ml_tag_config.json
    <Placement>/metadata/  training_results.json
    <Placement>/pipeline/  scoring.py   <- this file

Usage:
    from scoring import NovaMLPipeline
    pipeline = NovaMLPipeline()
    result_df = pipeline.run(df)          # NovaProbability + VoiceNonVoiceFlag
    tagged_df = pipeline.run_ml_tag(df)   # original columns + ml_tag only
"""

import json
import pickle
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_DEFAULT_RESOURCE_ROOT = Path(__file__).resolve().parent.parent
_SUBFOLDERS = ("config", "model", "scoring", "metadata", "")

# Wrapper keys nova-ml writes around what its own client reads as a bare list.
_UNWRAP_KEYS = {
    "column_map.json": "column_map",
    "feature_selection.json": "selected_columns",
    "column_config.json": "matched_columns",
    "subtask_mappings.json": "mappings",
}


# ── Column normalisation ─────────────────────────────────────────────────────

def _norm_col(name: str) -> str:
    """strip -> collapse whitespace -> remove special chars -> lowercase."""
    s = re.sub(r'\s+', ' ', str(name).strip())
    s = re.sub(r'[^\w\s]', '', s).strip()
    return s.lower()


def _dedupe_columns(cols) -> list:
    """Append _2, _3, ... to repeated names so every lookup stays a Series."""
    seen = {}
    out = []
    for c in cols:
        n = seen.get(c, 0)
        out.append(c if n == 0 else f"{c}_{n + 1}")
        seen[c] = n + 1
    return out


def _build_rename_map(col_map: list, df_columns) -> dict:
    norm_to_prod = {
        _norm_col(r["inventory"]): _norm_col(r["production"])
        for r in col_map
        if isinstance(r, dict)
        and r.get("include", True)
        and r.get("inventory") and r.get("production")
        and _norm_col(r["inventory"]) != _norm_col(r["production"])
    }
    rename = {}
    for col in df_columns:
        prod = norm_to_prod.get(_norm_col(col))
        if prod and prod != col:
            rename[col] = prod
    return rename


# ── Dtype coercion ───────────────────────────────────────────────────────────

def _apply_dtype_config(df, dtype_config):
    if not dtype_config:
        return df
    df = df.copy()
    norm_map, collided = {}, set()
    for c in df.columns:
        key = _norm_col(c)
        if key in norm_map and norm_map[key] != c:
            collided.add(key)
        else:
            norm_map[key] = c
    for col, spec in dtype_config.items():
        if not isinstance(spec, dict):
            continue
        key = _norm_col(col)
        actual_col = col if col in df.columns else (norm_map.get(key) if key not in collided else None)
        if actual_col is None:
            continue
        col = actual_col
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
                    s = s.str.replace(r'(?:[^\w\s]|_)+', '', regex=True).str.strip()
                    s = s.str.replace(' ', '_', regex=False)
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

def _eval_condition(df, col, op, value):
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if op == "is_null":     return series.isna()
    if op == "is_not_null": return series.notna()
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


def _apply_derived_config(df, derived):
    if not derived:
        return df
    df = df.copy()
    col_lower = {c.lower(): c for c in df.columns}

    def _res(name):
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


# ── Artifact loading ─────────────────────────────────────────────────────────

def _find(resource_root, filename):
    for sub in _SUBFOLDERS:
        candidate = resource_root / sub / filename if sub else resource_root / filename
        if candidate.exists():
            return candidate
    return None


def _read_json(resource_root, filename, default):
    """Read a config artifact, unwrapping nova-ml's documented wrapper key.

    A bare collection is returned unchanged, so this is a superset of the
    reference client's behaviour rather than a change to it.
    """
    path = _find(resource_root, filename)
    if path is None:
        return default
    with open(path, "r") as f:
        value = json.load(f)
    key = _UNWRAP_KEYS.get(filename)
    if key and isinstance(value, dict) and isinstance(value.get(key), (list, dict)):
        return value[key]
    return value


class NovaMLPipeline:
    """Loads a NoVA scoring package and scores raw inventory files."""

    def __init__(self, resource_root=None, model_path=None):
        self.resource_root = Path(resource_root) if resource_root else _DEFAULT_RESOURCE_ROOT
        self._load_artifacts(explicit_model_path=Path(model_path) if model_path else None)

    def _load_artifacts(self, explicit_model_path):
        d = self.resource_root

        self.column_map = _read_json(d, "column_map.json", default=[])
        self.dtype_config = _read_json(d, "dtype_config.json", default={})
        self.derived_config = _read_json(d, "derived_config.json", default=[])
        self.bucket_config = _read_json(d, "bucket_config.json", default={})
        self.grouping_config = _read_json(d, "grouping_config.json", default={})
        self.feature_selection = _read_json(d, "feature_selection.json", default=[])
        self.threshold_config = _read_json(d, "threshold_config.json", default={})
        self.training_results = _read_json(d, "training_results.json", default={})
        self.ml_tag_config = _read_json(d, "ml_tag_config.json", default={})

        fitted_path = _find(d, "fitted_transforms.pkl")
        if fitted_path is None:
            raise FileNotFoundError(
                f"fitted_transforms.pkl not found under {d} — the package is incomplete."
            )
        with open(fitted_path, "rb") as f:
            fitted = pickle.load(f)

        self.imputation_vals = {_norm_col(k): v for k, v in (fitted.get("imputation_vals") or {}).items()}
        self.outlier_bounds = {_norm_col(k): v for k, v in (fitted.get("outlier_bounds") or {}).items()}
        self.log_cols = [_norm_col(c) for c in (fitted.get("log_cols") or [])]
        self.label_encoders = {_norm_col(k): v for k, v in (fitted.get("label_encoders") or {}).items()}
        self.freq_maps = {_norm_col(k): v for k, v in (fitted.get("freq_maps") or {}).items()}
        # Both the key AND the generated column names are normalised. The
        # reference client normalises only the key, so for a one-hot fitted on a
        # bucketed/grouped column (named "<col>_Grouped" at training time) the
        # stored names keep their capital letter, never match the lowercase
        # dummies produced here, and every one-hot feature silently scores as
        # zero. See defect D4 in IMPLEMENTATION_GAP_ANALYSIS.md.
        self.onehot_cols = {
            _norm_col(k): [_norm_col(c) for c in (v or [])]
            for k, v in (fitted.get("onehot_cols") or {}).items()
        }
        self.scalers = {_norm_col(k): v for k, v in (fitted.get("scalers") or {}).items()}
        self.feature_names = [_norm_col(f) for f in (fitted.get("feature_names") or [])]

        self.best_model_id = self.training_results.get("best_model")
        if not self.best_model_id and self.training_results.get("results"):
            self.best_model_id = max(
                self.training_results["results"].items(),
                key=lambda x: x[1].get("test_metrics", {}).get("auc", 0),
            )[0]

        if explicit_model_path is not None:
            model_path = explicit_model_path
        else:
            if not self.best_model_id:
                raise FileNotFoundError(
                    "No 'best_model' key in training_results.json and no explicit "
                    "model_path was given — cannot resolve which model to load."
                )
            model_path = _find(d, f"{self.best_model_id}.pkl")
            if model_path is None:
                raise FileNotFoundError(
                    f"Trained model file '{self.best_model_id}.pkl' not found under {d}."
                )
        self.estimator = joblib.load(model_path)
        self.threshold = float(self.threshold_config.get(self.best_model_id, 0.5))

    # -- transformation --------------------------------------------------------

    def _prepare(self, df):
        """Everything up to (and including) the feature matrix. Reference logic."""
        df = df.reset_index(drop=True)
        orig_df = df.copy()

        rename_map = _build_rename_map(self.column_map, df.columns)
        if rename_map:
            df = df.rename(columns=rename_map)
        df.columns = _dedupe_columns([_norm_col(c) for c in df.columns])

        df["__row_idx__"] = np.arange(len(df))
        df = _apply_dtype_config(df, self.dtype_config)
        if "__row_idx__" in df.columns:
            surviving = df["__row_idx__"].tolist()
            df = df.drop(columns=["__row_idx__"])
            if len(surviving) < len(orig_df):
                orig_df = orig_df.iloc[surviving].reset_index(drop=True)
        df = df.reset_index(drop=True)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].where(df[col].astype(str).str.strip().ne(""), other=np.nan)

        numeric_cols = set(self.outlier_bounds) | set(self.log_cols) | {
            k for k, v in self.imputation_vals.items() if isinstance(v, (int, float))
        }
        for col in df.columns:
            if _norm_col(col) in numeric_cols and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.derived_config:
            scoring_ts = pd.Timestamp.now().normalize()
            injected = []
            for d in self.derived_config:
                if isinstance(d, dict) and d.get("col_type") == "date_diff" and d.get("use_run_date_for_scoring"):
                    for col_key in ("date_col", "reference_col"):
                        col = d.get(col_key)
                        if col and col not in df.columns:
                            df[col] = scoring_ts
                            injected.append(col)
            df = _apply_derived_config(df, self.derived_config)
            if injected:
                df = df.drop(columns=[c for c in injected if c in df.columns], errors="ignore")

        for col, cfg in (self.bucket_config or {}).items():
            col_n = _norm_col(col)
            if col_n not in df.columns:
                continue
            cuts, labels = cfg.get("cuts", []), cfg.get("labels", [])
            if not cuts or not labels:
                continue
            bins = [-np.inf] + [float(c) for c in cuts] + [np.inf]
            df[f"{col_n}_bucket"] = pd.cut(
                df[col_n], bins=bins, labels=labels, right=True, include_lowest=True
            ).astype(str)
            df = df.drop(columns=[col_n])

        for col, cfg in (self.grouping_config or {}).items():
            col_n = _norm_col(col)
            if col_n not in df.columns:
                continue
            kept_values = cfg.get("kept_values", [])
            others_label = cfg.get("others_label", "Other")
            null_label = cfg.get("null_label", "NA")
            grouped = df[col_n].astype(str).copy()
            grouped = grouped.where(grouped.isin(kept_values), other=others_label)
            grouped = grouped.where(df[col_n].notna(), other=null_label)
            df[f"{col_n}_grouped"] = grouped
            df = df.drop(columns=[col_n])

        raw_selected = [_norm_col(c) for c in self.feature_selection]
        bucket_rename = {_norm_col(k): f"{_norm_col(k)}_bucket" for k in (self.bucket_config or {})}
        grouping_rename = {_norm_col(k): f"{_norm_col(k)}_grouped" for k in (self.grouping_config or {})}
        selected_cols = [bucket_rename.get(c, grouping_rename.get(c, c)) for c in raw_selected]
        if selected_cols:
            keep = [c for c in selected_cols if c in df.columns]
            missing = [c for c in selected_cols if c not in df.columns]
            df = df[keep]
            for mc in missing:
                df[mc] = 0

        for col, fill_val in self.imputation_vals.items():
            if col in df.columns:
                if isinstance(fill_val, (int, float)) and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(fill_val)

        for col, bounds in self.outlier_bounds.items():
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])

        for col in self.log_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = np.log1p(df[col].clip(lower=0))

        for col, enc in self.label_encoders.items():
            if col in df.columns:
                class_map = {c: i for i, c in enumerate(enc.get("classes", []))}
                df[col] = df[col].astype(str).map(class_map).fillna(-1).astype(int)

        for col, freq_map in self.freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map({str(k): v for k, v in freq_map.items()}).fillna(0.0)

        for col, oh_cols in self.onehot_cols.items():
            if col in df.columns:
                dummies = pd.get_dummies(df[col].astype(str).fillna("NA"), prefix=col)
                df = df.drop(columns=[col])
                for oh_col in oh_cols:
                    if oh_col not in dummies.columns:
                        dummies[oh_col] = 0
                df = pd.concat([df, dummies[oh_cols]], axis=1)

        for col, scaler in self.scalers.items():
            if col in df.columns:
                try:
                    df[[col]] = scaler.transform(df[[col]])
                except Exception:
                    pass

        X = df.reindex(columns=self.feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return orig_df, X

    def predict_proba(self, df):
        """P(Non-Voice) per row, aligned to the surviving original rows."""
        orig_df, X = self._prepare(df)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X.values)[:, 1]
        else:
            proba = self.estimator.predict(X.values).astype(float)
        return orig_df, np.asarray(proba, dtype=float)

    # -- output modes ----------------------------------------------------------

    def run(self, df):
        """nova-ml parity: original columns + NovaProbability + VoiceNonVoiceFlag.

        VoiceNonVoiceFlag is 1 = Voice, 0 = Non-Voice — inverted from training's
        NonVoiceFlag, exactly as the tool itself does.
        """
        orig_df, proba = self.predict_proba(df)
        result_df = orig_df.copy()
        result_df["NovaProbability"] = np.round(proba, 4)
        result_df["VoiceNonVoiceFlag"] = (proba < self.threshold).astype(int)
        return result_df

    def run_ml_tag(self, df):
        """Original columns, original order, plus exactly one appended `ml_tag`.

        No probability and no Voice/Non-Voice text is exposed. The encoding comes
        from `scoring/ml_tag_config.json`, which is written only after a named
        approver confirms the convention; an unapproved package refuses to run
        in this mode rather than guessing.
        """
        config = self.ml_tag_config or {}
        if not config.get("approved"):
            raise RuntimeError(
                "ml_tag encoding has not been approved for this package. "
                "scoring/ml_tag_config.json must carry an approved encoding before "
                "run_ml_tag() can be used."
            )
        voice_value = config.get("voice_value", 1)
        non_voice_value = config.get("non_voice_value", 0)
        column_name = config.get("column_name", "ml_tag")

        original_columns = list(df.columns)
        original_rows = len(df)
        orig_df, proba = self.predict_proba(df)
        if len(orig_df) != original_rows:
            raise RuntimeError(
                f"Row count changed during scoring ({original_rows} in, {len(orig_df)} out). "
                "The package refuses to emit a misaligned ml_tag."
            )
        if column_name in original_columns:
            raise RuntimeError(
                f"The inventory already contains a column named {column_name!r}; "
                "appending would overwrite source data."
            )

        is_voice = proba < self.threshold
        result_df = orig_df.loc[:, original_columns].copy()
        result_df[column_name] = np.where(is_voice, voice_value, non_voice_value)
        return result_df


def score(df, resource_root=None, model_path=None):
    """One-shot wrapper matching the reference client's calling style."""
    return NovaMLPipeline(resource_root=resource_root, model_path=model_path).run(df)


def score_ml_tag(df, resource_root=None, model_path=None):
    """One-shot wrapper returning the original inventory plus only `ml_tag`."""
    return NovaMLPipeline(resource_root=resource_root, model_path=model_path).run_ml_tag(df)
