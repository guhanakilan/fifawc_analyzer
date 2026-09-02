"""Immutable dataset snapshot + manifest.

A snapshot is written once, hashed, and never modified. Everything downstream
(weights, training, comparison, export) refers to the snapshot id, so a result
can always be traced back to the exact rows that produced it.

Reports carry counts, hashes and column names — never raw row samples.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import labeling
from .data_profiler import match_column, read_dataset, sha256_file
from .nova_transform import TARGET, norm_col


class SnapshotError(ValueError):
    pass


@dataclass
class SnapshotDecisions:
    date_column: str
    target_mode: str                    # "existing" | "derive_from_subtask"
    target_column: str | None
    target_encoding: dict               # {"voice": 0, "non_voice": 1} for an existing column
    dedup_mode: str                     # "full_row" | "key_columns" | "none"
    dedup_keys: list[str]
    subtask_mappings: list[dict]
    subtask_keywords: list[str]
    allow_unmapped_default: bool
    historical_window_days: int | None
    approver: str


def atomic_write_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Write to a temp file in the same directory, then rename into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _coerce_target(series: pd.Series, encoding: dict) -> pd.Series:
    """Map an existing label column onto the internal 0=Voice / 1=Non-Voice."""
    voice_values = {str(v).strip().lower() for v in encoding.get("voice_values", [encoding.get("voice", 0)])}
    nv_values = {str(v).strip().lower() for v in encoding.get("non_voice_values", [encoding.get("non_voice", 1)])}
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[text.isin(voice_values)] = labeling.VOICE
    out[text.isin(nv_values)] = labeling.NON_VOICE
    return out


def build_snapshot(
    asset_paths: list[tuple[str, Path]],
    decisions: SnapshotDecisions,
    destination_dir: Path,
    snapshot_id: str,
    config_fingerprints: dict,
) -> dict:
    """Assemble, label, deduplicate and freeze the training dataset.

    `asset_paths` is [(role, path)] where role is 'historical', 'new' or 'combined'.
    Returns the manifest; the manifest and parquet are both written to disk.
    """
    if not asset_paths:
        raise SnapshotError("No training data assets were provided.")

    sources = []
    frames = []
    for role, path in asset_paths:
        frame = read_dataset(path)
        if frame.empty:
            raise SnapshotError(f"{path.name} contained no rows.")
        frame["__source_role__"] = role
        frames.append(frame)
        sources.append({
            "role": role,
            "filename": path.name,
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": int(frame.shape[1] - 1),
        })

    df = pd.concat(frames, ignore_index=True, sort=False)
    rows_loaded = int(len(df))

    # ── Date column ──────────────────────────────────────────────────────────
    date_col = match_column(df.columns, decisions.date_column)
    if date_col is None:
        raise SnapshotError(
            f"Approved date column '{decisions.date_column}' is not present in the uploaded data."
        )
    dates = pd.to_datetime(df[date_col], errors="coerce")
    unparseable_dates = int(dates.isna().sum())

    # ── Historical window ────────────────────────────────────────────────────
    window_dropped = 0
    if decisions.historical_window_days:
        newest = dates.max()
        if pd.notna(newest):
            cutoff = newest - pd.Timedelta(days=int(decisions.historical_window_days))
            keep = dates.isna() | (dates >= cutoff)
            window_dropped = int((~keep).sum())
            df = df[keep].reset_index(drop=True)
            dates = dates[keep].reset_index(drop=True)

    # ── Labels ───────────────────────────────────────────────────────────────
    label_stats: dict = {}
    if decisions.target_mode == "derive_from_subtask":
        df, label_stats = labeling.apply_subtask_mapping(
            df, decisions.subtask_mappings, decisions.subtask_keywords,
            allow_unmapped_default=decisions.allow_unmapped_default,
        )
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        target_col = match_column(df.columns, decisions.target_column or "")
        if target_col is None:
            raise SnapshotError(
                f"Approved target column '{decisions.target_column}' is not present in the uploaded data."
            )
        mapped = _coerce_target(df[target_col], decisions.target_encoding)
        unmapped = int(mapped.isna().sum())
        if unmapped:
            raise SnapshotError(
                f"{unmapped} row(s) carry a label value outside the approved encoding for "
                f"'{target_col}'. Fix the data or correct the approved encoding — the "
                "enhancer will not guess a label."
            )
        df = df.copy()
        df[TARGET] = mapped.astype(int)
        label_stats = {
            "total_rows": int(len(df)),
            "voice_count": int((df[TARGET] == labeling.VOICE).sum()),
            "non_voice_count": int((df[TARGET] == labeling.NON_VOICE).sum()),
            "ignored_count": 0,
            "source_column": target_col,
        }

    rows_after_labelling = int(len(df))

    # ── Deduplication ────────────────────────────────────────────────────────
    duplicates_removed = 0
    if decisions.dedup_mode == "full_row":
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = before - len(df)
    elif decisions.dedup_mode == "key_columns":
        if not decisions.dedup_keys:
            raise SnapshotError("Deduplication key columns were not supplied.")
        resolved = []
        for key in decisions.dedup_keys:
            hit = match_column(df.columns, key)
            if hit is None:
                raise SnapshotError(f"Deduplication key column '{key}' is not present in the data.")
            resolved.append(hit)
        before = len(df)
        # Keep the newest row per key so a correction supersedes the row it corrects.
        order = pd.to_datetime(df[date_col], errors="coerce")
        df = (
            df.assign(__order__=order)
              .sort_values("__order__", kind="stable", na_position="first")
              .drop_duplicates(subset=resolved, keep="last")
              .drop(columns="__order__")
              .sort_index()
              .reset_index(drop=True)
        )
        duplicates_removed = before - len(df)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    rows_final = int(len(df))
    if rows_final == 0:
        raise SnapshotError("Every row was removed by labelling, the window filter or deduplication.")
    if df[TARGET].nunique() < 2:
        raise SnapshotError(
            "The snapshot contains only one class. A classifier cannot be trained or compared on it."
        )

    # ── Freeze ───────────────────────────────────────────────────────────────
    parquet_path = destination_dir / f"{snapshot_id}.parquet"
    atomic_write_parquet(df, parquet_path)
    snapshot_sha = sha256_file(parquet_path)

    class_counts = {str(int(k)): int(v) for k, v in df[TARGET].value_counts().items()}
    monthly = []
    if dates.notna().any():
        monthly_frame = (
            df.assign(__period__=dates.dt.to_period("M"))
              .dropna(subset=["__period__"])
              .groupby("__period__")[TARGET].agg(["count", "sum"])
        )
        for period, row in monthly_frame.iterrows():
            monthly.append({
                "month": period.strftime("%b-%y"),
                "rows": int(row["count"]),
                "non_voice": int(row["sum"]),
                "voice": int(row["count"] - row["sum"]),
            })

    manifest = {
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha,
        "parquet_file": parquet_path.name,
        "sources": sources,
        "row_counts": {
            "loaded": rows_loaded,
            "after_historical_window": rows_loaded - window_dropped,
            "after_labelling": rows_after_labelling,
            "final": rows_final,
        },
        "exclusions": {
            "historical_window_days": decisions.historical_window_days,
            "rows_outside_window": window_dropped,
            "rows_ignored_by_subtask_mapping": label_stats.get("ignored_count", 0),
            "duplicate_rows_removed": duplicates_removed,
            "deduplication_mode": decisions.dedup_mode,
            "deduplication_keys": decisions.dedup_keys,
        },
        "date_column": date_col,
        "date_range": {
            "from": dates.min().isoformat() if dates.notna().any() else None,
            "to": dates.max().isoformat() if dates.notna().any() else None,
            "unparseable": unparseable_dates,
        },
        "target": {
            "column": TARGET,
            "encoding": {"voice": labeling.VOICE, "non_voice": labeling.NON_VOICE},
            "mode": decisions.target_mode,
            "source_column": label_stats.get("source_column"),
            "distribution": class_counts,
            "non_voice_rate_pct": round(
                class_counts.get(str(labeling.NON_VOICE), 0) / rows_final * 100, 2
            ),
        },
        "label_stats": label_stats,
        "monthly_label_trend": monthly,
        "columns": [str(c) for c in df.columns],
        "config_fingerprints": config_fingerprints,
        "approver": decisions.approver,
    }
    atomic_write_text(json.dumps(manifest, indent=2, default=str), destination_dir / f"{snapshot_id}.manifest.json")
    return manifest


def load_snapshot(destination_dir: Path, snapshot_id: str) -> pd.DataFrame:
    path = destination_dir / f"{snapshot_id}.parquet"
    if not path.exists():
        raise SnapshotError(f"Snapshot {snapshot_id} is not present in this workspace.")
    return pd.read_parquet(path)


def load_manifest(destination_dir: Path, snapshot_id: str) -> dict:
    path = destination_dir / f"{snapshot_id}.manifest.json"
    if not path.exists():
        raise SnapshotError(f"Snapshot manifest {snapshot_id} is missing.")
    return json.loads(path.read_text(encoding="utf-8"))
