"""Training-data reading and profiling.

Large delimited files are read in chunks so a 2 GB CSV is profiled without
being materialised whole. Parquet is read via its own metadata where possible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

SUPPORTED_DATA_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}

DATE_CANDIDATES = [
    "UpdatedDateTimeGMT", "UpdatedDateTime", "UpdatedDateTimeEST", "UpdatedDateTimeIST",
    "MLFlagDate", "CreatedDate", "DOSFrom",
]
TARGET_CANDIDATES = ["NonVoiceFlag", "ml_tag", "VoiceNonVoiceFlag", "Target", "Label"]
SUBTASK_CANDIDATES = ["SubTask", "Sub Task", "Sub-Task"]

_MAX_CATEGORY_REPORT = 25


class DataReadError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def match_column(columns, *candidates: str) -> str | None:
    lookup = {_norm(c): c for c in columns}
    for candidate in candidates:
        hit = lookup.get(_norm(candidate))
        if hit is not None:
            return hit
    return None


def _read_csv_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks, trying UTF-8 first and reporting encoding faults clearly."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            reader = pd.read_csv(
                path, chunksize=chunk_rows, low_memory=False,
                encoding=encoding, on_bad_lines="error",
            )
            first = True
            for chunk in reader:
                first = False
                yield chunk
            if first:
                yield pd.DataFrame()
            return
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            raise DataReadError(
                f"CSV is malformed and could not be parsed: {exc}. "
                "Check for unquoted delimiters or a ragged row."
            ) from exc
    raise DataReadError(
        "CSV could not be decoded as UTF-8, CP1252 or Latin-1. "
        f"Re-save it as UTF-8 and try again. ({last_error})"
    )


def iter_chunks(path: Path, chunk_rows: int = 200_000) -> Iterator[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _read_csv_chunks(path, chunk_rows)
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            yield pd.read_parquet(path)
            return
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:
            raise DataReadError(f"Parquet file could not be opened: {exc}") from exc
        for batch in parquet_file.iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()
    elif suffix in {".xlsx", ".xls"}:
        try:
            yield pd.read_excel(path)
        except Exception as exc:
            raise DataReadError(f"Excel workbook could not be read: {exc}") from exc
    else:
        raise DataReadError("Supported formats are Parquet, CSV, XLSX and XLS.")


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset fully. Used once the profile has confirmed it is workable."""
    frames = [chunk for chunk in iter_chunks(path) if len(chunk)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def profile_dataset(path: Path, chunk_rows: int = 200_000) -> dict:
    """Stream the file once and accumulate everything the readiness gate needs."""
    rows = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    null_counts: dict[str, int] = {}
    duplicate_hashes: dict[int, int] = {}
    target_counts: dict[str, int] = {}
    subtask_counts: dict[str, int] = {}
    min_date = max_date = None
    invalid_dates = 0
    missing_target = 0
    numeric_stats: dict[str, dict] = {}
    seen_rows = 0

    date_col = target_col = subtask_col = None

    for chunk in iter_chunks(path, chunk_rows):
        if not columns:
            columns = [str(c) for c in chunk.columns]
            dtypes = {str(c): str(t) for c, t in chunk.dtypes.items()}
            date_col = match_column(chunk.columns, *DATE_CANDIDATES)
            target_col = match_column(chunk.columns, *TARGET_CANDIDATES)
            subtask_col = match_column(chunk.columns, *SUBTASK_CANDIDATES)
            null_counts = {c: 0 for c in columns}
        if chunk.empty:
            continue

        rows += len(chunk)
        seen_rows += len(chunk)
        for column in chunk.columns:
            null_counts[str(column)] = null_counts.get(str(column), 0) + int(chunk[column].isna().sum())

        # Duplicate detection across chunks via row hashing.
        hashed = pd.util.hash_pandas_object(chunk, index=False)
        for value, count in hashed.value_counts().items():
            duplicate_hashes[int(value)] = duplicate_hashes.get(int(value), 0) + int(count)

        if date_col and date_col in chunk.columns:
            parsed = pd.to_datetime(chunk[date_col], errors="coerce")
            invalid_dates += int(parsed.isna().sum())
            valid = parsed.dropna()
            if len(valid):
                chunk_min, chunk_max = valid.min(), valid.max()
                min_date = chunk_min if min_date is None or chunk_min < min_date else min_date
                max_date = chunk_max if max_date is None or chunk_max > max_date else max_date

        if target_col and target_col in chunk.columns:
            missing_target += int(chunk[target_col].isna().sum())
            for value, count in chunk[target_col].value_counts(dropna=True).items():
                target_counts[str(value)] = target_counts.get(str(value), 0) + int(count)

        if subtask_col and subtask_col in chunk.columns:
            for value, count in chunk[subtask_col].astype(str).value_counts().items():
                subtask_counts[str(value)] = subtask_counts.get(str(value), 0) + int(count)

        for column in chunk.select_dtypes(include=[np.number]).columns:
            series = chunk[column].dropna().astype(float)
            if not len(series):
                continue
            stat = numeric_stats.setdefault(
                str(column), {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
            )
            stat["min"] = min(stat["min"], float(series.min()))
            stat["max"] = max(stat["max"], float(series.max()))
            stat["sum"] += float(series.sum())
            stat["count"] += int(len(series))

    duplicate_rows = sum(count - 1 for count in duplicate_hashes.values() if count > 1)

    class_distribution = dict(sorted(target_counts.items(), key=lambda kv: -kv[1])[:_MAX_CATEGORY_REPORT])
    top_subtasks = [
        {"subtask": name, "rows": count}
        for name, count in sorted(subtask_counts.items(), key=lambda kv: -kv[1])[:_MAX_CATEGORY_REPORT]
    ]

    return {
        "rows": rows,
        "columns": len(columns),
        "column_names": columns,
        "dtypes": dtypes,
        "null_counts": {k: v for k, v in null_counts.items() if v},
        "duplicate_rows": int(duplicate_rows),
        "date_column_detected": date_col,
        "target_column_detected": target_col,
        "subtask_column_detected": subtask_col,
        "min_date": min_date.isoformat() if min_date is not None else None,
        "max_date": max_date.isoformat() if max_date is not None else None,
        "invalid_dates": int(invalid_dates),
        "missing_target": int(missing_target) if target_col else None,
        "class_distribution": class_distribution,
        "distinct_subtasks": len(subtask_counts),
        "top_subtasks": top_subtasks,
        "numeric_summary": {
            k: {
                "min": round(v["min"], 4), "max": round(v["max"], 4),
                "mean": round(v["sum"] / v["count"], 4) if v["count"] else None,
            }
            for k, v in list(numeric_stats.items())[:60]
        },
        "date_candidates": [c for c in columns if match_column([c], *DATE_CANDIDATES)],
        "target_candidates": [c for c in columns if match_column([c], *TARGET_CANDIDATES)],
    }


def drift_report(profile: dict, expected_columns: list[str], column_map: list | None = None) -> dict:
    """Schema drift of the uploaded data against the champion's configuration.

    The champion's configuration names columns in *production* form, while an
    upload usually still carries *inventory* names. The rename map is applied
    first, so a column that is merely named differently is not reported as
    missing — that false alarm would send someone hunting for data they have.
    """
    from .nova_transform import build_rename_map, norm_col

    columns = [str(c) for c in profile.get("column_names", [])]
    if column_map:
        rename = build_rename_map(column_map, columns)
        columns = [rename.get(c, c) for c in columns]

    present = {norm_col(c) for c in columns}
    expected = {norm_col(c) for c in expected_columns}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return {
        "expected_columns": len(expected),
        "present_columns": len(present),
        "missing_columns": missing[:60],
        "missing_column_count": len(missing),
        "new_columns": extra[:60],
        "new_column_count": len(extra),
    }
