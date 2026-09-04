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

# Columns that a NoVA scoring run *writes*. Training on one of these would feed
# the model its own predictions as ground truth, which the project constraints
# forbid outright. They stay in TARGET_CANDIDATES so they are still detected and
# named, but choosing one requires an explicit, recorded acknowledgement.
MODEL_OUTPUT_COLUMNS = ["ml_tag", "VoiceNonVoiceFlag", "NovaProbability"]


def is_model_output(column: str) -> bool:
    """True when a column is one a scoring run produces rather than a human verifies."""
    return any(_norm(column) == _norm(name) for name in MODEL_OUTPUT_COLUMNS)
SUBTASK_CANDIDATES = ["SubTask", "Sub Task", "Sub-Task"]

_MAX_CATEGORY_REPORT = 25


class DataReadError(ValueError):
    pass


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


# ── Column lineage ───────────────────────────────────────────────────────────

# Bookkeeping columns this application adds when it combines uploads. They are
# not user data and reporting them as drift sends someone looking for a column
# they never supplied.
INTERNAL_COLUMNS = ("__source_role__", "NonVoiceFlag")


def _is_internal(column: str) -> bool:
    name = str(column)
    return name in INTERNAL_COLUMNS or (name.startswith("__") and name.endswith("__"))


def _map_entries(column_map: list | None) -> list[dict]:
    """Normalise column_map into {inventory, production, include} records."""
    entries = []
    for raw in column_map or []:
        if not isinstance(raw, dict):
            continue
        production = raw.get("production") or raw.get("inventory")
        if not production:
            continue
        entries.append({
            "inventory": raw.get("inventory") or production,
            "production": production,
            "include": bool(raw.get("include", True)),
        })
    return entries


def column_lineage(
    configs,
    uploaded_columns: list[str],
    fitted_feature_names: list[str] | None = None,
) -> dict:
    """Trace every column through the four lists a champion package carries.

    A package records the same columns four times, each meaning something
    different: what was mapped, what survived the Stage 03 filter, what was
    selected as a feature, and what the fitted model actually consumes. Showing
    only one of them answers only one question — this reports all four so the
    exact point a column left the pipeline is visible.

    One honest limit: Stage 05 EDA exclusions are not present in an export, so a
    column that is matched but not selected can be reported as having left
    during the build, but not *which* of those two steps dropped it.
    """
    from .nova_transform import build_rename_map, norm_col

    entries = _map_entries(configs.column_map)
    matched = [str(c) for c in (configs.column_config or [])]
    selected = [str(c) for c in (configs.feature_selection or [])]
    fitted = [str(c) for c in (fitted_feature_names or [])]

    # Derived features are computed, not uploaded. Reporting one as a missing
    # column sends someone hunting for data that never existed in the source —
    # and its *source* columns are required even when they are not features
    # themselves, which the raw layer lists do not show.
    derived_outputs: set[str] = set()
    derived_sources: dict[str, list[str]] = {}
    for spec in (configs.derived_config or []):
        if not isinstance(spec, dict):
            continue
        output = spec.get("output_col")
        if not output:
            continue
        derived_outputs.add(norm_col(output))
        for field in ("date_col", "reference_col", "numerator_col", "denominator_col",
                      "left_col", "right_col", "source_col"):
            source = spec.get(field)
            if source:
                derived_sources.setdefault(norm_col(source), []).append(str(output))

    raw_uploaded = [str(c) for c in uploaded_columns]
    rename = build_rename_map(configs.column_map, raw_uploaded) if configs.column_map else {}
    renamed_uploaded = [rename.get(c, c) for c in raw_uploaded]

    present = {norm_col(c) for c in renamed_uploaded}
    matched_set = {norm_col(c) for c in matched}
    selected_set = {norm_col(c) for c in selected}

    # A fitted name is either the column itself or a one-hot expansion of it.
    fitted_norm = [norm_col(c) for c in fitted]

    def fitted_for(column: str) -> list[str]:
        key = norm_col(column)
        return [
            original for original, name in zip(fitted, fitted_norm)
            if name == key or name.startswith(f"{key}_")
        ]

    # Every production column the package knows about, from any layer.
    known: list[str] = []
    seen: set[str] = set()
    for column in ([e["production"] for e in entries] + matched + selected
                   + sorted(derived_sources)):
        key = norm_col(column)
        if key not in seen:
            seen.add(key)
            known.append(column)

    include_flags = {norm_col(e["production"]): e["include"] for e in entries}
    inventory_names = {norm_col(e["production"]): e["inventory"] for e in entries}

    columns = []
    for column in known:
        key = norm_col(column)
        in_map = key in include_flags
        included = include_flags.get(key, True)
        in_matched = key in matched_set
        in_selected = key in selected_set
        expansions = fitted_for(column) if fitted else []

        feeds = derived_sources.get(key, [])
        is_derived = key in derived_outputs

        if is_derived:
            stage, note = (
                "derived",
                "Computed by derived_config during transformation — not an uploaded column.",
            )
        elif in_map and not included:
            stage, note = "excluded_at_mapping", "Marked include=false in column_map."
        elif feeds:
            stage, note = (
                "feeds_derived",
                f"Not a model feature itself, but required to compute {', '.join(feeds)}.",
            )
        elif not in_matched and in_map:
            stage, note = "dropped_at_matching", "Mapped, but not in the Stage 03 matched set."
        elif in_matched and not in_selected:
            stage, note = (
                "dropped_during_build",
                "Available at Stage 03 but absent from the final feature list — "
                "excluded during EDA or not selected. The export does not record which.",
            )
        elif in_selected:
            stage, note = "selected", "Used as a model feature."
        else:
            stage, note = "unknown", "Present in the package but in no recognised layer."

        columns.append({
            "column": column,
            "inventory_name": inventory_names.get(key),
            "in_column_map": in_map,
            "included_in_map": included,
            "in_matched": in_matched,
            "in_selected": in_selected,
            "fitted_features": expansions,
            "fitted_feature_count": len(expansions),
            "stage": stage,
            "note": note,
            "derived": is_derived,
            "feeds_derived": feeds,
            "required": bool((in_selected and not is_derived) or feeds),
            "present_in_upload": key in present,
        })

    order = {"selected": 0, "derived": 1, "feeds_derived": 2}
    columns.sort(key=lambda c: (order.get(c["stage"], 3), c["column"].lower()))

    removed = [c for c in columns if c["stage"] == "dropped_during_build"]
    required_missing = sorted(
        c["column"] for c in columns if c["required"] and not c["present_in_upload"]
    )
    optional_missing = sorted(
        c["column"] for c in columns
        if not c["required"] and not c["derived"]
        and c["in_matched"] and not c["present_in_upload"]
    )
    unexpected = sorted(
        c for c in renamed_uploaded
        if norm_col(c) not in seen
        and not is_model_output(c)
        and not _is_internal(c)
    )

    return {
        "layers": {
            "mapped": len(entries),
            "matched": len(matched_set),
            "selected": len(selected_set),
            "fitted": len(fitted),
            "fitted_available": bool(fitted),
            "derived": len(derived_outputs),
        },
        "columns": columns,
        "removed_during_build": [c["column"] for c in removed],
        "removed_during_build_count": len(removed),
        "missing_required": required_missing,
        "missing_required_count": len(required_missing),
        "missing_optional": optional_missing,
        "missing_optional_count": len(optional_missing),
        "unexpected_columns": unexpected[:60],
        "unexpected_column_count": len(unexpected),
    }
