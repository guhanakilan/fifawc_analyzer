"""SubTask -> NonVoiceFlag labelling, reproduced from the reference.

Source of truth: `backend/routers/flag.py::run_flag` in the reference app.

    SubTask mapped "Voice"     -> NonVoiceFlag = 0
    SubTask mapped "Non-Voice" -> NonVoiceFlag = 1
    SubTask mapped "Keyword"   -> ARComments contains any keyword -> 0, else 1
    SubTask mapped "Ignore"    -> row dropped
    SubTask absent from mapping-> 1 (default Non-Voice)

The "absent from mapping" default is the reference's behaviour, and it is
exactly why this application refuses to train on unmapped SubTasks: silently
defaulting a brand-new SubTask to Non-Voice manufactures labels nobody approved.
"""

from __future__ import annotations

import re

import pandas as pd

VOICE = 0
NON_VOICE = 1
FLAG_CHOICES = ("Voice", "Non-Voice", "Keyword", "Ignore")


def norm_text(value) -> str:
    """Lowercase, collapse whitespace/hyphens/underscores. Mirrors `flag._norm_text`."""
    return re.sub(r"[\s\-_]+", " ", str(value).strip().lower())


def suggest_flag(subtask_name: str, task_values) -> str:
    """Suggest a mapping for an unseen SubTask. Never applied automatically.

    Rule order is load-bearing and copied from `flag._suggest_subtask_flag`:
    "Non Voice" is checked before "Voice" because it contains it as a substring.
    """
    name_norm = norm_text(subtask_name)
    task_norms = [norm_text(t) for t in (task_values or []) if t]
    if "day to night" in name_norm:
        return "Ignore"
    if any("non workable" in t for t in task_norms):
        return "Ignore"
    if any("non voice" in t for t in task_norms):
        return "Non-Voice"
    if any("voice" in t for t in task_norms):
        return "Voice"
    if "non voice" in name_norm:
        return "Non-Voice"
    if "voice" in name_norm:
        return "Voice"
    return "Keyword"


def find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Locate a column by case/punctuation-insensitive name."""
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in lookup:
            return lookup[key]
    return None


def subtask_inventory(df: pd.DataFrame, mappings: list[dict]) -> dict:
    """Summarise SubTask coverage against a mapping, with suggestions for gaps."""
    subtask_col = find_column(df, "SubTask")
    if subtask_col is None:
        return {"subtask_column": None, "known": [], "unmapped": [], "total_subtasks": 0}

    task_col = find_column(df, "Task")
    mapped = {str(m.get("name")): m.get("flag") for m in mappings if isinstance(m, dict)}
    counts = df[subtask_col].astype(str).value_counts()

    known, unmapped = [], []
    for name, count in counts.items():
        row = {"subtask": name, "rows": int(count)}
        if task_col is not None:
            tasks = sorted(
                {str(t) for t in df.loc[df[subtask_col].astype(str) == name, task_col].dropna().unique()[:5]}
            )
            row["tasks"] = tasks
        else:
            row["tasks"] = []
        if name in mapped:
            row["flag"] = mapped[name]
            known.append(row)
        else:
            row["suggested_flag"] = suggest_flag(name, row["tasks"])
            unmapped.append(row)

    return {
        "subtask_column": subtask_col,
        "task_column": task_col,
        "known": known,
        "unmapped": unmapped,
        "total_subtasks": int(len(counts)),
    }


def apply_subtask_mapping(
    df: pd.DataFrame, mappings: list[dict], keywords: list[str],
    allow_unmapped_default: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Derive `NonVoiceFlag` from SubTask + ARComments.

    `allow_unmapped_default` must be an explicit, approved choice: with it False
    (the default) an unmapped SubTask raises instead of silently becoming
    Non-Voice.
    """
    subtask_col = find_column(df, "SubTask")
    if subtask_col is None:
        raise ValueError("No SubTask column found — cannot derive NonVoiceFlag from mappings.")

    mapping = {str(m["name"]): m["flag"] for m in mappings if isinstance(m, dict) and m.get("name")}
    present = set(df[subtask_col].astype(str).unique())
    unmapped = sorted(present - set(mapping))
    if unmapped and not allow_unmapped_default:
        raise ValueError(
            f"{len(unmapped)} SubTask value(s) have no approved mapping: "
            + ", ".join(unmapped[:5]) + ("…" if len(unmapped) > 5 else "")
        )

    out = df.copy().reset_index(drop=True)
    flags = out[subtask_col].astype(str).map(mapping).fillna("Non-Voice")
    out["NonVoiceFlag"] = None
    out.loc[flags == "Voice", "NonVoiceFlag"] = VOICE
    out.loc[flags == "Non-Voice", "NonVoiceFlag"] = NON_VOICE

    keyword_mask = flags == "Keyword"
    keyword_voice = keyword_nv = 0
    if keyword_mask.any():
        comment_col = find_column(out, "ARComments")
        keywords_lower = [k.lower() for k in (keywords or []) if k]
        if comment_col is not None and keywords_lower:
            comments = out.loc[keyword_mask, comment_col].fillna("").astype(str).str.lower()
            pattern = "|".join(re.escape(k) for k in keywords_lower)
            voice_match = comments.str.contains(pattern, regex=True, na=False)
            out.loc[keyword_mask, "NonVoiceFlag"] = voice_match.map({True: VOICE, False: NON_VOICE})
        else:
            out.loc[keyword_mask, "NonVoiceFlag"] = NON_VOICE
        keyword_voice = int((keyword_mask & (out["NonVoiceFlag"] == VOICE)).sum())
        keyword_nv = int((keyword_mask & (out["NonVoiceFlag"] == NON_VOICE)).sum())

    ignored = int(out["NonVoiceFlag"].isna().sum())
    out = out.dropna(subset=["NonVoiceFlag"]).reset_index(drop=True)
    out["NonVoiceFlag"] = out["NonVoiceFlag"].astype(int)

    stats = {
        "total_rows": int(len(out)),
        "voice_count": int((out["NonVoiceFlag"] == VOICE).sum()),
        "non_voice_count": int((out["NonVoiceFlag"] == NON_VOICE).sum()),
        "ignored_count": ignored,
        "keyword_voice_count": keyword_voice,
        "keyword_non_voice_count": keyword_nv,
        "unmapped_subtasks": unmapped,
        "unmapped_defaulted_to_non_voice": bool(unmapped and allow_unmapped_default),
    }
    return out, stats
