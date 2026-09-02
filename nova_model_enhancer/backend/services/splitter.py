"""Time-based and random split logic, matching the reference exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Same constants as `data/feature_pipeline.py`. The validation slice exists so
# threshold selection never touches the reported test set.
VAL_FRACTION_OF_TRAINVAL = 0.15
MIN_VAL_ROWS = 40
MIN_VAL_MINORITY_ROWS = 10


def _val_is_trustworthy(y_trainval: pd.Series, n_trainval: int) -> tuple[bool, int]:
    minority = int(y_trainval.value_counts().min()) if y_trainval.nunique() > 1 else 0
    would_be_rows = int(round(n_trainval * VAL_FRACTION_OF_TRAINVAL))
    would_be_minority = int(round(minority * VAL_FRACTION_OF_TRAINVAL))
    ok = would_be_rows >= MIN_VAL_ROWS and would_be_minority >= MIN_VAL_MINORITY_ROWS
    return ok, would_be_rows


def temporal_split_indices(dates: pd.Series, y_full: pd.Series, test_size: float):
    """Oldest rows -> train, newest -> test, val carved immediately before test.

    Rows with unparseable dates sort first so they always land in train: recency
    cannot be judged for them, so they must not define "recent performance".
    """
    n = len(dates)
    order = np.argsort(dates.fillna(pd.Timestamp.min).values, kind="stable")

    n_test = max(1, min(int(round(n * test_size)), n - 1))
    test_idx = order[n - n_test:]
    trainval_idx = order[: n - n_test]

    y_trainval = y_full.iloc[trainval_idx]
    n_trainval = len(trainval_idx)
    use_val, val_rows = _val_is_trustworthy(y_trainval, n_trainval)
    if use_val:
        val_idx = trainval_idx[n_trainval - val_rows:]
        train_idx = trainval_idx[: n_trainval - val_rows]
    else:
        train_idx, val_idx = trainval_idx, np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def random_split_indices(y_full: pd.Series, test_size: float, seed: int, stratify: bool = True):
    indices = np.arange(len(y_full))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed,
        stratify=y_full.values if stratify else None,
    )
    y_trainval = y_full.iloc[trainval_idx]
    use_val, _ = _val_is_trustworthy(y_trainval, len(trainval_idx))
    if use_val:
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=VAL_FRACTION_OF_TRAINVAL, random_state=seed,
            stratify=y_trainval.values if stratify else None,
        )
    else:
        train_idx, val_idx = trainval_idx, np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def describe_split(dates: pd.Series | None, train_idx, val_idx, test_idx) -> dict:
    """Human-checkable description, including the leakage guarantee."""
    def _window(idx):
        if dates is None or len(idx) == 0:
            return {"rows": int(len(idx)), "from": None, "to": None}
        window = dates.iloc[idx].dropna()
        return {
            "rows": int(len(idx)),
            "from": str(window.min()) if len(window) else None,
            "to": str(window.max()) if len(window) else None,
        }

    out = {
        "train": _window(train_idx),
        "validation": _window(val_idx),
        "test": _window(test_idx),
        "validation_used": bool(len(val_idx)),
    }
    if dates is not None and len(train_idx) and len(test_idx):
        train_max = dates.iloc[train_idx].dropna()
        test_min = dates.iloc[test_idx].dropna()
        if len(train_max) and len(test_min):
            out["no_future_leakage"] = bool(train_max.max() <= test_min.min())
    return out
