"""Item 9 — the threshold scale, floored at 0.50.

Both halves matter: the UI must not offer a value the API rejects, and the API
must reject one regardless of what the UI offers.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.services import evaluator
from backend.services.evaluator import (
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    THRESHOLD_STEP,
    clamp_threshold,
    threshold_grid,
    threshold_sweep,
)


def test_the_grid_is_exactly_what_was_agreed():
    assert (MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_STEP) == (0.50, 0.90, 0.05)
    assert threshold_grid() == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def test_the_sweep_and_the_selectable_grid_cannot_drift_apart():
    """A sweep that explores a threshold the UI cannot select is a trap."""
    y = np.resize([0, 1], 300)
    rng = np.random.default_rng(3)
    proba = np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1), 0.2), 0.001, 0.999)
    swept = [row["t"] for row in threshold_sweep(y, proba)["sweep"]]
    assert swept == threshold_grid()


@pytest.mark.parametrize("value,expected", [
    (0.50, 0.50), (0.63, 0.65), (0.62, 0.60), (0.90, 0.90), (0.8999, 0.90),
])
def test_a_value_between_steps_snaps_to_the_nearest(value, expected):
    assert clamp_threshold(value) == expected


@pytest.mark.parametrize("value", [0.0, 0.1, 0.49, 0.91, 1.0])
def test_anything_outside_the_range_is_refused(value):
    with pytest.raises(ValueError, match="outside the allowed range"):
        clamp_threshold(value)


def test_no_swept_threshold_is_below_a_coin_flip():
    """The reference swept from 0.10; below 0.5 the model calls Voice on rows
    it believes are Non-Voice, which is the reason for the floor."""
    y = np.resize([0, 1], 200)
    proba = np.linspace(0.01, 0.99, 200)
    assert all(row["t"] >= 0.5 for row in threshold_sweep(y, proba)["sweep"])


# ── The champion is scored honestly, not clamped ─────────────────────────────

def test_a_champion_below_the_floor_keeps_its_real_threshold():
    """Clamping it would report numbers for a model that is not in production."""
    from backend.services.pipeline import choose_threshold

    y_val = np.resize([0, 1], 200)
    y_test = np.resize([0, 1], 200)
    rng = np.random.default_rng(5)
    proba_val = np.clip(rng.normal(0.5 + 0.2 * (2 * y_val - 1), 0.2), 0.001, 0.999)
    proba_test = np.clip(rng.normal(0.5 + 0.2 * (2 * y_test - 1), 0.2), 0.001, 0.999)

    result = choose_threshold(y_val, proba_val, y_test, proba_test, 0.35, "f1")

    assert result["champion_below_floor"] is True
    champion_row = next(
        r for r in result["candidates"] if r["candidate"] == "champion_threshold"
    )
    assert champion_row["threshold"] == 0.35, "the champion must be scored where it runs"
    assert "not in production" in result["selection_note"]


def test_a_challenger_is_never_selected_below_the_floor():
    from backend.services.pipeline import choose_threshold

    y_val = np.resize([0, 1], 200)
    y_test = np.resize([0, 1], 200)
    rng = np.random.default_rng(6)
    proba_val = np.clip(rng.normal(0.5 + 0.2 * (2 * y_val - 1), 0.2), 0.001, 0.999)
    proba_test = np.clip(rng.normal(0.5 + 0.2 * (2 * y_test - 1), 0.2), 0.001, 0.999)

    result = choose_threshold(y_val, proba_val, y_test, proba_test, 0.20, "f1")
    assert result["selected_threshold"] >= MIN_THRESHOLD


def test_a_champion_on_the_grid_is_not_flagged():
    from backend.services.pipeline import choose_threshold

    y = np.resize([0, 1], 120)
    proba = np.clip(np.linspace(0.2, 0.8, 120), 0.001, 0.999)
    result = choose_threshold(y, proba, y, proba, 0.5, "f1")
    assert result["champion_below_floor"] is False
    assert "not in production" not in result["selection_note"]
