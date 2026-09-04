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


# ── Every model is scored at the champion's threshold ────────────────────────
#
# Threshold *selection* was removed: picking a different cutoff per model
# compared two operating points as well as two models, and flattered whichever
# got the more favourable one. The champion's threshold is now the operating
# point for everything.

def _probabilities(seed):
    y = np.resize([0, 1], 200)
    rng = np.random.default_rng(seed)
    proba = np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1), 0.2), 0.001, 0.999)
    return y, proba


def test_the_champions_threshold_is_the_operating_point():
    from backend.services.pipeline import choose_threshold

    y_val, proba_val = _probabilities(5)
    y_test, proba_test = _probabilities(6)

    result = choose_threshold(y_val, proba_val, y_test, proba_test, 0.5, "f1")
    assert result["selected_threshold"] == 0.5
    assert result["selected_candidate"] == "champion_threshold"


def test_an_unusual_champion_threshold_is_still_used_verbatim():
    """Including below the 0.50 floor: that is where the champion actually runs."""
    from backend.services.pipeline import choose_threshold

    y_val, proba_val = _probabilities(7)
    y_test, proba_test = _probabilities(8)

    result = choose_threshold(y_val, proba_val, y_test, proba_test, 0.35, "f1")
    assert result["selected_threshold"] == 0.35
    assert result["champion_below_floor"] is True


def test_no_alternative_threshold_can_be_selected():
    """The sweep is still reported, but it is advisory and never applied."""
    from backend.services.pipeline import choose_threshold

    y_val, proba_val = _probabilities(9)
    y_test, proba_test = _probabilities(10)

    result = choose_threshold(y_val, proba_val, y_test, proba_test, 0.7, "f1")
    assert [c["threshold"] for c in result["candidates"]] == [0.7]
    assert result["validation_sweep"] is not None, "the sweep stays available as guidance"
    assert "never applied" in result["selection_note"]


def test_champion_and_challenger_are_compared_at_one_cutoff():
    """The point of the change: a metric difference is the model, not the cutoff."""
    from backend.services.evaluator import metrics_at_threshold
    from backend.services.pipeline import choose_threshold

    y_test, champion_proba = _probabilities(11)
    _, challenger_proba = _probabilities(12)

    result = choose_threshold(y_test, challenger_proba, y_test, challenger_proba, 0.62, "f1")
    champion = metrics_at_threshold(y_test, champion_proba, 0.62)

    assert result["selected_threshold"] == 0.62
    assert champion["f1"] is not None
