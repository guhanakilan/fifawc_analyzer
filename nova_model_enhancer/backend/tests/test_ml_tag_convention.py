"""ml_tag is 1 = Voice, 0 = Non-Voice — confirmed by the project owner.

This is the one encoding that cannot be inferred from the package, and getting
it backwards inverts every routing decision downstream without any error. The
value is pinned here so a refactor cannot quietly flip it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_the_recommended_convention_is_one_equals_voice(client, champion_export):
    with champion_export["zip_path"].open("rb") as handle:
        job = client.post(
            "/api/packages/upload", files={"file": ("p.zip", handle, "application/zip")}
        ).json()["job_id"]

    body = client.get(f"/api/export/{job}/ml-tag").json()
    recommended = [c for c in body["candidate_conventions"] if c.get("recommended")]

    assert len(recommended) == 1, "exactly one convention must be marked confirmed"
    assert recommended[0]["voice_value"] == 1
    assert recommended[0]["non_voice_value"] == 0
    assert recommended[0]["column_name"] == "ml_tag"


def test_export_is_still_blocked_until_a_person_approves(client, champion_export):
    """Knowing the convention does not remove the approval gate."""
    with champion_export["zip_path"].open("rb") as handle:
        job = client.post(
            "/api/packages/upload", files={"file": ("p.zip", handle, "application/zip")}
        ).json()["job_id"]

    body = client.get(f"/api/export/{job}/ml-tag").json()
    assert body["blocked"] is True
    assert body["approved_config"] is None


def test_a_voice_row_is_tagged_one_end_to_end(tmp_path):
    """The runtime maps a Voice prediction to 1, not merely stores the config.

    The model predicts P(Non-Voice), so a probability below the threshold is a
    Voice row, and that row must carry ml_tag = 1.
    """
    import importlib.util

    source = Path(__file__).resolve().parents[1] / "scoring_runtime" / "scoring.py"
    spec = importlib.util.spec_from_file_location("shipped_scoring", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Stub(module.NovaMLPipeline):
        """Only the pieces run_ml_tag touches."""

        def __init__(self):
            self.ml_tag_config = {
                "approved": True, "column_name": "ml_tag",
                "voice_value": 1, "non_voice_value": 0,
            }
            self.threshold = 0.5

        def predict_proba(self, df):
            # Row 0 is confidently Voice (low P(Non-Voice)); row 1 is Non-Voice.
            return df, np.array([0.10, 0.90])

    frame = pd.DataFrame({"AccountID": ["A1", "A2"], "AmountBilled": [100.0, 200.0]})
    tagged = Stub().run_ml_tag(frame)

    assert list(tagged.columns) == ["AccountID", "AmountBilled", "ml_tag"]
    assert tagged["ml_tag"].tolist() == [1, 0], "a Voice row must be tagged 1"
    assert len(tagged) == 2

