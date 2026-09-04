"""Item 4 — the optional local model layers.

The property that matters: neither layer is ever required. A missing package,
a missing model file, or a model that raises mid-call must all degrade to the
deterministic rules rather than failing a retraining run.
"""

from __future__ import annotations

import builtins

import pytest

from backend.services import local_models


def test_status_reports_both_layers_without_either_installed():
    report = local_models.status_report()
    assert set(report) >= {"embeddings", "generation", "models_root"}
    for layer in ("embeddings", "generation"):
        assert "available" in report[layer]
        assert report[layer]["reason"] in ("package_missing", "model_missing", "ready")
        # An unavailable layer must say why, in words a person can act on.
        if not report[layer]["available"]:
            assert report[layer]["detail"]


def test_matching_returns_empty_when_the_layer_is_unavailable():
    """Empty, not an exception: the caller keeps its rule-based suggestion."""
    assert local_models.match_subtasks(["New Task"], ["Called Insurance"]) == {}


def test_matching_is_a_no_op_on_empty_input():
    assert local_models.match_subtasks([], ["Called Insurance"]) == {}
    assert local_models.match_subtasks(["New Task"], []) == {}


def test_explain_returns_none_when_the_layer_is_unavailable():
    assert local_models.explain("Summarise this run.") is None


def test_a_model_that_raises_is_swallowed_not_propagated(monkeypatch):
    """A broken optional model must not take a retraining run down with it."""
    class Exploding:
        def encode(self, *args, **kwargs):
            raise RuntimeError("CUDA is on fire")

    monkeypatch.setattr(local_models, "_load_embedder", lambda: Exploding())
    assert local_models.match_subtasks(["New Task"], ["Called Insurance"]) == {}


def test_generation_failure_is_swallowed(monkeypatch):
    class Exploding:
        def create_chat_completion(self, *args, **kwargs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(local_models, "_load_llm", lambda: Exploding())
    assert local_models.explain("anything") is None


def test_embedding_match_respects_the_similarity_floor(monkeypatch):
    """A weak match is no match: below the floor the rules keep the answer."""
    import numpy as np

    class Stub:
        def encode(self, texts, normalize_embeddings=True):
            # "Claim Status Chk" close to "Claim Status Check", far from "Called".
            table = {
                "Claim Status Chk": [1.0, 0.0],
                "Claim Status Check": [0.99, 0.14],
                "Called Insurance": [0.0, 1.0],
            }
            return np.array([table[t] for t in texts], dtype=float)

    monkeypatch.setattr(local_models, "_load_embedder", lambda: Stub())

    matched = local_models.match_subtasks(
        ["Claim Status Chk"], ["Claim Status Check", "Called Insurance"], minimum=0.55
    )
    assert matched["Claim Status Chk"]["nearest"] == "Claim Status Check"
    assert matched["Claim Status Chk"]["source"] == "embeddings"

    # Raise the floor above the real similarity and the match must disappear.
    assert local_models.match_subtasks(
        ["Claim Status Chk"], ["Claim Status Check", "Called Insurance"], minimum=0.999
    ) == {}


def test_models_root_is_overridable_for_a_locked_down_machine(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_ENHANCER_MODELS", str(tmp_path / "models"))
    assert local_models.models_root() == tmp_path / "models"
