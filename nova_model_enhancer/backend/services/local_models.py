"""Optional local models that sharpen rule suggestions. Never required.

Two layers sit on top of the rules engine, both off unless their package *and*
their model files are present on this machine:

  embeddings — sentence-transformers, for matching a new SubTask name against
               ones the champion already knows, by meaning rather than exact
               text ("ClaimStatusChk" ≈ "Claim Status Check").
  generation — a small local LLM via llama-cpp-python, for a written rationale.

Both run entirely on this machine. No request leaves it, which is what the
project brief requires: "no Bedrock, OpenAI API, paid cloud AI, or mandatory
external network service". Model files are fetched once, on explicit request,
and cached; nothing downloads on its own.

Every entry point degrades rather than raises. A missing package, a missing
model file or a model that errors mid-call all return "unavailable" with a
reason, and the caller falls back to the deterministic rules.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("nova_enhancer")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_LOCK = threading.Lock()
_EMBEDDER = None
_EMBEDDER_STATE: str | None = None
_LLM = None
_LLM_STATE: str | None = None


def models_root() -> Path:
    """Where model files are cached. Overridable for a locked-down machine."""
    override = os.environ.get("NOVA_ENHANCER_MODELS")
    if override:
        return Path(override)
    from ..config import workspace_root
    return workspace_root() / "models"


@dataclass
class LayerStatus:
    name: str
    available: bool
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name, "available": self.available,
            "reason": self.reason, "detail": self.detail,
        }


# ── Embeddings ───────────────────────────────────────────────────────────────

def embedding_status() -> LayerStatus:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return LayerStatus(
            "embeddings", False, "package_missing",
            "sentence-transformers is not installed. Install it to enable "
            "meaning-based SubTask matching; the rules work without it.",
        )
    cached = models_root() / "all-MiniLM-L6-v2"
    if not cached.is_dir():
        return LayerStatus(
            "embeddings", False, "model_missing",
            f"The embedding model has not been downloaded to {cached}. "
            "Fetch it once from the rules screen to enable this layer.",
        )
    return LayerStatus("embeddings", True, "ready", str(cached))


def _load_embedder():
    """Load the sentence-transformer once, from the local cache only."""
    global _EMBEDDER, _EMBEDDER_STATE
    with _LOCK:
        if _EMBEDDER is not None or _EMBEDDER_STATE == "failed":
            return _EMBEDDER
        status = embedding_status()
        if not status.available:
            _EMBEDDER_STATE = "failed"
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(str(models_root() / "all-MiniLM-L6-v2"))
            _EMBEDDER_STATE = "ready"
        except Exception as exc:  # noqa: BLE001 — optional layer, never fatal
            logger.warning("Embedding model failed to load: %s", exc)
            _EMBEDDER_STATE = "failed"
            _EMBEDDER = None
        return _EMBEDDER


def match_subtasks(unknown: list[str], known: list[str], minimum: float = 0.55) -> dict:
    """Nearest known SubTask for each unknown one, by embedding similarity.

    Returns {} when the layer is unavailable, so the caller keeps the rule-based
    suggestion rather than losing one.
    """
    if not unknown or not known:
        return {}
    model = _load_embedder()
    if model is None:
        return {}
    try:
        import numpy as np

        vectors_known = model.encode(known, normalize_embeddings=True)
        vectors_unknown = model.encode(unknown, normalize_embeddings=True)
        similarity = np.asarray(vectors_unknown) @ np.asarray(vectors_known).T

        matches: dict[str, dict] = {}
        for index, name in enumerate(unknown):
            row = similarity[index]
            best = int(row.argmax())
            score = float(row[best])
            if score >= minimum:
                matches[name] = {
                    "nearest": known[best],
                    "similarity": round(score, 4),
                    "source": "embeddings",
                }
        return matches
    except Exception as exc:  # noqa: BLE001 — optional layer, never fatal
        logger.warning("Embedding match failed: %s", exc)
        return {}


# ── Generation ───────────────────────────────────────────────────────────────

def llm_status() -> LayerStatus:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return LayerStatus(
            "generation", False, "package_missing",
            "llama-cpp-python is not installed. It needs a C++ toolchain on "
            "Windows unless a prebuilt wheel matches your Python. Everything "
            "works without it; it only adds written rationales.",
        )
    directory = models_root()
    candidates = sorted(directory.glob("*.gguf")) if directory.is_dir() else []
    if not candidates:
        return LayerStatus(
            "generation", False, "model_missing",
            f"No .gguf model found in {directory}. Place one there to enable "
            "written rationales.",
        )
    return LayerStatus("generation", True, "ready", candidates[0].name)


def _load_llm():
    global _LLM, _LLM_STATE
    with _LOCK:
        if _LLM is not None or _LLM_STATE == "failed":
            return _LLM
        status = llm_status()
        if not status.available:
            _LLM_STATE = "failed"
            return None
        try:
            from llama_cpp import Llama

            model_path = sorted(models_root().glob("*.gguf"))[0]
            _LLM = Llama(
                model_path=str(model_path), n_ctx=2048, verbose=False, n_threads=os.cpu_count(),
            )
            _LLM_STATE = "ready"
        except Exception as exc:  # noqa: BLE001 — optional layer, never fatal
            logger.warning("Local LLM failed to load: %s", exc)
            _LLM_STATE = "failed"
            _LLM = None
        return _LLM


def explain(prompt: str, max_tokens: int = 220) -> str | None:
    """A short written rationale, or None when the layer is unavailable.

    Temperature is 0 so the same run explains itself the same way twice; an
    explanation that changes between reads cannot be reviewed or approved.
    """
    model = _load_llm()
    if model is None:
        return None
    try:
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain data-readiness findings for a machine-learning "
                        "retraining tool. Be concise and factual. Use only the numbers "
                        "given to you. Never invent a figure. Never recommend promoting "
                        "a model."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (response["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as exc:  # noqa: BLE001 — optional layer, never fatal
        logger.warning("Local LLM generation failed: %s", exc)
        return None


def status_report() -> dict:
    """What each optional layer can do right now, and why not when it cannot."""
    return {
        "embeddings": embedding_status().as_dict(),
        "generation": llm_status().as_dict(),
        "models_root": str(models_root()),
        "note": (
            "Both layers are optional. Every recommendation is produced by the "
            "deterministic rules first; these only refine them."
        ),
    }
