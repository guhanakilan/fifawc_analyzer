"""The UI's stage mapping must cover every stage the backend can set.

A stage name the frontend does not know silently drops the user on Stage 1
when they reopen a saved job, which looks like lost work even though nothing
was lost. This test fails instead.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND = BACKEND.parent / "frontend" / "src"


def backend_stage_names() -> set[str]:
    names: set[str] = set()
    for path in BACKEND.rglob("*.py"):
        if "tests" in path.parts:
            continue
        names |= set(re.findall(r'current_stage\s*=\s*"([A-Z_]+)"', path.read_text(encoding="utf-8")))
        names |= set(re.findall(r'"current_stage":\s*"([A-Z_]+)"', path.read_text(encoding="utf-8")))
    return names


def test_every_backend_stage_has_a_ui_route():
    mapping = (FRONTEND / "App.jsx").read_text(encoding="utf-8")
    block = mapping.split("const STAGE_FOR_BACKEND = {", 1)[1].split("};", 1)[0]
    known = set(re.findall(r"([A-Z_]+):", block))
    missing = backend_stage_names() - known
    assert not missing, f"App.jsx STAGE_FOR_BACKEND is missing: {sorted(missing)}"


def test_every_backend_stage_has_a_home_screen_label():
    home = (FRONTEND / "stages" / "HomeScreen.jsx").read_text(encoding="utf-8")
    block = home.split("const STAGE_LABEL = {", 1)[1].split("};", 1)[0]
    known = set(re.findall(r"([A-Z_]+):", block))
    missing = backend_stage_names() - known
    assert not missing, f"HomeScreen.jsx STAGE_LABEL is missing: {sorted(missing)}"
