"""Every state setter a component calls must actually exist.

Removing a useState and leaving one of its setters behind is invisible to the
build — Vite compiles it happily and the page dies at runtime with
"setX is not defined". That happened when the four approver inputs were merged
into one operator identity: the comparison stage kept a setGateApprover call
and the whole page failed to render.

This is deliberately narrow. It is not a linter; it catches the one mistake
that a build cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"

SETTER_CALL = re.compile(r"\b(set[A-Z]\w*)\s*\(")
USE_STATE = re.compile(r"\[\s*\w+\s*,\s*(set[A-Z]\w*)\s*\]")
DECLARED = re.compile(r"\b(?:const|let|var|function)\s+(set[A-Z]\w*)\b")
# Setters can also arrive as props, destructured or referenced on an object.
PROP_DESTRUCTURE = re.compile(r"\{[^{}]*?\b(set[A-Z]\w*)\b[^{}]*?\}")
MEMBER_CALL = re.compile(r"\.(set[A-Z]\w*)\s*\(")

# Browser and standard-library functions that happen to match the pattern.
KNOWN_GLOBALS = {"setTimeout", "setInterval", "setImmediate"}


def _jsx_files():
    return sorted(FRONTEND.rglob("*.jsx"))


@pytest.mark.parametrize("path", _jsx_files(), ids=lambda p: p.name)
def test_every_setter_called_is_defined_somewhere_in_the_file(path):
    source = path.read_text(encoding="utf-8")

    called = set(SETTER_CALL.findall(source))
    # A method call like obj.setFoo() is not a bare identifier.
    called -= set(MEMBER_CALL.findall(source))
    called -= KNOWN_GLOBALS

    available = (
        set(USE_STATE.findall(source))
        | set(DECLARED.findall(source))
        | set(PROP_DESTRUCTURE.findall(source))
    )

    missing = sorted(called - available)
    assert not missing, (
        f"{path.name} calls {missing} but nothing in the file defines them — "
        "the page will fail at runtime while the build succeeds."
    )
