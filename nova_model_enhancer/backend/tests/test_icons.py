"""Every icon the UI asks for must exist in the inline set.

The app ships its own SVG icons because a blocked webfont CDN leaves the
literal ligature text on screen. A name with no entry silently renders the
fallback glyph — visible as a stray "?" — so it is caught here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_every_micon_name_has_an_inline_icon():
    icons_source = (FRONTEND / "components" / "icons.js").read_text(encoding="utf-8")
    body = icons_source.split("export const ICONS = {", 1)[1]
    available = set(re.findall(r"^\s{2}([a-z_0-9]+):\s*\{", body, re.MULTILINE))

    used: set[str] = set()
    for path in FRONTEND.rglob("*.jsx"):
        text = path.read_text(encoding="utf-8")
        # Literal names only; a dynamic name={cond ? "a" : "b"} yields both.
        for match in re.findall(r"<MIcon\b[^>]*?name=\{?([^>]*?)(?:size=|/>|>)", text, re.DOTALL):
            used |= set(re.findall(r'"([a-z_0-9]+)"', match))

    missing = sorted(used - available)
    assert not missing, f"icons.js has no entry for: {missing}"
