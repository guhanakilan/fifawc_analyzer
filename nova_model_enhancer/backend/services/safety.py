"""Identifier, filename and path safety helpers.

Every path the application builds is rooted at a directory it owns and keyed by
an identifier it generated. Client-supplied strings never reach the filesystem
without passing through here.
"""

import re
import unicodedata
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UnsafeIdentifier(ValueError):
    """Raised when an identifier could escape its owning directory."""


def assert_safe_id(value: str) -> str:
    """Accept only the identifier shape this application generates."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise UnsafeIdentifier("Identifier contains unsupported characters.")
    return value


def safe_filename(name: str, fallback: str = "upload") -> str:
    """Reduce an uploaded filename to a single safe basename.

    Directory components, control characters and Windows reserved names are all
    removed. The result is only ever used for display and for the suffix; stored
    files are named from generated ids.
    """
    base = Path(str(name or "")).name
    base = unicodedata.normalize("NFKC", base)
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    base = re.sub(r'[<>:"/\\|?*]', "_", base).strip(" .")
    if not base or base.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        return fallback
    return base[:200]


def resolve_within(root: Path, *parts: str) -> Path:
    """Join under `root` and prove the result did not escape it."""
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise UnsafeIdentifier("Resolved path escaped its owning directory.")
    return candidate


_PATH_RE = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w .~-]+[\\/])+[\w .~-]*")


def scrub(message: object, limit: int = 400) -> str:
    """Strip filesystem paths out of a message before it reaches the UI.

    Library exceptions routinely embed the file they were reading. That reveals
    the workspace layout to anyone who can see an error toast, and the path is
    never the part of the message a user can act on — the reason is.
    """
    text = str(message)
    text = _PATH_RE.sub("<path>", text)
    return text[:limit]
