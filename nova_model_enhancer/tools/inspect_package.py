"""Report what Stage 01 would make of a NoVA export ZIP, without running the app.

Runs the real package validator — the same code the upload endpoint uses — over
a ZIP and prints its verdict, every check, and the metadata it could read. Uses
only the standard library, so it works on a bare Python before setup.bat has
built the virtual environment.

Nothing is unpickled and nothing is written outside a temporary directory that
is deleted on exit. The ZIP itself is never modified.

Usage:
    python tools\\inspect_package.py "C:\\path\\to\\PLC984_..._export.zip"
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT.parent))

from nova_model_enhancer.backend.services.package_validator import (  # noqa: E402
    ALLOWED_SUFFIXES,
    BLOCKING_ARTIFACTS,
    WARNING_ARTIFACTS,
    PackageValidationError,
    validate_and_extract,
)

TICK = {"passed": "PASS", "warning": "WARN", "failed": "FAIL"}


def _raw_listing(zip_path: Path) -> None:
    """What is actually in the archive, before any interpretation."""
    print("ARCHIVE CONTENTS")
    print("-" * 78)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
    except Exception as exc:
        print(f"  could not be opened as a ZIP: {exc}")
        return

    suffixes: dict[str, int] = {}
    for member in sorted(members, key=lambda m: m.filename):
        name = member.filename.replace("\\", "/")
        suffix = Path(name).suffix.lower()
        suffixes[suffix or "(none)"] = suffixes.get(suffix or "(none)", 0) + 1
        flag = "" if suffix in ALLOWED_SUFFIXES else "   <-- extension not in the allow-list"
        print(f"  {member.file_size:>12,}  {name}{flag}")

    print()
    print(f"  {len(members)} files, {sum(m.file_size for m in members):,} bytes uncompressed")
    print("  extensions: " + ", ".join(f"{k} x{v}" for k, v in sorted(suffixes.items())))
    unknown = sorted(s for s in suffixes if s != "(none)" and s not in ALLOWED_SUFFIXES)
    if unknown:
        print()
        print("  NOTE: intake rejects the whole package on an unrecognised extension.")
        print(f"        Unrecognised here: {', '.join(unknown)}")
        print(f"        Currently allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}")


def _artifact_coverage(zip_path: Path) -> None:
    """Which logical artifacts the validator will find, by basename."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = [m.filename.replace("\\", "/") for m in archive.infolist() if not m.is_dir()]
    except Exception:
        return
    present = {Path(n).name for n in names}

    print()
    print("ARTIFACT COVERAGE")
    print("-" * 78)
    print("  Blocking — the package is unusable without these:")
    for artifact, why in sorted(BLOCKING_ARTIFACTS.items()):
        mark = "found  " if artifact in present else "MISSING"
        print(f"    [{mark}] {artifact}")
        if artifact not in present:
            print(f"              {why}")
    print("  Supporting — absence is a warning, not a blocker:")
    for artifact in sorted(WARNING_ARTIFACTS):
        mark = "found  " if artifact in present else "absent "
        print(f"    [{mark}] {artifact}")

    models = sorted(
        n for n in names
        if n.lower().endswith(".pkl") and Path(n).name != "fitted_transforms.pkl"
    )
    print(f"  Champion estimator candidates ({len(models)}):")
    for model in models:
        print(f"    {model}")
    if len(models) != 1:
        print("    ^ exactly one is required")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    zip_path = Path(argv[1]).expanduser()
    if not zip_path.is_file():
        print(f"No such file: {zip_path}")
        return 2

    print("=" * 78)
    print(f"  {zip_path.name}")
    print(f"  {zip_path.stat().st_size:,} bytes")
    print("=" * 78)
    print()

    _raw_listing(zip_path)
    _artifact_coverage(zip_path)

    print()
    print("VALIDATOR VERDICT")
    print("-" * 78)
    with tempfile.TemporaryDirectory(prefix="nova_inspect_") as tmp:
        try:
            result = validate_and_extract(zip_path, Path(tmp) / "extracted")
        except PackageValidationError as exc:
            print("  REJECTED OUTRIGHT — nothing would be stored.")
            print(f"  Reason: {exc}")
            print()
            print("  This is a hard rejection before any per-artifact check runs.")
            return 1
        except Exception as exc:
            print(f"  The validator itself raised {type(exc).__name__}: {exc}")
            print("  That is a defect in the enhancer, not in the package.")
            return 1

        for check in result["checks"]:
            scope = "blocking" if check["blocking"] else "advisory"
            print(f"  [{TICK.get(check['status'], '????')}] {check['label']}  ({scope})")
            print(f"         {check['detail']}")

        print()
        print(f"  ACCEPTED: {result['valid']}")
        if result["blocking_failures"]:
            print(f"  Blocking failures: {', '.join(result['blocking_failures'])}")
        if result["wrapper_directory"]:
            print(f"  Wrapper directory unwrapped: {result['wrapper_directory']}")

        print()
        print("METADATA READ FROM THE PACKAGE")
        print("-" * 78)
        for key, value in result["metadata"].items():
            print(f"  {key:24} {value}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
