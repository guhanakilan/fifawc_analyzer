"""Generate a synthetic PLC 984 dataset so the application can be tried immediately.

Nothing here is real data: every value is generated. The point is that the files
have the same *shape* as a real NoVA run export and a real labelled extract, so
all seven stages can be exercised before the genuine inputs are available.

Usage (from the nova_model_enhancer folder):
    .venv\\Scripts\\python.exe tools\\make_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
# The test fixtures import as `nova_model_enhancer.backend.…`, so the folder
# *containing* this application has to be importable.
sys.path.insert(0, str(APP_ROOT.parent))

from nova_model_enhancer.backend.tests import fixtures  # noqa: E402


def main() -> int:
    destination = APP_ROOT / "demo_data"
    destination.mkdir(parents=True, exist_ok=True)

    print("Generating a synthetic champion export (this trains a small model)…")
    result = fixtures.write_champion_export(destination / "PLC_984_nova_export.zip", rows=6000)

    labelled = destination / "PLC_984_labelled.parquet"
    result["raw"].to_parquet(labelled, index=False)

    inventory = destination / "PLC_984_inventory_sample.parquet"
    fixtures.make_inventory_sample(rows=400).to_parquet(inventory, index=False)

    metrics = result["training_results"]["results"][fixtures.CHAMPION_MODEL_ID]["test_metrics"]
    print()
    print(f"Written to {destination}")
    print(f"  PLC_984_nova_export.zip         Stage 01 — the champion package")
    print(f"  PLC_984_labelled.parquet        Stage 02 — {len(result['raw']):,} labelled rows")
    print(f"  PLC_984_inventory_sample.parquet Stage 07 — 400 unlabelled rows to score")
    print()
    print(f"The synthetic champion scores F1 {metrics['f1']}, AUC {metrics['auc']} on its own test split,")
    print("so there is a real model for the challengers to be measured against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
