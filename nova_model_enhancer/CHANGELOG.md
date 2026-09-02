# Changelog

## 1.1.0 — workbench UI parity and correctness fixes

### UI rebuilt against the reference workbench

The interface now uses the NoVA ML Workbench's own component vocabulary rather than a
lookalike: gradient-bordered cards, the dotted-globe brand mark generated exactly as the
reference generates it, `MetricCard` tiles with DM Mono values, the collapsible
INPUTS / PRODUCES / DECISIONS / NOTES stage-help banner, the same 240px staged sidebar,
the same 44px breadcrumb header, the same button variants, and framer-motion springs
throughout. Values are taken from the reference's own `C` palette and `nova-theme.css`.

Icons remain inline SVG rather than the Material Symbols webfont. That is the one
deliberate divergence: this application is localhost-first, and a webfont failure would
otherwise leave the literal ligature text ("check_circle") on screen. A 1.6 MB logo PNG
was dropped in favour of the procedural globe.

### Correctness fixes

* **Model output can no longer become ground truth.** Selecting `ml_tag`,
  `VoiceNonVoiceFlag` or `NovaProbability` as the training label is blocked unless a
  person explicitly acknowledges the rows were human-verified after scoring. Previously
  those columns were offered as ordinary label candidates, which would have trained the
  challenger to agree with the champion rather than with reality.
* **Label dtype no longer changes what a value means.** A float label column — which any
  column that ever held a null becomes — rendered as `"0.0"` and never matched an
  approved encoding of `"0"`, so the snapshot rejected every row of valid data. Values
  are canonicalised before comparison.
* **An encoding mapping one value to both classes is refused** instead of silently
  resolving to whichever list was checked last.
* **Unrecognised label values are named** in the error rather than only counted.
* **Filesystem paths are scrubbed** from every message that reaches the UI. Library
  exceptions embed the file they were reading, which exposed the workspace layout.
* **Period breakdowns match by period name**, not array position, so a period one model
  skipped can no longer shift every later row and pair the wrong numbers.
* **Comparison slices reset their index** after positional selection, so pairing depends
  on position rather than on pandas' alignment rules.

### Evidence

* `NOVA_ENHANCER_REFERENCE_SCORING` points Stage 07 at nova-ml's verbatim scoring
  client; the outcome is recorded in the validation report as non-blocking evidence.
* `backend/tests/test_reference_defects.py` asserts all four reference defects against
  the reference source itself, skipping cleanly when that tree is absent. D1 is now
  reproduced rather than inferred: the reference client raises
  `AttributeError: 'str' object has no attribute 'get'` on a real export.
* The second reference drop ("the updated NoVA ML workbench") was verified byte-identical
  to the first — same 93 files, same SHA-256 for each. No defect in it has been fixed.

**Tests:** 117 passing + 1 skipped. **Build:** clean. **Browser:** all seven stages
driven end to end with zero page errors.

## 1.0.0 — complete seven-stage application

Rebuilt on the three-stage intake prototype. The artifact contract, the transformation
chain and the target/output encodings were all derived from the reference NoVA ML
source rather than assumed; see `IMPLEMENTATION_GAP_ANALYSIS.md`.

### Stages 4-7, previously placeholders

* **Weight strategy** — multiplicative, capped, previewable sample weights across
  recency, human corrections, verified previous errors, rare SubTasks and class
  balance. The exact formula and the approver's name are persisted and exported.
* **Retrain & tune** — cancellable background training with Optuna TPE, 3-fold HPO CV
  with median pruning, 5-fold final CV, probability calibration on a held-out slice,
  and threshold optimisation done on validation and reported on test.
* **Comparison & approval** — champion and challengers scored on identical rows, with
  per-period and per-segment breakdowns, a rolling-origin backtest, an explicit
  promotion gate and a typed approval record.
* **Export** — the package is assembled, then unzipped, loaded and scored exactly as
  the deployment does. It is published only if that passes.

### Stages 1-3, hardened

* ZIP-bomb ratio detection, wrapper-directory unwrapping, duplicate-artifact detection
  and a blocking/warning split derived from real downstream need.
* Intake no longer touches any `.pkl`. The estimator is loaded only in an explicit
  compatibility step behind a local-trust acknowledgement.
* Datasets are streamed in chunks; a 2 GB CSV is profiled without being materialised.
* Readiness now requires approved decisions (date column, label source, deduplication
  key, SubTask mappings) and produces an immutable, hashed snapshot with a manifest.

### Architecture

* SQLite migrations with a recorded `schema_version`, replacing ad-hoc table creation.
* Background task state persisted in the database, so the UI recovers after a refresh
  and reports `interrupted` honestly after a backend restart.
* Workspace path resolved lazily rather than at import, so a test run can never write
  into the application directory.
* Atomic writes (temp file then rename) for every snapshot, manifest and export.
* SHA-256 recorded for uploads, snapshots and exports; audit trail for every decision.
* All paths built from generated ids; uploaded filenames are never used as paths.

### Frontend

* Rebuilt against the reference design language: DM Sans/DM Mono, the NoVA palette, a
  240px staged sidebar, light and dark themes.
* Icons are inline SVG rather than a webfont, so the UI is correct with no network.
* Real `/health` polling replaces the previous hardcoded "API connected" text.
* Active job restored from the URL and local storage after a refresh.
* Every disabled control states why it is disabled.

### Defects found in the reference application

Documented in `IMPLEMENTATION_GAP_ANALYSIS.md` §4 and worked around in the shipped
scoring runtime. D4 in particular — unnormalised one-hot column names causing every
one-hot feature to score as zero — was caught by the Stage 7 smoke test comparing the
built package against the enhancer's own predictions.

## 0.2.0 — handover state

Champion ZIP intake, training-data upload and a read-only readiness screen. Stages 4-7
were placeholders.
