# Changelog

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
