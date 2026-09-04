# Changelog

## 1.3.0 — champion parity and the escalating retrain loop

### Same and only the champion's pipeline

`features_config.json` was a blocking artifact, but validation checked only that
the file existed. An empty one passed, and the pipeline then fabricated a config —
Median imputation for every unconfigured numeric, Label encoding for every
categorical — so the challenger was trained through preprocessing the champion never
had. Intake now checks the file defines at least one rule, and the fabricator is gone.

Threshold selection is removed. Every model is scored at the champion's own operating
threshold, so a metric difference is the model and not the cutoff. Choosing per model
compared two operating points as well as two models. The validation sweep is still
shown as guidance and never applied.

Custom columns were already reproduced faithfully — diffed against the reference's own
`custom_cols._apply_derived_config`: all three types, all five date_part variants, same
fallback and case-resolution semantics. What was missing was a check that reproduction
succeeded, and writing it found a real gap: the reference treats an unrecognised
`col_type` as "condition", which with no branches yields a column of nulls, so a custom
column from a newer NoVA ML would appear to build and silently train as empty.

### Escalating retrain loop

Optional. Grows the champion's own family until a target is met — more trees, more
branches, learning rate falling and regularisation tightening as capacity rises.

Every round is scored on validation; the test split is read once, afterwards. On the
verification run validation F1 reached 0.747 and test came in at 0.729. That gap is
what a loop scored against its own target conceals, and it is now visible. Four stop
rules: target reached, no improvement, round and time caps, and gain smaller than the
run-to-run variation. The loop can fail, and says so.

### Optional SQL Server source

Read-only, Windows integrated authentication, no credential to store. The statement is
a SELECT built over a configured table or view with dates bound as parameters — no query
text reaches the endpoint and no stored procedure can be called. Strictly optional: with
no driver and no configuration the card says so and file upload is unaffected, which is
what the brief's "do not *require* direct SQL Server access" asks for.

### Fixes

* A call to `setGateApprover` survived the merge of the four approver inputs in 1.2.0.
  Vite compiles that happily and the comparison stage died at runtime, blank. A test now
  checks every state setter a component calls exists — the one class of error a build
  cannot catch.
* The escalation result was dropped between the pipeline and the comparison screen.
* The comparison endpoint took 5.2s: 1.2.0's bootstrap ran 400 sklearn calls per model
  across six models. Threshold metrics now come from resampled confusion counts as array
  operations — 0.76s to 0.015s per model, 5.2s to 0.69s overall, with a test asserting
  the fast path and a scorer loop agree exactly.
* The app reported version 1.0.0 in its footer, `/health` and every export's provenance
  record, two releases behind.


## 1.2.0 — ten corrections and enhancements

Each item below was agreed before implementation, and each was verified against the real
PLC984 export rather than only against fixtures.

### 1. Resumable runs and a jobs home screen

Candidates now checkpoint as they finish, so a run interrupted by a restart resumes and
retrains only what is missing. Resume is explicit — nothing heavy starts on its own when
the backend boots. A new home screen lists every saved job with its runs, models and
counts, and reopens one at the stage it reached. Nothing here is destructive: leaving a
job clears only this browser's pointer to it.

### 2. Four-layer column lineage

Drift is now traced through all four column lists a package carries — mapped, matched,
selected, fitted — so the point at which a column left the pipeline is visible. Testing
against the real export exposed two false alarms the old flat comparison would have
produced: `dosage_days` is *derived*, not uploaded, and `DOSFrom` / `UpdatedDateTimeGMT`
are required precisely because they are what it is derived from, despite not being
features. Between matched and selected the export records no reason, so those columns are
reported as dropped during the build with that limitation stated rather than guessed at.

### 3. Dark-mode logo and bundled fonts

The brand mark drew a hard-coded white plate behind its dots, which showed as a bright box
in dark mode. Removed. DM Sans and DM Mono are now served from the application itself
(101 KB, SIL Open Font Licence, licence text bundled) rather than the Google Fonts CDN —
offline, or behind a proxy, the two families were falling back to different substitutes,
which is what read as inconsistent typography. The stacks were declared 22 times over in
two spellings; they are now one pair of CSS variables.

### 4. Configurable rules engine for Stage 3

Recommendations for the date column, label source, dedup key, historical window and
SubTask mappings, each carrying the rule that produced it and the evidence behind it.
Interventions — four blocking, four warning — with every threshold stored per job and
changeable without a code change. Optional local models (embeddings, and a small local
LLM for written rationales) refine the suggestions when installed and degrade to the rules
when not; neither can block a run, and generation is pinned to temperature 0 so an
explanation can be reviewed.

### 5. Explicit training window

From/to date pickers over the approved date column, opening on the data's real span, with
a live preview of the rows kept, the resulting span and the class balance. A date-only
upper bound includes the whole of that day; an empty window fails loudly rather than
freezing an empty dataset; the older days-back setting still works untouched.

### 6. Weighting proposed from the data

The advisor measures the frozen snapshot — balance, span, recency, flag columns, rarest
SubTask — and proposes a strategy with a reason for each component, on or off. Below
2,000 rows or 90 days it recommends no weighting at all. A NoVA export carries no client
field, so `FacilityName` is treated as the client dimension and `PayerName` as a secondary
one when present, with a fallback to placement plus measured characteristics.

### 7. Performance

The backtest now runs 4 windows instead of 8 and fits its preprocessing once per window
rather than once per model: 5.8s to 1.9s. It is also opt-in and off by default, taking a
standard run from ~35s to ~29.7s. Parallel candidate training was implemented, measured
and left off — on 4 cores it is no faster (30.0s) and splitting cores two ways is markedly
slower (38.6s), because the tree models already parallelise internally. It remains
available as `max_parallel_candidates` for a machine with more cores than one fit can use.
A skipped backtest is now reported as "not assessed" rather than falling through the gate
silently.

### 8. Model comparison

Added McNemar significance testing and bootstrap confidence intervals; operating-point
metrics (precision at a target recall, recall at a target precision, top-decile lift); a
cost-weighted view at the approved 3:1 missed-Voice ratio; and a disagreement analysis
with a per-segment win/loss table. Plus a guidance panel suggesting concrete next actions
from the run's own evidence — never a recommendation to promote.

On the verification run these change the reading entirely: the challenger's higher F1
comes from 136 disagreements split 73/63 at p = 0.44, it costs *more* than the champion
(660 against 618, from 187 missed-Voice errors against 161), and it is worse on three of
six segments including the largest.

### 9. Threshold scale floored at 0.50

The grid is 0.50 to 0.90 in steps of 0.05, enforced identically in the optimiser's sweep,
the UI control and the API's validation — a test asserts the three cannot drift apart. The
champion keeps its real threshold even when that is below the floor, because reporting it
at any other value would describe a model that is not in production. A new read-only
rescore endpoint backs a threshold control that did not previously exist.

### 10. One operator identity

Four separate "type your name" boxes across Stages 3, 4, 6 and 7 are now one identity set
once per job in the header. Every decision still records its own approver and timestamp,
so the audit trail is unchanged. The substantive gates all remain: the local-trust
acknowledgement before any pickle is loaded, the two conditional acknowledgements that
appear only when the risk is present, and the typed promotion confirmation.


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
