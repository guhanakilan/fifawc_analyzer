# Implementation Status

Version 1.1.0. Verified on the synthetic PLC 984 fixture; a real pilot run is still
outstanding pending the inputs in §3.

## 1. Stage-by-stage

| Stage | Status | Verified by |
|---|---|---|
| 01 Champion Package | Complete | 15 validator tests + browser run |
| 02 Training Data | Complete | 8 intake tests + browser run |
| 03 Readiness & snapshot | Complete | 9 snapshot/labelling tests + browser run |
| 04 Weight Strategy | Complete | 5 weighting tests + browser run |
| 05 Retrain & Tune | Complete | determinism, cancellation, recovery tests + browser run |
| 06 Comparison & approval | Complete | gate tests + end-to-end journey |
| 07 Export & validation | Complete | package smoke test + browser run |

**Tests:** 117 passing + 1 skipped (`pytest backend/tests -q`, ~21 s). The skip is the
reference-defect evidence test, which runs only when the reference tree is provided.
**Frontend build:** clean (`npm run build`, 442 modules).
**Browser run:** all seven stages driven through Chromium against the real backend,
zero page errors, including a mid-flow refresh that restored the active job.

## 2. Minimum completion criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `setup.bat` works without PowerShell activation | Done — uses `.venv\Scripts\python.exe` and `npm.cmd` only |
| 2 | `run.bat` launches both halves | Done — verified with the same command line the script issues |
| 3 | Health status is real | Done — `/health` probes the workspace and reads the schema version; the UI polls it every 10 s |
| 4 | All seven stages perform real actions | Done |
| 5 | Refresh does not lose the active job | Done — job id in the URL and local storage, task state in SQLite |
| 6 | A synthetic end-to-end run completes | Done — `test_end_to_end.py` and the browser run |
| 7 | A real pilot run completes | **Blocked** — see §3 |
| 8 | Export scores identically through the loader | Done — 100% of rows identical to 4 dp on the shipped runtime; see §4 |
| 9 | Tests and frontend build pass | Done |
| 10 | README covers install, run, test, backup, export, rollback, troubleshooting | Done |

## 3. Inputs still outstanding

Items 5-9 of `UPLOAD_CHECKLIST.md` were not supplied:

1. A real completed NoVA run export ZIP for PLC 984.
2. A real labelled PLC 984 training dataset.
3. A real de-identified inventory sample.
4. The production Streamlit application, or at least its model-loader code.
5. The lead-approved rules file.

Until (1)-(3) arrive the end-to-end evidence rests on the synthetic fixture, which
reproduces the real artifact contract but not real data distributions. Until (4)
arrives, "Streamlit-compatible" means compatible with `scoring_client/scoring.py` from
the reference tree, which is the only loader supplied.

## 4. Business decisions still required

Each is implemented as a persisted, explicitly approved configuration with a blocking
gate. None has a silent default.

| Decision | Where | Current state |
|---|---|---|
| `ml_tag` encoding | Stage 07 | **Blocks export.** Two candidate conventions are offered with their consequences; nothing is applied until an approver picks one. |
| Deduplication key | Stage 03 | Must be chosen, or full-row dedup explicitly confirmed. |
| Primary metric, protected metrics, tolerances | Stage 06 | Pre-filled from the brief and labelled PROPOSED; every comparison reports `BLOCKED` until a named approver saves the gate. |
| Historical window and weight formula | Stage 04 | Previewed, then persisted with the approver's name. |
| New SubTask mappings | Stage 03 | Training pauses; suggestions shown, never applied. |
| Target encoding | Stage 03 | **Confirmed, not assumed** — `NonVoiceFlag` 0 = Voice, 1 = Non-Voice, read from `routers/flag.py::run_flag`. |

## 5. Proof of scoring compatibility

The Stage 07 smoke test unzips the built package to a temporary directory, imports
`pipeline/scoring.py` from it, and scores the inventory sample through it. On the last
verified run (300-row sample):

| Check | Result |
|---|---|
| Package loads | Estimator + 10 fitted features |
| nova-ml parity mode | 300 rows, appended `NovaProbability`, `VoiceNonVoiceFlag` |
| Package vs enhancer predictions | max abs difference 0.000000, 100% identical to 4 dp |
| Row count preserved | 300 in, 300 out |
| Original columns preserved in order | 9 columns, order unchanged |
| Exactly one appended column | `['ml_tag']` |
| No probability or flag text exposed | passed |
| `ml_tag` values | only the two approved values |
| `ml_tag` inversion | matches the approved convention |

A package that fails any blocking check is deleted, not published.

## 5a. Guards added after the first review

| Guard | Why |
|---|---|
| Model output cannot silently become the label | `ml_tag`, `VoiceNonVoiceFlag` and `NovaProbability` are written by a scoring run. Selecting one as the training label is blocked (409) unless a person explicitly acknowledges the rows were human-verified after scoring. Without this, the challenger learns to agree with the champion and every later metric measures agreement, not accuracy. |
| Label dtype cannot change what a value means | A label column that is float (which any column that ever held a null becomes) rendered as `"0.0"`, never matched an approved encoding of `"0"`, and the snapshot rejected every row of valid data. Values are now canonicalised before comparison. |
| An ambiguous encoding is refused | Mapping the same value to both Voice and Non-Voice is now an error rather than a silent last-writer-wins. |
| Errors do not leak the workspace layout | Library exceptions embed the file they were reading. Paths are scrubbed from anything that reaches the UI, keeping the actionable reason. |
| Unrecognised label values are named | The error now lists the offending values instead of only counting them. |
| Reference-loader evidence is runnable | `NOVA_ENHANCER_REFERENCE_SCORING` points Stage 07 at the verbatim nova-ml client; its result is recorded in the validation report as non-blocking evidence. |

## 6. Known limitations

* The four reference defects in `IMPLEMENTATION_GAP_ANALYSIS.md` §4 are worked around
  in the shipped runtime, not fixed in nova-ml. D4 (one-hot features scoring as zero)
  means a champion running under the unpatched reference client performs worse in
  production than the Stage 06 benchmark shows. The comparison is therefore
  conservative, but this should be said out loud when presenting the numbers.
* Per-segment metrics are mostly reported as "sample too small or single-class" when
  the segment column is `SubTask`, because the label is derived *from* SubTask and each
  segment is single-class by construction. Choose a different segment column for a
  meaningful breakdown.
* Training runs in threads inside the API process — right for a single-user localhost
  tool, not a multi-tenant queue.
* When the champion package carries no `features_config.json`, a default transform
  config is derived per column dtype. This is recorded as
  `used_default_features_config` in the run record rather than being silent.
* Retention and cleanup are deliberately not automated. Nothing is ever deleted without
  an explicit user action.
