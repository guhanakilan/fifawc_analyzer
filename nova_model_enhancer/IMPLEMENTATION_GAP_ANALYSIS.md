# Implementation Gap Analysis

Reference application: `nova_ml_reference_source.zip` (read-only).
Replacement application: NoVA Model Enhancer (this project).

**Reference version check.** A second reference drop,
`novamlenhancementmodelimprovement_5.zip` ("the updated NoVA ML workbench"), was
supplied later. It contains the same 93 files and every one has an identical SHA-256
to the first drop — the two trees are byte-identical. Nothing in this document changed
as a result, and every defect in §4 is still present in the version currently
considered current.

Every contract below was read out of the reference source, not assumed. File and
line references are to the reference tree as extracted from the handoff ZIP.

---

## 1. Baseline of the handed-over enhancer

Verified before any change was made:

| Check | Result |
|---|---|
| `pytest backend/tests -q` | 3 passed |
| `uvicorn backend.main:app` boots | yes |
| `GET /health` | `{"status":"ok","application":"NoVA Model Enhancer"}` |
| Stages implemented | 1 (champion ZIP), 2 (data upload), 3 (readiness display only) |
| Stages 4-7 | placeholder `ComingSoon` panel, no backend at all |

Structural problems found in the baseline that this rebuild fixes:

1. `App.jsx` held every stage in one 92-line minified-style file with no routing,
   no persistence and no recovery — a browser refresh lost the active job entirely.
2. The frontend `Topbar` printed the literal text **"API connected"** with no health
   request behind it (explicitly forbidden by the brief).
3. `ReadinessStage` derived pass/fail from hard-coded thresholds in JSX
   (`s.rows >= 2000`) with no backend record, no snapshot and no audit entry.
4. `database.py` created tables ad hoc with no schema version, so any column
   addition would silently break an existing workspace.
5. `data_profiler.read_dataset` loaded whole CSV/Excel files into memory — the
   stated 2 GB limit could not survive it.
6. `package_validator` demanded `config/features_config.json` as a *core*
   artifact. It is core for retraining, but the reference **scoring** path never
   reads it, so the blocker classification needed re-deriving per real downstream need.

---

## 2. The real artifact contract (derived from reference code)

### 2.1 Export ZIP layout — `backend/routers/export.py`

`_EXPORT_FILES` plus the model file give the authoritative layout:

```
model/{model_id}.pkl          <- NOT model.pkl; model_id is a composite key
model/fitted_transforms.pkl
config/column_map.json
config/subtask_mappings.json
config/column_config.json
config/dtype_config.json
config/derived_config.json
config/bucket_config.json
config/grouping_config.json
config/feature_selection.json
config/features_config.json
scoring/threshold_config.json
metadata/training_results.json
metadata/model_selection_config.json
metadata/pipeline_version.json
metadata/manifest.json        <- generated at export time
```

`model_id` is `f"{job_id}_{model_type}"` (`routers/training.py`, `_train_worker`),
e.g. `JOB_20250507_lgb`. The suggested `model/model.pkl` name in the master prompt
is **wrong for this loader** — `scoring_client/scoring.py` resolves the model as
`{training_results["best_model"]}.pkl`. The enhancer therefore keeps the composite
naming and additionally writes a `model.pkl` copy only when a loader-compatibility
alias is explicitly requested.

### 2.2 On-disk JSON shapes — `backend/data/file_store.py`

| File | Shape written by nova-ml |
|---|---|
| `column_map.json` | `{"column_map": [...], "coverage_threshold": float\|null}` |
| `column_config.json` | `{"matched_columns": [...]}` |
| `feature_selection.json` | `{"selected_columns": [...]}` |
| `subtask_mappings.json` | `{"mappings": [{"name","flag"}], "keywords": [...]}` |
| `dtype_config.json` | bare `{col: {"dtype","fallback"}}` |
| `derived_config.json` | bare `[...]` |
| `bucket_config.json` | bare `{col: {"cuts","labels"}}` |
| `grouping_config.json` | bare `{col: {"kept_values","others_label","null_label"}}` |
| `threshold_config.json` | bare `{model_id: float}` |
| `features_config.json` | `{"outlier_capping","log_transform","imputation","encoding","scaling","split","temporal_weight"}` |

### 2.3 Fitted state — `data/feature_pipeline.py::_fit_transform_by_indices`

`fitted_transforms.pkl` is a plain pickled dict with exactly these keys:
`outlier_bounds`, `imputation_vals`, `label_encoders`, `freq_maps`, `onehot_cols`,
`scalers`, `log_cols`, `feature_names`. `feature_names` fixes the model's input
column order and must never be regenerated independently of the estimator.

### 2.4 Target encoding — confirmed, not assumed

`backend/routers/flag.py::run_flag` is explicit:

* SubTask mapped `Voice` -> `NonVoiceFlag = 0`
* SubTask mapped `Non-Voice` -> `NonVoiceFlag = 1`
* SubTask mapped `Keyword` -> `ARComments` contains any keyword -> `0`, else `1`
* SubTask mapped `Ignore` -> row dropped
* SubTask absent from the mapping -> `1` (default Non-Voice)

So the internal target encoding **is** Voice = 0 / Non-Voice = 1, and
`P(class 1) = P(Non-Voice)`.

### 2.5 Output encoding — confirmed inverted

`scoring_client/scoring.py` step 12 appends **two** columns:

```
NovaProbability   = round(P(Non-Voice), 4)
VoiceNonVoiceFlag = int(proba < threshold)     # 1 = Voice, 0 = Non-Voice
```

This is inverted relative to the training target. Any `ml_tag` convention must be
stated explicitly rather than inherited — see the open decision in §5.

---

## 3. Behaviour the enhancer must reproduce

| Reference behaviour | Source | Enhancer implementation |
|---|---|---|
| `_norm_col` / `_dedupe_columns` column identity | `routers/eda.py`, `scoring.py` | `services/nova_transform.py` (byte-identical logic, shared by fit and score paths) |
| Column rename inventory -> production | `scoring.py::_build_rename_map` | `nova_transform.build_rename_map` |
| SubTask -> `NonVoiceFlag` labelling | `routers/flag.py::run_flag` | `services/labeling.py::apply_subtask_mapping` |
| Column filter, dtype cast, derived cols | `routers/eda.py::_load_eda_df` | `nova_transform.build_modelling_frame` |
| Bucket -> `{col}_Bucket`, grouping -> `{col}_Grouped` | `feature_pipeline._load_base_df` | same, same suffix casing |
| Rename-aware feature selection | `feature_pipeline._load_base_df` | same |
| Fit-on-train-only transform chain | `feature_pipeline._fit_transform_by_indices` | `nova_transform.fit_transforms` / `apply_transforms` |
| Temporal split, val carve-out (15% of train+val, min 40 rows / 10 minority) | `feature_pipeline._temporal_split_indices` | `services/splitter.py` (same constants) |
| Optuna TPE, 3-fold HPO CV, MedianPruner, 5-fold final CV | `routers/training.py` | `services/trainer.py` (same defaults and search spaces) |
| Probability calibration (isotonic >= 1000 val rows, else sigmoid, FrozenEstimator) | `routers/training.py::_calibrate_estimator` | `services/trainer.py` (same rule) |
| `gb` balance folded into `sample_weight` | `training.py::_effective_sample_weight` | `services/trainer.py` |
| Threshold sweep 0.10-0.90 step 0.05 with weighted composite | `routers/evaluation.py::_compute_threshold_sweep` | `services/evaluator.py` (same weights) |
| Rolling-origin backtest | `feature_pipeline.run_rolling_backtest_all` | `services/evaluator.py::rolling_backtest` |

---

## 4. Defects found in the reference application

These are reported, not fixed in the reference app (it stays read-only). Both are
load-bearing for Stage 7 validation.

**D1 — `column_map.json` wrapper.** `scoring.py::_build_rename_map` iterates the
loaded JSON as a list of row dicts. The real file is a dict with a `column_map`
key, so iteration yields the string `"column_map"` and `r.get(...)` raises
`AttributeError`. A verbatim reference loader cannot read a real nova-ml export.

*Reproduced, not inferred.* Pointing `NOVA_ENHANCER_REFERENCE_SCORING` at the
reference client and running a package this application built produces:

```
Reference client raised AttributeError: 'str' object has no attribute 'get'
```

`backend/tests/test_reference_defects.py::test_d1_reference_client_cannot_read_a_real_column_map`
asserts this against the reference source itself; it skips when the reference tree is
not present.

**D2 — `feature_selection.json` wrapper.** Same class of bug, but silent instead of
loud: iterating the dict yields `"selected_columns"`, that name matches no column,
the selection step drops every real feature and refills a single zero column, and
the model then scores a constant on every row.

**D3 — train/score transform order differ.** `feature_pipeline._fit_transform_by_indices`
applies outlier capping, then log, then imputation. `scoring_client/scoring.py`
applies imputation (step 6), then capping (7), then log (8). For any column that is
both log-transformed and imputed, the fill value learned on post-log training data is
inserted pre-log at scoring time and then logged again, so the deployed model sees a
different number than the one it was fitted on.

The enhancer keeps the *fitting* order identical to the reference (so the
`fitted_transforms.pkl` it writes is interchangeable with a nova-ml one) and uses the
*scoring* order whenever it applies an already-fitted state. Following the fitting
order at inference would make every number this application reports disagree with what
the deployed package actually produces.

**D4 — one-hot column names are not normalised.** `scoring.py` loads
`fitted_transforms.pkl` with `{_norm_col(k): v for k, v in onehot_cols.items()}` —
the dict key is normalised, the list of generated column names inside it is not.
Training names those columns after the source column, which for a bucketed or grouped
feature is `"<col>_Grouped"` with a capital letter. At scoring time the dummies are
generated from the already-lowercased column, producing `"<col>_grouped_<value>"`,
which never matches. The loader then adds each stored name as a zero column and the
final `reindex` to the normalised `feature_names` drops them again. **Every one-hot
feature therefore scores as zero**, silently, on any nova-ml model that one-hot
encodes a bucketed or grouped column.

This was found by the Stage 7 smoke test, not by inspection: the packaged loader and
the enhancer's own predictions disagreed on 98.4% of rows, and the disagreement
resolved to exactly this.

**How the enhancer responds.** It ships `pipeline/scoring.py` inside its export: the
reference client plus three documented changes — a shape-tolerant `_read_json` that
unwraps `{"column_map": [...]}` and `{"selected_columns": [...]}` (D1/D2), normalised
one-hot column names (D4), and the additional `ml_tag` output mode. Stage 7 runs the
smoke test through the shipped loader and, when a copy of the reference tree is placed
beside the application, through the verbatim reference loader as well, reporting each
result separately so the defects stay visible rather than being papered over.

**Consequence for the comparison.** Because the enhancer scores the champion through
the corrected chain, Stage 6 reports what the champion *would* achieve once the fixed
runtime is deployed. If a champion currently in production is running under the
unpatched reference client and uses one-hot encoding, its live performance is worse
than the benchmark figure shown. That makes the comparison conservative, never
flattering, but it should be stated when the numbers are presented.

---

## 5. Business decisions the enhancer must not invent

Each is implemented as a persisted, explicitly approved configuration with a
blocking gate — never a silent default.

| Decision | Enhancer behaviour |
|---|---|
| Final `ml_tag` encoding | Stage 7 blocks export until an encoding is chosen and typed-approved. No default is applied. |
| Deduplication key | Stage 3 requires the user to select the key columns (or explicitly choose full-row dedup). |
| Primary / protected metrics and promotion tolerances | Stage 6 gate config is user-editable and persisted with the approval record; the proposed values from the brief are pre-filled and labelled `PROPOSED`. |
| Historical window and weighting formula | Stage 4 previews and persists the exact approved formula plus approver identity. |
| New SubTask mapping | Stage 3 pauses and requires an authorised Voice / Non-Voice / Ignore decision per new SubTask. Suggestions are shown, never auto-applied. |
| Promotion | Always a recommendation; export requires a typed approval record. |

---

## 6. Inputs still outstanding

Items 5-9 of `UPLOAD_CHECKLIST.md` (real PLC 984 run ZIP, real labelled Parquet,
de-identified inventory sample, the production Streamlit loader, and the
lead-approved rules file) were not supplied. Until they arrive, end-to-end proof
runs on the synthetic PLC 984 fixture generated by
`backend/tests/fixtures.py`, which reproduces the real artifact contract above.
