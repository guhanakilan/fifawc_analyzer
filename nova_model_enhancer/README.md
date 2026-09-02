# NoVA Model Enhancer

A separate, fully local web application that takes the ZIP exported from a completed
NoVA ML run, retrains and tunes challenger models on verified data, compares them
fairly against the uploaded champion, and — only after a named person approves —
exports a validated scoring package.

The existing NoVA ML application is never modified. The uploaded champion, its
original data and every previously exported version are immutable.

* Backend: FastAPI + SQLite (metadata only; large files live on disk)
* Frontend: React + Vite, using the NoVA ML Workbench's own design language
* Models: scikit-learn, XGBoost, LightGBM, Optuna
* No Bedrock, no OpenAI, no paid cloud service, no mandatory outbound network call

---

## Install on Windows

You need Python 3.10+ and Node.js 18+ on PATH.

Tested on Python 3.10 and 3.11, and on Node 18 through 24. `setup.bat` checks the
Python version up front and tells you plainly if it is too old, rather than letting
pip fail halfway through with a resolver error.

1. Unzip this folder somewhere you can write to, e.g. `C:\nova\nova_model_enhancer`.
2. Double-click **`setup.bat`**. It creates `.venv`, installs the Python packages with
   `.venv\Scripts\python.exe`, and installs the frontend packages with `npm.cmd`.

`setup.bat` deliberately never runs `activate.ps1`, so a restricted PowerShell
execution policy cannot block it.

## Try it before you have real data

You need a completed NoVA run export to do anything, so if you do not have one yet,
double-click **`make_demo_data.bat`**. It generates a synthetic PLC 984 placement into
`demo_data\`:

| File | Feed it to |
|---|---|
| `PLC_984_nova_export.zip` | Stage 01 — Champion Package |
| `PLC_984_labelled.parquet` | Stage 02 — Training Data |
| `PLC_984_inventory_sample.parquet` | Stage 07 — scoring compatibility sample |

Nothing in it is real: every value is generated, and no patient-like data is involved.
The point is that it has the same *shape* as a real export, so all seven stages can be
exercised end to end. The synthetic champion is a genuinely trained model (F1 ≈ 0.73,
AUC ≈ 0.81 on its own test split), so the challengers have something real to beat.

The demo files are built on your machine by your own installation, which is why they
are generated rather than shipped — the application asks you to load only packages you
produced yourself, and that applies to its own demo too.

In Stage 03 use `UpdatedDateTimeGMT` as the date column and `AccountID` as the
deduplication key; every other decision is yours to make.

## Run

Double-click **`start.bat`**. Two console windows open and your browser is pointed at
the UI.

| | |
|---|---|
| UI | http://127.0.0.1:5174 |
| API | http://127.0.0.1:8081 |
| API docs | http://127.0.0.1:8081/docs |

Close both console windows to stop the application. Everything it wrote stays in
`workspace\`.

## The seven stages

| Stage | What it does | What it will not do |
|---|---|---|
| 01 Champion Package | Validates and extracts the NoVA export ZIP | Open any `.pkl` during intake |
| 02 Training Data | Streams and profiles labelled Parquet/CSV/Excel | Load a 2 GB CSV into memory |
| 03 Readiness | Confirms the rules, then freezes an immutable snapshot | Infer a deduplication key, map a new SubTask, or train on model output |
| 04 Weight Strategy | Previews and approves sample weights | Apply a weight nobody approved |
| 05 Retrain & Tune | Trains challengers in a cancellable background job | Reuse the champion's fitted state as a challenger's |
| 06 Comparison | Scores champion and challengers on identical rows | Pick a "best" model by an assumed metric |
| 07 Export | Builds, loads and scores the package before publishing | Deploy anything, or guess the `ml_tag` encoding |

## What the export contains

The layout is derived from the reference `routers/export.py` and
`scoring_client/scoring.py`, not assumed:

```
PLC_<placement>_V<nnn>_STREAMLIT_READY.zip
├── model/<model_id>.pkl          the deployable estimator
├── model/fitted_transforms.pkl   its matching fitted preprocessing state
├── config/                       column_map, dtype, derived, bucket, grouping,
│                                 feature_selection, features_config, subtask_mappings,
│                                 column_config
├── scoring/threshold_config.json decision threshold, keyed by model id
├── scoring/ml_tag_config.json    the approved ml_tag convention
├── metadata/                     training_results, manifest, dataset_manifest,
│                                 champion_comparison, validation_report,
│                                 approval_record, rollback_manifest, audit_trail,
│                                 model_selection_config, pipeline_version
├── pipeline/scoring.py           the scoring runtime
├── reports/retraining_report.xlsx
└── README.txt
```

The estimator keeps its composite `{model_id}.pkl` name. It is **not** renamed to
`model.pkl`: the deployed loader resolves the file as
`"{training_results.best_model}.pkl"`, so renaming it would break scoring.

### Deploying it

Unzip into the placement folder so `pipeline/` sits beside `config/`, `model/`,
`scoring/` and `metadata/` — the layout nova-ml's own client expects:

```python
from scoring import NovaMLPipeline

pipeline = NovaMLPipeline()
scored = pipeline.run(inventory_df)          # + NovaProbability, VoiceNonVoiceFlag
tagged = pipeline.run_ml_tag(inventory_df)   # original columns + ml_tag only
```

### Encodings — read before consuming the output

| Where | Column | Meaning |
|---|---|---|
| Training target | `NonVoiceFlag` | 0 = Voice, 1 = Non-Voice |
| Model output | `predict_proba[:, 1]` | P(Non-Voice) |
| `run()` | `VoiceNonVoiceFlag` | 1 = Voice, 0 = Non-Voice (inverted) |
| `run_ml_tag()` | `ml_tag` | whatever your approver confirmed in Stage 07 |

`run_ml_tag()` refuses to run on a package whose `ml_tag` convention was never
approved. It never returns a probability or Voice/Non-Voice text.

## Optional: prove compatibility against the verbatim nova-ml client

Stage 07 always validates the package through the scoring runtime it ships. It can
*additionally* run the built package through nova-ml's own unmodified
`scoring_client/scoring.py` and record the outcome as non-blocking evidence:

```bat
set NOVA_ENHANCER_REFERENCE_SCORING=C:\path\to\nova-ml\scoring_client\scoring.py
start.bat
```

Expect that check to report a warning: the reference client cannot read a real
nova-ml export (defect D1 in `IMPLEMENTATION_GAP_ANALYSIS.md`). That warning is the
point — it is recorded so the defect stays visible rather than being assumed fixed.

## Rollback

Every exported version is kept under `workspace\jobs\<job>\exports\` and listed in the
Versions panel and in `metadata/rollback_manifest.json`. To roll back, download the
earlier version and unzip it over the placement folder. This application never deploys
to any environment and never deletes a previous version.

## Backup

Copy the whole `workspace\` folder. It contains everything: the uploaded champion ZIPs,
the datasets, the frozen snapshots, the trained models, the exports, and
`enhancer.sqlite3` with the job metadata and the audit trail. Nothing else on the
machine is written to.

To relocate the workspace, set `NOVA_ENHANCER_WORKSPACE` before starting the backend:

```bat
set NOVA_ENHANCER_WORKSPACE=D:\nova-workspace
start.bat
```

## Tests

```bat
.venv\Scripts\python.exe -m pytest backend\tests -q
```

Run from the folder *containing* `nova_model_enhancer`, or from inside it — the test
bootstrap adds the repository root to `sys.path` either way. Tests always run against a
throwaway temporary workspace and never touch `workspace\`.

Frontend production build:

```bat
cd frontend
npm.cmd run build
```

## Troubleshooting

**"API unreachable" in the header.** The backend console window closed or never
started. The chip reflects a real `/health` request every 10 seconds — it is never
hardcoded, so if it says unreachable, it is.

**`setup.bat` fails at "Checking Python".** Python is not on PATH. Reinstall it with
"Add python.exe to PATH" ticked, or edit `setup.bat` to use the full interpreter path.

**PowerShell blocks a script.** Nothing here needs PowerShell. Use `setup.bat` and
`start.bat` from Explorer or `cmd.exe`.

**Port already in use.** Another copy is running, or something else holds 8081/5174.
Close the old console windows, or change the ports in `start.bat` and
`frontend\vite.config.js` together.

**A training task shows "interrupted".** The backend restarted while it was running.
Its state was recovered honestly rather than left spinning; start a new run.

**"xgboost / lightgbm cannot be trained".** The library is not installed in `.venv`.
Rerun `setup.bat`. Those families are excluded from the candidate plan rather than
silently substituted.

**Export validation failed.** The ZIP was discarded, not published. The panel lists
exactly which check failed. This is the intended behaviour: a package that cannot score
correctly must never be handed to a deployment.

**The UI looks unstyled.** The DM Sans webfont could not load (no outbound network).
The layout, icons and colours are all local, so only the typeface changes. Icons are
inline SVG rather than the Material Symbols webfont the reference app uses, precisely
so a missing network cannot leave raw ligature text on screen.

**"This column is model output".** You selected `ml_tag`, `VoiceNonVoiceFlag` or
`NovaProbability` as the training label. Those are written by a scoring run, so
training on them teaches the challenger to agree with the champion instead of with
reality. Pick a human-verified column, derive labels from SubTask mappings, or tick
the acknowledgement if those rows really were corrected by a person after scoring.

## Known limitations

* The four reference-application defects described in
  `IMPLEMENTATION_GAP_ANALYSIS.md` are worked around in the shipped scoring runtime,
  not fixed in nova-ml itself. Anyone deploying against the *unpatched* reference
  client should read §4 of that document first.
* Verified end to end against the synthetic PLC 984 fixture in
  `backend/tests/fixtures.py`. A real pilot run still needs the real inputs listed in
  `IMPLEMENTATION_STATUS.md`.
* Training runs in threads inside the API process. That is appropriate for a
  single-user localhost tool; it is not a multi-tenant job queue.
