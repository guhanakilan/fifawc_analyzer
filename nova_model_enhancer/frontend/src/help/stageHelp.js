// ─────────────────────────────────────────────────────────────────────────────
// stageHelp.js — in-app documentation for every enhancer stage.
//
// Same structure as the NoVA ML workbench's own stageHelp.js:
//   what      : one sentence on the stage's purpose
//   inputs    : what must exist before this stage runs
//   produces  : artifacts or state this stage creates
//   decisions : what the user is actively choosing here
//   gotchas   : silent failures and things to watch for
// ─────────────────────────────────────────────────────────────────────────────

const stageHelp = {
  champion: {
    what: "Upload the ZIP exported from a completed NoVA ML run. The archive is validated and copied into an immutable workspace before anything in it is used.",
    inputs: [
      "A Stage 12 (Model Export) ZIP from nova-ml, up to 500 MB",
    ],
    produces: [
      "retraining job id (RETRAIN_XXX) that scopes every later artifact",
      "an immutable copy of the ZIP plus its SHA-256 fingerprint",
      "an extracted, canonically-laid-out copy under the job workspace",
    ],
    decisions: [
      "Whether you trust this local package enough to load its model file",
    ],
    gotchas: [
      "Intake never unpickles anything — a .pkl is opened only in the compatibility check",
      "model.pkl is not the runtime name: nova-ml resolves {training_results.best_model}.pkl",
      "A missing supporting config is a warning, not a blocker, but it lowers transform fidelity",
    ],
  },

  data: {
    what: "Upload the verified labelled data the challengers will be trained on. Files are streamed, not loaded whole.",
    inputs: [
      "A valid champion package (Stage 01)",
      "Labelled history and/or new verified rows as Parquet, CSV, XLSX or XLS",
    ],
    produces: [
      "one profiled dataset asset per upload, with a SHA-256 fingerprint",
      "schema drift against the champion's own column configuration",
    ],
    decisions: [
      "Whether each file is combined history+new, history only, or new rows only",
    ],
    gotchas: [
      "Parquet is preferred — CSV type inference can silently change a column's dtype",
      "A column the champion expects but the data lacks scores as zero at inference; fix drift before training, not after",
      "Duplicates are counted across chunk boundaries, so the number is the real one",
    ],
  },

  readiness: {
    what: "Confirm the rules that decide what gets trained, then freeze the dataset into an immutable, hashed snapshot.",
    inputs: [
      "At least one uploaded dataset (Stage 02)",
    ],
    produces: [
      "SNAP_XXX — an immutable parquet snapshot plus its manifest",
      "row counts, exclusions, date range, label distribution and config fingerprints",
    ],
    decisions: [
      "Which column drives the temporal split and recency weighting",
      "Whether labels come from an existing column or from SubTask mappings",
      "How duplicate rows are resolved, and on which business key",
      "A Voice / Non-Voice / Ignore mapping for every new SubTask",
      "An optional historical window",
    ],
    gotchas: [
      "The target encoding is confirmed from the reference source, not assumed: NonVoiceFlag is 0 = Voice, 1 = Non-Voice",
      "nova-ml defaults an unmapped SubTask to Non-Voice; this application refuses to, because that manufactures labels nobody approved",
      "A deduplication key is never inferred — two rows sharing an account number may or may not be the same work item",
      "The snapshot is written once. Re-running this stage creates a new snapshot; it never edits the old one",
    ],
  },

  weights: {
    what: "Configure, preview and approve the sample weights. Nothing is applied until a named person approves it.",
    inputs: [
      "A frozen dataset snapshot (Stage 03)",
    ],
    produces: [
      "the exact approved weight formula, its distribution, and the approver's name",
    ],
    decisions: [
      "Whether to weight at all — an unweighted baseline is a valid, approvable choice",
      "Which components apply, their multipliers, and the combined cap",
    ],
    gotchas: [
      "The values pre-filled here are proposals from the project brief, not policy",
      "Components are multiplicative and then capped, so several firing on one row cannot compound without limit",
      "A component whose column is absent is skipped and reported, never silently dropped",
    ],
  },

  training: {
    what: "Refit preprocessing and train the challenger slate as a cancellable background job.",
    inputs: [
      "A frozen snapshot (Stage 03) and an approved weight strategy (Stage 04)",
    ],
    produces: [
      "RUN_XXX with one fitted estimator per candidate, shared fitted transforms, and per-candidate metrics",
      "a threshold analysis per candidate and a rolling-origin backtest",
    ],
    decisions: [
      "Split mode and percentages, random seed, parallelism",
      "Optuna trial budget and per-candidate timeout",
      "Which criterion selects the threshold",
    ],
    gotchas: [
      "Preprocessing is refit for every challenger — reusing the champion's fitted state would leak its training distribution",
      "Thresholds are chosen on validation and reported on test, so the test set never tunes anything",
      "Task state lives in the workspace database, so a browser refresh recovers it and a backend restart reports it as interrupted rather than leaving it spinning",
    ],
  },

  comparison: {
    what: "Score champion and challengers on identical rows, apply the approved promotion gate, and record a typed decision.",
    inputs: [
      "A completed run (Stage 05)",
    ],
    produces: [
      "a champion/challenger comparison on one shared benchmark",
      "a promotion gate result per candidate and a persisted approval record",
    ],
    decisions: [
      "The primary metric, protected metrics and every tolerance — before looking at the numbers",
      "Which candidate, if any, is approved for promotion",
    ],
    gotchas: [
      "Until the gate is approved every candidate reports BLOCKED — that is deliberate, not a failure",
      "BLOCKED is never downgraded to NOT RECOMMENDED: it means the comparison itself cannot be trusted",
      "Per-segment metrics on SubTask are single-class by construction when labels derive from SubTask; pick a different segment column",
      "Approving against the gate's recommendation is allowed but is recorded as an override",
    ],
  },

  export: {
    what: "Build the deployable package, then prove it by loading and scoring through it exactly as the deployment does.",
    inputs: [
      "A typed promotion approval (Stage 06)",
      "An approved ml_tag encoding",
      "Optionally, a de-identified inventory sample",
    ],
    produces: [
      "PLC_<placement>_V<nnn>_STREAMLIT_READY.zip with model, configs, scoring runtime and metadata",
      "a validation report, a retraining report workbook and a rollback manifest",
    ],
    decisions: [
      "What ml_tag actually means — which value is Voice and which is Non-Voice",
    ],
    gotchas: [
      "Export is blocked until the ml_tag encoding is approved; no default is applied",
      "A package failing any blocking check is deleted, not published",
      "Previous versions are never overwritten, and nothing is deployed automatically",
      "The shipped scoring runtime patches four defects in the reference client — see IMPLEMENTATION_GAP_ANALYSIS.md",
    ],
  },
};

export default stageHelp;
