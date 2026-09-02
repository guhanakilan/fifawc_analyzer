import React from "react";

import { api } from "../api.js";
import {
  ActionRow, Badge, Btn, C, Card, CheckRow, EmptyState, ErrorNotice, Field,
  FormGrid, MIcon, MetricGrid, Notice, Pill, ProgressBar, SectionTitle,
  SubHeading, Table,
} from "../nova/Components.jsx";
import { num, when } from "../format.js";
import { NoJob } from "./TrainingDataStage.jsx";

const TERMINAL = new Set(["complete", "failed", "cancelled", "interrupted"]);
const STATUS_TONE = {
  complete: "ok", failed: "bad", cancelled: "warn", interrupted: "warn",
  running: "info", queued: "muted",
};

export default function TrainingStage({ job, mark, go }) {
  const [options, setOptions] = React.useState(null);
  const [loadError, setLoadError] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [task, setTask] = React.useState(null);
  const [log, setLog] = React.useState([]);
  const [runs, setRuns] = React.useState([]);
  const [busy, setBusy] = React.useState(false);
  const [config, setConfig] = React.useState({
    mode: "temporal", train_pct: 70, val_pct: 15, test_pct: 15,
    n_trials: 20, timeout_seconds: "", n_jobs: -1, seed: 42,
    include_baseline: true, run_backtest: true, backtest_windows: "",
    threshold_criterion: "f1", second_family: "",
  });

  const loadRuns = React.useCallback(async () => {
    if (!job) return;
    try {
      const body = await api.runs(job.job_id);
      setRuns(body.runs);
      if (body.runs.length) mark("runId", body.runs[0].run_id);
    } catch (runsError) {
      setError(runsError);
    }
  }, [job, mark]);

  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    Promise.all([api.trainingOptions(job.job_id), api.tasks(job.job_id)])
      .then(([optionsBody, tasksBody]) => {
        if (!live) return;
        setOptions(optionsBody);
        setTask(tasksBody.tasks.find((t) => !TERMINAL.has(t.status)) || tasksBody.tasks[0] || null);
      })
      .catch((err) => live && setLoadError(err));
    loadRuns();
    return () => {
      live = false;
    };
  }, [job, loadRuns]);

  React.useEffect(() => {
    if (!task || TERMINAL.has(task.status)) return () => {};
    let live = true;
    const tick = async () => {
      try {
        const [latest, logBody] = await Promise.all([
          api.task(task.task_id), api.taskLog(task.task_id),
        ]);
        if (!live) return;
        setTask(latest);
        setLog(logBody.lines);
        if (TERMINAL.has(latest.status)) loadRuns();
      } catch {
        /* a transient poll failure is not worth a banner; the next tick retries */
      }
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [task, loadRuns]);

  if (!job) return <NoJob />;
  if (loadError) {
    return (
      <Card>
        <SectionTitle>Training unavailable</SectionTitle>
        <ErrorNotice error={loadError} title="Could not load the training options" />
      </Card>
    );
  }
  if (!options) {
    return <Card><SectionTitle sub="Reading the champion configuration…">Loading</SectionTitle></Card>;
  }

  const unavailable = Object.entries(options.available_model_types)
    .filter(([, ok]) => !ok).map(([key]) => key);
  const splitTotal = Number(config.train_pct) + Number(config.val_pct) + Number(config.test_pct);
  const running = task && !TERMINAL.has(task.status);
  const set = (key, value) => setConfig((current) => ({ ...current, [key]: value }));

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const started = await api.startTraining(job.job_id, {
        split: {
          mode: config.mode,
          train_pct: Number(config.train_pct),
          val_pct: Number(config.val_pct),
          test_pct: Number(config.test_pct),
          stratify: true,
          seed: Number(config.seed),
        },
        n_trials: Number(config.n_trials),
        timeout_seconds: config.timeout_seconds === "" ? null : Number(config.timeout_seconds),
        n_jobs: Number(config.n_jobs),
        seed: Number(config.seed),
        second_family: config.second_family || null,
        include_baseline: config.include_baseline,
        run_backtest: config.run_backtest,
        backtest_windows: config.backtest_windows === "" ? null : Number(config.backtest_windows),
        threshold_criterion: config.threshold_criterion,
        actor: "local-user",
      });
      setTask(await api.task(started.task_id));
      setLog([]);
    } catch (startError) {
      setError(startError);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    setError(null);
    try {
      await api.cancelTask(task.task_id);
      setTask(await api.task(task.task_id));
    } catch (cancelError) {
      setError(cancelError);
    }
  };

  return (
    <>
      <Card>
        <SectionTitle sub="Preprocessing is refit from scratch for every challenger. The champion's own fitted state is used only to score the champion.">
          Candidate plan
        </SectionTitle>
        <MetricGrid
          compact
          min={160}
          items={[
            { label: "Champion model", value: options.champion_model_id || "unresolved" },
            { label: "Champion family", value: (options.champion_family || "unknown").toUpperCase() },
            { label: "Champion threshold", value: options.champion_threshold, color: C.indigo },
          ]}
        />
        <SubHeading>What will be trained</SubHeading>
        <Table
          columns={[
            { key: "label", header: "Candidate" },
            { key: "model_type", header: "Family", render: (r) => r.model_type.toUpperCase() },
            {
              key: "mode", header: "Search",
              render: (r) => (r.mode === "tuned"
                ? <Pill tone="info">Optuna</Pill>
                : <Pill tone="muted">fixed params</Pill>),
            },
            { key: "n_trials", header: "Trials", className: "num", render: (r) => r.n_trials ?? "—" },
          ]}
          rows={options.default_candidate_plan}
          rowKey={(row) => row.candidate_id}
        />
        {unavailable.length > 0 && (
          <Notice tone="warn" title="Some model families cannot run here">
            {unavailable.map((key) => options.unavailable_note[key]).join(" ")} They are excluded from
            the plan rather than silently substituted.
          </Notice>
        )}
      </Card>

      <Card>
        <SectionTitle sub={options.split_note}>Run configuration</SectionTitle>

        <FormGrid min={190}>
          <Field
            label="Split mode" htmlFor="split-mode"
            hint="Temporal is the honest choice for a model that scores future work."
          >
            <select id="split-mode" value={config.mode} disabled={running}
              onChange={(e) => set("mode", e.target.value)}>
              <option value="temporal">Temporal — oldest train, newest test</option>
              <option value="random">Random stratified</option>
            </select>
          </Field>
          <Field label="Random seed" htmlFor="seed" hint="A fixed seed makes the whole run reproducible.">
            <input id="seed" type="number" value={config.seed} disabled={running}
              onChange={(e) => set("seed", e.target.value)} />
          </Field>
          <Field
            label="Parallelism (n_jobs)" htmlFor="jobs"
            hint="-1 uses every core. Lower it to keep the machine responsive."
          >
            <input id="jobs" type="number" value={config.n_jobs} disabled={running}
              onChange={(e) => set("n_jobs", e.target.value)} />
          </Field>
        </FormGrid>

        <FormGrid min={150} style={{ marginTop: 14 }}>
          <Field label="Train %" htmlFor="train-pct">
            <input id="train-pct" type="number" value={config.train_pct} disabled={running}
              onChange={(e) => set("train_pct", e.target.value)} />
          </Field>
          <Field
            label="Validation %" htmlFor="val-pct"
            hint="Used for calibration and threshold selection only."
          >
            <input id="val-pct" type="number" value={config.val_pct} disabled={running}
              onChange={(e) => set("val_pct", e.target.value)} />
          </Field>
          <Field label="Test %" htmlFor="test-pct">
            <input id="test-pct" type="number" value={config.test_pct} disabled={running}
              onChange={(e) => set("test_pct", e.target.value)} />
          </Field>
        </FormGrid>

        {Math.abs(splitTotal - 100) > 0.01 && (
          <Notice tone="warn" title="The split does not total 100%">
            Currently {splitTotal}%. Adjust before starting.
          </Notice>
        )}

        <FormGrid min={190} style={{ marginTop: 14 }}>
          <Field label="Optuna trials per tuned candidate" htmlFor="trials">
            <input id="trials" type="number" min="1" value={config.n_trials} disabled={running}
              onChange={(e) => set("n_trials", e.target.value)} />
          </Field>
          <Field label="Timeout per candidate (seconds)" htmlFor="timeout">
            <input id="timeout" type="number" min="1" placeholder="No timeout"
              value={config.timeout_seconds} disabled={running}
              onChange={(e) => set("timeout_seconds", e.target.value)} />
          </Field>
          <Field
            label="Threshold criterion" htmlFor="criterion"
            hint="Candidates are generated on validation and reported on test."
          >
            <select id="criterion" value={config.threshold_criterion} disabled={running}
              onChange={(e) => set("threshold_criterion", e.target.value)}>
              <option value="f1">F1</option>
              <option value="recall">Recall</option>
              <option value="precision">Precision</option>
              <option value="balanced_accuracy">Balanced accuracy</option>
              <option value="weighted_composite">Weighted composite</option>
            </select>
          </Field>
        </FormGrid>

        <div style={{ marginTop: 14 }}>
          <CheckRow checked={config.include_baseline} disabled={running}
            onChange={(value) => set("include_baseline", value)}>
            Include a logistic-regression baseline, so "better than the champion" has a floor to sit above.
          </CheckRow>
          <CheckRow checked={config.run_backtest} disabled={running}
            onChange={(value) => set("run_backtest", value)}>
            Run a rolling-origin backtest for stability over time (diagnostic; never fails the run).
          </CheckRow>
        </div>

        <ActionRow note="Training runs in the background. You can close this tab and come back — its state lives in the workspace database, not in this browser.">
          {running ? (
            <Btn variant="danger" onClick={cancel}>
              <MIcon name="stop_circle" size={15} /> Cancel run
            </Btn>
          ) : (
            <Btn
              onClick={start} busy={busy} busyLabel="Queuing…"
              disabledReason={
                !options.snapshot_ready ? "Build a dataset snapshot in Stage 03 first."
                  : !options.weight_strategy_approved ? "Approve a weight strategy in Stage 04 first."
                  : Math.abs(splitTotal - 100) > 0.01 ? "The split must total 100%."
                  : undefined
              }
            >
              <MIcon name="play_arrow" size={15} /> Start retraining
            </Btn>
          )}
        </ActionRow>
        <ErrorNotice error={error} title="The run could not start" />
      </Card>

      {task && <TaskCard task={task} log={log} />}

      <Card>
        <SectionTitle
          right={<Btn variant="ghost" small onClick={loadRuns}>
            <MIcon name="refresh" size={14} /> Refresh
          </Btn>}
        >
          Completed runs
        </SectionTitle>
        {runs.length === 0 ? (
          <EmptyState icon="science">No run has completed for this job yet.</EmptyState>
        ) : (
          <>
            <Table
              columns={[
                { key: "run_id", header: "Run" },
                { key: "created_at", header: "Completed", render: (r) => when(r.created_at) },
                { key: "split_mode", header: "Split" },
                { key: "feature_count", header: "Features", className: "num" },
                {
                  key: "candidates_trained", header: "Candidates",
                  render: (r) => r.candidates_trained.join(", "),
                },
              ]}
              rows={runs}
              rowKey={(row) => row.run_id}
            />
            <ActionRow>
              <Btn onClick={() => go("comparison")}>
                Compare against the champion <MIcon name="arrow_forward" size={15} />
              </Btn>
            </ActionRow>
          </>
        )}
      </Card>
    </>
  );
}

function TaskCard({ task, log }) {
  const tone = STATUS_TONE[task.status] || "muted";
  return (
    <Card borderSize={task.status === "complete" ? 2 : 1}>
      <SectionTitle
        sub={`${task.kind} · started ${when(task.created_at)}`}
        right={<Pill tone={tone}>{task.status}</Pill>}
      >
        Task {task.task_id}
      </SectionTitle>

      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "var(--nova-grey-dim)" }}>
          {task.message || task.status}
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 12, fontFamily: "'DM Mono',monospace" }}>
          {Math.round((task.progress || 0) * 100)}%
        </span>
      </div>
      <ProgressBar value={task.progress || 0} max={1} />

      {task.status === "interrupted" && (
        <Notice tone="warn" title="This task did not survive a backend restart">
          The process running it was stopped. Nothing was corrupted — start a new run when ready.
        </Notice>
      )}
      {task.status === "failed" && (
        <Notice tone="bad" title="The run failed">{task.error || task.message}</Notice>
      )}
      {task.status === "cancelled" && (
        <Notice tone="warn" title="Cancelled">
          The worker stopped at its next checkpoint. Artifacts already written are kept.
        </Notice>
      )}
      {task.status === "complete" && task.result && (
        <Notice tone="ok" title={`Run ${task.result.run_id} complete`}>
          Trained: {task.result.candidates_trained?.join(", ") || "—"}
          {task.result.candidates_skipped?.length
            ? ` · skipped: ${task.result.candidates_skipped.join(", ")}` : ""}
        </Notice>
      )}

      {log.length > 0 && (
        <>
          <SubHeading>Run log</SubHeading>
          <pre className="nova-log">{log.join("\n")}</pre>
        </>
      )}
    </Card>
  );
}
