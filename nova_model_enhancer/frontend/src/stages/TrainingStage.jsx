import React from "react";

import { api } from "../api.js";
import {
  Action, Empty, ErrorNotice, Icon, Metrics, Notice, Panel, Pill, Progress, Table,
} from "../components/Ui.jsx";
import { num, when } from "../format.js";

const TERMINAL = new Set(["complete", "failed", "cancelled", "interrupted"]);

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

  // Recover any task that was already running when this screen mounted.
  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    Promise.all([api.trainingOptions(job.job_id), api.tasks(job.job_id)])
      .then(([optionsBody, tasksBody]) => {
        if (!live) return;
        setOptions(optionsBody);
        const active =
          tasksBody.tasks.find((t) => !TERMINAL.has(t.status)) || tasksBody.tasks[0] || null;
        setTask(active);
      })
      .catch((err) => live && setLoadError(err));
    loadRuns();
    return () => {
      live = false;
    };
  }, [job, loadRuns]);

  // Poll while a task is live; state comes from the backend, never from memory.
  React.useEffect(() => {
    if (!task || TERMINAL.has(task.status)) return () => {};
    let live = true;
    const tick = async () => {
      try {
        const [latest, logBody] = await Promise.all([api.task(task.task_id), api.taskLog(task.task_id)]);
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

  if (!job) {
    return (
      <Panel title="No active job" icon="info">
        <Notice tone="warn" title="Start at Stage 01">Upload a champion package first.</Notice>
      </Panel>
    );
  }
  if (loadError) {
    return (
      <Panel title="Training unavailable" icon="report" tone="bad">
        <ErrorNotice error={loadError} title="Could not load the training options" />
      </Panel>
    );
  }
  if (!options) {
    return <Panel title="Loading" icon="hourglass_top"><p className="muted small">Reading the champion configuration…</p></Panel>;
  }

  const unavailable = Object.entries(options.available_model_types)
    .filter(([, ok]) => !ok)
    .map(([key]) => key);
  const splitTotal = Number(config.train_pct) + Number(config.val_pct) + Number(config.test_pct);
  const running = task && !TERMINAL.has(task.status);

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

  const set = (key, value) => setConfig((current) => ({ ...current, [key]: value }));

  return (
    <>
      <Panel
        title="Candidate plan"
        subtitle="Preprocessing is refit from scratch for every challenger. The champion's own fitted state is used only to score the champion."
        icon="model_training"
      >
        <Metrics
          cols={3}
          items={[
            { label: "Champion model", value: options.champion_model_id || "unresolved" },
            { label: "Champion family", value: (options.champion_family || "unknown").toUpperCase() },
            { label: "Champion threshold", value: options.champion_threshold },
          ]}
        />
        <div className="section-title">What will be trained</div>
        <Table
          columns={[
            { key: "label", header: "Candidate" },
            { key: "model_type", header: "Family", render: (r) => r.model_type.toUpperCase() },
            {
              key: "mode", header: "Search",
              render: (r) => (r.mode === "tuned" ? <Pill tone="info">Optuna</Pill> : <Pill tone="muted">fixed params</Pill>),
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
      </Panel>

      <Panel title="Run configuration" icon="tune" subtitle={options.split_note}>
        <div className="grid-3">
          <div className="field">
            <label htmlFor="split-mode">Split mode</label>
            <select id="split-mode" value={config.mode} onChange={(e) => set("mode", e.target.value)} disabled={running}>
              <option value="temporal">Temporal — oldest train, newest test</option>
              <option value="random">Random stratified</option>
            </select>
            <span className="hint">
              Temporal is the honest choice for a model that scores future work.
            </span>
          </div>
          <div className="field">
            <label htmlFor="seed">Random seed</label>
            <input id="seed" type="number" value={config.seed} onChange={(e) => set("seed", e.target.value)} disabled={running} />
            <span className="hint">A fixed seed makes the whole run reproducible.</span>
          </div>
          <div className="field">
            <label htmlFor="jobs">Parallelism (n_jobs)</label>
            <input id="jobs" type="number" value={config.n_jobs} onChange={(e) => set("n_jobs", e.target.value)} disabled={running} />
            <span className="hint">-1 uses every core. Lower it to keep the machine responsive.</span>
          </div>
        </div>
        <div className="grid-3">
          <div className="field">
            <label htmlFor="train-pct">Train %</label>
            <input id="train-pct" type="number" value={config.train_pct} onChange={(e) => set("train_pct", e.target.value)} disabled={running} />
          </div>
          <div className="field">
            <label htmlFor="val-pct">Validation %</label>
            <input id="val-pct" type="number" value={config.val_pct} onChange={(e) => set("val_pct", e.target.value)} disabled={running} />
            <span className="hint">Used for calibration and threshold selection only.</span>
          </div>
          <div className="field">
            <label htmlFor="test-pct">Test %</label>
            <input id="test-pct" type="number" value={config.test_pct} onChange={(e) => set("test_pct", e.target.value)} disabled={running} />
          </div>
        </div>
        {Math.abs(splitTotal - 100) > 0.01 && (
          <Notice tone="warn" title="The split does not total 100%">
            Currently {splitTotal}%. Adjust before starting.
          </Notice>
        )}
        <div className="grid-3">
          <div className="field">
            <label htmlFor="trials">Optuna trials per tuned candidate</label>
            <input id="trials" type="number" min="1" value={config.n_trials} onChange={(e) => set("n_trials", e.target.value)} disabled={running} />
          </div>
          <div className="field">
            <label htmlFor="timeout">Timeout per candidate (seconds)</label>
            <input id="timeout" type="number" min="1" placeholder="No timeout" value={config.timeout_seconds}
              onChange={(e) => set("timeout_seconds", e.target.value)} disabled={running} />
          </div>
          <div className="field">
            <label htmlFor="criterion">Threshold criterion</label>
            <select id="criterion" value={config.threshold_criterion} onChange={(e) => set("threshold_criterion", e.target.value)} disabled={running}>
              <option value="f1">F1</option>
              <option value="recall">Recall</option>
              <option value="precision">Precision</option>
              <option value="balanced_accuracy">Balanced accuracy</option>
              <option value="weighted_composite">Weighted composite</option>
            </select>
            <span className="hint">Candidates are generated on validation and reported on test.</span>
          </div>
        </div>
        <label className="checkbox">
          <input type="checkbox" checked={config.include_baseline} disabled={running}
            onChange={(e) => set("include_baseline", e.target.checked)} />
          <span>Include a logistic-regression baseline, so "better than the champion" has a floor to sit above.</span>
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={config.run_backtest} disabled={running}
            onChange={(e) => set("run_backtest", e.target.checked)} />
          <span>Run a rolling-origin backtest for stability over time (diagnostic; never fails the run).</span>
        </label>

        <div className="btn-row end">
          <span className="btn-note">
            Training runs in the background. You can close this tab and come back — its state lives in
            the workspace database, not in this browser.
          </span>
          {running ? (
            <button type="button" className="btn danger" onClick={cancel}>
              <Icon name="stop_circle" size={15} /> Cancel run
            </button>
          ) : (
            <Action
              onClick={start}
              busy={busy}
              busyLabel="Queuing…"
              disabledReason={
                !options.snapshot_ready ? "Build a dataset snapshot in Stage 03 first."
                  : !options.weight_strategy_approved ? "Approve a weight strategy in Stage 04 first."
                  : Math.abs(splitTotal - 100) > 0.01 ? "The split must total 100%."
                  : undefined
              }
            >
              <Icon name="play_arrow" size={15} /> Start retraining
            </Action>
          )}
        </div>
        <ErrorNotice error={error} title="The run could not start" />
      </Panel>

      {task && <TaskPanel task={task} log={log} />}

      <Panel title="Completed runs" icon="history"
        actions={<button type="button" className="btn ghost" onClick={loadRuns}><Icon name="refresh" size={15} /> Refresh</button>}>
        {runs.length === 0 ? (
          <Empty icon="science">No run has completed for this job yet.</Empty>
        ) : (
          <>
            <Table
              columns={[
                { key: "run_id", header: "Run" },
                { key: "created_at", header: "Completed", render: (r) => when(r.created_at) },
                { key: "split_mode", header: "Split" },
                { key: "feature_count", header: "Features", className: "num" },
                { key: "candidates_trained", header: "Candidates", render: (r) => r.candidates_trained.join(", ") },
              ]}
              rows={runs}
              rowKey={(row) => row.run_id}
            />
            <div className="btn-row end">
              <Action onClick={() => go("comparison")}>
                Compare against the champion <Icon name="arrow_forward" size={15} />
              </Action>
            </div>
          </>
        )}
      </Panel>
    </>
  );
}

function TaskPanel({ task, log }) {
  const tone = {
    complete: "ok", failed: "bad", cancelled: "warn", interrupted: "warn",
  }[task.status] || "";
  return (
    <Panel
      title={`Task ${task.task_id}`}
      subtitle={`${task.kind} · started ${when(task.created_at)}`}
      icon="terminal"
      tone={tone}
      actions={<Pill tone={tone || "info"}>{task.status}</Pill>}
    >
      <Progress value={task.progress} label={task.message || task.status} />

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
            ? ` · skipped: ${task.result.candidates_skipped.join(", ")}`
            : ""}
        </Notice>
      )}

      {log.length > 0 && (
        <>
          <div className="section-title">Run log</div>
          <pre className="log">{log.join("\n")}</pre>
        </>
      )}
    </Panel>
  );
}
