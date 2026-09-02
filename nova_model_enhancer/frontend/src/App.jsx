import React from "react";

import { ApiError, api } from "./api.js";
import { Icon, Notice, Panel } from "./components/Ui.jsx";
import ChampionStage from "./stages/ChampionStage.jsx";
import ComparisonStage from "./stages/ComparisonStage.jsx";
import ExportStage from "./stages/ExportStage.jsx";
import ReadinessStage from "./stages/ReadinessStage.jsx";
import TrainingDataStage from "./stages/TrainingDataStage.jsx";
import TrainingStage from "./stages/TrainingStage.jsx";
import WeightStage from "./stages/WeightStage.jsx";

const STAGES = [
  { id: "champion", num: "01", label: "Champion Package", icon: "inventory_2",
    title: "Import the champion model",
    blurb: "Upload the ZIP exported from a completed NoVA ML run. The archive is validated before any model file is opened." },
  { id: "data", num: "02", label: "Training Data", icon: "database",
    title: "Add verified labelled data",
    blurb: "Upload historical and new verified rows. Schema, dates, labels, duplicates and drift are profiled by streaming the file." },
  { id: "readiness", num: "03", label: "Readiness", icon: "fact_check",
    title: "Confirm the rules, then freeze the dataset",
    blurb: "Approve the date column, the label source, the deduplication key and every SubTask mapping. The snapshot is then written once and hashed." },
  { id: "weights", num: "04", label: "Weight Strategy", icon: "balance",
    title: "Configure sample weights",
    blurb: "Preview the effect of recency, correction, rarity and class-balance weighting before anyone approves it." },
  { id: "training", num: "05", label: "Retrain & Tune", icon: "play_circle",
    title: "Train and tune challengers",
    blurb: "Preprocessing is refit for every challenger, models are tuned with Optuna, probabilities are calibrated and the threshold is optimised separately." },
  { id: "comparison", num: "06", label: "Model Comparison", icon: "compare_arrows",
    title: "Compare, then decide",
    blurb: "Champion and challengers are scored on identical rows. Promotion is a recommendation that a named approver must confirm." },
  { id: "export", num: "07", label: "Export", icon: "deployed_code",
    title: "Build the deployable package",
    blurb: "The ZIP is assembled, then loaded and scored exactly as the deployment does before it is published." },
];

const STORAGE_KEY = "nova-enhancer:job";
const THEME_KEY = "nova-enhancer:theme";

function useTheme() {
  const [theme, setTheme] = React.useState(() => {
    try {
      return localStorage.getItem(THEME_KEY) || "light";
    } catch {
      return "light";
    }
  });
  React.useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* a browser with storage disabled still gets a working theme */
    }
  }, [theme]);
  return [theme, () => setTheme((current) => (current === "dark" ? "light" : "dark"))];
}

/** Polls the real /health endpoint. Nothing in the header is hardcoded. */
function useHealth() {
  const [health, setHealth] = React.useState({ state: "checking" });
  React.useEffect(() => {
    let live = true;
    const check = async () => {
      try {
        const body = await api.health();
        if (live) setHealth({ state: body.status, body });
      } catch (error) {
        if (live) setHealth({ state: "unreachable", error });
      }
    };
    check();
    const timer = setInterval(check, 10000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);
  return health;
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const health = useHealth();
  const [stage, setStage] = React.useState("champion");
  const [job, setJob] = React.useState(null);
  const [progress, setProgress] = React.useState({});
  const [restoring, setRestoring] = React.useState(true);
  const [restoreError, setRestoreError] = React.useState(null);

  // ── Restore the active job after a refresh ────────────────────────────────
  React.useEffect(() => {
    let live = true;
    const stored = (() => {
      try {
        return localStorage.getItem(STORAGE_KEY);
      } catch {
        return null;
      }
    })();
    const fromUrl = new URLSearchParams(window.location.search).get("job");
    const jobId = fromUrl || stored;
    if (!jobId) {
      setRestoring(false);
      return () => {};
    }
    api
      .job(jobId)
      .then((restored) => {
        if (!live) return;
        setJob(restored);
        const fromHash = window.location.hash.replace("#", "");
        if (STAGES.some((s) => s.id === fromHash)) setStage(fromHash);
      })
      .catch((error) => {
        if (!live) return;
        if (error instanceof ApiError && error.status === 404) {
          try {
            localStorage.removeItem(STORAGE_KEY);
          } catch {
            /* nothing to clear */
          }
        } else {
          setRestoreError(error);
        }
      })
      .finally(() => live && setRestoring(false));
    return () => {
      live = false;
    };
  }, []);

  // ── Keep the URL and storage in step with the active job and stage ────────
  React.useEffect(() => {
    if (!job) return;
    try {
      localStorage.setItem(STORAGE_KEY, job.job_id);
    } catch {
      /* storage may be unavailable; the URL still carries the job */
    }
    const url = new URL(window.location.href);
    url.searchParams.set("job", job.job_id);
    url.hash = stage;
    window.history.replaceState(null, "", url);
  }, [job, stage]);

  const mark = React.useCallback((key, value) => {
    setProgress((current) => ({ ...current, [key]: value }));
  }, []);

  const startOver = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* nothing to clear */
    }
    const url = new URL(window.location.href);
    url.searchParams.delete("job");
    url.hash = "champion";
    window.history.replaceState(null, "", url);
    setJob(null);
    setProgress({});
    setStage("champion");
  };

  const packageReady = Boolean(job?.validation?.valid);
  const reachable = STAGES.reduce((furthest, stageDef, index) => {
    if (index === 0) return 0;
    if (stageDef.id === "data") return packageReady ? index : furthest;
    if (stageDef.id === "readiness") return progress.dataUploaded ? index : furthest;
    if (stageDef.id === "weights") return progress.snapshotId ? index : furthest;
    if (stageDef.id === "training") return progress.weightsApproved ? index : furthest;
    if (stageDef.id === "comparison") return progress.runId ? index : furthest;
    if (stageDef.id === "export") return progress.approvedRunId ? index : furthest;
    return furthest;
  }, 0);

  const activeIndex = STAGES.findIndex((s) => s.id === stage);
  const definition = STAGES[activeIndex] || STAGES[0];

  const stageProps = { job, setJob, progress, mark, go: setStage };

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Retraining stages">
        <div className="sidebar-logo">
          <img src="/ags-nova-logo.png" alt="" width={30} height={30} />
          <div style={{ minWidth: 0 }}>
            <div className="name">NoVA Enhancer</div>
            <div className="sub">Model Retraining</div>
          </div>
        </div>
        <div className="sidebar-section">Automated retraining</div>
        <div className="sidebar-steps">
          {STAGES.map((stageDef, index) => {
            const locked = index > reachable;
            const done = index < reachable;
            return (
              <button
                key={stageDef.id}
                type="button"
                className={`step ${done ? "done" : ""}`}
                aria-current={stage === stageDef.id ? "step" : undefined}
                disabled={locked}
                title={locked ? "Complete the previous stage first." : undefined}
                onClick={() => setStage(stageDef.id)}
              >
                <span className="step-icon">
                  <Icon name={done ? "check" : stageDef.icon} size={15} />
                </span>
                <span style={{ minWidth: 0 }}>
                  <span className="num">STAGE {stageDef.num}</span>
                  <span className="label">{stageDef.label}</span>
                </span>
              </button>
            );
          })}
        </div>
        <div className="sidebar-foot">
          v{health.body?.version || "—"} · local workspace
        </div>
      </nav>

      <div className="workspace">
        <header className="header">
          <div className="title">NoVA Model Enhancer</div>
          <div className="context">
            <HealthChip health={health} />
            {job && (
              <>
                <span className="chip">
                  <Icon name="badge" size={13} />
                  {job.job_id}
                </span>
                {job.placement_id != null && (
                  <span className="chip">
                    <Icon name="location_on" size={13} />
                    PLC {job.placement_id}
                  </span>
                )}
                <button type="button" className="btn ghost" onClick={startOver}>
                  <Icon name="restart_alt" size={15} /> New job
                </button>
              </>
            )}
            <button
              type="button"
              className="icon-button"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              <Icon name={theme === "dark" ? "light_mode" : "dark_mode"} size={16} />
            </button>
          </div>
        </header>

        <main className="page">
          <div className="hero">
            <div className="no">{definition.num}</div>
            <div>
              <div className="kicker">Model enhancement pipeline</div>
              <h1>{definition.title}</h1>
              <p>{definition.blurb}</p>
            </div>
          </div>

          {health.state === "unreachable" && (
            <Notice tone="bad" title="The backend is not responding">
              {health.error?.message} Nothing on this screen reflects live state until it returns.
            </Notice>
          )}
          {health.state === "degraded" && health.body?.detail && (
            <Notice tone="warn" title="The backend is running with a limitation">
              {health.body.detail}
            </Notice>
          )}
          {restoreError && (
            <Notice tone="warn" title="Could not restore the previous job">
              {restoreError.message}
            </Notice>
          )}

          {restoring ? (
            <Panel title="Restoring your session" icon="hourglass_top">
              <p className="muted small">Looking up the job this browser was last working on…</p>
            </Panel>
          ) : (
            <>
              {stage === "champion" && <ChampionStage {...stageProps} />}
              {stage === "data" && <TrainingDataStage {...stageProps} />}
              {stage === "readiness" && <ReadinessStage {...stageProps} />}
              {stage === "weights" && <WeightStage {...stageProps} />}
              {stage === "training" && <TrainingStage {...stageProps} />}
              {stage === "comparison" && <ComparisonStage {...stageProps} />}
              {stage === "export" && <ExportStage {...stageProps} />}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function HealthChip({ health }) {
  if (health.state === "checking") {
    return (
      <span className="chip">
        <span className="dot" /> Checking API…
      </span>
    );
  }
  if (health.state === "unreachable") {
    return (
      <span className="chip bad">
        <span className="dot" /> API unreachable
      </span>
    );
  }
  if (health.state === "degraded") {
    return (
      <span className="chip warn">
        <span className="dot" /> API degraded
      </span>
    );
  }
  return (
    <span className="chip ok">
      <span className="dot" /> API healthy
    </span>
  );
}
