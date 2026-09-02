import { AnimatePresence, motion } from "framer-motion";
import React from "react";

import { ApiError, api } from "./api.js";
import stageHelp from "./help/stageHelp.js";
import {
  Btn, C, MIcon, Notice, NovaGlobeLogo, StageHelpBanner, StatusDot,
} from "./nova/Components.jsx";
import { SPRING_FAST } from "./nova/palette.js";
import { ThemeProvider, useTheme } from "./nova/theme.jsx";
import ChampionStage from "./stages/ChampionStage.jsx";
import ComparisonStage from "./stages/ComparisonStage.jsx";
import ExportStage from "./stages/ExportStage.jsx";
import ReadinessStage from "./stages/ReadinessStage.jsx";
import TrainingDataStage from "./stages/TrainingDataStage.jsx";
import TrainingStage from "./stages/TrainingStage.jsx";
import WeightStage from "./stages/WeightStage.jsx";

export const STAGES = [
  { id: "champion", num: "01", label: "Champion Package", icon: "inventory_2" },
  { id: "data", num: "02", label: "Training Data", icon: "database" },
  { id: "readiness", num: "03", label: "Readiness", icon: "fact_check" },
  { id: "weights", num: "04", label: "Weight Strategy", icon: "balance" },
  { id: "training", num: "05", label: "Retrain & Tune", icon: "play_circle" },
  { id: "comparison", num: "06", label: "Model Comparison", icon: "compare_arrows" },
  { id: "export", num: "07", label: "Export", icon: "deployed_code" },
];

const STORAGE_KEY = "nova-enhancer:job";

export default function App() {
  return (
    <ThemeProvider>
      <Workbench />
    </ThemeProvider>
  );
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

function Workbench() {
  const { theme } = useTheme();
  const isDark = theme === "dark";
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
    const jobId = new URLSearchParams(window.location.search).get("job") || stored;
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
  const furthest = STAGES.reduce((reach, stageDef, index) => {
    if (index === 0) return 0;
    if (stageDef.id === "data") return packageReady ? index : reach;
    if (stageDef.id === "readiness") return progress.dataUploaded ? index : reach;
    if (stageDef.id === "weights") return progress.snapshotId ? index : reach;
    if (stageDef.id === "training") return progress.weightsApproved ? index : reach;
    if (stageDef.id === "comparison") return progress.runId ? index : reach;
    if (stageDef.id === "export") return progress.approvedRunId ? index : reach;
    return reach;
  }, 0);

  const definition = STAGES.find((s) => s.id === stage) || STAGES[0];
  const stageProps = { job, setJob, progress, mark, go: setStage };

  return (
    <div style={{ minHeight: "100vh", background: "var(--nova-view-bg)" }}>
      <Sidebar
        active={stage}
        furthest={furthest}
        onNav={setStage}
        version={health.body?.version}
      />
      <Header
        stage={definition}
        job={job}
        health={health}
        onStartOver={job ? startOver : null}
      />

      <main style={{ marginLeft: 240, paddingTop: 44 }}>
        <div style={{ maxWidth: 1180, margin: "0 auto", padding: "22px 26px 72px" }}>
          <StageBanner definition={definition} />
          <StageHelpBanner help={stageHelp[definition.id]} />

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

          <AnimatePresence mode="wait">
            <motion.div
              key={stage}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18 }}
            >
              {restoring ? (
                <Notice tone="info" title="Restoring your session">
                  Looking up the job this browser was last working on…
                </Notice>
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
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar({ active, furthest, onNav, version }) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const accent = isDark ? C.green : C.indigo;

  return (
    <nav
      aria-label="Retraining stages"
      style={{
        width: 240, minHeight: "100vh", background: "var(--nova-sidebar-bg)",
        display: "flex", flexDirection: "column",
        borderRight: "1px solid var(--nova-sidebar-border)",
        position: "fixed", left: 0, top: 0, bottom: 0, zIndex: 100,
        fontFamily: "'DM Mono',monospace", transition: "var(--nova-transition)",
      }}
    >
      <div style={{
        height: 44, display: "flex", alignItems: "center", padding: "0 20px",
        borderBottom: "1px solid var(--nova-sidebar-border)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
          <NovaGlobeLogo size={28} />
          <div style={{
            color: "var(--nova-sidebar-logo-text)", fontSize: 12.5, fontWeight: 800,
            letterSpacing: -0.3, whiteSpace: "nowrap",
          }}>
            NoVa Model Enhancer
          </div>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "10px 0" }}>
        {STAGES.map((stageDef, index) => {
          const isActive = active === stageDef.id;
          const isCompleted = index < furthest;
          const locked = index > furthest;
          return (
            <motion.button
              key={stageDef.id}
              type="button"
              onClick={() => !locked && onNav(stageDef.id)}
              disabled={locked}
              aria-current={isActive ? "step" : undefined}
              title={locked ? "Complete the previous stage first." : undefined}
              whileHover={!locked && !isActive
                ? { background: `linear-gradient(90deg, ${accent}0A 0%, transparent 80%)`, x: 1 }
                : {}}
              transition={SPRING_FAST}
              style={{
                display: "flex", alignItems: "center", gap: 10, padding: "9px 16px",
                width: "100%", textAlign: "left", border: 0,
                borderLeft: `3px solid ${isActive ? accent : "transparent"}`,
                cursor: locked ? "not-allowed" : "pointer",
                background: isActive
                  ? (isDark
                    ? `linear-gradient(90deg, ${accent}28 0%, ${accent}08 55%, transparent 100%)`
                    : `linear-gradient(90deg, ${accent}48 0%, ${accent}18 55%, transparent 100%)`)
                  : "transparent",
                boxShadow: isActive
                  ? (isDark ? `inset 4px 0 18px ${accent}18` : `inset 4px 0 24px ${accent}30`)
                  : "none",
                opacity: locked ? 0.45 : 1,
                transition: "background 0.2s, box-shadow 0.2s, opacity 0.2s",
              }}
            >
              <span style={{
                width: 26, height: 26, borderRadius: 6, display: "flex",
                alignItems: "center", justifyContent: "center", flexShrink: 0,
                background: isCompleted ? accent : isActive ? `${accent}22` : "var(--nova-sidebar-icon-inactive-bg)",
                color: isCompleted ? "#fff" : isActive ? accent : "var(--nova-sidebar-icon-inactive-cl)",
                boxShadow: isActive ? `0 0 10px ${accent}40` : "none",
              }}>
                <MIcon name={isCompleted ? "check" : stageDef.icon} size={15} />
              </span>
              <span style={{ minWidth: 0 }}>
                <span style={{
                  display: "block", fontSize: 8, letterSpacing: 0.9,
                  color: "var(--nova-sidebar-text-dim)",
                }}>
                  STAGE {stageDef.num}
                </span>
                <span style={{
                  display: "block", fontSize: 11, fontWeight: isActive ? 800 : 600, letterSpacing: 0.1,
                  color: isActive ? accent
                    : isCompleted ? "var(--nova-sidebar-text-completed)"
                    : "var(--nova-sidebar-text-default)",
                  textShadow: isActive && isDark ? `0 0 12px ${accent}90` : "none",
                }}>
                  {stageDef.label}
                </span>
              </span>
            </motion.button>
          );
        })}
      </div>

      <div style={{ padding: "12px 16px", borderTop: "1px solid var(--nova-sidebar-border)" }}>
        <div style={{ fontSize: 10, color: "var(--nova-sidebar-text-dim)", letterSpacing: 0.5 }}>
          v{version || "—"} · local workspace
        </div>
      </div>
    </nav>
  );
}

// ── Header ───────────────────────────────────────────────────────────────────

function Header({ stage, job, health, onStartOver }) {
  const { theme, toggle } = useTheme();
  const isDark = theme === "dark";

  const healthTone =
    health.state === "ok" ? "ok"
      : health.state === "degraded" ? "warn"
      : health.state === "unreachable" ? "bad" : "muted";
  const healthLabel =
    health.state === "ok" ? "API healthy"
      : health.state === "degraded" ? "API degraded"
      : health.state === "unreachable" ? "API unreachable" : "Checking API…";

  const chip = {
    display: "inline-flex", alignItems: "center", gap: 5,
    border: "1px solid var(--nova-header-border)", borderRadius: 6,
    padding: "3px 9px", fontSize: 11, fontFamily: "'DM Mono',monospace",
    color: "var(--nova-header-text)", whiteSpace: "nowrap",
  };

  return (
    <div style={{
      height: 44, background: "var(--nova-header-bg)",
      borderBottom: "1px solid var(--nova-header-border)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 24px", position: "fixed", top: 0, left: 240, right: 0, zIndex: 99,
      gap: 14, transition: "var(--nova-transition)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span style={{ color: "var(--nova-header-text-dim)", fontSize: 12 }}>Enhancer</span>
        <span style={{ color: "var(--nova-header-text-dim)" }}>›</span>
        <span style={{ color: "var(--nova-header-text-dim)", fontSize: 12 }}>Workbench</span>
        <span style={{ color: "var(--nova-header-text-dim)" }}>›</span>
        <span style={{
          color: "var(--nova-header-text)", fontSize: 13, fontWeight: 700,
          fontFamily: "'DM Mono',monospace", whiteSpace: "nowrap",
          overflow: "hidden", textOverflow: "ellipsis",
        }}>
          {stage.label}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={chip} title={health.body?.detail || healthLabel}>
          <StatusDot tone={healthTone} pulse={health.state === "ok"} />
          {healthLabel}
        </span>
        {job && (
          <>
            <span style={chip} title="Retraining job">
              <span style={{ opacity: 0.55, fontSize: 10 }}>job</span>
              {job.job_id}
            </span>
            {job.placement_id != null && (
              <span style={chip} title="Placement">
                <span style={{ opacity: 0.55, fontSize: 10 }}>plc</span>
                {job.placement_id}
              </span>
            )}
          </>
        )}
        {onStartOver && (
          <Btn variant="ghost" small onClick={onStartOver}>
            <MIcon name="restart_alt" size={14} /> New job
          </Btn>
        )}
        <motion.button
          type="button"
          onClick={toggle}
          aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
          whileTap={{ scale: 0.94 }}
          transition={SPRING_FAST}
          style={{
            width: 28, height: 28, borderRadius: 6, display: "grid", placeItems: "center",
            background: "var(--nova-toggle-bg)", border: "1px solid var(--nova-toggle-border)",
            color: "var(--nova-toggle-color)", cursor: "pointer",
          }}
        >
          <MIcon name={isDark ? "light_mode" : "dark_mode"} size={15} />
        </motion.button>
      </div>
    </div>
  );
}

// ── Stage banner ─────────────────────────────────────────────────────────────

const BANNER = {
  champion: ["Import the champion model", "Upload the ZIP exported from a completed NoVA ML run. The archive is validated before any model file is opened."],
  data: ["Add verified labelled data", "Upload historical and new verified rows. Schema, dates, labels, duplicates and drift are profiled by streaming the file."],
  readiness: ["Confirm the rules, then freeze the dataset", "Approve the date column, the label source, the deduplication key and every SubTask mapping. The snapshot is then written once and hashed."],
  weights: ["Configure sample weights", "Preview the effect of recency, correction, rarity and class-balance weighting before anyone approves it."],
  training: ["Train and tune challengers", "Preprocessing is refit for every challenger, models are tuned with Optuna, probabilities are calibrated and the threshold is optimised separately."],
  comparison: ["Compare, then decide", "Champion and challengers are scored on identical rows. Promotion is a recommendation that a named approver must confirm."],
  export: ["Build the deployable package", "The ZIP is assembled, then loaded and scored exactly as the deployment does before it is published."],
};

function StageBanner({ definition }) {
  const [title, blurb] = BANNER[definition.id] || ["", ""];
  return (
    <div style={{
      display: "flex", gap: 20, alignItems: "center", marginBottom: 16,
      background: "var(--nova-hero-bg)", border: "1px solid var(--nova-hero-border)",
      borderRadius: 12, padding: "18px 22px",
    }}>
      <div style={{
        fontSize: 32, fontWeight: 900, color: C.indigo, fontFamily: "'DM Mono',monospace",
        letterSpacing: -1.5, paddingRight: 18, borderRight: "1px solid var(--nova-hero-border)",
      }}>
        {definition.num}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{
          fontSize: 9, letterSpacing: 1.4, fontWeight: 800, color: "var(--nova-hero-sub)",
          fontFamily: "'DM Mono',monospace", textTransform: "uppercase",
        }}>
          Model enhancement pipeline
        </div>
        <h1 style={{
          fontSize: 20, margin: "5px 0 5px", color: "var(--nova-hero-title)", letterSpacing: -0.3,
        }}>
          {title}
        </h1>
        <p style={{
          margin: 0, fontSize: 12.5, color: "var(--nova-hero-sub)", maxWidth: "82ch", lineHeight: 1.55,
        }}>
          {blurb}
        </p>
      </div>
    </div>
  );
}
