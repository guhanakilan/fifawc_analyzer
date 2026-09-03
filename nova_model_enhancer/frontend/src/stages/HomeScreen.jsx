/* The landing screen: every job this workspace holds, and what is inside it.
 *
 * Everything shown here is read from durable state on the backend — the SQLite
 * tables and the run artifacts on disk — never from browser storage. Closing
 * the tab, restarting the backend or moving to another machine's copy of the
 * workspace all leave this list intact.
 */

import React from "react";

import { api } from "../api.js";
import {
  Badge, Btn, C, Card, EmptyState, ErrorNotice, MIcon, Notice, Pill,
  SectionTitle, StatusDot,
} from "../nova/Components.jsx";
import { metric, when } from "../format.js";

const STAGE_LABEL = {
  CHAMPION_PACKAGE: "Champion package",
  TRAINING_DATA: "Training data",
  READINESS: "Readiness review",
  WEIGHT_STRATEGY: "Weight strategy",
  RETRAIN_TUNE: "Retrain & tune",
  COMPARISON: "Model comparison",
  EXPORT: "Export",
};

/** Status → the dot tone used across the workbench. */
function toneFor(job) {
  if (job.runs?.running) return "warn";
  if (job.resumable) return "warn";
  if (job.export_count > 0) return "ok";
  return "info";
}

function JobRuns({ jobId, open }) {
  const [runs, setRuns] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (!open) return undefined;
    let live = true;
    api
      .runs(jobId)
      .then((body) => live && setRuns(body.runs || []))
      .catch((err) => live && setError(err));
    return () => {
      live = false;
    };
  }, [jobId, open]);

  if (!open) return null;
  if (error) return <ErrorNotice error={error} title="Could not read this job's runs" />;
  if (runs === null) {
    return (
      <div style={{ padding: "10px 2px", color: C.textMid, fontSize: 12 }}>
        Reading run history…
      </div>
    );
  }
  if (!runs.length) {
    return (
      <div style={{ padding: "10px 2px", color: C.textMid, fontSize: 12 }}>
        No completed retraining runs in this job yet.
      </div>
    );
  }

  return (
    <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
      {runs.map((run) => (
        <div
          key={run.run_id}
          style={{
            border: "1px solid var(--nova-input-border)",
            borderRadius: 8,
            padding: "10px 12px",
            background: "var(--nova-input-bg)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <span style={{ fontFamily: "var(--nova-font-mono)", fontSize: 12, fontWeight: 600 }}>
              {run.run_id}
            </span>
            <span style={{ color: C.textMid, fontSize: 11 }}>{when(run.created_at)}</span>
          </div>
          <div style={{ marginTop: 8, display: "grid", gap: 4 }}>
            {(run.models || []).map((model) => (
              <div
                key={model.candidate_id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  fontSize: 12,
                  flexWrap: "wrap",
                }}
              >
                <span style={{ minWidth: 150 }}>{model.label}</span>
                {model.skipped ? (
                  <Pill tone="muted">not trained</Pill>
                ) : (
                  <>
                    <Badge small>F1 {metric(model.test_metrics?.f1)}</Badge>
                    <span style={{ color: C.textMid }}>
                      AUC {metric(model.test_metrics?.auc)} · threshold{" "}
                      {model.selected_threshold ?? "—"}
                    </span>
                  </>
                )}
              </div>
            ))}
            {run.champion_metrics && (
              <div style={{ fontSize: 12, color: C.textMid, marginTop: 2 }}>
                Champion {run.champion_model_id}: F1 {metric(run.champion_metrics.f1)} at
                threshold {run.champion_threshold}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function HomeScreen({ onOpenJob, onNewJob }) {
  const [jobs, setJobs] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [expanded, setExpanded] = React.useState(null);

  const load = React.useCallback(() => {
    setError(null);
    api
      .overview()
      .then((body) => setJobs(body.jobs || []))
      .catch(setError);
  }, []);

  React.useEffect(load, [load]);

  return (
    <>
      <SectionTitle
        sub="Everything below is saved on this machine and survives a restart. Open a job to pick up exactly where you left off."
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn variant="ghost" onClick={load}>
              <MIcon name="refresh" size={14} /> Refresh
            </Btn>
            <Btn onClick={onNewJob}>
              <MIcon name="add" size={15} /> New job
            </Btn>
          </div>
        }
      >
        Your retraining jobs
      </SectionTitle>

      {error && <ErrorNotice error={error} title="Could not list your jobs" />}

      {jobs === null && !error && (
        <Notice tone="info" title="Loading">
          Reading saved jobs from this workspace…
        </Notice>
      )}

      {jobs !== null && jobs.length === 0 && (
        <EmptyState icon="inbox">
          No jobs yet. Start one by uploading a completed NoVA ML export package.
        </EmptyState>
      )}

      {(jobs || []).map((job) => {
        const isOpen = expanded === job.job_id;
        return (
          <Card key={job.job_id}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 14,
                alignItems: "flex-start",
                flexWrap: "wrap",
              }}
            >
              <div style={{ minWidth: 260, flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <StatusDot tone={toneFor(job)} pulse={Boolean(job.runs?.running)} />
                  <span style={{ fontWeight: 700, fontSize: 15 }}>
                    {job.placement_id ? `Placement ${job.placement_id}` : job.job_id}
                  </span>
                  {job.resumable && <Pill tone="warn">resumable</Pill>}
                  {job.runs?.running > 0 && <Pill tone="warn">running</Pill>}
                  {job.export_count > 0 && <Pill tone="ok">{job.export_count} exported</Pill>}
                </div>
                <div style={{ color: C.textMid, fontSize: 12, marginTop: 4 }}>
                  {job.original_filename}
                </div>
                <div
                  style={{
                    color: C.textMid,
                    fontSize: 11,
                    marginTop: 6,
                    fontFamily: "var(--nova-font-mono)",
                  }}
                >
                  {job.job_id} · updated {when(job.updated_at)}
                </div>
              </div>

              <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
                <Stat label="Stage" value={STAGE_LABEL[job.current_stage] || job.current_stage} />
                <Stat label="Runs" value={job.runs?.complete ?? 0} />
                <Stat label="Datasets" value={job.dataset_count ?? 0} />
                <Btn onClick={() => onOpenJob(job)}>
                  Open <MIcon name="arrow_forward" size={15} />
                </Btn>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <Btn variant="ghost" small onClick={() => setExpanded(isOpen ? null : job.job_id)}>
                <MIcon name={isOpen ? "expand_less" : "expand_more"} size={14} />{" "}
                {isOpen ? "Hide runs" : "Show runs and models"}
              </Btn>
              <JobRuns jobId={job.job_id} open={isOpen} />
            </div>
          </Card>
        );
      })}
    </>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ textAlign: "right" }}>
      <div style={{ fontSize: 10, letterSpacing: 0.6, color: C.textMid, textTransform: "uppercase" }}>
        {label}
      </div>
      <div style={{ fontFamily: "var(--nova-font-mono)", fontWeight: 600, fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}
