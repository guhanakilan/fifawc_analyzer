import React from "react";

import { api } from "../api.js";
import {
  ActionRow, ApprovalIdentity, Badge, Btn, C, Card, CheckRow, EmptyState, ErrorNotice, Field,
  FormGrid, MIcon, MetricGrid, Notice, Pill, SectionTitle, SubHeading, Table,
} from "../nova/Components.jsx";
import { useTheme } from "../nova/theme.jsx";
import { delta, metric, num, when } from "../format.js";
import { NoJob } from "./TrainingDataStage.jsx";

const METRIC_KEYS = [
  ["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"],
  ["specificity", "Specificity"], ["accuracy", "Accuracy"],
  ["auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["brier_score", "Brier"],
];

const GATE_TONE = {
  RECOMMENDED: "ok", NOT_RECOMMENDED: "warn", BLOCKED: "bad", APPROVED: "ok",
};

export default function ComparisonStage({ job, mark, go, operator }) {
  const [runs, setRuns] = React.useState([]);
  const [runId, setRunId] = React.useState(null);
  const [gate, setGate] = React.useState(null);
  const [gateForm, setGateForm] = React.useState(null);
  const [comparison, setComparison] = React.useState(null);
  const [selected, setSelected] = React.useState(null);
  const [typed, setTyped] = React.useState("");
  const [notes, setNotes] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [loadError, setLoadError] = React.useState(null);

  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    Promise.all([api.runs(job.job_id), api.gate(job.job_id)])
      .then(([runsBody, gateBody]) => {
        if (!live) return;
        setRuns(runsBody.runs);
        setRunId(runsBody.runs[0]?.run_id || null);
        setGate(gateBody);
        setGateForm({ ...gateBody.gate });
      })
      .catch((err) => live && setLoadError(err));
    return () => {
      live = false;
    };
  }, [job]);

  const loadComparison = React.useCallback(async () => {
    if (!job || !runId) return;
    setError(null);
    try {
      const body = await api.comparison(job.job_id, runId);
      setComparison(body);
      setSelected((current) => current || body.leading_candidate);
      if (body.approval?.decision === "APPROVED") mark("approvedRunId", runId);
    } catch (comparisonError) {
      setError(comparisonError);
    }
  }, [job, runId, mark]);

  React.useEffect(() => {
    loadComparison();
  }, [loadComparison]);

  if (!job) return <NoJob />;
  if (loadError) {
    return (
      <Card>
        <SectionTitle>Comparison unavailable</SectionTitle>
        <ErrorNotice error={loadError} />
      </Card>
    );
  }
  if (!runs.length) {
    return (
      <Card>
        <SectionTitle>Nothing to compare yet</SectionTitle>
        <EmptyState icon="science">Complete a retraining run in Stage 05 first.</EmptyState>
      </Card>
    );
  }

  const saveGate = async () => {
    setBusy(true);
    setError(null);
    try {
      const { approved, approver: _ignored, ...payload } = gateForm;
      await api.saveGate(job.job_id, { gate: payload, approver: (operator || "").trim() });
      const refreshed = await api.gate(job.job_id);
      setGate(refreshed);
      setGateForm({ ...refreshed.gate });
      await loadComparison();
    } catch (gateError) {
      setError(gateError);
    } finally {
      setBusy(false);
    }
  };

  const approve = async (decision) => {
    setBusy(true);
    setError(null);
    try {
      const record = await api.approvePromotion(job.job_id, {
        run_id: runId, candidate_id: selected, decision,
        approver: (operator || "").trim(), typed_confirmation: typed.trim(), notes,
      });
      setComparison((current) => ({ ...current, approval: record }));
      if (decision === "APPROVED") mark("approvedRunId", runId);
    } catch (approveError) {
      setError(approveError);
    } finally {
      setBusy(false);
    }
  };

  const candidates = comparison
    ? Object.entries(comparison.candidates).filter(([, value]) => !value.skipped) : [];
  const championMetrics = comparison?.champion?.test_metrics || {};
  const gateResult = comparison?.gate_results?.[selected];
  const approval = comparison?.approval;

  return (
    <>
      <Card borderSize={gate?.gate?.approved ? 2 : 1}>
        <SectionTitle
          sub="What counts as better must be decided before looking at the numbers, not after."
          right={<Badge color={gate?.gate?.approved ? C.green : "#F59E0B"}
            bg={gate?.gate?.approved ? "#E3F5EC" : "#FDF2DD"}>
            {gate?.gate?.approved ? "APPROVED" : "NOT APPROVED"}
          </Badge>}
        >
          Promotion gate
        </SectionTitle>

        {!gate?.gate?.approved && (
          <Notice tone="warn" title="Every comparison is BLOCKED until this gate is approved">
            {gate?.proposed_note}
          </Notice>
        )}

        {gateForm && (
          <>
            <FormGrid min={190}>
              <Field label="Primary metric" htmlFor="primary-metric">
                <select id="primary-metric" value={gateForm.primary_metric}
                  onChange={(e) => setGateForm((c) => ({ ...c, primary_metric: e.target.value }))}>
                  {gate.primary_metric_choices.map((choice) => (
                    <option key={choice} value={choice}>{choice}</option>
                  ))}
                </select>
              </Field>
              <Field label="Minimum improvement (%)" htmlFor="min-improve">
                <input id="min-improve" type="number" step="0.1" value={gateForm.min_primary_improvement_pct}
                  onChange={(e) => setGateForm((c) => ({
                    ...c, min_primary_improvement_pct: Number(e.target.value),
                  }))} />
              </Field>
              <Field label="Max historical regression (%)" htmlFor="hist-regress">
                <input id="hist-regress" type="number" step="0.1"
                  value={gateForm.max_historical_primary_regression_pct ?? ""}
                  onChange={(e) => setGateForm((c) => ({
                    ...c,
                    max_historical_primary_regression_pct:
                      e.target.value === "" ? null : Number(e.target.value),
                  }))} />
              </Field>
            </FormGrid>

            <SubHeading>Protected metrics</SubHeading>
            {(gateForm.protected_metrics || []).map((protectedMetric, index) => (
              <FormGrid min={180} key={index} style={{ marginBottom: 10, alignItems: "end" }}>
                <Field label="Metric" htmlFor={`protected-${index}`}>
                  <select id={`protected-${index}`} value={protectedMetric.metric}
                    onChange={(e) => {
                      const next = [...gateForm.protected_metrics];
                      next[index] = { ...next[index], metric: e.target.value };
                      setGateForm((c) => ({ ...c, protected_metrics: next }));
                    }}>
                    {METRIC_KEYS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                  </select>
                </Field>
                <Field label="Max regression (%)" htmlFor={`tolerance-${index}`}>
                  <input id={`tolerance-${index}`} type="number" step="0.1"
                    value={protectedMetric.max_regression_pct}
                    onChange={(e) => {
                      const next = [...gateForm.protected_metrics];
                      next[index] = { ...next[index], max_regression_pct: Number(e.target.value) };
                      setGateForm((c) => ({ ...c, protected_metrics: next }));
                    }} />
                </Field>
                <div>
                  <Btn variant="ghost" small onClick={() => setGateForm((c) => ({
                    ...c, protected_metrics: c.protected_metrics.filter((_, i) => i !== index),
                  }))}>
                    <MIcon name="delete" size={14} /> Remove
                  </Btn>
                </div>
              </FormGrid>
            ))}
            <Btn variant="ghost" small onClick={() => setGateForm((c) => ({
              ...c,
              protected_metrics: [...(c.protected_metrics || []), { metric: "precision", max_regression_pct: 1.0 }],
            }))}>
              <MIcon name="add" size={14} /> Add a protected metric
            </Btn>

            <FormGrid min={190} style={{ marginTop: 16 }}>
              <Field label="Segment column for breakdowns" htmlFor="segment-col">
                <input id="segment-col" type="text" value={gateForm.segment_column || ""}
                  onChange={(e) => setGateForm((c) => ({ ...c, segment_column: e.target.value || null }))} />
              </Field>
              <Field
                label="Minimum rows per segment" htmlFor="min-segment"
                hint="Smaller segments are listed as skipped rather than reported noisily."
              >
                <input id="min-segment" type="number" min="10" value={gateForm.min_segment_rows}
                  onChange={(e) => setGateForm((c) => ({ ...c, min_segment_rows: Number(e.target.value) }))} />
              </Field>
            </FormGrid>
            <ApprovalIdentity operator={operator} what="This promotion gate" />

            <div style={{ marginTop: 12 }}>
              <CheckRow checked={gateForm.require_backtest_pass}
                onChange={(value) => setGateForm((c) => ({ ...c, require_backtest_pass: value }))}>
                Require the rolling backtest to have completed.
              </CheckRow>
              <CheckRow checked={gateForm.require_package_validation}
                onChange={(value) => setGateForm((c) => ({ ...c, require_package_validation: value }))}>
                Require export package validation to pass before recommending promotion.
              </CheckRow>
            </div>

            <ActionRow>
              <Btn onClick={saveGate} busy={busy} busyLabel="Saving…"
                disabledReason={!operator?.trim() ? "Enter your name in the header first." : undefined}>
                <MIcon name="how_to_reg" size={15} /> Approve this gate
              </Btn>
            </ActionRow>
          </>
        )}
      </Card>

      <Card>
        <SectionTitle
          sub={comparison?.benchmark?.note}
          right={
            <select value={runId || ""} onChange={(e) => { setRunId(e.target.value); setSelected(null); }}
              aria-label="Run to compare" style={{ width: "auto", minWidth: 230 }}>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} · {when(run.created_at)}
                </option>
              ))}
            </select>
          }
        >
          Benchmark
        </SectionTitle>

        <ErrorNotice error={error} />
        {!comparison ? (
          <p style={{ fontSize: 12, color: "var(--nova-grey-dim)" }}>Loading the comparison…</p>
        ) : (
          <>
            <MetricGrid
              compact
              min={150}
              items={[
                { label: "Benchmark rows", value: num(comparison.benchmark.rows) },
                { label: "Actual Non-Voice", value: num(comparison.benchmark.actual_non_voice) },
                { label: "Actual Voice", value: num(comparison.benchmark.actual_voice) },
                {
                  label: "Window", value: comparison.benchmark.date_from?.slice(0, 10) || "—",
                  sub: comparison.benchmark.date_to?.slice(0, 10),
                },
              ]}
            />

            {comparison.split?.no_future_leakage === true && (
              <Notice tone="ok" title="No future leakage">
                Every training row predates every test row, so these numbers reflect predicting
                genuinely unseen future work.
              </Notice>
            )}
            {comparison.data_quality_blockers?.length > 0 && (
              <Notice tone="bad" title="Data-quality blockers">
                <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
                  {comparison.data_quality_blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
                </ul>
              </Notice>
            )}

            <SubHeading>Champion versus challengers on identical rows</SubHeading>
            <MetricTable
              championMetrics={championMetrics}
              championLabel={comparison.champion.model_id}
              championThreshold={comparison.champion.threshold}
              candidates={candidates}
              selected={selected}
              onSelect={setSelected}
              primary={gate?.gate?.primary_metric || "f1"}
            />

            <SubHeading>Confusion matrix at the selected threshold</SubHeading>
            <Table
              columns={[
                { key: "model", header: "Model" },
                { key: "tp", header: "TP", className: "num" },
                { key: "fp", header: "FP", className: "num" },
                { key: "fn", header: "FN", className: "num" },
                { key: "tn", header: "TN", className: "num" },
                { key: "pred_nv", header: "Predicted NV", className: "num" },
                { key: "latency", header: "µs / row", className: "num" },
              ]}
              rows={[
                cmRow(`CHAMPION · ${comparison.champion.model_id}`, championMetrics, comparison.champion.latency),
                ...candidates.map(([id, value]) => cmRow(id, value.test_metrics, value.latency)),
              ]}
              rowKey={(row) => row.model}
            />

            <ThresholdExplorer
              jobId={job.job_id}
              runId={comparison.run_id || runId}
              comparison={comparison}
              selected={selected}
            />
            <EscalationSection comparison={comparison} selected={selected} />
            <SignificanceSection comparison={comparison} selected={selected} />
            <OperatingSection comparison={comparison} selected={selected} />
            <DisagreementSection comparison={comparison} selected={selected} />
            <PeriodSection comparison={comparison} selected={selected} />
            <SegmentSection comparison={comparison} selected={selected} />
            <BacktestSection backtest={comparison.backtest} />
          </>
        )}
      </Card>

      {comparison?.guidance?.length > 0 && <GuidanceSection guidance={comparison.guidance} />}

      {comparison && selected && (
        <Card borderSize={2}>
          <SectionTitle
            sub="Promotion is a recommendation. Nothing is promoted without a typed approval."
            right={<Pill tone={GATE_TONE[gateResult?.status] || "muted"}>{gateResult?.status || "—"}</Pill>}
          >
            Decision for {selected}
          </SectionTitle>

          {gateResult?.blockers?.length > 0 && (
            <Notice tone="bad" title="Blocked">
              <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
                {gateResult.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
              </ul>
            </Notice>
          )}

          {gateResult?.rules?.length > 0 && (
            <Table
              columns={[
                { key: "rule", header: "Rule" },
                { key: "champion", header: "Champion", className: "num", render: (r) => metric(r.champion) },
                { key: "challenger", header: "Challenger", className: "num", render: (r) => metric(r.challenger) },
                {
                  key: "delta_pct", header: "Δ%", className: "num",
                  render: (r) => (r.delta_pct == null
                    ? "—" : `${r.delta_pct > 0 ? "+" : ""}${r.delta_pct.toFixed(2)}%`),
                  tone: (r) => (r.delta_pct == null ? null : r.delta_pct >= 0 ? "ok" : "bad"),
                },
                {
                  key: "passed", header: "Result",
                  // null is a third state: the rule could not be evaluated. Showing
                  // it as "fail" would be wrong, and as "pass" would be worse.
                  render: (r) => (
                    r.passed === null || r.passed === undefined
                      ? <Pill tone="muted">not assessed</Pill>
                      : <Pill tone={r.passed ? "ok" : "bad"}>{r.passed ? "pass" : "fail"}</Pill>
                  ),
                },
              ]}
              rows={gateResult.rules}
              rowKey={(row) => row.rule}
            />
          )}

          {approval ? (
            <Notice
              tone={approval.decision === "APPROVED" ? "ok" : "warn"}
              title={`${approval.decision} by ${approval.approver} on ${when(approval.approved_at)}`}
            >
              <p style={{ margin: 0 }}>{approval.notes || "No notes recorded."}</p>
              {approval.override_of_recommendation && (
                <p style={{ margin: "6px 0 0" }}>
                  <strong>This was an override:</strong> the gate did not recommend promotion.
                </p>
              )}
              {approval.decision === "APPROVED" && (
                <ActionRow>
                  <Btn onClick={() => go("export")}>
                    Build the export package <MIcon name="arrow_forward" size={15} />
                  </Btn>
                </ActionRow>
              )}
            </Notice>
          ) : (
            <>
              <ApprovalIdentity operator={operator} what="This promotion decision" />
              <FormGrid min={200}>
                <Field
                  label="Type the candidate id to confirm" htmlFor="promo-typed"
                  hint={<>Exactly <code>{selected}</code>.</>}
                >
                  <input id="promo-typed" type="text" placeholder={selected} value={typed}
                    onChange={(e) => setTyped(e.target.value)}
                    style={{ fontFamily: "var(--nova-font-mono)" }} />
                </Field>
                <Field label="Notes" htmlFor="promo-notes">
                  <input id="promo-notes" type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
                </Field>
              </FormGrid>
              <ActionRow>
                <Btn variant="ghost" onClick={() => approve("REJECTED")}
                  disabledReason={
                    !operator?.trim() ? "Enter your name in the header first."
                      : typed.trim() !== selected ? "Type the candidate id exactly." : undefined
                  }>
                  <MIcon name="block" size={15} /> Record rejection
                </Btn>
                <Btn onClick={() => approve("APPROVED")} busy={busy} busyLabel="Recording…"
                  disabledReason={
                    !operator?.trim() ? "Enter your name in the header first."
                      : typed.trim() !== selected ? "Type the candidate id exactly."
                      : gateResult?.status === "BLOCKED" ? "This candidate is blocked; resolve the blockers first."
                      : undefined
                  }>
                  <MIcon name="how_to_reg" size={15} /> Approve promotion
                </Btn>
              </ActionRow>
            </>
          )}
        </Card>
      )}
    </>
  );
}

function cmRow(model, metrics, latency) {
  const cm = metrics?.confusion_matrix || {};
  return {
    model, tp: num(cm.tp), fp: num(cm.fp), fn: num(cm.fn), tn: num(cm.tn),
    pred_nv: num(metrics?.predicted_non_voice),
    latency: latency ? latency.microseconds_per_row : "—",
  };
}

function MetricTable({ championMetrics, championLabel, championThreshold, candidates, selected, onSelect, primary }) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const border = isDark ? "#2A2C4A" : "#E4E8F0";
  const th = {
    padding: "8px 11px", background: isDark ? "#1A1C2E" : "#F5F7FC",
    borderBottom: `1px solid ${border}`, whiteSpace: "nowrap", fontSize: 9.5,
    letterSpacing: 0.8, textTransform: "uppercase", fontFamily: "var(--nova-font-mono)",
    fontWeight: 700, color: isDark ? "#8892A0" : C.indigo, position: "sticky", top: 0,
  };
  const td = {
    padding: "7px 11px", borderBottom: `1px solid ${border}`, whiteSpace: "nowrap",
    textAlign: "right", fontFamily: "var(--nova-font-mono)",
  };

  return (
    <div style={{ overflowX: "auto", border: `1px solid ${border}`, borderRadius: 8 }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
        <thead>
          <tr>
            <th scope="col" style={{ ...th, textAlign: "left" }}>Model</th>
            <th scope="col" style={th}>Threshold</th>
            {METRIC_KEYS.map(([key, label]) => (
              <th key={key} scope="col" style={th}>{label}{key === primary ? " ★" : ""}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ ...td, textAlign: "left", fontFamily: "inherit" }}>
              <strong>CHAMPION</strong> · <code>{championLabel}</code>
            </td>
            <td style={td}>{championThreshold}</td>
            {METRIC_KEYS.map(([key]) => (
              <td style={td} key={key}>{metric(championMetrics[key])}</td>
            ))}
          </tr>
          {candidates.map(([id, value]) => (
            <tr key={id} style={selected === id ? { background: isDark ? "#1B2436" : "#EEF3FF" } : undefined}>
              <td style={{ ...td, textAlign: "left", fontFamily: "inherit" }}>
                <label style={{ display: "flex", gap: 6, alignItems: "center", cursor: "pointer" }}>
                  <input type="radio" name="candidate" checked={selected === id}
                    onChange={() => onSelect(id)} style={{ accentColor: C.green, width: "auto" }} />
                  <code>{id}</code>
                </label>
              </td>
              <td style={td}>{value.selected_threshold}</td>
              {METRIC_KEYS.map(([key]) => {
                const change = delta(value.test_metrics?.[key], championMetrics[key]);
                const better = key === "brier_score" ? change < 0 : change > 0;
                const meaningful = change != null && Math.abs(change) >= 0.05;
                return (
                  <td
                    key={key}
                    style={{
                      ...td,
                      color: meaningful ? (better ? C.green : "#EF4444") : undefined,
                      fontWeight: meaningful ? 700 : 400,
                    }}
                  >
                    {metric(value.test_metrics?.[key])}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PeriodSection({ comparison, selected }) {
  const champion = comparison.period_breakdown?.champion || [];
  const challenger = comparison.period_breakdown?.[selected] || [];
  if (!champion.length) return null;
  // Matched by period name, never by array position: a period the challenger
  // skipped would otherwise shift every later row and pair the wrong numbers.
  const byPeriod = Object.fromEntries(challenger.map((row) => [row.period, row]));
  const rows = champion.map((row) => {
    const pair = byPeriod[row.period];
    return {
      period: row.period, rows: row.rows, champion_f1: row.f1,
      challenger_f1: pair?.f1, skipped: row.skipped || pair?.skipped,
    };
  });
  return (
    <>
      <SubHeading>Performance by period</SubHeading>
      <Table
        columns={[
          { key: "period", header: "Month" },
          { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
          {
            key: "champion_f1", header: "Champion F1", className: "num",
            render: (r) => (r.skipped ? "—" : metric(r.champion_f1)),
          },
          {
            key: "challenger_f1", header: "Challenger F1", className: "num",
            render: (r) => (r.skipped ? "—" : metric(r.challenger_f1)),
          },
          { key: "skipped", header: "Note", render: (r) => r.skipped || "" },
        ]}
        rows={rows}
        rowKey={(row) => row.period}
      />
    </>
  );
}

function SegmentSection({ comparison, selected }) {
  const champion = comparison.segment_breakdown?.champion || [];
  const challenger = comparison.segment_breakdown?.[selected] || [];
  if (!champion.length) return null;
  const byName = Object.fromEntries(challenger.map((row) => [row.segment, row]));
  const allSkipped = champion.every((row) => row.skipped);
  return (
    <>
      <SubHeading>Performance by segment</SubHeading>
      {allSkipped && (
        <Notice tone="info" title="Every segment was skipped">
          When labels are derived from SubTask, each SubTask segment is single-class by
          construction, so a per-SubTask metric is undefined. Choose a different segment column in
          the promotion gate for a meaningful breakdown.
        </Notice>
      )}
      <Table
        columns={[
          { key: "segment", header: "Segment" },
          { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
          {
            key: "champion_f1", header: "Champion F1", className: "num",
            render: (r) => (r.skipped ? "—" : metric(r.f1)),
          },
          {
            key: "challenger_f1", header: "Challenger F1", className: "num",
            render: (r) => (r.skipped ? "—" : metric(byName[r.segment]?.f1)),
          },
          { key: "skipped", header: "Note", render: (r) => r.skipped || "" },
        ]}
        rows={champion}
        rowKey={(row) => row.segment}
      />
    </>
  );
}

function BacktestSection({ backtest }) {
  const entries = Object.entries(backtest || {});
  if (!entries.length) return null;
  return (
    <>
      <SubHeading>Rolling-origin backtest — stability over time</SubHeading>
      {entries.map(([modelType, result]) => (
        <div key={modelType} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
            <strong style={{ fontSize: 12 }}>{modelType.toUpperCase()}</strong>
            {result.error
              ? <Pill tone="warn">skipped</Pill>
              : <Pill tone="info">{result.n_windows} windows</Pill>}
          </div>
          {result.error ? (
            <p style={{ fontSize: 12, color: "var(--nova-grey-dim)", margin: 0 }}>{result.error}</p>
          ) : (
            <Table
              columns={[
                { key: "window", header: "Window", className: "num" },
                { key: "train_rows", header: "Train", className: "num", render: (r) => num(r.train_rows) },
                { key: "test_rows", header: "Test", className: "num", render: (r) => num(r.test_rows) },
                { key: "f1", header: "F1", className: "num", render: (r) => (r.skipped ? "—" : metric(r.f1)) },
                { key: "recall", header: "Recall", className: "num", render: (r) => (r.skipped ? "—" : metric(r.recall)) },
                { key: "auc", header: "AUC", className: "num", render: (r) => (r.skipped ? "—" : metric(r.auc)) },
                { key: "skipped", header: "Note", render: (r) => r.skipped || "" },
              ]}
              rows={result.results || []}
              rowKey={(row) => row.window}
            />
          )}
          {result.summary?.f1 && (
            <p style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 6 }}>
              F1 across windows: mean {metric(result.summary.f1.mean)}, spread{" "}
              {metric(result.summary.f1.min)}–{metric(result.summary.f1.max)} (std{" "}
              {metric(result.summary.f1.std)}). A wide spread means this placement is unstable over
              time, whatever a single split reports.
            </p>
          )}
        </div>
      ))}
    </>
  );
}

/* ── Item 8 sections ───────────────────────────────────────────────────────
 *
 * These answer the questions the headline table cannot: is the difference real,
 * how does the model behave at a chosen operating point, what do its errors
 * cost, and exactly where does it differ from the champion.
 */

function SignificanceSection({ comparison, selected }) {
  const candidate = comparison.candidates?.[selected];
  const significance = candidate?.significance;
  const interval = candidate?.confidence_interval;
  const cost = candidate?.cost;
  const championCost = comparison.champion?.cost;
  if (!significance && !interval) return null;

  const notSignificant = significance && significance.p_value !== null && !significance.significant;
  const costDelta =
    cost && championCost ? cost.total_cost - championCost.total_cost : null;

  return (
    <>
      <SubHeading>Is the difference real?</SubHeading>

      {notSignificant && (
        <Notice tone="warn" title="This difference is within what chance would produce">
          {significance.interpretation}
        </Notice>
      )}
      {significance?.significant && (
        <Notice tone="ok" title="This difference is statistically significant">
          {significance.interpretation}
        </Notice>
      )}

      <MetricGrid
        compact
        min={150}
        items={[
          {
            label: "p-value",
            value: significance?.p_value != null ? significance.p_value.toFixed(4) : "—",
            sub: "McNemar, paired rows",
            color: notSignificant ? "#F59E0B" : C.green,
          },
          {
            label: "Rows they differ on",
            value: significance ? num(significance.discordant) : "—",
            sub: significance
              ? `${significance.challenger_only_correct} challenger · ${significance.champion_only_correct} champion`
              : undefined,
          },
          {
            label: `${interval?.metric || "metric"} 95% CI`,
            value: interval?.low != null ? `${interval.low} – ${interval.high}` : "—",
            sub: interval?.point != null ? `point ${interval.point}` : undefined,
          },
          {
            label: "Cost vs champion",
            value: costDelta == null ? "—" : `${costDelta > 0 ? "+" : ""}${costDelta.toFixed(0)}`,
            color: costDelta == null ? C.navy : costDelta > 0 ? "#EF4444" : C.green,
            sub: cost ? `${cost.cost_ratio}:1 missed-Voice weighting` : undefined,
          },
        ]}
      />

      {cost && championCost && (
        <Table
          columns={[
            { key: "model", header: "Model" },
            { key: "missed", header: "Missed Voice", className: "num" },
            { key: "wasted", header: "Wasted Voice", className: "num" },
            { key: "total", header: "Weighted cost", className: "num" },
            { key: "per1000", header: "Per 1,000 rows", className: "num" },
          ]}
          rows={[
            {
              model: `CHAMPION · ${comparison.champion.model_id}`,
              missed: num(championCost.missed_voice),
              wasted: num(championCost.wasted_voice),
              total: num(championCost.total_cost),
              per1000: num(championCost.cost_per_1000_rows),
            },
            {
              model: selected,
              missed: num(cost.missed_voice),
              wasted: num(cost.wasted_voice),
              total: num(cost.total_cost),
              per1000: num(cost.cost_per_1000_rows),
            },
          ]}
          rowKey={(row) => row.model}
        />
      )}
      {cost && (
        <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 6 }}>
          {cost.note} A missed Voice means the model said Non-Voice for an account that
          actually needed a call, so the claim stalls.
        </div>
      )}
    </>
  );
}

function OperatingSection({ comparison, selected }) {
  const points = comparison.candidates?.[selected]?.operating_points;
  const championPoints = comparison.champion?.operating_points;
  if (!points) return null;
  if (!points.available) {
    return (
      <>
        <SubHeading>Operating points</SubHeading>
        <Notice tone="info" title="Not available">{points.reason}</Notice>
      </>
    );
  }

  const row = (label, source) => ({
    model: label,
    atRecall: source?.precision_at_recall
      ? `${source.precision_at_recall.precision} @ thr ${source.precision_at_recall.threshold}`
      : "not reachable",
    atPrecision: source?.recall_at_precision
      ? `${source.recall_at_precision.recall} @ thr ${source.recall_at_precision.threshold}`
      : "not reachable",
    lift: source?.top_decile?.lift ?? "—",
  });

  return (
    <>
      <SubHeading>Operating points</SubHeading>
      <Table
        columns={[
          { key: "model", header: "Model" },
          {
            key: "atRecall",
            header: `Precision at recall ≥ ${points.recall_target}`,
          },
          {
            key: "atPrecision",
            header: `Recall at precision ≥ ${points.precision_target}`,
          },
          { key: "lift", header: "Top-decile lift", className: "num" },
        ]}
        rows={[
          row(`CHAMPION · ${comparison.champion.model_id}`, championPoints),
          row(selected, points),
        ]}
        rowKey={(r) => r.model}
      />
      <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 6 }}>
        Top-decile lift is how much richer the highest-scoring 10% of accounts is than the
        base rate — the prioritisation value, separate from accuracy.
      </div>
    </>
  );
}

function DisagreementSection({ comparison, selected }) {
  const detail = comparison.candidates?.[selected]?.disagreement;
  if (!detail) return null;
  if (!detail.rows) {
    return (
      <>
        <SubHeading>Where they differ</SubHeading>
        <Notice tone="info" title="No disagreement">{detail.note}</Notice>
      </>
    );
  }

  const losing = (detail.by_segment || []).filter((s) => s.net < 0);

  return (
    <>
      <SubHeading>Where they differ</SubHeading>
      <MetricGrid
        compact
        min={150}
        items={[
          { label: "Rows", value: num(detail.rows), sub: `${detail.pct_of_test}% of the test set` },
          { label: "Challenger right", value: num(detail.challenger_correct) },
          { label: "Champion right", value: num(detail.champion_correct) },
          {
            label: "Net",
            value: `${detail.net_to_challenger > 0 ? "+" : ""}${detail.net_to_challenger}`,
            color: detail.net_to_challenger >= 0 ? C.green : "#EF4444",
          },
        ]}
      />

      {losing.length > 0 && (
        <Notice tone="warn" title={`The challenger is worse on ${losing.length} segment(s)`}>
          A model that is better overall can still be worse where it matters most:{" "}
          {losing.map((s) => `${s.segment} (${s.net})`).join(", ")}.
        </Notice>
      )}

      {detail.by_segment && (
        <Table
          columns={[
            { key: "segment", header: "Segment" },
            { key: "rows", header: "Rows", className: "num" },
            { key: "disagreements", header: "Differ", className: "num" },
            { key: "challenger_wins", header: "Challenger", className: "num" },
            { key: "champion_wins", header: "Champion", className: "num" },
            {
              key: "net",
              header: "Net",
              className: "num",
              tone: (r) => (r.net > 0 ? "ok" : r.net < 0 ? "bad" : "muted"),
            },
          ]}
          rows={detail.by_segment}
          rowKey={(r) => r.segment}
        />
      )}
    </>
  );
}

const PRIORITY_TONE = { high: "bad", medium: "warn", low: "muted" };

function GuidanceSection({ guidance }) {
  return (
    <Card>
      <SectionTitle sub="Read from this run. Suggestions for the next one — never a recommendation to promote, which stays your decision behind the gate.">
        What would help next
      </SectionTitle>
      {guidance.map((item) => (
        <div
          key={item.title}
          style={{
            border: "1px solid var(--nova-input-border)",
            borderRadius: 8,
            padding: "11px 13px",
            marginBottom: 9,
            background: "var(--nova-input-bg)",
          }}
        >
          <div style={{ display: "flex", gap: 9, alignItems: "center", flexWrap: "wrap" }}>
            <Pill tone={PRIORITY_TONE[item.priority] || "muted"}>{item.priority}</Pill>
            <span style={{ fontWeight: 700, fontSize: 12.5 }}>{item.title}</span>
          </div>
          <div style={{ marginTop: 6, fontSize: 12, color: "var(--nova-grey-dim)" }}>
            {item.why}
          </div>
          <div style={{ marginTop: 5, fontSize: 12 }}>
            <strong>Try:</strong> {item.action}
          </div>
        </div>
      ))}
    </Card>
  );
}

/* ── Threshold explorer ────────────────────────────────────────────────────
 *
 * The decision threshold is floored at 0.50 and stepped at 0.05. Below a coin
 * flip the model would be calling Voice on rows it believes are Non-Voice,
 * which is not an operating point to reach by accident.
 *
 * Exploring here changes nothing: the saved probabilities are re-thresholded on
 * the server and the run's own selected threshold is untouched.
 */

function ThresholdExplorer({ jobId, runId, comparison, selected }) {
  const [grid, setGrid] = React.useState(null);
  const [value, setValue] = React.useState(null);
  const [scored, setScored] = React.useState(null);
  const [error, setError] = React.useState(null);

  const candidate = comparison.candidates?.[selected];
  const chosen = candidate?.selected_threshold;
  const range = candidate?.threshold_analysis?.threshold_range;
  const championBelowFloor = candidate?.threshold_analysis?.champion_below_floor;

  React.useEffect(() => {
    api.thresholds().then(setGrid).catch(() => setGrid(null));
  }, []);

  React.useEffect(() => {
    setValue(chosen ?? null);
    setScored(null);
  }, [chosen, selected]);

  React.useEffect(() => {
    if (!value || !runId || !selected) return () => {};
    let live = true;
    setError(null);
    api
      .rescore(jobId, runId, { candidate_id: selected, threshold: value })
      .then((body) => live && setScored(body))
      .catch((err) => live && setError(err));
    return () => {
      live = false;
    };
  }, [jobId, runId, selected, value]);

  if (!grid) return null;

  const metrics = scored?.test_metrics;
  const baseline = candidate?.test_metrics;
  const moved = value != null && chosen != null && Math.abs(value - chosen) > 1e-9;

  return (
    <>
      <SubHeading>Decision threshold</SubHeading>

      {championBelowFloor && (
        <Notice tone="warn" title="The champion runs below this floor">
          {candidate.threshold_analysis.selection_note}
        </Notice>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
        {grid.grid.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setValue(t)}
            style={{
              padding: "5px 11px",
              borderRadius: 6,
              cursor: "pointer",
              fontFamily: "var(--nova-font-mono)",
              fontSize: 12,
              fontWeight: 700,
              border: `1.5px solid ${value === t ? C.green : "var(--nova-input-border)"}`,
              background: value === t ? `${C.green}1A` : "var(--nova-input-bg)",
              color: value === t ? C.green : "inherit",
            }}
          >
            {t.toFixed(2)}
            {Math.abs(t - (chosen ?? -1)) < 1e-9 ? " ★" : ""}
          </button>
        ))}
      </div>

      <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginBottom: 10 }}>
        {grid.note} ★ marks the threshold this run selected.
        {moved && " Exploring here changes nothing that is saved."}
      </div>

      <ErrorNotice error={error} title="Could not rescore at that threshold" />

      {metrics && (
        <MetricGrid
          compact
          min={140}
          items={[
            {
              label: "F1", value: metric(metrics.f1),
              sub: baseline ? `selected ${metric(baseline.f1)}` : undefined,
              color: baseline && metrics.f1 < baseline.f1 ? "#EF4444" : C.green,
            },
            { label: "Precision", value: metric(metrics.precision) },
            { label: "Recall", value: metric(metrics.recall) },
            {
              label: "Weighted cost",
              value: num(scored.cost?.total_cost),
              sub: `${scored.cost?.cost_ratio}:1 missed-Voice`,
            },
            { label: "Missed Voice", value: num(scored.cost?.missed_voice) },
          ]}
        />
      )}
      {range && (
        <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)" }}>
          Allowed range {range.min}–{range.max}, step {range.step}. The backend rejects anything
          outside it, so this control cannot offer a value the API would refuse.
        </div>
      )}
    </>
  );
}

/* ── Escalating retrain loop ───────────────────────────────────────────────
 *
 * Shown where the promotion decision is made, because the two facts that
 * matter most are here: whether the target was actually reached, and that the
 * search never saw the test split.
 */

const STOP_TONE = {
  target_reached: "ok",
  no_improvement: "warn",
  time_budget: "warn",
  rounds_exhausted: "warn",
  gain_within_noise: "warn",
  cancelled: "muted",
};

function EscalationSection({ comparison, selected }) {
  const detail = comparison.candidates?.[selected]?.autotune;
  if (!detail) return null;

  const reached = detail.target_reached;

  return (
    <>
      <SubHeading>Escalating retrain loop</SubHeading>

      <Notice
        tone={reached ? "ok" : "warn"}
        title={
          detail.target_value == null
            ? `Best of ${detail.rounds_run} round(s) taken`
            : reached
              ? `Target ${detail.target_metric} ${detail.target_value} reached`
              : `Target ${detail.target_metric} ${detail.target_value} was not reached`
        }
      >
        {detail.note}
      </Notice>

      <MetricGrid
        compact
        min={150}
        items={[
          {
            label: `Best validation ${detail.target_metric}`,
            value: metric(detail.best_score),
            sub: `round ${detail.best_round ?? "—"}`,
          },
          { label: "Rounds run", value: `${detail.rounds_run} of ${detail.rounds_planned}` },
          {
            label: "Stopped because",
            value: (detail.stop_reason || "").replace(/_/g, " "),
            color: reached ? C.green : "#F59E0B",
          },
          { label: "Search time", value: `${detail.elapsed_seconds ?? "—"}s` },
        ]}
      />

      {detail.best_params && (
        <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 6 }}>
          Winning settings:{" "}
          <span style={{ fontFamily: "var(--nova-font-mono)" }}>
            {Object.entries(detail.best_params)
              .map(([k, v]) => `${k}=${v}`)
              .join("  ")}
          </span>
        </div>
      )}
      <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 4 }}>
        The validation score above is what the search optimised. The test metrics elsewhere
        on this page are the honest estimate — expect them to be a little lower, and treat
        that gap as the cost of searching rather than as a fault.
      </div>
    </>
  );
}
