import React from "react";

import { api } from "../api.js";
import {
  Action, Empty, ErrorNotice, Icon, Metrics, Notice, Panel, Pill, Table,
} from "../components/Ui.jsx";
import { delta, metric, num, when } from "../format.js";

const METRIC_KEYS = [
  ["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"],
  ["specificity", "Specificity"], ["accuracy", "Accuracy"],
  ["auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["brier_score", "Brier"],
];

const GATE_TONE = {
  RECOMMENDED: "ok", NOT_RECOMMENDED: "warn", BLOCKED: "bad", APPROVED: "ok",
};

export default function ComparisonStage({ job, progress, mark, go }) {
  const [runs, setRuns] = React.useState([]);
  const [runId, setRunId] = React.useState(null);
  const [gate, setGate] = React.useState(null);
  const [gateForm, setGateForm] = React.useState(null);
  const [gateApprover, setGateApprover] = React.useState("");
  const [comparison, setComparison] = React.useState(null);
  const [selected, setSelected] = React.useState(null);
  const [approver, setApprover] = React.useState("");
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
        setGateApprover(gateBody.approved_by || "");
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
      setSelected(body.leading_candidate);
      if (body.approval?.decision === "APPROVED") mark("approvedRunId", runId);
    } catch (comparisonError) {
      setError(comparisonError);
    }
  }, [job, runId, mark]);

  React.useEffect(() => {
    loadComparison();
  }, [loadComparison]);

  if (!job) {
    return (
      <Panel title="No active job" icon="info">
        <Notice tone="warn" title="Start at Stage 01">Upload a champion package first.</Notice>
      </Panel>
    );
  }
  if (loadError) {
    return (
      <Panel title="Comparison unavailable" icon="report" tone="bad">
        <ErrorNotice error={loadError} />
      </Panel>
    );
  }
  if (!runs.length) {
    return (
      <Panel title="Nothing to compare yet" icon="compare_arrows">
        <Empty icon="science">Complete a retraining run in Stage 05 first.</Empty>
      </Panel>
    );
  }

  const saveGate = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.saveGate(job.job_id, { gate: stripGate(gateForm), approver: gateApprover.trim() });
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
        approver: approver.trim(), typed_confirmation: typed.trim(), notes,
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
    ? Object.entries(comparison.candidates).filter(([, value]) => !value.skipped)
    : [];
  const championMetrics = comparison?.champion?.test_metrics || {};
  const gateResult = comparison?.gate_results?.[selected];
  const approval = comparison?.approval;

  return (
    <>
      <Panel
        title="Promotion gate"
        subtitle="What counts as better must be decided before looking at the numbers, not after."
        icon="gavel"
        tone={gate?.gate?.approved ? "ok" : "warn"}
        actions={<Pill tone={gate?.gate?.approved ? "ok" : "warn"}>{gate?.gate?.approved ? "Approved" : "Not approved"}</Pill>}
      >
        {!gate?.gate?.approved && (
          <Notice tone="warn" title="Every comparison is BLOCKED until this gate is approved">
            {gate?.proposed_note}
          </Notice>
        )}
        {gateForm && (
          <>
            <div className="grid-3">
              <div className="field">
                <label htmlFor="primary-metric">Primary metric</label>
                <select id="primary-metric" value={gateForm.primary_metric}
                  onChange={(e) => setGateForm((c) => ({ ...c, primary_metric: e.target.value }))}>
                  {gate.primary_metric_choices.map((choice) => (
                    <option key={choice} value={choice}>{choice}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="min-improve">Minimum improvement (%)</label>
                <input id="min-improve" type="number" step="0.1" value={gateForm.min_primary_improvement_pct}
                  onChange={(e) => setGateForm((c) => ({ ...c, min_primary_improvement_pct: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label htmlFor="hist-regress">Max historical regression (%)</label>
                <input id="hist-regress" type="number" step="0.1" value={gateForm.max_historical_primary_regression_pct ?? ""}
                  onChange={(e) => setGateForm((c) => ({
                    ...c,
                    max_historical_primary_regression_pct: e.target.value === "" ? null : Number(e.target.value),
                  }))} />
              </div>
            </div>
            <div className="section-title">Protected metrics</div>
            {(gateForm.protected_metrics || []).map((protectedMetric, index) => (
              <div className="grid-3" key={index}>
                <div className="field">
                  <label htmlFor={`protected-${index}`}>Metric</label>
                  <select id={`protected-${index}`} value={protectedMetric.metric}
                    onChange={(e) => {
                      const next = [...gateForm.protected_metrics];
                      next[index] = { ...next[index], metric: e.target.value };
                      setGateForm((c) => ({ ...c, protected_metrics: next }));
                    }}>
                    {METRIC_KEYS.map(([key, label]) => (
                      <option key={key} value={key}>{label}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor={`tolerance-${index}`}>Max regression (%)</label>
                  <input id={`tolerance-${index}`} type="number" step="0.1" value={protectedMetric.max_regression_pct}
                    onChange={(e) => {
                      const next = [...gateForm.protected_metrics];
                      next[index] = { ...next[index], max_regression_pct: Number(e.target.value) };
                      setGateForm((c) => ({ ...c, protected_metrics: next }));
                    }} />
                </div>
                <div className="field" style={{ justifyContent: "flex-end" }}>
                  <button type="button" className="btn ghost"
                    onClick={() => setGateForm((c) => ({
                      ...c, protected_metrics: c.protected_metrics.filter((_, i) => i !== index),
                    }))}>
                    <Icon name="delete" size={14} /> Remove
                  </button>
                </div>
              </div>
            ))}
            <button type="button" className="btn ghost"
              onClick={() => setGateForm((c) => ({
                ...c, protected_metrics: [...(c.protected_metrics || []), { metric: "precision", max_regression_pct: 1.0 }],
              }))}>
              <Icon name="add" size={14} /> Add a protected metric
            </button>

            <div className="grid-3" style={{ marginTop: 12 }}>
              <div className="field">
                <label htmlFor="segment-col">Segment column for breakdowns</label>
                <input id="segment-col" type="text" value={gateForm.segment_column || ""}
                  onChange={(e) => setGateForm((c) => ({ ...c, segment_column: e.target.value || null }))} />
              </div>
              <div className="field">
                <label htmlFor="min-segment">Minimum rows per segment</label>
                <input id="min-segment" type="number" min="10" value={gateForm.min_segment_rows}
                  onChange={(e) => setGateForm((c) => ({ ...c, min_segment_rows: Number(e.target.value) }))} />
                <span className="hint">Smaller segments are listed as skipped rather than reported noisily.</span>
              </div>
              <div className="field">
                <label htmlFor="gate-approver">Approved by</label>
                <input id="gate-approver" type="text" placeholder="Your name" value={gateApprover}
                  onChange={(e) => setGateApprover(e.target.value)} />
              </div>
            </div>
            <label className="checkbox">
              <input type="checkbox" checked={gateForm.require_backtest_pass}
                onChange={(e) => setGateForm((c) => ({ ...c, require_backtest_pass: e.target.checked }))} />
              <span>Require the rolling backtest to have completed.</span>
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={gateForm.require_package_validation}
                onChange={(e) => setGateForm((c) => ({ ...c, require_package_validation: e.target.checked }))} />
              <span>Require export package validation to pass before recommending promotion.</span>
            </label>

            <div className="btn-row end">
              <Action onClick={saveGate} busy={busy} busyLabel="Saving…"
                disabledReason={!gateApprover.trim() ? "Enter an approver name." : undefined}>
                <Icon name="how_to_reg" size={15} /> Approve this gate
              </Action>
            </div>
          </>
        )}
      </Panel>

      <Panel title="Benchmark" icon="compare_arrows"
        subtitle={comparison?.benchmark?.note}
        actions={
          <select value={runId || ""} onChange={(e) => setRunId(e.target.value)} aria-label="Run to compare">
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} · {when(run.created_at)}
              </option>
            ))}
          </select>
        }>
        <ErrorNotice error={error} />
        {!comparison ? (
          <p className="muted small">Loading the comparison…</p>
        ) : (
          <>
            <Metrics
              cols={4}
              items={[
                { label: "Benchmark rows", value: num(comparison.benchmark.rows) },
                { label: "Actual Non-Voice", value: num(comparison.benchmark.actual_non_voice) },
                { label: "Actual Voice", value: num(comparison.benchmark.actual_voice) },
                {
                  label: "Window",
                  value: comparison.benchmark.date_from?.slice(0, 10) || "—",
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

            <div className="section-title">Champion versus challengers on identical rows</div>
            <MetricTable
              championMetrics={championMetrics}
              championLabel={comparison.champion.model_id}
              championThreshold={comparison.champion.threshold}
              candidates={candidates}
              selected={selected}
              onSelect={setSelected}
              primary={gate?.gate?.primary_metric || "f1"}
            />

            <div className="section-title">Confusion matrix at the selected threshold</div>
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

            <PeriodSection comparison={comparison} selected={selected} />
            <SegmentSection comparison={comparison} selected={selected} />
            <BacktestSection backtest={comparison.backtest} />
          </>
        )}
      </Panel>

      {comparison && selected && (
        <Panel
          title={`Decision for ${selected}`}
          subtitle="Promotion is a recommendation. Nothing is promoted without a typed approval."
          icon="how_to_vote"
          tone={GATE_TONE[approval?.decision === "APPROVED" ? "APPROVED" : gateResult?.status] || ""}
          actions={<Pill tone={GATE_TONE[gateResult?.status] || "muted"}>{gateResult?.status || "—"}</Pill>}
        >
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
                  render: (r) => (r.delta_pct == null ? "—" : `${r.delta_pct > 0 ? "+" : ""}${r.delta_pct.toFixed(2)}%`),
                },
                {
                  key: "passed", header: "Result",
                  render: (r) => <Pill tone={r.passed ? "ok" : "bad"}>{r.passed ? "pass" : "fail"}</Pill>,
                },
              ]}
              rows={gateResult.rules}
              rowKey={(row) => row.rule}
            />
          )}

          {approval ? (
            <Notice tone={approval.decision === "APPROVED" ? "ok" : "warn"}
              title={`${approval.decision} by ${approval.approver} on ${when(approval.approved_at)}`}>
              <p>{approval.notes || "No notes recorded."}</p>
              {approval.override_of_recommendation && (
                <p><strong>This was an override:</strong> the gate did not recommend promotion.</p>
              )}
              {approval.decision === "APPROVED" && (
                <div className="btn-row end">
                  <Action onClick={() => go("export")}>
                    Build the export package <Icon name="arrow_forward" size={15} />
                  </Action>
                </div>
              )}
            </Notice>
          ) : (
            <>
              <div className="grid-3">
                <div className="field">
                  <label htmlFor="promo-approver">Approved by</label>
                  <input id="promo-approver" type="text" placeholder="Your name" value={approver}
                    onChange={(e) => setApprover(e.target.value)} />
                </div>
                <div className="field">
                  <label htmlFor="promo-typed">Type the candidate id to confirm</label>
                  <input id="promo-typed" type="text" placeholder={selected} value={typed}
                    onChange={(e) => setTyped(e.target.value)} className="mono" />
                  <span className="hint">Exactly <code>{selected}</code>.</span>
                </div>
                <div className="field">
                  <label htmlFor="promo-notes">Notes</label>
                  <input id="promo-notes" type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
                </div>
              </div>
              <div className="btn-row end">
                <button type="button" className="btn ghost" disabled={busy || !approver.trim() || typed.trim() !== selected}
                  onClick={() => approve("REJECTED")}>
                  <Icon name="block" size={15} /> Record rejection
                </button>
                <Action onClick={() => approve("APPROVED")} busy={busy} busyLabel="Recording…"
                  disabledReason={
                    !approver.trim() ? "Enter an approver name."
                      : typed.trim() !== selected ? "Type the candidate id exactly."
                      : gateResult?.status === "BLOCKED" ? "This candidate is blocked; resolve the blockers first."
                      : undefined
                  }>
                  <Icon name="how_to_reg" size={15} /> Approve promotion
                </Action>
              </div>
            </>
          )}
        </Panel>
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
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Threshold</th>
            {METRIC_KEYS.map(([key, label]) => (
              <th key={key} scope="col">{label}{key === primary ? " ★" : ""}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>CHAMPION</strong> · <span className="mono">{championLabel}</span></td>
            <td className="num">{championThreshold}</td>
            {METRIC_KEYS.map(([key]) => (
              <td className="num" key={key}>{metric(championMetrics[key])}</td>
            ))}
          </tr>
          {candidates.map(([id, value]) => (
            <tr key={id} className={selected === id ? "highlight" : ""}>
              <td>
                <label style={{ display: "flex", gap: 6, alignItems: "center", cursor: "pointer" }}>
                  <input type="radio" name="candidate" checked={selected === id} onChange={() => onSelect(id)} />
                  <span className="mono">{id}</span>
                </label>
              </td>
              <td className="num">{value.selected_threshold}</td>
              {METRIC_KEYS.map(([key]) => {
                const change = delta(value.test_metrics?.[key], championMetrics[key]);
                const better = key === "brier_score" ? change < 0 : change > 0;
                return (
                  <td className={`num ${change == null || Math.abs(change) < 0.05 ? "" : better ? "win" : "lose"}`} key={key}>
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
  const challengerByPeriod = Object.fromEntries(challenger.map((row) => [row.period, row]));
  const rows = champion.map((row) => {
    const pair = challengerByPeriod[row.period];
    return {
      period: row.period,
      rows: row.rows,
      champion_f1: row.f1,
      challenger_f1: pair?.f1,
      skipped: row.skipped || pair?.skipped,
    };
  });
  return (
    <>
      <div className="section-title">Performance by period</div>
      <Table
        columns={[
          { key: "period", header: "Month" },
          { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
          { key: "champion_f1", header: "Champion F1", className: "num", render: (r) => (r.skipped ? "—" : metric(r.champion_f1)) },
          { key: "challenger_f1", header: "Challenger F1", className: "num", render: (r) => (r.skipped ? "—" : metric(r.challenger_f1)) },
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
  return (
    <>
      <div className="section-title">Performance by segment</div>
      <Table
        columns={[
          { key: "segment", header: "Segment" },
          { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
          { key: "champion_f1", header: "Champion F1", className: "num", render: (r) => (r.skipped ? "—" : metric(r.f1)) },
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
      <div className="section-title">Rolling-origin backtest — stability over time</div>
      {entries.map(([modelType, result]) => (
        <div key={modelType} style={{ marginBottom: 12 }}>
          <div className="row" style={{ marginBottom: 6 }}>
            <strong>{modelType.toUpperCase()}</strong>
            {result.error ? <Pill tone="warn">skipped</Pill> : <Pill tone="info">{result.n_windows} windows</Pill>}
          </div>
          {result.error ? (
            <p className="muted small">{result.error}</p>
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
            <p className="muted small" style={{ marginTop: 6 }}>
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

function stripGate(gate) {
  const { approved, approver, ...rest } = gate;
  return rest;
}
