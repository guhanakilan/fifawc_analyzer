import React from "react";

import { api } from "../api.js";
import {
  Action, ErrorNotice, Icon, Metrics, Notice, Panel, Pill, Table,
} from "../components/Ui.jsx";
import { num, pct } from "../format.js";

const COMPONENTS = [
  { key: "recency", label: "Recent rows", proposed: 1.5,
    describe: "Rows within N days of the newest row in the snapshot." },
  { key: "human_correction", label: "Human-corrected rows", proposed: 3.0,
    describe: "Rows a person reviewed and corrected. Needs a flag column.", needsColumn: true },
  { key: "verified_error", label: "Previously misclassified (verified)", proposed: 2.5,
    describe: "Rows the previous champion got wrong, confirmed by a person.", needsColumn: true },
  { key: "rare_subtask", label: "Rare approved SubTasks", proposed: 2.0,
    describe: "SubTasks whose share of the dataset is below the threshold." },
  { key: "class_balance", label: "Class balance", proposed: null,
    describe: "sklearn's 'balanced' weighting, applied multiplicatively on top." },
];

export default function WeightStage({ job, mark, go }) {
  const [options, setOptions] = React.useState(null);
  const [strategy, setStrategy] = React.useState(null);
  const [preview, setPreview] = React.useState(null);
  const [approved, setApproved] = React.useState(null);
  const [approver, setApprover] = React.useState("");
  const [notes, setNotes] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [loadError, setLoadError] = React.useState(null);

  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    api
      .weightOptions(job.job_id)
      .then((body) => {
        if (!live) return;
        setOptions(body);
        const existing = body.approved_strategy?.strategy;
        setStrategy(existing || structuredClone(body.proposed_defaults));
        if (body.approved_strategy) {
          setApproved(body.approved_strategy);
          setApprover(body.approved_by || "");
          mark("weightsApproved", true);
        }
      })
      .catch((err) => live && setLoadError(err));
    return () => {
      live = false;
    };
  }, [job, mark]);

  if (!job) {
    return (
      <Panel title="No active job" icon="info">
        <Notice tone="warn" title="Start at Stage 01">Upload a champion package first.</Notice>
      </Panel>
    );
  }
  if (loadError) {
    return (
      <Panel title="Weight configuration unavailable" icon="report" tone="bad">
        <ErrorNotice error={loadError} title="Could not load the snapshot" />
      </Panel>
    );
  }
  if (!options || !strategy) {
    return <Panel title="Loading" icon="hourglass_top"><p className="muted small">Reading the snapshot…</p></Panel>;
  }

  const setComponent = (key, patch) =>
    setStrategy((current) => ({
      ...current,
      components: { ...current.components, [key]: { ...current.components[key], ...patch } },
    }));

  const runPreview = async () => {
    setBusy(true);
    setError(null);
    try {
      setPreview(await api.previewWeights(job.job_id, strategy));
    } catch (previewError) {
      setError(previewError);
    } finally {
      setBusy(false);
    }
  };

  const approve = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.approveWeights(job.job_id, {
        strategy,
        approver: approver.trim(),
        notes,
      });
      setApproved(result);
      setPreview(null);
      mark("weightsApproved", true);
    } catch (approveError) {
      setError(approveError);
    } finally {
      setBusy(false);
    }
  };

  const summary = preview?.summary || approved?.summary;

  return (
    <>
      <Panel
        title="Sample weight strategy"
        subtitle={`${num(options.snapshot_rows)} snapshot rows · ${options.date_range?.from?.slice(0, 10) || "—"} to ${options.date_range?.to?.slice(0, 10) || "—"}`}
        icon="balance"
      >
        <Notice tone="info" title="The numbers below are proposals, not policy">
          {options.proposed_note} Every component is multiplied onto a base of 1.0 and the product is
          capped, so several components firing on one row cannot compound without limit.
        </Notice>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={strategy.enabled}
            onChange={(e) => setStrategy((c) => ({ ...c, enabled: e.target.checked }))}
          />
          <span>
            <strong>Apply sample weighting.</strong> Leave this off to train on an unweighted baseline —
            that is a valid, approvable choice.
          </span>
        </label>

        <div className="grid-3">
          <div className="field">
            <label htmlFor="base">Base weight for every row</label>
            <input id="base" type="number" step="0.1" min="0.1" value={strategy.historical_base}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, historical_base: Number(e.target.value) }))} />
          </div>
          <div className="field">
            <label htmlFor="cap">Combined cap</label>
            <input id="cap" type="number" step="0.5" min="1" value={strategy.cap}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, cap: Number(e.target.value) }))} />
            <span className="hint">No row can exceed this, however many components fire on it.</span>
          </div>
          <div className="field">
            <label htmlFor="normalise">Normalise</label>
            <select id="normalise" value={strategy.normalise_mean_to_one ? "yes" : "no"}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, normalise_mean_to_one: e.target.value === "yes" }))}>
              <option value="yes">Scale so the mean weight is 1.0</option>
              <option value="no">Leave the raw weights</option>
            </select>
            <span className="hint">Normalising keeps the loss scale comparable across experiments.</span>
          </div>
        </div>

        <div className="section-title">Components</div>
        {COMPONENTS.map((component) => {
          const config = strategy.components[component.key] || {};
          return (
            <div key={component.key} style={{ borderTop: "1px solid var(--card-border)", paddingTop: 10, marginTop: 10 }}>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={Boolean(config.enabled)}
                  disabled={!strategy.enabled}
                  onChange={(e) => setComponent(component.key, { enabled: e.target.checked })}
                />
                <span>
                  <strong>{component.label}</strong>
                  {component.proposed !== null && (
                    <Pill tone="muted"> proposed ×{component.proposed}</Pill>
                  )}
                  <div className="muted small">{component.describe}</div>
                </span>
              </label>
              {config.enabled && (
                <div className="grid-3" style={{ marginLeft: 24 }}>
                  {component.key !== "class_balance" && (
                    <div className="field">
                      <label htmlFor={`${component.key}-weight`}>Multiplier</label>
                      <input id={`${component.key}-weight`} type="number" step="0.1" min="0.1"
                        value={config.weight ?? component.proposed}
                        onChange={(e) => setComponent(component.key, { weight: Number(e.target.value) })} />
                    </div>
                  )}
                  {component.key === "recency" && (
                    <div className="field">
                      <label htmlFor="recent-days">Recent means within (days)</label>
                      <input id="recent-days" type="number" min="1" value={config.recent_days ?? 90}
                        onChange={(e) => setComponent("recency", { recent_days: Number(e.target.value) })} />
                    </div>
                  )}
                  {component.key === "rare_subtask" && (
                    <div className="field">
                      <label htmlFor="rare-share">Rare means share below (%)</label>
                      <input id="rare-share" type="number" step="0.1" min="0.1" value={config.max_share_pct ?? 1}
                        onChange={(e) => setComponent("rare_subtask", { max_share_pct: Number(e.target.value) })} />
                    </div>
                  )}
                  {component.needsColumn && (
                    <div className="field">
                      <label htmlFor={`${component.key}-column`}>Flag column</label>
                      <select id={`${component.key}-column`} value={config.column || ""}
                        onChange={(e) => setComponent(component.key, { column: e.target.value || null })}>
                        <option value="">— none available —</option>
                        {(options.candidate_flag_columns || []).map((column) => (
                          <option key={column} value={column}>{column}</option>
                        ))}
                      </select>
                      {!options.candidate_flag_columns?.length && (
                        <span className="hint">
                          No column in the snapshot looks like a correction or error flag. Without one,
                          this component is skipped and the skip is reported.
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        <div className="btn-row end">
          <span className="btn-note">Preview first: a strategy is only applied once someone approves it.</span>
          <Action className="btn ghost" onClick={runPreview} busy={busy} busyLabel="Computing…">
            <Icon name="visibility" size={15} /> Preview weights
          </Action>
        </div>
        <ErrorNotice error={error} title="The weight strategy could not be applied" />
      </Panel>

      {summary && (
        <Panel
          title={approved ? "Approved weight strategy" : "Preview — not yet approved"}
          subtitle={approved?.formula || preview?.formula}
          icon={approved ? "verified" : "visibility"}
          tone={approved ? "ok" : ""}
        >
          <Metrics
            items={[
              { label: "Rows", value: num(summary.distribution.rows) },
              { label: "Min", value: summary.distribution.min },
              { label: "Mean", value: summary.distribution.mean },
              { label: "Median", value: summary.distribution.median },
              { label: "Max", value: summary.distribution.max },
              { label: "Std dev", value: summary.distribution.std },
              { label: "Rows above 1.0", value: num(summary.distribution.rows_above_1) },
              {
                label: "Rows hitting the cap",
                value: num(summary.distribution.capped_rows),
                sub: summary.distribution.capped_rows ? "Capped rather than compounding" : undefined,
              },
            ]}
          />

          {summary.effective_class_balance && (
            <>
              <div className="section-title">Effective class balance after weighting</div>
              <Table
                columns={[
                  { key: "cls", header: "Class", render: (r) => (r.cls === "1" ? "1 · Non-Voice" : "0 · Voice") },
                  { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
                  { key: "raw", header: "Raw share", className: "num", render: (r) => pct(r.raw_share_pct) },
                  { key: "weighted", header: "Weighted share", className: "num", render: (r) => pct(r.weighted_share_pct) },
                ]}
                rows={Object.entries(summary.effective_class_balance).map(([cls, value]) => ({ cls, ...value }))}
                rowKey={(row) => row.cls}
              />
            </>
          )}

          {summary.applied?.length > 0 && (
            <>
              <div className="section-title">Components applied</div>
              <Table
                columns={[
                  { key: "component", header: "Component" },
                  { key: "multiplier", header: "Multiplier", className: "num", render: (r) => (r.multiplier ? `×${r.multiplier}` : "—") },
                  { key: "rows", header: "Rows affected", className: "num", render: (r) => num(r.rows) },
                  { key: "detail", header: "Rule" },
                ]}
                rows={summary.applied}
                rowKey={(row) => row.component}
              />
            </>
          )}

          {summary.skipped?.length > 0 && (
            <Notice tone="warn" title="Components that could not be applied">
              <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
                {summary.skipped.map((skip) => (
                  <li key={skip.component}>
                    <strong>{skip.component}</strong>: {skip.reason}
                  </li>
                ))}
              </ul>
            </Notice>
          )}

          {summary.histogram && (
            <>
              <div className="section-title">Weight distribution</div>
              <div className="stack">
                {summary.histogram.map((bucket) => {
                  const maxRows = Math.max(...summary.histogram.map((b) => b.rows), 1);
                  return (
                    <div className="row" key={`${bucket.from}-${bucket.to}`}>
                      <span className="mono small" style={{ width: 110 }}>
                        {bucket.from.toFixed(2)}–{bucket.to.toFixed(2)}
                      </span>
                      <div className="progress" style={{ flex: 1 }}>
                        <div style={{ width: `${(bucket.rows / maxRows) * 100}%` }} />
                      </div>
                      <span className="mono small" style={{ width: 70, textAlign: "right" }}>
                        {num(bucket.rows)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {!approved && (
            <>
              <div className="section-title">Approval</div>
              <div className="grid-2">
                <div className="field">
                  <label htmlFor="weight-approver">Approved by</label>
                  <input id="weight-approver" type="text" placeholder="Your name" value={approver}
                    onChange={(e) => setApprover(e.target.value)} />
                </div>
                <div className="field">
                  <label htmlFor="weight-notes">Notes (optional)</label>
                  <input id="weight-notes" type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
                </div>
              </div>
              <div className="btn-row end">
                <span className="btn-note">
                  The exact formula and your name are stored with the run and written into the export.
                </span>
                <Action onClick={approve} busy={busy} busyLabel="Saving…"
                  disabledReason={!approver.trim() ? "Enter an approver name." : undefined}>
                  <Icon name="how_to_reg" size={15} /> Approve this strategy
                </Action>
              </div>
            </>
          )}

          {approved && (
            <>
              <Notice tone="ok" title={`Approved by ${approved.approver || approver}`}>
                <code>{approved.formula}</code>
              </Notice>
              <div className="btn-row end">
                <Action onClick={() => go("training")}>
                  Go to retraining <Icon name="arrow_forward" size={15} />
                </Action>
              </div>
            </>
          )}
        </Panel>
      )}
    </>
  );
}
