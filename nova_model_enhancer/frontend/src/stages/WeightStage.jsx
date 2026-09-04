import React from "react";

import { api } from "../api.js";
import {
  ActionRow, Badge, Btn, C, Card, CheckRow, ErrorNotice, Field, FormGrid,
  MIcon, MetricGrid, Notice, Pill, ProgressBar, SectionTitle, SubHeading, Table,
} from "../nova/Components.jsx";
import { num, pct } from "../format.js";
import { NoJob } from "./TrainingDataStage.jsx";

const COMPONENTS = [
  {
    key: "recency", label: "Recent rows", proposed: 1.5,
    describe: "Rows within N days of the newest row in the snapshot.",
  },
  {
    key: "human_correction", label: "Human-corrected rows", proposed: 3.0, needsColumn: true,
    describe: "Rows a person reviewed and corrected. Needs a flag column in the data.",
  },
  {
    key: "verified_error", label: "Previously misclassified (verified)", proposed: 2.5, needsColumn: true,
    describe: "Rows the previous champion got wrong, confirmed by a person.",
  },
  {
    key: "rare_subtask", label: "Rare approved SubTasks", proposed: 2.0,
    describe: "SubTasks whose share of the dataset is below the threshold.",
  },
  {
    key: "class_balance", label: "Class balance", proposed: null,
    describe: "sklearn's 'balanced' weighting, applied multiplicatively on top.",
  },
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
    api.weightOptions(job.job_id)
      .then((body) => {
        if (!live) return;
        setOptions(body);
        setStrategy(
          body.approved_strategy?.strategy
          || structuredClone(body.advice?.strategy || body.proposed_defaults)
        );
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

  if (!job) return <NoJob />;
  if (loadError) {
    return (
      <Card>
        <SectionTitle>Weight configuration unavailable</SectionTitle>
        <ErrorNotice error={loadError} title="Could not load the snapshot" />
      </Card>
    );
  }
  if (!options || !strategy) {
    return <Card><SectionTitle sub="Reading the snapshot…">Loading</SectionTitle></Card>;
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
        strategy, approver: approver.trim(), notes,
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
      <Card>
        <SectionTitle
          sub={`${num(options.snapshot_rows)} snapshot rows · ${options.date_range?.from?.slice(0, 10) || "—"} to ${options.date_range?.to?.slice(0, 10) || "—"}`}
        >
          Sample weight strategy
        </SectionTitle>

        <Notice tone="info" title="The numbers below are proposals, not policy">
          {options.proposed_note} Every component is multiplied onto a base of 1.0 and the product is
          capped, so several components firing on one row cannot compound without limit.
        </Notice>

        {options.advice && (
          <WeightAdvice advice={options.advice} dimensions={options.dimensions} />
        )}

        <CheckRow
          checked={strategy.enabled}
          onChange={(value) => setStrategy((c) => ({ ...c, enabled: value }))}
          title="Apply sample weighting."
        >
          Leave this off to train on an unweighted baseline — that is a valid, approvable choice.
        </CheckRow>

        <FormGrid min={180}>
          <Field label="Base weight for every row" htmlFor="base">
            <input id="base" type="number" step="0.1" min="0.1" value={strategy.historical_base}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, historical_base: Number(e.target.value) }))} />
          </Field>
          <Field
            label="Combined cap" htmlFor="cap"
            hint="No row can exceed this, however many components fire on it."
          >
            <input id="cap" type="number" step="0.5" min="1" value={strategy.cap}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, cap: Number(e.target.value) }))} />
          </Field>
          <Field
            label="Normalise" htmlFor="normalise"
            hint="Normalising keeps the loss scale comparable across experiments."
          >
            <select id="normalise" value={strategy.normalise_mean_to_one ? "yes" : "no"}
              disabled={!strategy.enabled}
              onChange={(e) => setStrategy((c) => ({ ...c, normalise_mean_to_one: e.target.value === "yes" }))}>
              <option value="yes">Scale so the mean weight is 1.0</option>
              <option value="no">Leave the raw weights</option>
            </select>
          </Field>
        </FormGrid>

        <SubHeading>Components</SubHeading>
        {COMPONENTS.map((component) => {
          const config = strategy.components[component.key] || {};
          return (
            <div
              key={component.key}
              style={{ borderTop: "1px solid var(--nova-header-border)", paddingTop: 12, marginTop: 12 }}
            >
              <CheckRow
                checked={Boolean(config.enabled)}
                disabled={!strategy.enabled}
                onChange={(value) => setComponent(component.key, { enabled: value })}
              >
                <strong style={{ color: "var(--nova-header-text)" }}>{component.label}</strong>{" "}
                {component.proposed !== null && <Badge small>proposed ×{component.proposed}</Badge>}
                <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 2 }}>
                  {component.describe}
                </div>
              </CheckRow>

              {config.enabled && (
                <FormGrid min={170} style={{ marginLeft: 26, marginBottom: 4 }}>
                  {component.key !== "class_balance" && (
                    <Field label="Multiplier" htmlFor={`${component.key}-weight`}>
                      <input id={`${component.key}-weight`} type="number" step="0.1" min="0.1"
                        value={config.weight ?? component.proposed}
                        onChange={(e) => setComponent(component.key, { weight: Number(e.target.value) })} />
                    </Field>
                  )}
                  {component.key === "recency" && (
                    <Field label="Recent means within (days)" htmlFor="recent-days">
                      <input id="recent-days" type="number" min="1" value={config.recent_days ?? 90}
                        onChange={(e) => setComponent("recency", { recent_days: Number(e.target.value) })} />
                    </Field>
                  )}
                  {component.key === "rare_subtask" && (
                    <Field label="Rare means share below (%)" htmlFor="rare-share">
                      <input id="rare-share" type="number" step="0.1" min="0.1" value={config.max_share_pct ?? 1}
                        onChange={(e) => setComponent("rare_subtask", { max_share_pct: Number(e.target.value) })} />
                    </Field>
                  )}
                  {component.needsColumn && (
                    <Field
                      label="Flag column" htmlFor={`${component.key}-column`}
                      hint={options.candidate_flag_columns?.length
                        ? undefined
                        : "No column in the snapshot looks like a correction or error flag. Without one, this component is skipped and the skip is reported."}
                    >
                      <select id={`${component.key}-column`} value={config.column || ""}
                        onChange={(e) => setComponent(component.key, { column: e.target.value || null })}>
                        <option value="">— none selected —</option>
                        {(options.candidate_flag_columns || []).map((column) => (
                          <option key={column} value={column}>{column}</option>
                        ))}
                      </select>
                    </Field>
                  )}
                </FormGrid>
              )}
            </div>
          );
        })}

        <ActionRow note="Preview first: a strategy is only applied once someone approves it.">
          <Btn variant="secondary" onClick={runPreview} busy={busy} busyLabel="Computing…">
            <MIcon name="visibility" size={15} /> Preview weights
          </Btn>
        </ActionRow>
        <ErrorNotice error={error} title="The weight strategy could not be applied" />
      </Card>

      {summary && (
        <Card borderSize={approved ? 2 : 1}>
          <SectionTitle
            sub={approved?.formula || preview?.formula}
            right={<Badge color={approved ? C.green : C.indigo} bg={approved ? "#E3F5EC" : "#EEF1FF"}>
              {approved ? "APPROVED" : "PREVIEW"}
            </Badge>}
          >
            {approved ? "Approved weight strategy" : "Preview — not yet approved"}
          </SectionTitle>

          <MetricGrid
            compact
            min={120}
            items={[
              { label: "Rows", value: num(summary.distribution.rows) },
              { label: "Min", value: summary.distribution.min },
              { label: "Mean", value: summary.distribution.mean },
              { label: "Median", value: summary.distribution.median },
              { label: "Max", value: summary.distribution.max },
              { label: "Above 1.0", value: num(summary.distribution.rows_above_1) },
              {
                label: "Capped", value: num(summary.distribution.capped_rows),
                color: summary.distribution.capped_rows ? "#F59E0B" : C.navy,
              },
            ]}
          />

          {summary.effective_class_balance && (
            <>
              <SubHeading>Effective class balance after weighting</SubHeading>
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
              <SubHeading>Components applied</SubHeading>
              <Table
                columns={[
                  { key: "component", header: "Component" },
                  {
                    key: "multiplier", header: "Multiplier", className: "num",
                    render: (r) => (r.multiplier ? `×${r.multiplier}` : "—"),
                  },
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
                  <li key={skip.component}><strong>{skip.component}</strong>: {skip.reason}</li>
                ))}
              </ul>
            </Notice>
          )}

          {summary.histogram && (
            <>
              <SubHeading>Weight distribution</SubHeading>
              <div style={{ display: "grid", gap: 6 }}>
                {summary.histogram.map((bucket) => {
                  const maxRows = Math.max(...summary.histogram.map((b) => b.rows), 1);
                  return (
                    <div key={`${bucket.from}-${bucket.to}`}
                      style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      <span style={{ width: 108, fontSize: 11, fontFamily: "var(--nova-font-mono)", color: "var(--nova-grey-dim)" }}>
                        {bucket.from.toFixed(2)}–{bucket.to.toFixed(2)}
                      </span>
                      <div style={{ flex: 1 }}><ProgressBar value={bucket.rows} max={maxRows} /></div>
                      <span style={{ width: 66, textAlign: "right", fontSize: 11, fontFamily: "var(--nova-font-mono)" }}>
                        {num(bucket.rows)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {!approved ? (
            <>
              <SubHeading>Approval</SubHeading>
              <FormGrid>
                <Field label="Approved by" htmlFor="weight-approver">
                  <input id="weight-approver" type="text" placeholder="Your name" value={approver}
                    onChange={(e) => setApprover(e.target.value)} />
                </Field>
                <Field label="Notes (optional)" htmlFor="weight-notes">
                  <input id="weight-notes" type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
                </Field>
              </FormGrid>
              <ActionRow note="The exact formula and your name are stored with the run and written into the export.">
                <Btn onClick={approve} busy={busy} busyLabel="Saving…"
                  disabledReason={!approver.trim() ? "Enter an approver name." : undefined}>
                  <MIcon name="how_to_reg" size={15} /> Approve this strategy
                </Btn>
              </ActionRow>
            </>
          ) : (
            <>
              <Notice tone="ok" title={`Approved by ${approved.approver || approver}`}>
                <code>{approved.formula}</code>
              </Notice>
              <ActionRow>
                <Btn onClick={() => go("training")}>
                  Go to retraining <MIcon name="arrow_forward" size={15} />
                </Btn>
              </ActionRow>
            </>
          )}
        </Card>
      )}
    </>
  );
}

/* ── Weighting proposal ────────────────────────────────────────────────────
 *
 * Derived from this snapshot's own characteristics rather than a single static
 * default, and every component states why it is on or off so the proposal can
 * be argued with. "No weighting" is a legitimate recommendation, not a failure
 * to produce one.
 */

function WeightAdvice({ advice, dimensions }) {
  const [open, setOpen] = React.useState(false);
  const facts = advice.facts || {};

  return (
    <Card>
      <SectionTitle
        sub={advice.why}
        right={
          <>
            <Pill tone={advice.recommend_weighting ? "ok" : "muted"}>{advice.headline}</Pill>
            <Btn variant="ghost" small onClick={() => setOpen((v) => !v)}>
              <MIcon name={open ? "expand_less" : "expand_more"} size={14} />{" "}
              {open ? "Hide reasoning" : "Why"}
            </Btn>
          </>
        }
      >
        Proposed for this data
      </SectionTitle>

      <MetricGrid
        compact
        min={140}
        items={[
          { label: "Rows", value: num(facts.rows) },
          { label: "Span", value: facts.span_days != null ? `${num(facts.span_days)}d` : "—" },
          { label: "Majority class", value: facts.majority_class_pct != null ? `${facts.majority_class_pct}%` : "—" },
          {
            label: "Balance drift",
            value: facts.balance_drift_pts != null ? `${facts.balance_drift_pts} pts` : "—",
          },
          {
            label: "Rarest SubTask",
            value: facts.rarest_subtask_pct != null ? `${facts.rarest_subtask_pct}%` : "—",
          },
        ]}
      />

      {open && (
        <Table
          columns={[
            { key: "component", header: "Component" },
            {
              key: "enabled",
              header: "Proposed",
              render: (row) => (
                <Pill tone={row.enabled ? "ok" : "muted"}>{row.enabled ? "on" : "off"}</Pill>
              ),
            },
            { key: "why", header: "Why" },
          ]}
          rows={advice.reasons || []}
          rowKey={(row) => row.component}
          empty="No components evaluated."
        />
      )}

      {dimensions && (
        <div style={{ marginTop: 10, fontSize: 11.5, color: "var(--nova-grey-dim)" }}>
          Client dimension: <strong>{dimensions.client || "none detected"}</strong>
          {dimensions.secondary ? <> · secondary: <strong>{dimensions.secondary}</strong></> : null}
          <div style={{ marginTop: 3 }}>{dimensions.note}</div>
        </div>
      )}
    </Card>
  );
}
