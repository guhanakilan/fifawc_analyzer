import React from "react";

import { api } from "../api.js";
import {
  Action, Empty, ErrorNotice, Icon, Metrics, Notice, Panel, Pill, Table,
} from "../components/Ui.jsx";
import { num, pct, shortHash, when } from "../format.js";

const FLAGS = ["Voice", "Non-Voice", "Keyword", "Ignore"];

export default function ReadinessStage({ job, mark, go }) {
  const [review, setReview] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [saveError, setSaveError] = React.useState(null);
  const [snapshot, setSnapshot] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [form, setForm] = React.useState(null);
  const [mappings, setMappings] = React.useState({});

  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    setLoading(true);
    Promise.all([
      api.review(job.job_id),
      api.snapshot(job.job_id).catch(() => null),
    ])
      .then(([reviewBody, snapshotBody]) => {
        if (!live) return;
        setReview(reviewBody);
        setSnapshot(snapshotBody);
        if (snapshotBody) mark("snapshotId", snapshotBody.snapshot_id);
        const saved = reviewBody.saved_decisions;
        setForm({
          date_column: saved?.date_column || reviewBody.detected_date_column || "",
          target_mode: saved?.target_mode || (reviewBody.champion_has_subtask_mappings
            ? "derive_from_subtask" : "existing"),
          target_column: saved?.target_column || reviewBody.detected_target_column || "",
          voice_values: (saved?.target_encoding?.voice_values || ["0"]).join(","),
          non_voice_values: (saved?.target_encoding?.non_voice_values || ["1"]).join(","),
          dedup_mode: saved?.dedup_mode || "",
          dedup_keys: saved?.dedup_keys || [],
          historical_window_days: saved?.historical_window_days ?? "",
          allow_unmapped_default: Boolean(saved?.allow_unmapped_default),
          approver: saved?.approver || "",
        });
        const seed = {};
        (reviewBody.subtask_review?.unmapped || []).forEach((row) => {
          seed[row.subtask] = row.suggested_flag;
        });
        setMappings(seed);
      })
      .catch((loadError) => live && setError(loadError))
      .finally(() => live && setLoading(false));
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
  if (loading || !form) {
    return <Panel title="Loading the dataset review" icon="hourglass_top"><p className="muted small">Reading the uploaded data…</p></Panel>;
  }
  if (error) {
    return (
      <Panel title="Readiness review unavailable" icon="report" tone="bad">
        <ErrorNotice error={error} title="Could not build the review" />
      </Panel>
    );
  }

  const subtasks = review.subtask_review || {};
  const unmapped = subtasks.unmapped || [];
  const unresolved = unmapped.filter((row) => !mappings[row.subtask] || mappings[row.subtask] === "");

  const blockers = [];
  if (!form.date_column) blockers.push("Choose the date column that drives the temporal split.");
  if (form.target_mode === "existing" && !form.target_column) {
    blockers.push("Choose the label column, or switch to deriving labels from SubTask mappings.");
  }
  if (!form.dedup_mode) blockers.push("Choose how duplicate rows are resolved.");
  if (form.dedup_mode === "key_columns" && form.dedup_keys.length === 0) {
    blockers.push("Select at least one deduplication key column.");
  }
  if (form.target_mode === "derive_from_subtask" && unresolved.length && !form.allow_unmapped_default) {
    blockers.push(`Map ${unresolved.length} new SubTask value(s), or explicitly accept the Non-Voice default.`);
  }
  if (!form.approver.trim()) blockers.push("Enter the name of the person approving these rules.");

  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  const save = async () => {
    setBusy(true);
    setSaveError(null);
    try {
      const combined = [
        ...(subtasks.known || []).map((row) => ({ name: row.subtask, flag: row.flag })),
        ...unmapped
          .filter((row) => mappings[row.subtask])
          .map((row) => ({ name: row.subtask, flag: mappings[row.subtask] })),
      ];
      await api.saveDecisions(job.job_id, {
        date_column: form.date_column,
        target_mode: form.target_mode,
        target_column: form.target_mode === "existing" ? form.target_column : null,
        target_encoding: {
          voice_values: form.voice_values.split(",").map((v) => v.trim()).filter(Boolean),
          non_voice_values: form.non_voice_values.split(",").map((v) => v.trim()).filter(Boolean),
        },
        dedup_mode: form.dedup_mode,
        dedup_keys: form.dedup_keys,
        subtask_mappings: combined,
        subtask_keywords: [],
        allow_unmapped_default: form.allow_unmapped_default,
        historical_window_days: form.historical_window_days === "" ? null : Number(form.historical_window_days),
        approver: form.approver.trim(),
      });
      const manifest = await api.buildSnapshot(job.job_id);
      setSnapshot(manifest);
      mark("snapshotId", manifest.snapshot_id);
    } catch (buildError) {
      setSaveError(buildError);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Panel
        title="What the data looks like"
        subtitle={`${num(review.row_count)} rows across ${review.columns.length} columns.`}
        icon="query_stats"
      >
        <Metrics
          items={[
            { label: "Rows", value: num(review.row_count) },
            { label: "Columns", value: num(review.columns.length) },
            { label: "Full-row duplicates", value: num(review.duplicates.full_row_duplicates) },
            { label: "Distinct SubTasks", value: num(subtasks.total_subtasks) },
            { label: "Unmapped SubTasks", value: num(unmapped.length) },
            {
              label: "Missing expected columns",
              value: num(review.schema_drift.missing_column_count),
              sub: review.schema_drift.missing_column_count
                ? review.schema_drift.missing_columns.slice(0, 4).join(", ")
                : undefined,
            },
          ]}
          cols={3}
        />
        <Notice tone="info" title="Target encoding, confirmed from the reference implementation">
          <code>NonVoiceFlag</code> is <strong>0 = Voice</strong>, <strong>1 = Non-Voice</strong>. This
          comes from <code>routers/flag.py::run_flag</code> in the NoVA source, not from an assumption.
        </Notice>
      </Panel>

      <Panel
        title="Decisions this application will not make for you"
        subtitle="Each of these changes what gets trained, so each needs a person behind it."
        icon="rule"
      >
        <div className="grid-2">
          <div className="field">
            <label htmlFor="date-column">Date column (drives the temporal split and recency)</label>
            <select id="date-column" value={form.date_column} onChange={(e) => set("date_column", e.target.value)}>
              <option value="">— choose a column —</option>
              {review.columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                  {review.date_candidates.includes(column) ? "  (detected)" : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="window-days">Historical window (days, optional)</label>
            <input
              id="window-days"
              type="number"
              min="1"
              placeholder="Leave empty to use every row"
              value={form.historical_window_days}
              onChange={(e) => set("historical_window_days", e.target.value)}
            />
            <span className="hint">Rows older than this many days before the newest row are excluded.</span>
          </div>
        </div>

        <div className="section-title">Where the label comes from</div>
        <div className="grid-2">
          <div className="field">
            <label htmlFor="target-mode">Label source</label>
            <select id="target-mode" value={form.target_mode} onChange={(e) => set("target_mode", e.target.value)}>
              <option value="derive_from_subtask">
                Derive from SubTask mappings (the reference's own rules)
              </option>
              <option value="existing">Use a label column already in the data</option>
            </select>
          </div>
          {form.target_mode === "existing" ? (
            <div className="field">
              <label htmlFor="target-column">Label column</label>
              <select id="target-column" value={form.target_column} onChange={(e) => set("target_column", e.target.value)}>
                <option value="">— choose a column —</option>
                {review.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                    {review.target_candidates.includes(column) ? "  (detected)" : ""}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div />
          )}
        </div>
        {form.target_mode === "existing" && (
          <div className="grid-2">
            <div className="field">
              <label htmlFor="voice-values">Values that mean Voice</label>
              <input id="voice-values" type="text" value={form.voice_values} onChange={(e) => set("voice_values", e.target.value)} />
              <span className="hint">Comma-separated. Any value outside these two lists stops the snapshot.</span>
            </div>
            <div className="field">
              <label htmlFor="nv-values">Values that mean Non-Voice</label>
              <input id="nv-values" type="text" value={form.non_voice_values} onChange={(e) => set("non_voice_values", e.target.value)} />
            </div>
          </div>
        )}

        <div className="section-title">Deduplication</div>
        <Notice tone="info" title="A business key is never inferred">
          {num(review.duplicates.full_row_duplicates)} exact duplicate rows are present. Whether two
          rows sharing an account number are the same work item is a business rule, so choose it here.
        </Notice>
        <div className="grid-2">
          <div className="field">
            <label htmlFor="dedup-mode">How duplicates are resolved</label>
            <select id="dedup-mode" value={form.dedup_mode} onChange={(e) => set("dedup_mode", e.target.value)}>
              <option value="">— choose —</option>
              <option value="full_row">Drop exact duplicate rows only</option>
              <option value="key_columns">Keep the newest row per business key</option>
              <option value="none">Keep every row as uploaded</option>
            </select>
          </div>
          {form.dedup_mode === "key_columns" && (
            <div className="field">
              <label htmlFor="dedup-keys">Key columns</label>
              <select
                id="dedup-keys"
                multiple
                size={6}
                value={form.dedup_keys}
                onChange={(e) => set("dedup_keys", Array.from(e.target.selectedOptions, (o) => o.value))}
              >
                {review.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                    {review.duplicates.key_column_candidates.includes(column) ? "  (likely)" : ""}
                  </option>
                ))}
              </select>
              <span className="hint">Ctrl/Cmd-click to select several. The newest row per key is kept.</span>
            </div>
          )}
        </div>

        {form.target_mode === "derive_from_subtask" && (
          <>
            <div className="section-title">SubTask review</div>
            {unmapped.length === 0 ? (
              <Notice tone="ok" title="Every SubTask is already mapped">
                All {num(subtasks.total_subtasks)} SubTask values are covered by the champion package's
                approved mappings.
              </Notice>
            ) : (
              <>
                <Notice tone="warn" title={`${unmapped.length} SubTask value(s) have no approved mapping`}>
                  The reference defaults an unmapped SubTask to Non-Voice. That would manufacture labels
                  nobody approved, so training is paused until you decide. Suggestions are shown but
                  never applied on your behalf.
                </Notice>
                <Table
                  columns={[
                    { key: "subtask", header: "SubTask" },
                    { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
                    { key: "tasks", header: "Seen with Task", render: (r) => r.tasks?.join(", ") || "—" },
                    { key: "suggested_flag", header: "Suggested", render: (r) => <Pill tone="info">{r.suggested_flag}</Pill> },
                    {
                      key: "decision",
                      header: "Your decision",
                      render: (row) => (
                        <select
                          aria-label={`Mapping for ${row.subtask}`}
                          value={mappings[row.subtask] || ""}
                          onChange={(e) => setMappings((c) => ({ ...c, [row.subtask]: e.target.value }))}
                        >
                          <option value="">— decide —</option>
                          {FLAGS.map((flag) => (
                            <option key={flag} value={flag}>{flag}</option>
                          ))}
                        </select>
                      ),
                    },
                  ]}
                  rows={unmapped}
                  rowKey={(row) => row.subtask}
                />
                <label className="checkbox" style={{ marginTop: 10 }}>
                  <input
                    type="checkbox"
                    checked={form.allow_unmapped_default}
                    onChange={(e) => set("allow_unmapped_default", e.target.checked)}
                  />
                  <span>
                    Accept the reference default (unmapped → Non-Voice) for anything left undecided.
                    This is recorded as a data-quality blocker in Stage 6.
                  </span>
                </label>
              </>
            )}
          </>
        )}

        <div className="section-title">Approval</div>
        <div className="field" style={{ maxWidth: 340 }}>
          <label htmlFor="approver">Approved by</label>
          <input
            id="approver"
            type="text"
            placeholder="Your name"
            value={form.approver}
            onChange={(e) => set("approver", e.target.value)}
          />
          <span className="hint">Stored with the decisions and written into the dataset manifest.</span>
        </div>

        {blockers.length > 0 && (
          <Notice tone="warn" title="Still to decide">
            <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
              {blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
          </Notice>
        )}

        <div className="btn-row end">
          <Action
            onClick={save}
            busy={busy}
            busyLabel="Building the snapshot…"
            disabledReason={blockers.length ? blockers[0] : undefined}
          >
            <Icon name="lock" size={15} /> Save decisions &amp; freeze snapshot
          </Action>
        </div>
        <ErrorNotice error={saveError} title="The snapshot was not built" />
      </Panel>

      {snapshot && <SnapshotPanel snapshot={snapshot} onContinue={() => go("weights")} />}
    </>
  );
}

function SnapshotPanel({ snapshot, onContinue }) {
  const counts = snapshot.row_counts || {};
  const exclusions = snapshot.exclusions || {};
  return (
    <Panel
      title="Immutable dataset snapshot"
      subtitle={`${snapshot.snapshot_id} — written once and hashed. Every later result refers to this snapshot.`}
      icon="lock"
      tone="ok"
    >
      <Metrics
        items={[
          { label: "Rows loaded", value: num(counts.loaded) },
          { label: "After labelling", value: num(counts.after_labelling) },
          { label: "Final rows", value: num(counts.final) },
          { label: "Non-Voice rate", value: pct(snapshot.target?.non_voice_rate_pct) },
          { label: "Duplicates removed", value: num(exclusions.duplicate_rows_removed) },
          { label: "Ignored by mapping", value: num(exclusions.rows_ignored_by_subtask_mapping) },
          { label: "Outside window", value: num(exclusions.rows_outside_window) },
          { label: "SHA-256", value: shortHash(snapshot.snapshot_sha256) },
        ]}
      />
      <div className="section-title">Date range and labels</div>
      <dl className="kv">
        <dt>From</dt><dd>{when(snapshot.date_range?.from)}</dd>
        <dt>To</dt><dd>{when(snapshot.date_range?.to)}</dd>
        <dt>Date column</dt><dd className="mono">{snapshot.date_column}</dd>
        <dt>Target</dt>
        <dd className="mono">
          NonVoiceFlag — 0 = Voice, 1 = Non-Voice · {JSON.stringify(snapshot.target?.distribution)}
        </dd>
        <dt>Dedup</dt>
        <dd>
          {exclusions.deduplication_mode}
          {exclusions.deduplication_keys?.length ? ` on ${exclusions.deduplication_keys.join(", ")}` : ""}
        </dd>
        <dt>Approved by</dt><dd>{snapshot.approver}</dd>
      </dl>

      {snapshot.monthly_label_trend?.length > 0 && (
        <>
          <div className="section-title">Label balance over time</div>
          <Table
            columns={[
              { key: "month", header: "Month" },
              { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
              { key: "voice", header: "Voice", className: "num", render: (r) => num(r.voice) },
              { key: "non_voice", header: "Non-Voice", className: "num", render: (r) => num(r.non_voice) },
              {
                key: "rate", header: "Non-Voice %", className: "num",
                render: (r) => pct(r.rows ? (r.non_voice / r.rows) * 100 : null, 1),
              },
            ]}
            rows={snapshot.monthly_label_trend}
            rowKey={(row) => row.month}
          />
        </>
      )}

      <div className="btn-row end">
        <Action onClick={onContinue}>
          Configure sample weights <Icon name="arrow_forward" size={15} />
        </Action>
      </div>
    </Panel>
  );
}
