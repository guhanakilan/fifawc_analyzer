import React from "react";

import { api } from "../api.js";
import {
  ActionRow, ApprovalIdentity, Badge, Btn, C, Card, CheckRow, EmptyState, ErrorNotice, Field,
  FormGrid, MIcon, MetricGrid, Notice, Pill, SectionTitle, SubHeading, Table,
} from "../nova/Components.jsx";
import { num, pct, shortHash, when } from "../format.js";
import { NoJob } from "./TrainingDataStage.jsx";

const FLAGS = ["Voice", "Non-Voice", "Keyword", "Ignore"];

export default function ReadinessStage({ job, mark, go, operator }) {
  const [review, setReview] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [saveError, setSaveError] = React.useState(null);
  const [snapshot, setSnapshot] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [form, setForm] = React.useState(null);
  const [mappings, setMappings] = React.useState({});
  const [advice, setAdvice] = React.useState(null);
  const [checks, setChecks] = React.useState(null);
  const [applied, setApplied] = React.useState({});

  React.useEffect(() => {
    let live = true;
    if (!job) return () => {};
    setLoading(true);
    Promise.all([
      api.review(job.job_id),
      api.snapshot(job.job_id).catch(() => null),
      api.recommendations(job.job_id).catch(() => null),
      api.interventions(job.job_id).catch(() => null),
    ])
      .then(([reviewBody, snapshotBody, adviceBody, checksBody]) => {
        setAdvice(adviceBody);
        setChecks(checksBody);
        if (!live) return;
        setReview(reviewBody);
        setSnapshot(snapshotBody);
        if (snapshotBody) mark("snapshotId", snapshotBody.snapshot_id);
        const saved = reviewBody.saved_decisions;
        setForm({
          date_column: saved?.date_column || reviewBody.detected_date_column || "",
          target_mode: saved?.target_mode
            || (reviewBody.champion_has_subtask_mappings ? "derive_from_subtask" : "existing"),
          target_column: saved?.target_column || reviewBody.detected_target_column || "",
          voice_values: (saved?.target_encoding?.voice_values || ["0"]).join(","),
          non_voice_values: (saved?.target_encoding?.non_voice_values || ["1"]).join(","),
          dedup_mode: saved?.dedup_mode || "",
          dedup_keys: saved?.dedup_keys || [],
          historical_window_days: saved?.historical_window_days ?? "",
          date_from: saved?.date_from || "",
          date_to: saved?.date_to || "",
          allow_unmapped_default: Boolean(saved?.allow_unmapped_default),
          acknowledge_model_output_target: Boolean(saved?.acknowledge_model_output_target),
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

  if (!job) return <NoJob />;
  if (loading || !form) {
    return (
      <Card>
        <SectionTitle sub="Reading the uploaded data…">Loading the dataset review</SectionTitle>
      </Card>
    );
  }
  if (error) {
    return (
      <Card>
        <SectionTitle>Readiness review unavailable</SectionTitle>
        <ErrorNotice error={error} title="Could not build the review" />
      </Card>
    );
  }

  const subtasks = review.subtask_review || {};
  const unmapped = subtasks.unmapped || [];
  const unresolved = unmapped.filter((row) => !mappings[row.subtask]);

  const blockers = [];
  if (!form.date_column) blockers.push("Choose the date column that drives the temporal split.");
  if (form.target_mode === "existing" && !form.target_column) {
    blockers.push("Choose the label column, or switch to deriving labels from SubTask mappings.");
  }
  const modelOutputs = review.model_output_columns_present || [];
  const targetIsModelOutput =
    form.target_mode === "existing" && modelOutputs.includes(form.target_column);
  if (targetIsModelOutput && !form.acknowledge_model_output_target) {
    blockers.push(
      `'${form.target_column}' is written by a scoring run. Confirm it has since been human-verified, or choose another label source.`,
    );
  }
  if (!form.dedup_mode) blockers.push("Choose how duplicate rows are resolved.");
  if (form.dedup_mode === "key_columns" && form.dedup_keys.length === 0) {
    blockers.push("Select at least one deduplication key column.");
  }
  if (form.target_mode === "derive_from_subtask" && unresolved.length && !form.allow_unmapped_default) {
    blockers.push(`Map ${unresolved.length} new SubTask value(s), or explicitly accept the Non-Voice default.`);
  }
  if (!operator?.trim()) blockers.push("Enter your name in the header before approving.");

  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  const save = async () => {
    setBusy(true);
    setSaveError(null);
    try {
      const combined = [
        ...(subtasks.known || []).map((row) => ({ name: row.subtask, flag: row.flag })),
        ...unmapped.filter((row) => mappings[row.subtask])
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
        acknowledge_model_output_target: form.acknowledge_model_output_target,
        historical_window_days: form.historical_window_days === ""
          ? null : Number(form.historical_window_days),
        date_from: form.date_from || null,
        date_to: form.date_to || null,
        approver: operator.trim(),
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
      <Card>
        <SectionTitle sub={`${num(review.row_count)} rows across ${review.columns.length} columns.`}>
          What the data looks like
        </SectionTitle>
        <MetricGrid
          compact
          min={140}
          items={[
            { label: "Rows", value: num(review.row_count) },
            { label: "Columns", value: num(review.columns.length) },
            {
              label: "Exact duplicates", value: num(review.duplicates.full_row_duplicates),
              color: review.duplicates.full_row_duplicates ? "#F59E0B" : C.navy,
            },
            { label: "SubTasks", value: num(subtasks.total_subtasks) },
            {
              label: "Unmapped", value: num(unmapped.length),
              color: unmapped.length ? "#F59E0B" : C.green,
            },
            {
              label: "Missing required",
              value: num(review.column_lineage?.missing_required_count ?? 0),
              color: review.column_lineage?.missing_required_count ? "#F59E0B" : C.green,
              sub: review.column_lineage?.missing_required_count
                ? review.column_lineage.missing_required.slice(0, 3).join(", ") : undefined,
            },
          ]}
        />
        <Notice tone="info" title="Target encoding, confirmed from the reference implementation">
          <code>NonVoiceFlag</code> is <strong>0 = Voice</strong>, <strong>1 = Non-Voice</strong>.
          Read from <code>routers/flag.py::run_flag</code> in the NoVA source, not assumed.
        </Notice>
      </Card>

      {checks && <InterventionChecks checks={checks} />}

      {advice && (
        <Recommendations
          advice={advice}
          form={form}
          applied={applied}
          onApply={(field, value) => {
            set(field, value);
            setApplied((current) => ({ ...current, [field]: true }));
          }}
        />
      )}

      {review.column_lineage && <ColumnLineage lineage={review.column_lineage} />}

      <Card borderSize={2}>
        <SectionTitle sub="Each of these changes what gets trained, so each needs a person behind it.">
          Decisions this application will not make for you
        </SectionTitle>

        <FormGrid>
          <Field label="Date column (drives the temporal split and recency)" htmlFor="date-column">
            <select id="date-column" value={form.date_column} onChange={(e) => set("date_column", e.target.value)}>
              <option value="">— choose a column —</option>
              {review.columns.map((column) => (
                <option key={column} value={column}>
                  {column}{review.date_candidates.includes(column) ? "  (detected)" : ""}
                </option>
              ))}
            </select>
          </Field>
        </FormGrid>

        <TrainingWindow
          jobId={job.job_id}
          form={form}
          set={set}
          span={review.date_spans?.[form.date_column]}
        />

        <SubHeading>Where the label comes from</SubHeading>
        <FormGrid>
          <Field label="Label source" htmlFor="target-mode">
            <select id="target-mode" value={form.target_mode} onChange={(e) => set("target_mode", e.target.value)}>
              <option value="derive_from_subtask">Derive from SubTask mappings (the reference's own rules)</option>
              <option value="existing">Use a label column already in the data</option>
            </select>
          </Field>
          {form.target_mode === "existing" ? (
            <Field label="Label column" htmlFor="target-column">
              <select id="target-column" value={form.target_column} onChange={(e) => set("target_column", e.target.value)}>
                <option value="">— choose a column —</option>
                {review.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                    {(review.model_output_columns_present || []).includes(column)
                      ? "  ⚠ model output"
                      : review.target_candidates.includes(column) ? "  (detected)" : ""}
                  </option>
                ))}
              </select>
            </Field>
          ) : <div />}
        </FormGrid>

        {targetIsModelOutput && (
          <>
            <Notice tone="bad" title="This column is model output, not verified ground truth">
              {review.model_output_warning} Training on <code>{form.target_column}</code> would teach
              the challenger to agree with the champion rather than with reality, and every metric
              that follows would measure agreement instead of accuracy.
            </Notice>
            <CheckRow
              checked={form.acknowledge_model_output_target}
              onChange={(value) => set("acknowledge_model_output_target", value)}
            >
              I confirm these rows were reviewed and corrected by a person after scoring, so this
              column is verified ground truth rather than raw model output.
            </CheckRow>
          </>
        )}

        {form.target_mode === "existing" && (
          <FormGrid style={{ marginTop: 12 }}>
            <Field
              label="Values that mean Voice" htmlFor="voice-values"
              hint="Comma-separated. Any value outside these two lists stops the snapshot rather than being guessed."
            >
              <input id="voice-values" type="text" value={form.voice_values}
                onChange={(e) => set("voice_values", e.target.value)} />
            </Field>
            <Field label="Values that mean Non-Voice" htmlFor="nv-values">
              <input id="nv-values" type="text" value={form.non_voice_values}
                onChange={(e) => set("non_voice_values", e.target.value)} />
            </Field>
          </FormGrid>
        )}

        <SubHeading>Deduplication</SubHeading>
        <Notice tone="info" title="A business key is never inferred">
          {num(review.duplicates.full_row_duplicates)} exact duplicate rows are present. Whether two
          rows sharing an account number are the same work item is a business rule, so choose it here.
        </Notice>
        <FormGrid>
          <Field label="How duplicates are resolved" htmlFor="dedup-mode">
            <select id="dedup-mode" value={form.dedup_mode} onChange={(e) => set("dedup_mode", e.target.value)}>
              <option value="">— choose —</option>
              <option value="full_row">Drop exact duplicate rows only</option>
              <option value="key_columns">Keep the newest row per business key</option>
              <option value="none">Keep every row as uploaded</option>
            </select>
          </Field>
          {form.dedup_mode === "key_columns" && (
            <Field
              label="Key columns" htmlFor="dedup-keys"
              hint="Ctrl/Cmd-click to select several. The newest row per key is kept, so a correction supersedes what it corrects."
            >
              <select
                id="dedup-keys" multiple size={6} value={form.dedup_keys}
                onChange={(e) => set("dedup_keys", Array.from(e.target.selectedOptions, (o) => o.value))}
              >
                {review.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}{review.duplicates.key_column_candidates.includes(column) ? "  (likely)" : ""}
                  </option>
                ))}
              </select>
            </Field>
          )}
        </FormGrid>

        {form.target_mode === "derive_from_subtask" && (
          <>
            <SubHeading>SubTask review</SubHeading>
            {unmapped.length === 0 ? (
              <Notice tone="ok" title="Every SubTask is already mapped">
                All {num(subtasks.total_subtasks)} SubTask values are covered by the champion
                package's approved mappings.
              </Notice>
            ) : (
              <>
                <Notice tone="warn" title={`${unmapped.length} SubTask value(s) have no approved mapping`}>
                  The reference defaults an unmapped SubTask to Non-Voice. That would manufacture
                  labels nobody approved, so training is paused until you decide. Suggestions are
                  shown but never applied on your behalf.
                </Notice>
                <Table
                  columns={[
                    { key: "subtask", header: "SubTask" },
                    { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
                    { key: "tasks", header: "Seen with Task", render: (r) => r.tasks?.join(", ") || "—" },
                    {
                      key: "suggested_flag", header: "Suggested",
                      render: (r) => <Pill tone="info">{r.suggested_flag}</Pill>,
                    },
                    {
                      key: "decision", header: "Your decision",
                      render: (row) => (
                        <select
                          aria-label={`Mapping for ${row.subtask}`}
                          value={mappings[row.subtask] || ""}
                          onChange={(e) => setMappings((c) => ({ ...c, [row.subtask]: e.target.value }))}
                          style={{ minWidth: 120 }}
                        >
                          <option value="">— decide —</option>
                          {FLAGS.map((flag) => <option key={flag} value={flag}>{flag}</option>)}
                        </select>
                      ),
                    },
                  ]}
                  rows={unmapped}
                  rowKey={(row) => row.subtask}
                />
                <div style={{ marginTop: 12 }}>
                  <CheckRow
                    checked={form.allow_unmapped_default}
                    onChange={(value) => set("allow_unmapped_default", value)}
                  >
                    Accept the reference default (unmapped → Non-Voice) for anything left undecided.
                    This is recorded as a data-quality blocker in Stage 06.
                  </CheckRow>
                </div>
              </>
            )}
          </>
        )}

        <SubHeading>Approval</SubHeading>
        <ApprovalIdentity operator={operator} what="These readiness decisions" />

        {blockers.length > 0 && (
          <Notice tone="warn" title="Still to decide">
            <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
              {blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
            </ul>
          </Notice>
        )}

        <ActionRow>
          <Btn
            onClick={save} busy={busy} busyLabel="Building the snapshot…"
            disabledReason={blockers.length ? blockers[0] : undefined}
          >
            <MIcon name="lock" size={15} /> Save decisions &amp; freeze snapshot
          </Btn>
        </ActionRow>
        <ErrorNotice error={saveError} title="The snapshot was not built" />
      </Card>

      {snapshot && <SnapshotCard snapshot={snapshot} onContinue={() => go("weights")} />}
    </>
  );
}

function SnapshotCard({ snapshot, onContinue }) {
  const counts = snapshot.row_counts || {};
  const exclusions = snapshot.exclusions || {};
  return (
    <Card borderSize={2}>
      <SectionTitle
        sub={`${snapshot.snapshot_id} — written once and hashed. Every later result refers to this snapshot.`}
        right={<Badge color={C.green} bg="#E3F5EC">FROZEN</Badge>}
      >
        Immutable dataset snapshot
      </SectionTitle>

      <MetricGrid
        compact
        min={140}
        items={[
          { label: "Rows loaded", value: num(counts.loaded) },
          { label: "After labelling", value: num(counts.after_labelling) },
          { label: "Final rows", value: num(counts.final), color: C.green },
          { label: "Non-Voice rate", value: pct(snapshot.target?.non_voice_rate_pct), color: C.indigo },
          { label: "Dupes removed", value: num(exclusions.duplicate_rows_removed) },
          { label: "Ignored", value: num(exclusions.rows_ignored_by_subtask_mapping) },
        ]}
      />

      <SubHeading>Provenance</SubHeading>
      <dl style={{ display: "grid", gridTemplateColumns: "max-content 1fr", gap: "5px 16px", fontSize: 12, margin: 0 }}>
        <Dt>SHA-256</Dt><Dd>{snapshot.snapshot_sha256}</Dd>
        <Dt>Date range</Dt><Dd>{when(snapshot.date_range?.from)} → {when(snapshot.date_range?.to)}</Dd>
        <Dt>Date column</Dt><Dd><code>{snapshot.date_column}</code></Dd>
        <Dt>Target</Dt>
        <Dd>
          <code>NonVoiceFlag</code> — 0 = Voice, 1 = Non-Voice ·{" "}
          {JSON.stringify(snapshot.target?.distribution)}
        </Dd>
        <Dt>Dedup</Dt>
        <Dd>
          {exclusions.deduplication_mode}
          {exclusions.deduplication_keys?.length ? ` on ${exclusions.deduplication_keys.join(", ")}` : ""}
        </Dd>
        <Dt>Approved by</Dt><Dd>{snapshot.approver}</Dd>
      </dl>

      {snapshot.monthly_label_trend?.length > 0 && (
        <>
          <SubHeading>Label balance over time</SubHeading>
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

      <ActionRow>
        <Btn onClick={onContinue}>
          Configure sample weights <MIcon name="arrow_forward" size={15} />
        </Btn>
      </ActionRow>
    </Card>
  );
}

const Dt = ({ children }) => (
  <dt style={{ color: "var(--nova-grey-dim)", fontFamily: "var(--nova-font-mono)", fontSize: 11 }}>{children}</dt>
);
const Dd = ({ children }) => <dd style={{ margin: 0, overflowWrap: "anywhere" }}>{children}</dd>;

/* ── Column lineage ────────────────────────────────────────────────────────
 *
 * A champion package records its columns four times over, each list meaning
 * something different. Showing one of them answers one question; showing the
 * chain answers the one people actually ask — where did this column go?
 */

const STAGE_META = {
  selected: { tone: "ok", label: "Model feature" },
  derived: { tone: "info", label: "Derived" },
  feeds_derived: { tone: "info", label: "Feeds derived" },
  dropped_during_build: { tone: "warn", label: "Dropped during build" },
  dropped_at_matching: { tone: "muted", label: "Dropped at matching" },
  excluded_at_mapping: { tone: "muted", label: "Excluded at mapping" },
  unknown: { tone: "muted", label: "Unclassified" },
};

function ColumnLineage({ lineage }) {
  const [showAll, setShowAll] = React.useState(false);
  const layers = lineage.layers || {};
  const rows = showAll
    ? lineage.columns
    : lineage.columns.filter((c) => c.required || c.stage === "dropped_during_build");

  return (
    <Card>
      <SectionTitle
        sub="Every column traced through the four lists the champion package carries, so you can see exactly where each one left the pipeline."
        right={
          <Btn variant="ghost" small onClick={() => setShowAll((v) => !v)}>
            <MIcon name={showAll ? "filter_list" : "list"} size={14} />{" "}
            {showAll ? "Only what matters" : `Show all ${lineage.columns.length}`}
          </Btn>
        }
      >
        Column lineage
      </SectionTitle>

      <MetricGrid
        compact
        min={130}
        items={[
          { label: "Mapped", value: num(layers.mapped) },
          { label: "Matched", value: num(layers.matched) },
          { label: "Selected", value: num(layers.selected) },
          { label: "Derived", value: num(layers.derived) },
          {
            label: "Fitted",
            value: layers.fitted_available ? num(layers.fitted) : "—",
            sub: layers.fitted_available ? undefined : "needs model load",
          },
        ]}
      />

      {!layers.fitted_available && (
        <Notice tone="info" title="The fitted layer is not loaded">
          The model's real input vector lives inside the pickle, which is only read in the
          trust-gated compatibility step. Run that in Stage 1 to add the fourth layer here.
        </Notice>
      )}

      {lineage.missing_required_count > 0 && (
        <Notice tone="bad" title={`${lineage.missing_required_count} required column(s) are not in your data`}>
          {lineage.missing_required.join(", ")} — training cannot reproduce the champion's
          features without these.
        </Notice>
      )}

      {lineage.removed_during_build_count > 0 && (
        <Notice tone="warn" title={`${lineage.removed_during_build_count} column(s) were dropped when the champion was built`}>
          {lineage.removed_during_build.join(", ")}. These were available at the Stage 03 filter
          but are absent from the final feature list. The export does not record whether they
          were excluded during EDA or simply not selected, so this cannot say which.
        </Notice>
      )}

      {lineage.unexpected_column_count > 0 && (
        <Notice tone="info" title={`${lineage.unexpected_column_count} column(s) in your data are new`}>
          {lineage.unexpected_columns.join(", ")}. The champion never saw these; they are ignored
          unless the feature list changes.
        </Notice>
      )}

      <Table
        columns={[
          { key: "column", header: "Column" },
          {
            key: "stage",
            header: "Where it ends up",
            render: (row) => {
              const meta = STAGE_META[row.stage] || STAGE_META.unknown;
              return <Pill tone={meta.tone}>{meta.label}</Pill>;
            },
          },
          {
            key: "present_in_upload",
            header: "In your data",
            render: (row) =>
              row.derived ? (
                <span style={{ color: "var(--nova-grey-dim)" }}>computed</span>
              ) : (
                <Pill tone={row.present_in_upload ? "ok" : row.required ? "bad" : "muted"}>
                  {row.present_in_upload ? "yes" : "no"}
                </Pill>
              ),
          },
          {
            key: "fitted_feature_count",
            header: "Fitted features",
            render: (row) => (layers.fitted_available ? row.fitted_feature_count : "—"),
          },
          { key: "note", header: "Why" },
        ]}
        rows={rows}
        rowKey={(row) => row.column}
        empty="No columns to trace."
      />
    </Card>
  );
}

/* ── Intervention checks ───────────────────────────────────────────────────
 *
 * Rules, not hard-coded logic: every threshold here comes from the job's rule
 * set and can be changed without touching this file. A finding states what was
 * measured against what threshold, so it can be argued with on the facts.
 */

function InterventionChecks({ checks }) {
  if (checks.clear) {
    return (
      <Notice tone="ok" title="No readiness rules are firing on this data">
        Every configured check passed. The measured values are still recorded with your
        decisions, so this run can be explained later.
      </Notice>
    );
  }

  const blocks = checks.findings.filter((f) => f.action === "block");
  const warns = checks.findings.filter((f) => f.action === "warn");

  return (
    <Card borderSize={blocks.length ? 2 : 1}>
      <SectionTitle
        sub="Configured rules evaluated against this upload. A block needs a decision from you; a warning is visible but passable."
        right={
          <>
            {blocks.length > 0 && <Pill tone="bad">{blocks.length} blocking</Pill>}
            {warns.length > 0 && <Pill tone="warn">{warns.length} warning</Pill>}
          </>
        }
      >
        Readiness checks
      </SectionTitle>

      {checks.findings.map((finding) => (
        <Notice
          key={finding.id}
          tone={finding.action === "block" ? "bad" : "warn"}
          title={finding.label}
        >
          <div>{finding.why}</div>
          <div style={{ marginTop: 6, fontFamily: "var(--nova-font-mono)", fontSize: 11.5 }}>
            measured {String(finding.measured)}
            {finding.threshold !== null && finding.threshold !== undefined
              ? ` · threshold ${finding.threshold}${finding.unit ? ` ${finding.unit}` : ""}`
              : ""}
          </div>
        </Notice>
      ))}
    </Card>
  );
}

/* ── Recommendations ───────────────────────────────────────────────────────
 *
 * Advisory only. Nothing is applied until you press Use it, and the decisions
 * form below is still the thing that gets saved.
 */

const CONFIDENCE_TONE = { high: "ok", medium: "warn", low: "muted" };

function describeValue(value) {
  if (value === null || value === undefined || value === "") return "no recommendation";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  return String(value);
}

function currentlyMatches(form, field, value) {
  const current = form?.[field];
  if (Array.isArray(value)) {
    return Array.isArray(current) && current.join(",") === value.join(",");
  }
  return String(current ?? "") === String(value ?? "");
}

function Recommendations({ advice, form, applied, onApply }) {
  const models = advice.local_models || {};
  const layers = [models.embeddings, models.generation].filter(Boolean);

  return (
    <Card>
      <SectionTitle sub={advice.note}>Suggestions</SectionTitle>

      {advice.suggestions.length === 0 && (
        <EmptyState icon="inbox">
          Nothing to suggest for this data yet.
        </EmptyState>
      )}

      {advice.suggestions.map((suggestion) => {
        const matches = currentlyMatches(form, suggestion.field, suggestion.value);
        const actionable = suggestion.value !== null && suggestion.value !== undefined;
        return (
          <div
            key={`${suggestion.field}-${suggestion.rule}`}
            style={{
              border: "1px solid var(--nova-input-border)",
              borderRadius: 8,
              padding: "11px 13px",
              marginBottom: 9,
              background: "var(--nova-input-bg)",
            }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontWeight: 700, fontSize: 12.5 }}>{suggestion.field}</span>
              <Badge small>{describeValue(suggestion.value)}</Badge>
              <Pill tone={CONFIDENCE_TONE[suggestion.confidence] || "muted"}>
                {suggestion.confidence} confidence
              </Pill>
              <Pill tone="muted">{suggestion.source}</Pill>
              <span style={{ flex: 1 }} />
              {actionable && (
                matches ? (
                  <Pill tone="ok">in use</Pill>
                ) : (
                  <Btn variant="secondary" small onClick={() => onApply(suggestion.field, suggestion.value)}>
                    <MIcon name="check" size={14} /> Use it
                  </Btn>
                )
              )}
            </div>
            <div style={{ marginTop: 6, fontSize: 12, color: "var(--nova-grey-dim)" }}>
              {suggestion.why}
            </div>
            <div style={{ marginTop: 4, fontSize: 11, fontFamily: "var(--nova-font-mono)", color: "var(--nova-grey-dim)" }}>
              rule: {suggestion.rule}
              {applied[suggestion.field] && !matches ? " · overridden since" : ""}
            </div>
          </div>
        );
      })}

      {advice.subtask_suggestions?.length > 0 && (
        <>
          <SubHeading>Unmapped SubTasks</SubHeading>
          <Table
            columns={[
              { key: "subtask", header: "SubTask" },
              { key: "rows", header: "Rows", className: "num" },
              { key: "suggested_flag", header: "Suggested" },
              {
                key: "source",
                header: "From",
                render: (row) => <Pill tone={row.source === "embeddings" ? "info" : "muted"}>{row.source}</Pill>,
              },
              { key: "why", header: "Why" },
            ]}
            rows={advice.subtask_suggestions}
            rowKey={(row) => row.subtask}
            empty="No unmapped SubTasks."
          />
        </>
      )}

      <div style={{ marginTop: 12, fontSize: 11.5, color: "var(--nova-grey-dim)" }}>
        {layers.map((layer) => (
          <div key={layer.name}>
            <strong>{layer.name}</strong>: {layer.available ? "available" : "not installed"}
            {layer.available ? "" : ` — ${layer.detail}`}
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ── Training window ───────────────────────────────────────────────────────
 *
 * An explicit from/to range over the approved date column. The preview is the
 * point: narrowing the window changes what the model is trained on, and that
 * consequence should be visible while choosing it rather than discovered after
 * the dataset is frozen.
 */

function TrainingWindow({ jobId, form, set, span }) {
  const [preview, setPreview] = React.useState(null);
  const [previewError, setPreviewError] = React.useState(null);
  const [checking, setChecking] = React.useState(false);

  const { date_column: dateColumn, date_from: from, date_to: to } = form;

  React.useEffect(() => {
    if (!dateColumn) {
      setPreview(null);
      return () => {};
    }
    let live = true;
    setChecking(true);
    setPreviewError(null);
    const timer = setTimeout(() => {
      api
        .windowPreview(jobId, {
          date_column: dateColumn,
          date_from: from || null,
          date_to: to || null,
        })
        .then((body) => live && setPreview(body))
        .catch((error) => live && setPreviewError(error))
        .finally(() => live && setChecking(false));
    }, 350);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [jobId, dateColumn, from, to]);

  const narrowed = preview && preview.rows_excluded > 0;

  return (
    <>
      <SubHeading>Training window</SubHeading>
      {!dateColumn && (
        <Notice tone="info" title="Choose a date column first">
          The window is applied to whichever date column you approve above.
        </Notice>
      )}

      {dateColumn && (
        <>
          <FormGrid>
            <Field
              label="From"
              htmlFor="window-from"
              hint={span ? `Earliest row: ${span.from}` : "Earliest row in your data"}
            >
              <input
                id="window-from" type="date"
                min={span?.from} max={span?.to}
                value={form.date_from}
                onChange={(e) => set("date_from", e.target.value)}
              />
            </Field>
            <Field
              label="To"
              htmlFor="window-to"
              hint={span ? `Latest row: ${span.to}` : "Latest row in your data"}
            >
              <input
                id="window-to" type="date"
                min={span?.from} max={span?.to}
                value={form.date_to}
                onChange={(e) => set("date_to", e.target.value)}
              />
            </Field>
          </FormGrid>

          <ActionRow note="Leave both empty to train on every row.">
            <Btn
              variant="ghost" small
              disabled={!form.date_from && !form.date_to}
              onClick={() => {
                set("date_from", "");
                set("date_to", "");
              }}
            >
              <MIcon name="restart_alt" size={14} /> Use everything
            </Btn>
            {span && (
              <Btn
                variant="ghost" small
                onClick={() => {
                  set("date_from", span.from);
                  set("date_to", span.to);
                }}
              >
                <MIcon name="history" size={14} /> Full span
              </Btn>
            )}
          </ActionRow>

          <ErrorNotice error={previewError} title="That window could not be previewed" />

          {preview && !previewError && (
            <MetricGrid
              compact
              min={150}
              items={[
                {
                  label: "Rows selected",
                  value: num(preview.rows_selected),
                  color: preview.rows_selected === 0 ? "#EF4444" : C.navy,
                  sub: narrowed ? `${num(preview.rows_excluded)} excluded` : "all rows",
                },
                { label: "From", value: preview.actual_from || "—" },
                { label: "To", value: preview.actual_to || "—" },
                {
                  label: "Voice",
                  value: preview.class_balance ? pct(preview.class_balance.voice_pct, 1) : "—",
                  sub: preview.class_balance ? undefined : "needs mapped SubTasks",
                },
                {
                  label: "Non-Voice",
                  value: preview.class_balance ? pct(preview.class_balance.non_voice_pct, 1) : "—",
                },
              ]}
            />
          )}

          {preview?.rows_selected === 0 && (
            <Notice tone="bad" title="This window contains no rows">
              Widen the range, or check that the date column you approved is the one this
              window refers to. The snapshot cannot be built from an empty selection.
            </Notice>
          )}

          {checking && (
            <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)" }}>Checking window…</div>
          )}
        </>
      )}
    </>
  );
}
