import React from "react";

import { api } from "../api.js";
import {
  ActionRow, ApprovalIdentity, Badge, Btn, C, Card, CheckList, DropZone, EmptyState, ErrorNotice,
  Field, FormGrid, MIcon, MetricGrid, Notice, SectionTitle, SubHeading, Table,
} from "../nova/Components.jsx";
import { bytes, num, shortHash, when } from "../format.js";
import { NoJob } from "./TrainingDataStage.jsx";

export default function ExportStage({ job, operator }) {
  const [mlTag, setMlTag] = React.useState(null);
  const [form, setForm] = React.useState({
    column_name: "ml_tag", voice_value: 1, non_voice_value: 0, notes: "",
  });
  const [inventory, setInventory] = React.useState(null);
  const [inventoryFile, setInventoryFile] = React.useState(null);
  const [runs, setRuns] = React.useState([]);
  const [runId, setRunId] = React.useState(null);
  const [comparison, setComparison] = React.useState(null);
  const [exports, setExports] = React.useState([]);
  const [built, setBuilt] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [loadError, setLoadError] = React.useState(null);

  const refresh = React.useCallback(async () => {
    if (!job) return;
    try {
      const [tagBody, runsBody, exportsBody] = await Promise.all([
        api.mlTag(job.job_id), api.runs(job.job_id), api.exports(job.job_id),
      ]);
      setMlTag(tagBody);
      setRuns(runsBody.runs);
      setExports(exportsBody.exports);
      if (tagBody.approved_config) setForm((c) => ({ ...c, ...tagBody.approved_config }));
      setRunId((current) => current || runsBody.runs[0]?.run_id || null);
    } catch (err) {
      setLoadError(err);
    }
  }, [job]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  React.useEffect(() => {
    let live = true;
    if (!job || !runId) return () => {};
    api.comparison(job.job_id, runId)
      .then((body) => live && setComparison(body))
      .catch(() => live && setComparison(null));
    return () => {
      live = false;
    };
  }, [job, runId]);

  if (!job) return <NoJob />;
  if (loadError) {
    return (
      <Card>
        <SectionTitle>Export unavailable</SectionTitle>
        <ErrorNotice error={loadError} />
      </Card>
    );
  }

  const approval = comparison?.approval;
  const approvedCandidate = approval?.decision === "APPROVED" ? approval.selected_candidate_id : null;
  const tagApproved = Boolean(mlTag?.approved_config);

  const approveTag = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.approveMlTag(job.job_id, {
        column_name: form.column_name,
        voice_value: Number(form.voice_value),
        non_voice_value: Number(form.non_voice_value),
        approver: (operator || "").trim(),
        notes: form.notes,
      });
      await refresh();
    } catch (tagError) {
      setError(tagError);
    } finally {
      setBusy(false);
    }
  };

  const uploadInventory = async () => {
    setBusy(true);
    setError(null);
    try {
      setInventory(await api.uploadInventory(job.job_id, inventoryFile));
      setInventoryFile(null);
    } catch (uploadError) {
      setError(uploadError);
    } finally {
      setBusy(false);
    }
  };

  const build = async () => {
    setBusy(true);
    setError(null);
    setBuilt(null);
    try {
      const result = await api.buildExport(job.job_id, {
        run_id: runId, candidate_id: approvedCandidate, actor: "local-user",
      });
      setBuilt(result);
      await refresh();
    } catch (buildError) {
      setError(buildError);
    } finally {
      setBusy(false);
    }
  };

  const buildBlocker =
    !approvedCandidate ? "Approve a model for promotion in Stage 06 first."
      : !tagApproved ? "Confirm the ml_tag encoding above first."
      : undefined;

  return (
    <>
      <Card borderSize={tagApproved ? 2 : 1}>
        <SectionTitle
          sub="A business decision this application will not make on your behalf."
          right={<Badge color={tagApproved ? C.green : "#F59E0B"}
            bg={tagApproved ? "#E3F5EC" : "#FDF2DD"}>
            {tagApproved ? "APPROVED" : "BLOCKED"}
          </Badge>}
        >
          The ml_tag encoding
        </SectionTitle>

        {tagApproved ? (
          <Notice tone="ok" title={`Approved by ${mlTag.approved_by}`}>
            <code>{mlTag.approved_config.column_name}</code>: {mlTag.approved_config.voice_value} =
            Voice, {mlTag.approved_config.non_voice_value} = Non-Voice. This is written into{" "}
            <code>scoring/ml_tag_config.json</code>, and the packaged runtime refuses to emit the
            column under any other convention.
          </Notice>
        ) : (
          <>
            <Notice tone="warn" title="Why this is blocked by default">
              {mlTag?.decision_required}
            </Notice>
            {mlTag?.candidate_conventions?.map((convention) => (
              <label key={convention.label} style={{
                display: "flex", gap: 9, alignItems: "flex-start", fontSize: 12.5,
                marginBottom: 10, cursor: "pointer",
              }}>
                <input
                  type="radio" name="convention"
                  checked={Number(form.voice_value) === convention.voice_value
                    && Number(form.non_voice_value) === convention.non_voice_value}
                  onChange={() => setForm((c) => ({
                    ...c, voice_value: convention.voice_value, non_voice_value: convention.non_voice_value,
                  }))}
                  style={{ marginTop: 2, accentColor: C.green, width: "auto" }}
                />
                <span>
                  <strong style={{ color: "var(--nova-header-text)" }}>{convention.label}</strong>
                  <div style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 2 }}>
                    {convention.voice_value} = Voice, {convention.non_voice_value} = Non-Voice ·{" "}
                    {convention.note}
                  </div>
                </span>
              </label>
            ))}

            <FormGrid min={190}>
              <Field label="Appended column name" htmlFor="tag-column">
                <input id="tag-column" type="text" value={form.column_name}
                  onChange={(e) => setForm((c) => ({ ...c, column_name: e.target.value }))} />
              </Field>
              <Field label="Notes" htmlFor="tag-notes">
                <input id="tag-notes" type="text" value={form.notes}
                  onChange={(e) => setForm((c) => ({ ...c, notes: e.target.value }))} />
              </Field>
            </FormGrid>

            <ApprovalIdentity operator={operator} what="This ml_tag encoding" />

            <ActionRow>
              <Btn onClick={approveTag} busy={busy} busyLabel="Saving…"
                disabledReason={!operator?.trim() ? "Enter your name in the header first." : undefined}>
                <MIcon name="how_to_reg" size={15} /> Confirm this encoding
              </Btn>
            </ActionRow>
          </>
        )}
      </Card>

      <Card>
        <SectionTitle sub="A de-identified inventory extract, used to prove the built package scores correctly.">
          Scoring compatibility sample
        </SectionTitle>
        <DropZone
          accept=".parquet,.csv,.xlsx,.xls"
          label="Choose a de-identified inventory sample"
          hint="Parquet, CSV or Excel · optional but strongly recommended"
          file={inventoryFile}
          onFile={setInventoryFile}
          disabled={busy}
        />
        <ActionRow note="Without a sample, the check falls back to rows taken from the snapshot. The validation report always records which was used.">
          <Btn variant="secondary" onClick={uploadInventory} busy={busy} busyLabel="Reading…"
            disabledReason={!inventoryFile ? "Choose a file first." : undefined}>
            <MIcon name="upload" size={15} /> Upload sample
          </Btn>
        </ActionRow>
        {inventory && (
          <Notice tone="ok" title={`${num(inventory.rows)} rows, ${inventory.columns} columns`}>
            The package must return exactly these rows and columns plus one appended{" "}
            <code>{form.column_name}</code>.
          </Notice>
        )}
        <ErrorNotice error={error && !error.body?.detail?.validation ? error : null} />
      </Card>

      <Card>
        <SectionTitle
          sub="The ZIP is assembled, then unzipped and scored exactly as the deployment does. It is only published if that passes."
          right={runs.length > 1 ? (
            <select value={runId || ""} onChange={(e) => setRunId(e.target.value)}
              aria-label="Run to export" style={{ width: "auto", minWidth: 200 }}>
              {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}
            </select>
          ) : null}
        >
          Build the package
        </SectionTitle>

        {approvedCandidate ? (
          <MetricGrid
            compact
            min={170}
            items={[
              { label: "Approved candidate", value: approvedCandidate, color: C.green },
              { label: "Approved by", value: approval.approver, sub: when(approval.approved_at) },
              { label: "Threshold", value: approval.selected_threshold, color: C.indigo },
            ]}
          />
        ) : (
          <Notice tone="warn" title="No approved model for this run">
            Stage 06 must record a typed promotion approval before a package can be built.
          </Notice>
        )}

        <ActionRow>
          <Btn onClick={build} busy={busy} busyLabel="Building and validating…" disabledReason={buildBlocker}>
            <MIcon name="build" size={15} /> Build &amp; validate package
          </Btn>
        </ActionRow>

        {error?.body?.detail?.validation && (
          <>
            <Notice tone="bad" title="Validation failed — the ZIP was discarded, not published">
              {error.body.detail.message}
            </Notice>
            <CheckList checks={error.body.detail.validation.checks} />
          </>
        )}

        {built && (
          <>
            <Notice tone="ok" title={`${built.zip_name} built and validated`}>
              Version {built.version} · {bytes(built.size_bytes)} · sha256{" "}
              {shortHash(built.zip_sha256, 24)}
            </Notice>
            <SubHeading>Package validation record</SubHeading>
            <CheckList checks={built.validation.checks} />
            {built.validation.prediction_agreement?.rows > 0 && (
              <MetricGrid
                compact
                min={160}
                items={[
                  { label: "Rows scored", value: num(built.validation.prediction_agreement.rows) },
                  {
                    label: "Identical to 4dp",
                    value: `${built.validation.prediction_agreement.agreement_pct}%`,
                    color: C.green,
                  },
                  {
                    label: "Max difference",
                    value: built.validation.prediction_agreement.max_absolute_difference,
                  },
                ]}
              />
            )}
            <p style={{ fontSize: 11.5, color: "var(--nova-grey-dim)", marginTop: 10 }}>
              Inventory used: {built.validation.inventory_source}. Reference predictions:{" "}
              {built.validation.expected_prediction_source}.
            </p>
          </>
        )}
      </Card>

      <Card>
        <SectionTitle sub="Every version is retained. This application never deploys anything by itself.">
          Versions and rollback
        </SectionTitle>
        {exports.length === 0 ? (
          <EmptyState icon="archive">No package has been published for this job yet.</EmptyState>
        ) : (
          <Table
            columns={[
              { key: "version", header: "Version", className: "num" },
              { key: "zip_name", header: "File" },
              { key: "model_id", header: "Model" },
              { key: "created_at", header: "Built", render: (r) => when(r.created_at) },
              { key: "approver", header: "Approved by", render: (r) => r.approval?.approver || "—" },
              { key: "zip_sha256", header: "SHA-256", render: (r) => shortHash(r.zip_sha256) },
              {
                key: "download", header: "",
                render: (row) => (row.exists ? (
                  <a href={api.downloadUrl(row.export_id)} style={{
                    display: "inline-flex", alignItems: "center", gap: 5, color: C.indigo,
                    fontWeight: 700, textDecoration: "none", fontSize: 12,
                  }}>
                    <MIcon name="download" size={14} /> Download
                  </a>
                ) : (
                  <span style={{ color: "var(--nova-grey-dim)" }}>file missing</span>
                )),
              },
            ]}
            rows={exports}
            rowKey={(row) => row.export_id}
          />
        )}
      </Card>
    </>
  );
}
