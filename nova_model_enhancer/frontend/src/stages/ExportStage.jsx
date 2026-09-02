import React from "react";

import { api } from "../api.js";
import {
  Action, CheckList, DropZone, Empty, ErrorNotice, Icon, Metrics, Notice, Panel, Pill, Table,
} from "../components/Ui.jsx";
import { bytes, num, shortHash, when } from "../format.js";

export default function ExportStage({ job }) {
  const [mlTag, setMlTag] = React.useState(null);
  const [form, setForm] = React.useState({
    column_name: "ml_tag", voice_value: 1, non_voice_value: 0, approver: "", notes: "",
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
      const first = runsBody.runs[0]?.run_id || null;
      setRunId((current) => current || first);
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
    api
      .comparison(job.job_id, runId)
      .then((body) => live && setComparison(body))
      .catch(() => live && setComparison(null));
    return () => {
      live = false;
    };
  }, [job, runId]);

  if (!job) {
    return (
      <Panel title="No active job" icon="info">
        <Notice tone="warn" title="Start at Stage 01">Upload a champion package first.</Notice>
      </Panel>
    );
  }
  if (loadError) {
    return (
      <Panel title="Export unavailable" icon="report" tone="bad">
        <ErrorNotice error={loadError} />
      </Panel>
    );
  }

  const approval = comparison?.approval;
  const approvedCandidate = approval?.decision === "APPROVED" ? approval.selected_candidate_id : null;

  const approveTag = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.approveMlTag(job.job_id, {
        column_name: form.column_name,
        voice_value: Number(form.voice_value),
        non_voice_value: Number(form.non_voice_value),
        approver: form.approver.trim(),
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
      : !mlTag?.approved_config ? "Confirm the ml_tag encoding below first."
      : undefined;

  return (
    <>
      <Panel
        title="The ml_tag encoding"
        subtitle="A business decision this application will not make on your behalf."
        icon="label"
        tone={mlTag?.approved_config ? "ok" : "warn"}
        actions={<Pill tone={mlTag?.approved_config ? "ok" : "warn"}>{mlTag?.approved_config ? "Approved" : "Blocked"}</Pill>}
      >
        {mlTag?.approved_config ? (
          <Notice tone="ok" title="This decision has been recorded">
            The encoding below is what the built package will emit, and it is written into
            <code> scoring/ml_tag_config.json</code> so the runtime can refuse any other convention.
          </Notice>
        ) : (
          <Notice tone="warn" title="Why this is blocked by default">
            {mlTag?.decision_required}
          </Notice>
        )}

        {mlTag?.candidate_conventions?.map((convention) => (
          <label className="checkbox" key={convention.label}>
            <input
              type="radio"
              name="convention"
              checked={
                Number(form.voice_value) === convention.voice_value &&
                Number(form.non_voice_value) === convention.non_voice_value
              }
              disabled={Boolean(mlTag?.approved_config)}
              onChange={() => setForm((c) => ({
                ...c, voice_value: convention.voice_value, non_voice_value: convention.non_voice_value,
              }))}
            />
            <span>
              <strong>{convention.label}</strong>
              <div className="muted small">
                {convention.voice_value} = Voice, {convention.non_voice_value} = Non-Voice · {convention.note}
              </div>
            </span>
          </label>
        ))}

        <div className="grid-3">
          <div className="field">
            <label htmlFor="tag-column">Appended column name</label>
            <input id="tag-column" type="text" value={form.column_name} disabled={Boolean(mlTag?.approved_config)}
              onChange={(e) => setForm((c) => ({ ...c, column_name: e.target.value }))} />
          </div>
          <div className="field">
            <label htmlFor="tag-approver">Approved by</label>
            <input id="tag-approver" type="text" placeholder="Your name" value={form.approver}
              disabled={Boolean(mlTag?.approved_config)}
              onChange={(e) => setForm((c) => ({ ...c, approver: e.target.value }))} />
          </div>
          <div className="field">
            <label htmlFor="tag-notes">Notes</label>
            <input id="tag-notes" type="text" value={form.notes} disabled={Boolean(mlTag?.approved_config)}
              onChange={(e) => setForm((c) => ({ ...c, notes: e.target.value }))} />
          </div>
        </div>

        {mlTag?.approved_config ? (
          <Notice tone="ok" title={`Approved by ${mlTag.approved_by}`}>
            <code>{mlTag.approved_config.column_name}</code>: {mlTag.approved_config.voice_value} = Voice,{" "}
            {mlTag.approved_config.non_voice_value} = Non-Voice. The package refuses to emit this column
            under any other convention.
          </Notice>
        ) : (
          <div className="btn-row end">
            <Action onClick={approveTag} busy={busy} busyLabel="Saving…"
              disabledReason={!form.approver.trim() ? "Enter an approver name." : undefined}>
              <Icon name="how_to_reg" size={15} /> Confirm this encoding
            </Action>
          </div>
        )}
      </Panel>

      <Panel
        title="Scoring compatibility sample"
        subtitle="A de-identified inventory extract, used to prove the built package scores correctly."
        icon="science"
      >
        <DropZone
          accept=".parquet,.csv,.xlsx,.xls"
          label="Choose a de-identified inventory sample"
          hint="Parquet, CSV or Excel · optional but strongly recommended"
          file={inventoryFile}
          onFile={setInventoryFile}
          disabled={busy}
        />
        <div className="btn-row end">
          <span className="btn-note">
            Without a sample, the check falls back to rows taken from the snapshot. The validation
            report always records which was used.
          </span>
          <Action className="btn ghost" onClick={uploadInventory} busy={busy} busyLabel="Reading…"
            disabledReason={!inventoryFile ? "Choose a file first." : undefined}>
            <Icon name="upload" size={15} /> Upload sample
          </Action>
        </div>
        {inventory && (
          <Notice tone="ok" title={`${num(inventory.rows)} rows, ${inventory.columns} columns`}>
            The package must return exactly these rows and columns plus one appended{" "}
            <code>{form.column_name}</code>.
          </Notice>
        )}
        <ErrorNotice error={error} />
      </Panel>

      <Panel
        title="Build the package"
        subtitle="The ZIP is assembled, then unzipped and scored exactly as the deployment does. It is only published if that passes."
        icon="deployed_code"
        actions={
          runs.length > 1 && (
            <select value={runId || ""} onChange={(e) => setRunId(e.target.value)} aria-label="Run to export">
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>{run.run_id}</option>
              ))}
            </select>
          )
        }
      >
        {approvedCandidate ? (
          <Metrics
            cols={3}
            items={[
              { label: "Approved candidate", value: approvedCandidate },
              { label: "Approved by", value: approval.approver, sub: when(approval.approved_at) },
              { label: "Threshold", value: approval.selected_threshold },
            ]}
          />
        ) : (
          <Notice tone="warn" title="No approved model for this run">
            Stage 06 must record a typed promotion approval before a package can be built.
          </Notice>
        )}

        <div className="btn-row end">
          <Action onClick={build} busy={busy} busyLabel="Building and validating…" disabledReason={buildBlocker}>
            <Icon name="build" size={15} /> Build &amp; validate package
          </Action>
        </div>
        {error?.body?.detail?.validation && (
          <>
            <Notice tone="bad" title="Validation failed — the ZIP was discarded, not published">
              {error.body.detail.message}
            </Notice>
            <CheckList checks={error.body.detail.validation.checks} />
          </>
        )}
        {error && !error.body?.detail?.validation && <ErrorNotice error={error} />}

        {built && (
          <>
            <Notice tone="ok" title={`${built.zip_name} built and validated`}>
              Version {built.version} · {bytes(built.size_bytes)} · sha256 {shortHash(built.zip_sha256, 24)}
            </Notice>
            <div className="section-title">Package validation record</div>
            <CheckList checks={built.validation.checks} />
            {built.validation.prediction_agreement?.rows > 0 && (
              <Metrics
                cols={3}
                items={[
                  { label: "Rows scored", value: num(built.validation.prediction_agreement.rows) },
                  { label: "Identical to 4dp", value: `${built.validation.prediction_agreement.agreement_pct}%` },
                  { label: "Max difference", value: built.validation.prediction_agreement.max_absolute_difference },
                ]}
              />
            )}
            <p className="muted small">
              Inventory used: {built.validation.inventory_source}. Reference predictions:{" "}
              {built.validation.expected_prediction_source}.
            </p>
          </>
        )}
      </Panel>

      <Panel title="Versions and rollback" icon="inventory"
        subtitle="Every version is retained. This application never deploys anything by itself.">
        {exports.length === 0 ? (
          <Empty icon="archive">No package has been published for this job yet.</Empty>
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
                render: (row) =>
                  row.exists ? (
                    <a className="btn ghost" href={api.downloadUrl(row.export_id)}>
                      <Icon name="download" size={14} /> Download
                    </a>
                  ) : (
                    <span className="muted small">file missing</span>
                  ),
              },
            ]}
            rows={exports}
            rowKey={(row) => row.export_id}
          />
        )}
      </Panel>
    </>
  );
}
