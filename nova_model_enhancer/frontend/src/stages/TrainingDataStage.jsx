import React from "react";

import { api } from "../api.js";
import {
  Action, DropZone, Empty, ErrorNotice, Icon, Metrics, Notice, Panel, Table,
} from "../components/Ui.jsx";
import { num, shortHash, when } from "../format.js";

const ROLES = [
  { value: "combined", label: "Combined historical + new" },
  { value: "historical", label: "Historical only" },
  { value: "new", label: "New verified rows only" },
];

export default function TrainingDataStage({ job, mark, go }) {
  const [file, setFile] = React.useState(null);
  const [role, setRole] = React.useState("combined");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [assets, setAssets] = React.useState([]);
  const [loading, setLoading] = React.useState(true);

  const refresh = React.useCallback(async () => {
    if (!job) return;
    try {
      const body = await api.trainingData(job.job_id);
      setAssets(body.assets);
      mark("dataUploaded", body.assets.length > 0);
    } catch (loadError) {
      setError(loadError);
    } finally {
      setLoading(false);
    }
  }, [job, mark]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  if (!job) {
    return (
      <Panel title="No active job" icon="info">
        <Notice tone="warn" title="Start at Stage 01">
          Upload a valid champion package before adding training data.
        </Notice>
      </Panel>
    );
  }

  const upload = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.uploadTrainingData(job.job_id, file, role);
      setFile(null);
      await refresh();
    } catch (uploadError) {
      setError(uploadError);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (assetId) => {
    setError(null);
    try {
      await api.removeTrainingData(job.job_id, assetId);
      await refresh();
    } catch (removeError) {
      setError(removeError);
    }
  };

  const totalRows = assets.reduce((sum, asset) => sum + asset.rows_count, 0);

  return (
    <>
      <Panel
        title="Labelled training data"
        subtitle="Parquet is preferred. CSV and Excel are also accepted; large delimited files are streamed rather than loaded whole."
        icon="database"
      >
        <div className="grid-2">
          <div className="field">
            <label htmlFor="data-role">What this file contains</label>
            <select id="data-role" value={role} onChange={(event) => setRole(event.target.value)}>
              {ROLES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <span className="hint">
              Upload several files if your history and your new verified rows are separate. They are
              concatenated when the snapshot is built.
            </span>
          </div>
          <div />
        </div>
        <DropZone
          accept=".parquet,.csv,.xlsx,.xls"
          label="Choose a labelled dataset"
          hint="Parquet, CSV, XLSX or XLS · maximum 2 GB"
          file={file}
          onFile={(chosen) => {
            setFile(chosen);
            setError(null);
          }}
          disabled={busy}
        />
        <div className="btn-row end">
          <Action
            onClick={upload}
            busy={busy}
            busyLabel="Reading and profiling…"
            disabledReason={!file ? "Choose a file first." : undefined}
          >
            <Icon name="upload" size={15} /> Upload &amp; profile
          </Action>
        </div>
        <ErrorNotice error={error} title="The file could not be used" />
      </Panel>

      <Panel
        title="Uploaded datasets"
        subtitle={
          assets.length
            ? `${assets.length} file(s), ${num(totalRows)} rows in total.`
            : "Nothing uploaded yet."
        }
        icon="folder_open"
        actions={
          <button type="button" className="btn ghost" onClick={refresh}>
            <Icon name="refresh" size={15} /> Refresh
          </button>
        }
      >
        {loading ? (
          <p className="muted small">Loading…</p>
        ) : assets.length === 0 ? (
          <Empty icon="database">Upload at least one labelled dataset to continue.</Empty>
        ) : (
          assets.map((asset) => <AssetCard key={asset.asset_id} asset={asset} onRemove={remove} />)
        )}

        {assets.length > 0 && (
          <div className="btn-row end">
            <Action onClick={() => go("readiness")}>
              Review readiness <Icon name="arrow_forward" size={15} />
            </Action>
          </div>
        )}
      </Panel>
    </>
  );
}

function AssetCard({ asset, onRemove }) {
  const summary = asset.summary || {};
  const drift = summary.schema_drift;
  return (
    <div style={{ marginBottom: 18 }}>
      <div className="row" style={{ marginBottom: 8 }}>
        <strong>{asset.original_filename}</strong>
        <span className="chip">{asset.role}</span>
        <span className="chip mono">{asset.file_type}</span>
        <span className="spacer" />
        <span className="small muted mono">sha256 {shortHash(asset.sha256)}</span>
        <button type="button" className="btn danger" onClick={() => onRemove(asset.asset_id)}>
          <Icon name="delete" size={14} /> Remove
        </button>
      </div>
      <Metrics
        items={[
          { label: "Rows", value: num(summary.rows) },
          { label: "Columns", value: num(summary.columns) },
          { label: "Duplicate rows", value: num(summary.duplicate_rows) },
          { label: "Distinct SubTasks", value: num(summary.distinct_subtasks) },
          { label: "Date column", value: summary.date_column_detected || "None detected" },
          {
            label: "Date range",
            value: summary.min_date ? when(summary.min_date) : "—",
            sub: summary.max_date ? `to ${when(summary.max_date)}` : undefined,
          },
          {
            label: "Unparseable dates",
            value: num(summary.invalid_dates),
            sub: summary.invalid_dates ? "These rows sort oldest and stay in train" : undefined,
          },
          {
            label: "Label column",
            value: summary.target_column_detected || "None detected",
            sub: summary.target_column_detected
              ? undefined
              : "Labels will be derived from SubTask mappings",
          },
        ]}
      />

      {summary.class_distribution && Object.keys(summary.class_distribution).length > 0 && (
        <>
          <div className="section-title">Label distribution as uploaded</div>
          <div className="row">
            {Object.entries(summary.class_distribution).map(([key, value]) => (
              <span className="chip" key={key}>
                {key}: {num(value)}
              </span>
            ))}
          </div>
        </>
      )}

      {drift && (
        <>
          <div className="section-title">Schema drift against the champion configuration</div>
          {drift.missing_column_count === 0 && drift.new_column_count === 0 ? (
            <Notice tone="ok" title="No drift detected">
              Every column the champion configuration expects is present.
            </Notice>
          ) : (
            <Notice
              tone={drift.missing_column_count ? "warn" : "info"}
              title={`${drift.missing_column_count} expected column(s) missing, ${drift.new_column_count} new column(s)`}
            >
              {drift.missing_column_count > 0 && (
                <p>
                  <strong>Missing:</strong> {drift.missing_columns.join(", ")}. A missing feature is
                  filled with zero at scoring time, which silently weakens the model — resolve it
                  before training rather than after.
                </p>
              )}
              {drift.new_column_count > 0 && (
                <p>
                  <strong>New:</strong> {drift.new_columns.slice(0, 20).join(", ")}. New columns are
                  ignored unless the champion's feature selection names them.
                </p>
              )}
            </Notice>
          )}
        </>
      )}

      {summary.top_subtasks?.length > 0 && (
        <>
          <div className="section-title">Most frequent SubTasks</div>
          <Table
            columns={[
              { key: "subtask", header: "SubTask" },
              { key: "rows", header: "Rows", className: "num", render: (r) => num(r.rows) },
            ]}
            rows={summary.top_subtasks.slice(0, 8)}
            rowKey={(row) => row.subtask}
          />
        </>
      )}
    </div>
  );
}
