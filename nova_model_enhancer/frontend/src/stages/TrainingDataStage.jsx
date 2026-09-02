import React from "react";

import { api } from "../api.js";
import {
  ActionRow, Badge, Btn, C, Card, DropZone, EmptyState, ErrorNotice, Field,
  FormGrid, MIcon, MetricGrid, Notice, SectionTitle, SubHeading, Table,
} from "../nova/Components.jsx";
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

  if (!job) return <NoJob />;

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
      <Card>
        <SectionTitle sub="Parquet is preferred. CSV and Excel are also accepted; large delimited files are streamed rather than loaded whole.">
          Labelled training data
        </SectionTitle>

        <FormGrid style={{ marginBottom: 14 }}>
          <Field
            label="What this file contains"
            htmlFor="data-role"
            hint="Upload several files if your history and your new verified rows are separate. They are concatenated when the snapshot is built."
          >
            <select id="data-role" value={role} onChange={(event) => setRole(event.target.value)}>
              {ROLES.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </Field>
          <div />
        </FormGrid>

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
        <ActionRow>
          <Btn
            onClick={upload}
            busy={busy}
            busyLabel="Reading and profiling…"
            disabledReason={!file ? "Choose a file first." : undefined}
          >
            <MIcon name="upload" size={15} /> Upload &amp; profile
          </Btn>
        </ActionRow>
        <ErrorNotice error={error} title="The file could not be used" />
      </Card>

      <Card>
        <SectionTitle
          sub={assets.length
            ? `${assets.length} file(s), ${num(totalRows)} rows in total.`
            : "Nothing uploaded yet."}
          right={<Btn variant="ghost" small onClick={refresh}>
            <MIcon name="refresh" size={14} /> Refresh
          </Btn>}
        >
          Uploaded datasets
        </SectionTitle>

        {loading ? (
          <p style={{ fontSize: 12, color: "var(--nova-grey-dim)" }}>Loading…</p>
        ) : assets.length === 0 ? (
          <EmptyState icon="database">Upload at least one labelled dataset to continue.</EmptyState>
        ) : (
          assets.map((asset) => <AssetBlock key={asset.asset_id} asset={asset} onRemove={remove} />)
        )}

        {assets.length > 0 && (
          <ActionRow>
            <Btn onClick={() => go("readiness")}>
              Review readiness <MIcon name="arrow_forward" size={15} />
            </Btn>
          </ActionRow>
        )}
      </Card>
    </>
  );
}

function AssetBlock({ asset, onRemove }) {
  const summary = asset.summary || {};
  const drift = summary.schema_drift;
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
        <strong style={{ fontSize: 13 }}>{asset.original_filename}</strong>
        <Badge small>{asset.role}</Badge>
        <Badge small color={C.indigo} bg="#EEF1FF">{asset.file_type}</Badge>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--nova-grey-dim)", fontFamily: "'DM Mono',monospace" }}>
          sha256 {shortHash(asset.sha256)}
        </span>
        <Btn variant="danger" small onClick={() => onRemove(asset.asset_id)}>
          <MIcon name="delete" size={14} /> Remove
        </Btn>
      </div>

      <MetricGrid
        compact
        min={135}
        items={[
          { label: "Rows", value: num(summary.rows) },
          { label: "Columns", value: num(summary.columns) },
          { label: "Duplicates", value: num(summary.duplicate_rows), color: summary.duplicate_rows ? "#F59E0B" : C.navy },
          { label: "SubTasks", value: num(summary.distinct_subtasks) },
          { label: "Date column", value: summary.date_column_detected || "none" },
          {
            label: "Bad dates",
            value: num(summary.invalid_dates),
            color: summary.invalid_dates ? "#F59E0B" : C.navy,
            sub: summary.invalid_dates ? "sort oldest, stay in train" : undefined,
          },
        ]}
      />

      <div style={{ fontSize: 12, color: "var(--nova-grey-dim)", marginTop: 10 }}>
        {summary.min_date
          ? `Date range ${when(summary.min_date)} → ${when(summary.max_date)}`
          : "No parseable dates in this file."}
        {summary.target_column_detected
          ? ` · label column detected: ${summary.target_column_detected}`
          : " · no label column detected, labels will be derived from SubTask mappings"}
      </div>

      {summary.class_distribution && Object.keys(summary.class_distribution).length > 0 && (
        <>
          <SubHeading>Label distribution as uploaded</SubHeading>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {Object.entries(summary.class_distribution).map(([key, value]) => (
              <Badge key={key}>{key}: {num(value)}</Badge>
            ))}
          </div>
        </>
      )}

      {drift && (
        <>
          <SubHeading>Schema drift against the champion configuration</SubHeading>
          {drift.missing_column_count === 0 && drift.new_column_count === 0 ? (
            <Notice tone="ok" title="No drift detected">
              Every column the champion configuration expects is present, after applying its own
              inventory→production rename map.
            </Notice>
          ) : (
            <Notice
              tone={drift.missing_column_count ? "warn" : "info"}
              title={`${drift.missing_column_count} expected column(s) missing, ${drift.new_column_count} new column(s)`}
            >
              {drift.missing_column_count > 0 && (
                <p style={{ margin: "0 0 6px" }}>
                  <strong>Missing:</strong> {drift.missing_columns.join(", ")}. A missing feature is
                  filled with zero at scoring time, which silently weakens the model — resolve it
                  before training rather than after.
                </p>
              )}
              {drift.new_column_count > 0 && (
                <p style={{ margin: 0 }}>
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
          <SubHeading>Most frequent SubTasks</SubHeading>
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

export function NoJob() {
  return (
    <Card>
      <SectionTitle sub="Upload a valid champion package in Stage 01 before continuing.">
        No active job
      </SectionTitle>
      <EmptyState icon="inventory_2">This stage needs a champion package to work against.</EmptyState>
    </Card>
  );
}
