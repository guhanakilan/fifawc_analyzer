import React from "react";

import { api } from "../api.js";
import {
  ActionRow, Badge, Btn, C, Card, CheckList, DropZone, ErrorNotice, MIcon,
  MetricGrid, Notice, SectionTitle, SubHeading,
} from "../nova/Components.jsx";
import { CheckRow } from "../nova/Components.jsx";
import { bytes, metric, shortHash, when } from "../format.js";

export default function ChampionStage({ job, setJob, go }) {
  const [file, setFile] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [trust, setTrust] = React.useState(false);
  const [compat, setCompat] = React.useState(null);
  const [compatBusy, setCompatBusy] = React.useState(false);
  const [compatError, setCompatError] = React.useState(null);

  const upload = async () => {
    setBusy(true);
    setError(null);
    setCompat(null);
    try {
      setJob(await api.uploadPackage(file));
    } catch (uploadError) {
      setError(uploadError);
    } finally {
      setBusy(false);
    }
  };

  const runCompatibility = async () => {
    setCompatBusy(true);
    setCompatError(null);
    try {
      setCompat(await api.compatibility(job.job_id, {
        trust_local_package: true, actor: "local-user",
      }));
    } catch (checkError) {
      setCompatError(checkError);
    } finally {
      setCompatBusy(false);
    }
  };

  const validation = job?.validation;
  const meta = validation?.metadata || {};
  const valid = Boolean(validation?.valid);

  return (
    <>
      <Card>
        <SectionTitle sub="The uploaded ZIP is copied into a new immutable workspace and never modified.">
          NoVA export package
        </SectionTitle>
        <DropZone
          accept=".zip"
          label="Drop a NoVA export ZIP here"
          hint="or click to browse · maximum 500 MB"
          file={file}
          onFile={(chosen) => {
            setFile(chosen);
            setError(null);
          }}
          disabled={busy}
        />
        <ActionRow note="Nothing inside the archive is unpickled during this step. The estimator is loaded only in the compatibility check below.">
          <Btn
            onClick={upload}
            busy={busy}
            busyLabel="Inspecting the package…"
            disabledReason={!file ? "Choose a ZIP file first." : undefined}
          >
            <MIcon name="upload" size={15} /> Upload &amp; validate
          </Btn>
        </ActionRow>
        <ErrorNotice error={error} title="The package was rejected" />
      </Card>

      {job && (
        <Card borderSize={2}>
          <SectionTitle
            sub={valid
              ? `Retraining job ${job.job_id} created.`
              : "One or more blocking checks failed. Fix the package and upload it again."}
            right={<Badge color={valid ? C.green : C.red} bg={valid ? "#E3F5EC" : "#FDECEA"}>
              {valid ? "ACCEPTED" : "REJECTED"}
            </Badge>}
          >
            {valid ? "Champion package accepted" : "Package cannot be used"}
          </SectionTitle>

          <MetricGrid
            compact
            min={140}
            items={[
              { label: "Placement", value: meta.placement_id ?? "—", sub: meta.placement_id ? undefined : "not in manifest" },
              { label: "Source run", value: meta.run_id ?? "—" },
              { label: "Champion", value: meta.model_id ?? "unresolved" },
              { label: "Family", value: (meta.model_family || "unknown").toUpperCase() },
              { label: "Threshold", value: meta.threshold ?? "—", color: C.indigo },
              { label: "Features", value: meta.feature_count ?? "—", color: C.indigo },
            ]}
          />

          <SubHeading>Package identity</SubHeading>
          <dl style={{ display: "grid", gridTemplateColumns: "max-content 1fr", gap: "5px 16px", fontSize: 12, margin: 0 }}>
            <dt style={{ color: "var(--nova-grey-dim)", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>SHA-256</dt>
            <dd style={{ margin: 0, overflowWrap: "anywhere" }}>{job.package_sha256}</dd>
            <dt style={{ color: "var(--nova-grey-dim)", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>Size</dt>
            <dd style={{ margin: 0 }}>{bytes(meta.extracted_bytes)} extracted from {meta.file_count} files</dd>
            <dt style={{ color: "var(--nova-grey-dim)", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>Training window</dt>
            <dd style={{ margin: 0 }}>
              {meta.training_from ? `${when(meta.training_from)} → ${when(meta.training_to)}` : "not recorded"}
            </dd>
            <dt style={{ color: "var(--nova-grey-dim)", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>Split mode</dt>
            <dd style={{ margin: 0 }}>{meta.split_mode || "not recorded"}</dd>
          </dl>

          <SubHeading>Source metrics recorded in the package</SubHeading>
          {Object.keys(meta.source_metrics || {}).length ? (
            <MetricGrid
              compact
              min={120}
              items={["f1", "precision", "recall", "auc"].map((key) => ({
                label: key.toUpperCase(),
                value: metric(meta.source_metrics[key]),
                color: C.indigo,
              }))}
            />
          ) : (
            <p style={{ fontSize: 12, color: "var(--nova-grey-dim)", margin: 0 }}>
              The package carries no test metrics for the champion. Stage 06 still benchmarks it
              directly on the same rows as the challengers, so the comparison does not depend on this.
            </p>
          )}

          <SubHeading>Validation checks</SubHeading>
          <CheckList checks={validation.checks} />

          {!!validation.missing_supporting_files?.length && (
            <Notice tone="warn" title="Supporting configuration not included">
              {validation.missing_supporting_files.join(", ")}. These do not block intake, but each
              one reduces how faithfully the original transformation chain can be reproduced.
            </Notice>
          )}
          {validation.wrapper_directory && (
            <Notice tone="info" title="Wrapper directory unwrapped">
              The archive nested everything inside <code>{validation.wrapper_directory}/</code>. It was
              removed so the artifacts sit at their expected paths.
            </Notice>
          )}
        </Card>
      )}

      {valid && (
        <Card>
          <SectionTitle
            sub="Loads the estimator and its fitted transform state. This is the only step that unpickles the package."
            right={compat ? <Badge color={C.green} bg="#E3F5EC">LOADED</Badge> : null}
          >
            Model compatibility check
          </SectionTitle>

          <Notice tone="warn" title="Loading a model file executes code from the package">
            Only continue for a package produced by your own NoVA installation and stored on this
            machine. Nothing is uploaded anywhere.
          </Notice>

          <CheckRow checked={trust} onChange={setTrust}>
            I trust this local package and accept that loading it will execute code contained in it.
          </CheckRow>

          <ActionRow>
            <Btn
              onClick={runCompatibility}
              busy={compatBusy}
              busyLabel="Loading the champion…"
              disabledReason={!trust ? "Confirm the local-trust acknowledgement first." : undefined}
            >
              <MIcon name="play_arrow" size={15} /> Run compatibility check
            </Btn>
          </ActionRow>
          <ErrorNotice error={compatError} title="The champion could not be loaded" />

          {compat && (
            <>
              <SubHeading>Loaded champion</SubHeading>
              <MetricGrid
                compact
                min={140}
                items={[
                  { label: "Estimator", value: compat.model_class },
                  { label: "Threshold", value: compat.threshold, color: C.indigo },
                  { label: "Features", value: compat.feature_count, color: C.indigo },
                  {
                    label: "Probabilities",
                    value: compat.supports_predict_proba ? "yes" : "no",
                    color: compat.supports_predict_proba ? C.green : C.red,
                    sub: compat.supports_predict_proba ? "predict_proba" : "threshold tuning limited",
                  },
                  { label: "SubTask maps", value: compat.subtask_mappings },
                  { label: "Keywords", value: compat.subtask_keywords },
                  {
                    label: "Transform cfg",
                    value: compat.has_features_config ? "present" : "absent",
                    color: compat.has_features_config ? C.green : "#F59E0B",
                    sub: compat.has_features_config ? undefined : "a default will be derived and flagged",
                  },
                ]}
              />
              <SubHeading>Fitted transform state</SubHeading>
              <MetricGrid
                compact
                min={120}
                items={Object.entries(compat.fitted_state).map(([key, value]) => ({
                  label: key.replace(/_/g, " "),
                  value,
                }))}
              />
              <ActionRow>
                <Btn onClick={() => go("data")}>
                  Continue to training data <MIcon name="arrow_forward" size={15} />
                </Btn>
              </ActionRow>
            </>
          )}
        </Card>
      )}
    </>
  );
}
