import React from "react";

import { api } from "../api.js";
import {
  Action, CheckList, DropZone, ErrorNotice, Icon, Metrics, Notice, Panel, Pill,
} from "../components/Ui.jsx";
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
      setCompat(await api.compatibility(job.job_id, { trust_local_package: true, actor: "local-user" }));
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
      <Panel
        title="NoVA export package"
        subtitle="The uploaded ZIP is copied into a new immutable workspace and never modified."
        icon="folder_zip"
      >
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
        <div className="btn-row end">
          <span className="btn-note">
            Nothing inside the archive is unpickled during this step. The estimator is loaded only in
            the compatibility check below.
          </span>
          <Action
            onClick={upload}
            busy={busy}
            busyLabel="Inspecting the package…"
            disabledReason={!file ? "Choose a ZIP file first." : undefined}
          >
            <Icon name="upload" size={15} /> Upload &amp; validate
          </Action>
        </div>
        <ErrorNotice error={error} title="The package was rejected" />
      </Panel>

      {job && (
        <Panel
          title={valid ? "Champion package accepted" : "Package cannot be used"}
          subtitle={
            valid
              ? `Retraining job ${job.job_id} created.`
              : "One or more blocking checks failed. Fix the package and upload it again."
          }
          icon={valid ? "verified" : "report"}
          tone={valid ? "ok" : "bad"}
        >
          <Metrics
            items={[
              { label: "Placement", value: meta.placement_id ?? "Not in manifest" },
              { label: "Source run", value: meta.run_id ?? "Not in manifest" },
              { label: "Champion model", value: meta.model_id ?? "Unresolved" },
              { label: "Family", value: (meta.model_family || "unknown").toUpperCase() },
              { label: "Threshold", value: meta.threshold ?? "Not recorded" },
              { label: "Features", value: meta.feature_count ?? "—" },
              {
                label: "Training window",
                value: meta.training_from ? `${when(meta.training_from)} →` : "—",
                sub: meta.training_to ? when(meta.training_to) : undefined,
              },
              { label: "Package SHA-256", value: shortHash(job.package_sha256), sub: `${bytes(meta.extracted_bytes)} extracted` },
            ]}
          />

          <div className="section-title">Source metrics recorded in the package</div>
          {Object.keys(meta.source_metrics || {}).length ? (
            <Metrics
              cols={4}
              items={["f1", "precision", "recall", "auc"].map((key) => ({
                label: key.toUpperCase(),
                value: metric(meta.source_metrics[key]),
              }))}
            />
          ) : (
            <p className="muted small">
              The package carries no test metrics for the champion. Stage 6 will still benchmark it
              directly on the same rows as the challengers.
            </p>
          )}

          <div className="section-title">Validation checks</div>
          <CheckList checks={validation.checks} />

          {!!validation.missing_supporting_files?.length && (
            <Notice tone="warn" title="Supporting configuration not included">
              {validation.missing_supporting_files.join(", ")}. These do not block intake, but each one
              reduces how faithfully the original transformation chain can be reproduced.
            </Notice>
          )}
          {validation.wrapper_directory && (
            <Notice tone="info" title="Wrapper directory unwrapped">
              The archive nested everything inside <code>{validation.wrapper_directory}/</code>. It was
              removed so the artifacts sit at their expected paths.
            </Notice>
          )}
        </Panel>
      )}

      {valid && (
        <Panel
          title="Model compatibility check"
          subtitle="Loads the estimator and its fitted transform state. This is the only step that unpickles the package."
          icon="policy"
          tone={compat ? "ok" : ""}
        >
          <Notice tone="warn" title="Loading a model file executes code from the package">
            Only continue for a package produced by your own NoVA installation and stored on this
            machine. Nothing is uploaded anywhere.
          </Notice>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={trust}
              onChange={(event) => setTrust(event.target.checked)}
            />
            <span>
              I trust this local package and accept that loading it will execute code contained in it.
            </span>
          </label>
          <div className="btn-row end">
            <Action
              onClick={runCompatibility}
              busy={compatBusy}
              busyLabel="Loading the champion…"
              disabledReason={!trust ? "Confirm the local-trust acknowledgement first." : undefined}
            >
              <Icon name="play_arrow" size={15} /> Run compatibility check
            </Action>
          </div>
          <ErrorNotice error={compatError} title="The champion could not be loaded" />

          {compat && (
            <>
              <div className="section-title">Loaded champion</div>
              <Metrics
                items={[
                  { label: "Estimator class", value: compat.model_class },
                  { label: "Model id", value: compat.model_id },
                  { label: "Threshold", value: compat.threshold },
                  { label: "Feature count", value: compat.feature_count },
                  {
                    label: "Probability output",
                    value: compat.supports_predict_proba ? "predict_proba" : "predict only",
                    sub: compat.supports_predict_proba ? undefined : "Threshold tuning is limited",
                  },
                  { label: "SubTask mappings", value: compat.subtask_mappings },
                  { label: "Keywords", value: compat.subtask_keywords },
                  {
                    label: "Transform config",
                    value: compat.has_features_config ? "present" : "absent",
                    sub: compat.has_features_config ? undefined : "A default will be derived and flagged",
                  },
                ]}
              />
              <div className="section-title">Fitted transform state</div>
              <Metrics
                cols={4}
                items={Object.entries(compat.fitted_state).map(([key, value]) => ({
                  label: key.replace(/_/g, " "),
                  value,
                }))}
              />
              <div className="btn-row end">
                <Action onClick={() => go("data")}>
                  Continue to training data <Icon name="arrow_forward" size={15} />
                </Action>
              </div>
            </>
          )}
        </Panel>
      )}
    </>
  );
}
