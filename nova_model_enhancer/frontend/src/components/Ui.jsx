import React from "react";

import { FALLBACK_ICON, ICONS } from "./icons.js";

export const Icon = ({ name, size = 18, className = "", ...rest }) => {
  const icon = ICONS[name] || FALLBACK_ICON;
  return (
    <svg
      className={`mi ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={icon.d} />
    </svg>
  );
};

export function Panel({ title, subtitle, icon, tone = "", actions, children }) {
  return (
    <section className={`panel ${tone ? `tone-${tone}` : ""}`}>
      <div className="panel-head">
        {icon && (
          <span className="badge">
            <Icon name={icon} size={17} />
          </span>
        )}
        <div style={{ minWidth: 0 }}>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {actions && <div className="actions">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

export function Metrics({ items, cols = 4 }) {
  return (
    <div className={`metrics ${cols === 3 ? "cols-3" : cols === 2 ? "cols-2" : ""}`}>
      {items.map((item) => (
        <div className="metric" key={item.label}>
          <div className="k">{item.label}</div>
          <div className="v">{item.value ?? "—"}</div>
          {item.sub && <div className="sub">{item.sub}</div>}
        </div>
      ))}
    </div>
  );
}

const CHECK_ICON = { passed: "check_circle", warning: "warning", failed: "cancel" };
const CHECK_CLASS = { passed: "pass", warning: "warn", failed: "fail" };

export function CheckList({ checks }) {
  return (
    <div className="check-list">
      {checks.map((check) => (
        <div className={`check ${CHECK_CLASS[check.status] || "warn"}`} key={check.key || check.label}>
          <Icon name={CHECK_ICON[check.status] || "help"} size={17} />
          <div>
            <b>{check.label}</b>
            <small>{check.detail}</small>
          </div>
          <span className="tag">
            {check.status}
            {check.blocking === false && check.status !== "passed" ? " · non-blocking" : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

export function Notice({ tone = "info", title, children, icon }) {
  const fallback = { info: "info", ok: "check_circle", warn: "warning", bad: "error" }[tone];
  return (
    <div className={`notice ${tone}`} role={tone === "bad" ? "alert" : undefined}>
      <Icon name={icon || fallback} size={17} />
      <div>
        {title && <b>{title}</b>}
        <div>{children}</div>
      </div>
    </div>
  );
}

export function Pill({ tone = "muted", children }) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

export function Empty({ icon = "inbox", children }) {
  return (
    <div className="empty">
      <Icon name={icon} size={28} />
      {children}
    </div>
  );
}

export function Field({ label, hint, htmlFor, children }) {
  return (
    <div className="field">
      {label && <label htmlFor={htmlFor}>{label}</label>}
      {children}
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

export function DropZone({ accept, label, hint, file, onFile, disabled, id }) {
  const inputRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);

  const pick = (chosen) => {
    if (chosen) onFile(chosen);
  };

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""}`}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label={label}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if (!disabled && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        if (!disabled) pick(event.dataTransfer.files?.[0]);
      }}
      style={disabled ? { opacity: 0.5, cursor: "not-allowed" } : undefined}
    >
      <Icon name="cloud_upload" size={26} />
      <strong>{file ? file.name : label}</strong>
      <span>{file ? `${(file.size / 1048576).toFixed(2)} MB selected` : hint}</span>
      <input
        id={id}
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={(event) => pick(event.target.files?.[0])}
      />
    </div>
  );
}

export function Table({ columns, rows, rowKey, highlight }) {
  if (!rows.length) return <Empty>Nothing to show yet.</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} scope="col">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey ? rowKey(row, index) : index} className={highlight?.(row) ? "highlight" : ""}>
              {columns.map((column) => (
                <td key={column.key} className={column.className}>
                  {column.render ? column.render(row) : row[column.key] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Progress({ value, label }) {
  const percent = Math.round((value || 0) * 100);
  return (
    <div>
      <div className="row" style={{ marginBottom: 6 }}>
        <span className="small">{label}</span>
        <span className="spacer" />
        <span className="small mono">{percent}%</span>
      </div>
      <div
        className="progress"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export function ErrorNotice({ error, title = "That did not work" }) {
  if (!error) return null;
  return (
    <Notice tone="bad" title={title}>
      {error.message || String(error)}
    </Notice>
  );
}

/** A button that is disabled with a stated reason rather than silently inert. */
export function Action({ children, disabledReason, busy, busyLabel, ...rest }) {
  const disabled = Boolean(disabledReason) || busy;
  return (
    <button
      {...rest}
      className={rest.className || "btn primary"}
      disabled={disabled}
      title={disabledReason || rest.title}
      aria-disabled={disabled}
    >
      {busy ? busyLabel || "Working…" : children}
    </button>
  );
}
