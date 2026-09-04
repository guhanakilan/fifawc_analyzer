/* The NoVA workbench component set.
 *
 * Card, SectionTitle, Badge, Btn, MetricCard, ProgressBar, StatusDot and
 * StageHelpBanner reproduce the reference application's own components — same
 * geometry, same gradient borders, same DM Mono typography, same spring
 * motion — so the enhancer reads as part of the same product.
 *
 * Icons are the one deliberate difference: the reference pulls Material
 * Symbols from Google Fonts, which on a machine with no outbound network
 * leaves the literal ligature text ("check_circle") on screen. This
 * application is localhost-first, so the same glyphs are drawn inline.
 */

import { AnimatePresence, motion } from "framer-motion";
import React from "react";

import { FALLBACK_ICON, ICONS } from "../components/icons.js";
import { C, SPRING, SPRING_FAST, STATUS } from "./palette.js";
import { useTheme } from "./theme.jsx";

export { C, STATUS };

// ── Icon ─────────────────────────────────────────────────────────────────────

export const MIcon = ({ name, size = 16, style }) => {
  const icon = ICONS[name] || FALLBACK_ICON;
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" focusable="false"
      style={{ flexShrink: 0, display: "inline-block", verticalAlign: "middle", ...style }}
    >
      <path d={icon.d} />
    </svg>
  );
};

// ── Brand mark ───────────────────────────────────────────────────────────────

/** Dotted-sphere logo, generated exactly as the reference generates it. */
export const NovaGlobeLogo = ({ size = 32 }) => {
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.44;
  const baseDotR = size * 0.042;
  const dots = [];
  const latSteps = 9;
  const lonSteps = 18;
  for (let i = 0; i < latSteps; i += 1) {
    const lat = -Math.PI / 2 + (Math.PI * i) / (latSteps - 1);
    const y = cy - r * Math.sin(lat);
    const rowR = r * Math.cos(lat);
    const cols = Math.max(1, Math.round(lonSteps * Math.cos(lat)));
    for (let j = 0; j < cols; j += 1) {
      const lon = cols === 1 ? 0 : -Math.PI + (2 * Math.PI * j) / cols;
      const x = cx + rowR * Math.cos(lon);
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist > r + baseDotR) continue;
      const scale = 0.55 + 0.45 * Math.cos(lon) * Math.cos(lat);
      dots.push(<circle key={`${i}-${j}`} cx={x} cy={y} r={Math.max(0.5, baseDotR * scale)} fill={C.green} />);
    }
  }
  // No background plate: a hard-coded white one showed as a bright box against
  // the dark theme. The dots carry the mark on whatever surface it sits on.
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      {dots}
    </svg>
  );
};

// ── Card ─────────────────────────────────────────────────────────────────────

export const Card = ({ children, style = {}, borderSize = 1, ...rest }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const bg = isDark ? "#161727" : C.bgCard;
  const grad = borderSize === 2
    ? `linear-gradient(135deg, ${C.green}, ${C.indigo})`
    : `linear-gradient(135deg, ${C.green}40, ${C.indigo}40)`;
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING}
      style={{
        background: `linear-gradient(${bg}, ${bg}) padding-box, ${grad} border-box`,
        borderRadius: 10,
        border: `${borderSize}px solid transparent`,
        padding: 20,
        boxShadow: "var(--nova-card-shadow)",
        marginBottom: 16,
        ...style,
      }}
      {...rest}
    >
      {children}
    </motion.section>
  );
};

// ── Typography ───────────────────────────────────────────────────────────────

export const SectionTitle = ({ children, sub, right }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <div style={{ marginBottom: 20, display: "flex", alignItems: "flex-start", gap: 12 }}>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{
          fontSize: 18, fontWeight: 800, fontFamily: "var(--nova-font-mono)",
          letterSpacing: -0.3, color: isDark ? "#E8EAF6" : C.navy,
        }}>
          {children}
        </div>
        {sub && (
          <div style={{ fontSize: 13, marginTop: 4, color: isDark ? "#8892A0" : C.greyDim }}>
            {sub}
          </div>
        )}
      </div>
      {right && <div style={{ display: "flex", gap: 8, alignItems: "center" }}>{right}</div>}
    </div>
  );
};

export const SubHeading = ({ children }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <div style={{
      fontSize: 10, fontWeight: 800, letterSpacing: 1, textTransform: "uppercase",
      fontFamily: "var(--nova-font-mono)", color: isDark ? "#8892A0" : C.indigo,
      margin: "22px 0 10px", display: "flex", alignItems: "center", gap: 10,
    }}>
      <span>{children}</span>
      <span style={{ flex: 1, height: 1, background: isDark ? "#2A2C4A" : `${C.navy}14` }} />
    </div>
  );
};

// ── Badge / status ───────────────────────────────────────────────────────────

export const Badge = ({ children, color = C.indigo, bg = "#EEF1FF", small }) => (
  <span style={{
    background: bg, color, borderRadius: 4,
    padding: small ? "2px 6px" : "3px 10px",
    fontSize: small ? 11 : 12, fontWeight: 700,
    fontFamily: "var(--nova-font-mono)", letterSpacing: 0.3, whiteSpace: "nowrap",
  }}>
    {children}
  </span>
);

const TONE_BG = {
  ok: "#E3F5EC", warn: "#FDF2DD", bad: "#FDECEA", info: "#EEF1FF", muted: "#EEF0F5",
};
const TONE_BG_DARK = {
  ok: "#12301F", warn: "#33280F", bad: "#3A1C1A", info: "#1E2440", muted: "#242637",
};

export const Pill = ({ tone = "muted", children }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <Badge
      small
      color={STATUS[tone] || C.grey}
      bg={(isDark ? TONE_BG_DARK : TONE_BG)[tone] || TONE_BG.muted}
    >
      {children}
    </Badge>
  );
};

export const StatusDot = ({ tone = "muted", pulse }) => (
  <span style={{
    display: "inline-block", width: 8, height: 8, borderRadius: "50%",
    background: STATUS[tone] || C.grey, marginRight: 6,
    boxShadow: pulse ? `0 0 6px ${STATUS[tone]}` : "none",
  }} />
);

// ── Buttons ──────────────────────────────────────────────────────────────────

export const Btn = ({
  children, onClick, variant = "primary", small, disabled, disabledReason,
  busy, busyLabel, type = "button", style = {}, ...rest
}) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const isDisabled = Boolean(disabled || disabledReason || busy);

  const base = {
    primary: { background: isDark ? C.green : C.navy, color: isDark ? "#051A0E" : "#fff", border: "none" },
    secondary: { background: "transparent", color: C.green, border: `1.5px solid ${C.green}` },
    danger: { background: C.red, color: "#fff", border: "none" },
    ghost: {
      background: isDark ? "#1E2033" : "#F1F4F8",
      color: isDark ? "#C8CDEF" : C.navy,
      border: `1px solid ${isDark ? "#2E3150" : "#E4E8F0"}`,
    },
    violet: { background: "#7C3AED", color: "#fff", border: "none" },
  };
  const hover = {
    primary: { boxShadow: `0 4px 14px ${(isDark ? C.green : C.navy)}50` },
    secondary: { background: `${C.green}12` },
    danger: { boxShadow: `0 4px 14px ${C.red}50` },
    ghost: { background: isDark ? "#262941" : "#E8EDF5" },
    violet: { boxShadow: "0 4px 14px rgba(124,58,237,0.5)" },
  };

  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={isDisabled}
      title={disabledReason || rest.title}
      aria-disabled={isDisabled}
      whileHover={isDisabled ? {} : hover[variant]}
      whileTap={isDisabled ? {} : { scale: 0.96 }}
      transition={SPRING_FAST}
      style={{
        ...base[variant],
        borderRadius: 7,
        padding: small ? "6px 14px" : "9px 20px",
        fontSize: small ? 12 : 13,
        fontWeight: 700,
        cursor: isDisabled ? "not-allowed" : "pointer",
        opacity: isDisabled ? 0.5 : 1,
        fontFamily: "inherit",
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        ...style,
      }}
      {...rest}
    >
      {busy ? busyLabel || "Working…" : children}
    </motion.button>
  );
};

/** Right-aligned action row with an optional explanatory note on the left. */
export const ActionRow = ({ note, children }) => (
  <div style={{
    display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap",
    justifyContent: "flex-end", marginTop: 18,
  }}>
    {note && (
      <span style={{ fontSize: 12, color: "var(--nova-grey-dim)", marginRight: "auto", maxWidth: "62ch" }}>
        {note}
      </span>
    )}
    {children}
  </div>
);

// ── Metrics ──────────────────────────────────────────────────────────────────

export const MetricCard = ({ label, value, sub, color = C.navy, compact }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const valueColor = isDark && color === C.navy ? "#E8EAF6" : color;
  const bg = isDark ? "#161727" : "#fff";
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      whileHover={{
        y: -2,
        boxShadow: isDark ? "0 8px 24px rgba(0,0,0,0.35)" : "0 8px 24px rgba(40,41,61,0.12)",
      }}
      transition={{ type: "spring", stiffness: 340, damping: 28 }}
      style={{
        textAlign: "center", padding: compact ? "12px 10px" : "18px 12px", borderRadius: 14,
        border: "2px solid transparent", minWidth: 0,
        background: `linear-gradient(${bg}, ${bg}) padding-box, linear-gradient(135deg, ${C.green}, ${C.indigo}) border-box`,
      }}
    >
      <div style={{
        fontSize: compact ? 19 : 28, fontWeight: 900, color: valueColor,
        fontFamily: "var(--nova-font-mono)", letterSpacing: -1, overflowWrap: "anywhere",
        lineHeight: 1.15,
      }}>
        {value ?? "—"}
      </div>
      <div style={{
        fontSize: compact ? 10 : 12, fontWeight: 700, marginTop: 4,
        color: isDark ? "#8892A0" : C.navy, textTransform: "uppercase", letterSpacing: 0.5,
      }}>
        {label}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: isDark ? "#8892A0" : C.greyDim, marginTop: 3 }}>
          {sub}
        </div>
      )}
    </motion.div>
  );
};

export const MetricGrid = ({ items, min = 150, compact }) => (
  <div style={{
    display: "grid", gap: 10,
    gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))`,
  }}>
    {items.map((item) => (
      <MetricCard key={item.label} compact={compact} {...item} />
    ))}
  </div>
);

export const ProgressBar = ({ value, max = 1, color = C.green, height = 6 }) => (
  <div style={{ background: "#E8EDF5", borderRadius: 99, height, overflow: "hidden" }}>
    <div style={{
      width: `${Math.max(0, Math.min(100, Math.round((value / max) * 100)))}%`,
      height: "100%", background: color, borderRadius: 99, transition: "width 0.4s ease",
    }} />
  </div>
);

// ── Notices ──────────────────────────────────────────────────────────────────

const NOTICE_ICON = { ok: "check_circle", warn: "warning", bad: "error", info: "info" };

export const Notice = ({ tone = "info", title, children, icon }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const color = STATUS[tone] || C.indigo;
  const bg = (isDark ? TONE_BG_DARK : TONE_BG)[tone] || TONE_BG.info;
  return (
    <div
      role={tone === "bad" ? "alert" : undefined}
      style={{
        display: "flex", gap: 10, alignItems: "flex-start", background: bg,
        borderLeft: `3px solid ${color}`, borderRadius: 8, padding: "11px 13px",
        margin: "14px 0", fontSize: 12.5, lineHeight: 1.55,
        color: isDark ? "#C9CFE4" : "#334155",
      }}
    >
      <span style={{ color, marginTop: 1 }}>
        <MIcon name={icon || NOTICE_ICON[tone]} size={16} />
      </span>
      <div style={{ minWidth: 0 }}>
        {title && <div style={{ fontWeight: 800, color, marginBottom: 3 }}>{title}</div>}
        <div>{children}</div>
      </div>
    </div>
  );
};

export const ErrorNotice = ({ error, title = "That did not work" }) =>
  error ? <Notice tone="bad" title={title}>{error.message || String(error)}</Notice> : null;

export const EmptyState = ({ icon = "inbox", children }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <div style={{
      border: `1px dashed ${isDark ? "#2E3150" : "#D8DCF0"}`, borderRadius: 10,
      padding: "28px 18px", textAlign: "center",
      color: isDark ? "#8892A0" : C.greyDim, fontSize: 12.5,
    }}>
      <div style={{ marginBottom: 8, opacity: 0.6 }}>
        <MIcon name={icon} size={28} />
      </div>
      {children}
    </div>
  );
};

// ── Checks ───────────────────────────────────────────────────────────────────

const CHECK_ICON = { passed: "check_circle", warning: "warning", failed: "cancel" };
const CHECK_TONE = { passed: "ok", warning: "warn", failed: "bad" };

export const CheckList = ({ checks }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {checks.map((check) => {
        const tone = CHECK_TONE[check.status] || "warn";
        const color = STATUS[tone];
        return (
          <div
            key={check.key || check.label}
            style={{
              display: "grid", gridTemplateColumns: "20px 1fr auto", gap: 10,
              alignItems: "start", padding: "10px 12px", borderRadius: 8,
              border: `1px solid ${isDark ? "#2A2C4A" : "#E4E8F0"}`,
              background: isDark ? "#131426" : "#FCFDFF",
            }}
          >
            <span style={{ color, marginTop: 1 }}><MIcon name={CHECK_ICON[check.status] || "help"} size={16} /></span>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: isDark ? "#E8EAF6" : C.navy }}>
                {check.label}
              </div>
              <div style={{
                fontSize: 11.5, marginTop: 2, overflowWrap: "anywhere",
                color: isDark ? "#8892A0" : C.greyDim,
              }}>
                {check.detail}
              </div>
            </div>
            <Pill tone={tone}>
              {check.status}
              {check.blocking === false && check.status !== "passed" ? " · non-blocking" : ""}
            </Pill>
          </div>
        );
      })}
    </div>
  );
};

// ── Table ────────────────────────────────────────────────────────────────────

export const Table = ({ columns, rows, rowKey, highlight, empty = "Nothing to show yet." }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  if (!rows?.length) return <EmptyState>{empty}</EmptyState>;
  const border = isDark ? "#2A2C4A" : "#E4E8F0";
  return (
    <div style={{ overflowX: "auto", border: `1px solid ${border}`, borderRadius: 8 }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                style={{
                  padding: "8px 11px", textAlign: column.className === "num" ? "right" : "left",
                  background: isDark ? "#1A1C2E" : "#F5F7FC",
                  borderBottom: `1px solid ${border}`, whiteSpace: "nowrap",
                  fontSize: 9.5, letterSpacing: 0.8, textTransform: "uppercase",
                  fontFamily: "var(--nova-font-mono)", fontWeight: 700,
                  color: isDark ? "#8892A0" : C.indigo, position: "sticky", top: 0,
                }}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={rowKey ? rowKey(row, index) : index}
              style={highlight?.(row) ? { background: isDark ? "#1B2436" : "#EEF3FF" } : undefined}
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  style={{
                    padding: "7px 11px", borderBottom: `1px solid ${border}`, whiteSpace: "nowrap",
                    textAlign: column.className === "num" ? "right" : "left",
                    fontFamily: column.className === "num" ? "var(--nova-font-mono)" : "inherit",
                    color: column.tone?.(row)
                      ? STATUS[column.tone(row)]
                      : (isDark ? "#C9CFE4" : "#334155"),
                    fontWeight: column.tone?.(row) ? 700 : 400,
                  }}
                >
                  {column.render ? column.render(row) : row[column.key] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ── Stage help banner ────────────────────────────────────────────────────────

export const StageHelpBanner = ({ help }) => {
  const [open, setOpen] = React.useState(false);
  const { theme } = useTheme();
  const isDark = theme === "dark";
  if (!help) return null;

  const sections = [
    { key: "inputs", label: "INPUTS", color: "#3B82F6", items: help.inputs || [] },
    { key: "produces", label: "PRODUCES", color: "#10B981", items: help.produces || [] },
    { key: "decisions", label: "DECISIONS", color: "#8B5CF6", items: help.decisions || [] },
    { key: "gotchas", label: "NOTES !", color: "#EF4444", items: help.gotchas || [] },
  ].filter((section) => section.items.length > 0);

  const BG = isDark ? "#161727" : "#F0F4FF";
  const BG_INNER = isDark ? "#0F1020" : "#fff";
  const DIV_INNER = isDark ? "#1E2040" : `${C.navy}12`;
  const COL_ICON = isDark ? C.green : C.navy;
  const COL_TEXT = isDark ? "#8892A0" : C.greyDim;
  const COL_ITEM = isDark ? "#9CA3AF" : "#475569";

  return (
    <div style={{
      marginBottom: 16, borderRadius: 8, overflow: "hidden", border: "2px solid transparent",
      background: `linear-gradient(${BG}, ${BG}) padding-box, linear-gradient(135deg, ${C.green}, ${C.indigo}) border-box`,
    }}>
      <motion.button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        whileHover={{ opacity: 0.85 }}
        whileTap={{ scale: 0.995 }}
        transition={{ type: "spring", stiffness: 420, damping: 30 }}
        style={{
          display: "flex", alignItems: "center", gap: 8, padding: "9px 14px",
          cursor: "pointer", userSelect: "none", width: "100%", background: "transparent",
          border: 0, textAlign: "left",
        }}
      >
        <span style={{ fontSize: 13, color: COL_ICON, flexShrink: 0 }}>ℹ</span>
        <span style={{ fontSize: 12, color: COL_TEXT, flex: 1, lineHeight: 1.4 }}>{help.what}</span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ type: "spring", stiffness: 380, damping: 24 }}
          style={{ fontSize: 10, color: COL_ICON, flexShrink: 0, display: "inline-block" }}
        >
          ▾
        </motion.span>
      </motion.button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="help-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 340, damping: 32 }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ padding: "0 14px 14px", borderTop: `1px solid ${DIV_INNER}`, background: BG_INNER }}>
              <div style={{
                display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "12px 24px", paddingTop: 12,
              }}>
                {sections.map((section) => (
                  <div key={section.key}>
                    <div style={{
                      fontSize: 9, fontWeight: 800, color: section.color, letterSpacing: 0.8,
                      marginBottom: 5, textTransform: "uppercase",
                    }}>
                      {section.label}
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 14 }}>
                      {section.items.map((item) => (
                        <li key={item} style={{ fontSize: 12, color: COL_ITEM, lineHeight: 1.6, marginBottom: 2 }}>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ── Form primitives ──────────────────────────────────────────────────────────

export const Field = ({ label, hint, htmlFor, children, style }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 0, ...style }}>
      {label && (
        <label htmlFor={htmlFor} style={{
          fontSize: 11, fontWeight: 700, color: isDark ? "#A8B0C8" : C.navy,
        }}>
          {label}
        </label>
      )}
      {children}
      {hint && (
        <span style={{ fontSize: 11, color: isDark ? "#7A8397" : C.greyDim, lineHeight: 1.45 }}>
          {hint}
        </span>
      )}
    </div>
  );
};

export const FormGrid = ({ children, min = 220, style }) => (
  <div style={{
    display: "grid", gap: 14, gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))`,
    ...style,
  }}>
    {children}
  </div>
);

export const CheckRow = ({ checked, onChange, disabled, title, children }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  return (
    <label style={{
      display: "flex", alignItems: "flex-start", gap: 9, fontSize: 12.5,
      marginBottom: 9, cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.55 : 1, color: isDark ? "#C9CFE4" : "#334155",
    }}>
      <input
        type="checkbox" checked={checked} disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        style={{ marginTop: 2, accentColor: C.green }}
      />
      <span>
        {title && <strong style={{ color: isDark ? "#E8EAF6" : C.navy }}>{title} </strong>}
        {children}
      </span>
    </label>
  );
};

// ── Drop zone ────────────────────────────────────────────────────────────────

export const DropZone = ({ accept, label, hint, file, onFile, disabled, id }) => {
  const inputRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const pick = (chosen) => {
    if (chosen) onFile(chosen);
  };

  return (
    <motion.div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label={label}
      whileHover={disabled ? {} : { borderColor: C.green }}
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
      style={{
        width: "100%", minHeight: 122, borderRadius: 10, padding: 18,
        border: `1.5px dashed ${dragging ? C.green : (isDark ? "#2E3150" : "#C8CEE2")}`,
        background: dragging
          ? (isDark ? "#12251B" : "#F2FBF6")
          : (isDark ? "#131426" : "#FBFCFF"),
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", gap: 6, textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.55 : 1,
      }}
    >
      <span style={{ color: C.green }}><MIcon name="cloud_upload" size={26} /></span>
      <strong style={{ fontSize: 13, color: isDark ? "#E8EAF6" : C.navy }}>
        {file ? file.name : label}
      </strong>
      <span style={{ fontSize: 11.5, color: isDark ? "#7A8397" : C.greyDim }}>
        {file ? `${(file.size / 1048576).toFixed(2)} MB selected` : hint}
      </span>
      <input
        id={id} ref={inputRef} type="file" accept={accept} disabled={disabled}
        onChange={(event) => pick(event.target.files?.[0])}
        style={{ display: "none" }}
      />
    </motion.div>
  );
};

/* The operator identity, set once in the header. Shown here so it is obvious
 * whose name goes on this approval, without asking for it a fourth time. */
export function ApprovalIdentity({ operator, what }) {
  if (!operator) {
    return (
      <Notice tone="warn" title="Enter your name before approving">
        Approvals are recorded against a person. Type your name in the header — it is used
        for every approval on this job, and each decision still records it individually.
      </Notice>
    );
  }
  return (
    <div style={{ fontSize: 12, color: "var(--nova-grey-dim)", marginBottom: 10 }}>
      <MIcon name="badge" size={13} /> {what} will be recorded against{" "}
      <strong style={{ color: "inherit" }}>{operator}</strong>. Change it in the header if
      that is not you.
    </div>
  );
}
