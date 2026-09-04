/* The NoVA ML Workbench palette and theme tokens.
 *
 * Values are taken verbatim from the reference application
 * (frontend/src/App.jsx's `C` object and frontend/src/nova-theme.css) so this
 * application is visually the same product, not merely a similar one.
 */

export const C = {
  navy: "#28293D",
  navyDark: "#1E1F30",
  navyMid: "#32334A",
  green: "#05C16E",
  greenDim: "#05C16E40",
  indigo: "#415395",
  grey: "#C1C8D0",
  greyDim: "var(--nova-grey-dim)",
  bg: "#F1F4F8",
  bgCard: "#FFFFFF",
  red: "#FF7F7F",
  mint: "#98DAB6",
  text: "#2C2D3F",
  textMid: "var(--nova-text-mid)",
};

/* Semantic colours for status. The reference uses amber #F59E0B for "Running"
 * and C.red / C.green for terminal states; the same mapping is used here. */
export const STATUS = {
  ok: C.green,
  warn: "#F59E0B",
  bad: "#EF4444",
  info: C.indigo,
  muted: C.grey,
};

export const SPRING = { type: "spring", stiffness: 340, damping: 30 };
export const SPRING_FAST = { type: "spring", stiffness: 420, damping: 26 };
