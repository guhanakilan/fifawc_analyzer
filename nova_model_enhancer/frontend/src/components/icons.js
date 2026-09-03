/* Inline SVG icon set.
 *
 * The reference application pulls Material Symbols from Google Fonts. This
 * application is localhost-first and must survive a machine with no outbound
 * network, where a webfont failure leaves the literal ligature text
 * ("check_circle") on screen. These are drawn inline instead, in a single
 * 24x24 stroke style, so every icon renders offline.
 *
 * Names are kept identical to the Material Symbols names the reference uses, so
 * the mapping between the two applications stays obvious.
 */

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
const F = { fill: "currentColor", stroke: "none" };

export const ICON_STYLE = { stroke: S, fill: F };

export const ICONS = {
  // ── Stages ────────────────────────────────────────────────────────────────
  inventory_2: { d: "M3 7h18M5 7v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V7M4 7l2-3h12l2 3M10 12h4" },
  database: { d: "M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3ZM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" },
  fact_check: { d: "M4 4h16v16H4zM7 8h5M7 12h5M7 16h5M15.5 11.5l1.5 1.5 3-3" },
  balance: { d: "M12 4v16M7 20h10M12 6 4 9l3 6h2l3-6ZM12 6l8 3-3 6h-2l-3-6Z" },
  play_circle: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM10 8.5l6 3.5-6 3.5z" },
  compare_arrows: { d: "M3 9h13l-3-3M21 15H8l3 3" },
  deployed_code: { d: "M12 3 4 7v10l8 4 8-4V7l-8-4ZM4 7l8 4 8-4M12 11v10" },

  // ── Actions and status ────────────────────────────────────────────────────
  check: { d: "M4 12.5 9.5 18 20 6.5" },
  check_circle: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM8 12.2l2.7 2.8L16 9.5" },
  cancel: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM9 9l6 6M15 9l-6 6" },
  warning: { d: "M12 3.5 22 20H2L12 3.5ZM12 10v4.5M12 17.2v.1" },
  error: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM12 7.5V13M12 16.2v.1" },
  info: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM12 11v5.5M12 7.8v.1" },
  help: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM9.6 9.5a2.5 2.5 0 1 1 3.4 2.3c-.7.3-1 .9-1 1.6v.4M12 17.2v.1" },
  report: { d: "M8 3h8l5 5v8l-5 5H8l-5-5V8l5-5ZM12 7.5V13M12 16.2v.1" },
  verified: { d: "m12 2.5 2.4 2.1 3.2-.2.6 3.1 2.6 1.8-1.4 2.9 1.4 2.9-2.6 1.8-.6 3.1-3.2-.2L12 21.5l-2.4-2.1-3.2.2-.6-3.1L3.2 14.7l1.4-2.9-1.4-2.9 2.6-1.8.6-3.1 3.2.2ZM8.6 12.2l2.5 2.5 4.3-4.6" },
  policy: { d: "M12 3 4 6v6c0 4.4 3.4 7.9 8 9 4.6-1.1 8-4.6 8-9V6l-8-3ZM9.5 12l1.9 2 3.3-3.6" },
  gavel: { d: "m4.5 19.5 7-7M3 21h7M13.5 3.5 20.5 10.5M11 6 18 13M9.5 7.5 16.5 14.5" },
  rule: { d: "M3 6.5 5 8.5 9 4.5M3 16.5 5 18.5 9 14.5M13 6.5h8M13 16.5h8" },
  how_to_reg: { d: "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM2.5 20c0-3.3 2.9-5.5 6.5-5.5 1.2 0 2.3.2 3.2.7M14.5 17.5l2 2 4-4.5" },
  how_to_vote: { d: "M12 3 6.5 8.5 9 11l5.5-5.5L12 3ZM3 15l3-3h12l3 3v5H3v-5ZM3 15h18" },
  block: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM5.6 5.6l12.8 12.8" },
  add: { d: "M12 5v14M5 12h14" },
  delete: { d: "M4 6.5h16M9.5 6.5V4h5v2.5M6.5 6.5 7.5 20h9l1-13.5M10 10v6M14 10v6" },
  refresh: { d: "M20 12a8 8 0 1 1-2.6-5.9M20 4v4.5h-4.5" },
  restart_alt: { d: "M12 5V2L8.5 5.5 12 9V6a6 6 0 1 1-5.6 3.9" },
  upload: { d: "M12 16V4M7.5 8.5 12 4l4.5 4.5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" },
  cloud_upload: { d: "M7 18a4 4 0 0 1-.4-8A6 6 0 0 1 18 10.5a3.8 3.8 0 0 1-.6 7.5M12 20v-9M9 14l3-3 3 3" },
  download: { d: "M12 4v12M7.5 11.5 12 16l4.5-4.5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" },
  folder_zip: { d: "M3 6.5a1 1 0 0 1 1-1h5l2 2.5h9a1 1 0 0 1 1 1V19a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6.5ZM14 10h2M14 12.5h2M14 15h2" },
  folder_open: { d: "M3 6.5a1 1 0 0 1 1-1h5l2 2.5h9a1 1 0 0 1 1 1V19a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6.5Z" },
  archive: { d: "M3 4.5h18v4H3zM5 8.5V20h14V8.5M9.5 12.5h5" },
  inventory: { d: "M4 4h16v5H4zM6 9v11h12V9M10 12.5h4" },
  badge: { d: "M9 3.5h6v2h4v15H5v-15h4v-2ZM12 11a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM8 17c0-2 1.8-3.2 4-3.2s4 1.2 4 3.2" },
  location_on: { d: "M12 21s7-5.7 7-11a7 7 0 1 0-14 0c0 5.3 7 11 7 11ZM12 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" },
  light_mode: { d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8ZM12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8" },
  dark_mode: { d: "M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" },
  play_arrow: { d: "M7 4.5 19 12 7 19.5z" },
  stop_circle: { d: "M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18ZM9.5 9.5h5v5h-5z" },
  build: { d: "m14.5 3.5 3 3-2.5 2.5-3-3a4.5 4.5 0 0 1 2.5-2.5ZM12 9 4.5 16.5a2.1 2.1 0 0 0 3 3L15 12" },
  tune: { d: "M4 7h9M17 7h3M4 17h3M11 17h9M15 4.5v5M9 14.5v5" },
  terminal: { d: "M3 4.5h18v15H3zM7 9.5l3 2.5-3 2.5M12.5 15h4.5" },
  history: { d: "M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5M12 8v4.5l3 1.8" },
  science: { d: "M10 3.5v6L4.5 18a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9.5v-6M8.5 3.5h7M7 14.5h10" },
  model_training: { d: "M12 3a9 9 0 1 0 8.5 6M17 3.5v4.5h4.5M8.5 15V11M12 15V9M15.5 15v-2.5" },
  query_stats: { d: "M4 20V10M9 20V6M14 20v-5M19 20v-9M3 8.5 8.5 3l4 4 6-6" },
  visibility: { d: "M12 5.5c5 0 9 4.2 9 6.5s-4 6.5-9 6.5S3 14.3 3 12s4-6.5 9-6.5ZM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" },
  lock: { d: "M6 10.5h12v9H6zM8.5 10.5V7.5a3.5 3.5 0 1 1 7 0v3M12 14v2.5" },
  label: { d: "M3 6h11.5l5 6-5 6H3zM8 12h.1" },
  hourglass_top: { d: "M7 3h10M7 21h10M7 3v3.5c0 2.2 2.2 3.5 5 5.5 2.8-2 5-3.3 5-5.5V3M7 21v-3.5c0-2.2 2.2-3.5 5-5.5 2.8 2 5 3.3 5 5.5V21" },
  arrow_forward: { d: "M4 12h15M13 6l6 6-6 6" },
  inbox: { d: "M3 13.5 5.5 5h13L21 13.5V19a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5.5ZM3 13.5h5a4 4 0 0 0 8 0h5" },
  home: { d: "M4 10.5 12 4l8 6.5V19a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1v-8.5Z" },
  expand_more: { d: "M6 9.5 12 15.5 18 9.5" },
  expand_less: { d: "M6 14.5 12 8.5 18 14.5" },
  list: { d: "M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" },
  filter_list: { d: "M4 6h16M7 12h10M10 18h4" },
};

export const FALLBACK_ICON = ICONS.help;
