export const nf = new Intl.NumberFormat();

export function num(value, fallback = "—") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return nf.format(value);
}

export function pct(value, digits = 2, fallback = "—") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return `${Number(value).toFixed(digits)}%`;
}

export function metric(value, digits = 4, fallback = "—") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return Number(value).toFixed(digits);
}

export function when(value, fallback = "—") {
  if (!value) return fallback;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function day(value, fallback = "—") {
  if (!value) return fallback;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString();
}

export function bytes(value, fallback = "—") {
  if (value === null || value === undefined) return fallback;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

export function delta(challenger, champion) {
  if (challenger == null || champion == null) return null;
  if (champion === 0) return null;
  return ((challenger - champion) / Math.abs(champion)) * 100;
}

export function shortHash(value, length = 16) {
  return value ? `${String(value).slice(0, length)}…` : "—";
}
