export const API_BASE =
  import.meta.env?.VITE_NOVA_ENHANCER_API || "http://127.0.0.1:8081";

/** Extracts a human-readable message from a FastAPI error body. */
function describe(body, fallback) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  }
  return fallback;
}

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Is the backend window still open?`,
      0,
      null,
    );
  }
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
  }
  if (!response.ok) {
    throw new ApiError(
      describe(body, `${response.status} ${response.statusText}`),
      response.status,
      body,
    );
  }
  return body;
}

export const api = {
  health: () => request("/health"),

  jobs: () => request("/api/packages/jobs"),
  overview: () => request("/api/packages/overview"),
  job: (jobId) => request(`/api/packages/jobs/${jobId}`),
  jobProgress: (jobId) => request(`/api/packages/jobs/${jobId}/progress`),
  uploadPackage: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/packages/upload", { method: "POST", body: form });
  },
  compatibility: (jobId, payload) =>
    request(`/api/packages/jobs/${jobId}/compatibility`, json(payload)),

  trainingData: (jobId) => request(`/api/training-data/${jobId}`),
  uploadTrainingData: (jobId, file, role) => {
    const form = new FormData();
    form.append("file", file);
    form.append("role", role);
    return request(`/api/training-data/${jobId}/upload`, { method: "POST", body: form });
  },
  removeTrainingData: (jobId, assetId) =>
    request(`/api/training-data/${jobId}/${assetId}`, { method: "DELETE" }),

  review: (jobId) => request(`/api/readiness/${jobId}/review`),
  saveDecisions: (jobId, payload) => request(`/api/readiness/${jobId}/decisions`, json(payload)),
  buildSnapshot: (jobId) => request(`/api/readiness/${jobId}/snapshot`, { method: "POST" }),
  snapshot: (jobId) => request(`/api/readiness/${jobId}/snapshot`),

  weightOptions: (jobId) => request(`/api/weights/${jobId}/options`),
  previewWeights: (jobId, strategy) => request(`/api/weights/${jobId}/preview`, json({ strategy })),
  approveWeights: (jobId, payload) => request(`/api/weights/${jobId}/approve`, json(payload)),

  trainingOptions: (jobId) => request(`/api/training/${jobId}/options`),
  startTraining: (jobId, payload) => request(`/api/training/${jobId}/start`, json(payload)),
  tasks: (jobId) => request(`/api/training/${jobId}/tasks`),
  task: (taskId) => request(`/api/training/tasks/${taskId}`),
  taskLog: (taskId) => request(`/api/training/tasks/${taskId}/log`),
  cancelTask: (taskId) => request(`/api/training/tasks/${taskId}/cancel`, { method: "POST" }),
  runs: (jobId) => request(`/api/training/${jobId}/runs`),
  resumable: (jobId) => request(`/api/training/${jobId}/resumable`),
  resumeTask: (taskId) =>
    request(`/api/training/tasks/${taskId}/resume`, { method: "POST" }),

  gate: (jobId) => request(`/api/comparison/${jobId}/gate`),
  saveGate: (jobId, payload) => request(`/api/comparison/${jobId}/gate`, json(payload)),
  comparison: (jobId, runId) => request(`/api/comparison/${jobId}/runs/${runId}`),
  approvePromotion: (jobId, payload) => request(`/api/comparison/${jobId}/approve`, json(payload)),

  mlTag: (jobId) => request(`/api/export/${jobId}/ml-tag`),
  approveMlTag: (jobId, payload) => request(`/api/export/${jobId}/ml-tag`, json(payload)),
  uploadInventory: (jobId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/export/${jobId}/inventory-sample`, { method: "POST", body: form });
  },
  buildExport: (jobId, payload) => request(`/api/export/${jobId}/build`, json(payload)),
  exports: (jobId) => request(`/api/export/${jobId}/exports`),
  downloadUrl: (exportId) => `${API_BASE}/api/export/download/${exportId}`,

  audit: (jobId) => request(`/api/audit/${jobId}`),
};

function json(payload) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}
