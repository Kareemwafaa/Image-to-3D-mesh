import { Injectable } from '@angular/core';

// Client for the FastAPI backend (backend/ in this repo).
// Agreed contract:
//   POST /generate              multipart "image" (+ optional remove_background) -> 202 {job_id}
//   GET  /jobs/{job_id}         -> {job_id, status, progress, result_url, error, ...}
//   GET  /jobs/{job_id}/result  -> .glb (model/gltf-binary)
//
// The backend URL is read at runtime from /config.json, so a deployment can
// point at a different backend without rebuilding the app.

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export interface Job {
  jobId: string;
  status: JobStatus;
  progress: number | null;
  /** Pipeline stage while running: segmenting, multiview, reconstructing, meshing. */
  stage: string | null;
  resultUrl: string | null;
  /** PNG grid of the generated views, available once multi-view diffusion finishes. */
  viewsUrl: string | null;
  error: string | null;
  createdAt: number | null;
}

export interface GenerateOptions {
  removeBackground: boolean;
  /** Marching-cubes grid resolution; higher is more detailed and slower. */
  resolution?: number;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const DEFAULT_API_URL = 'http://localhost:8000';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private apiUrlPromise?: Promise<string>;

  private apiUrl(): Promise<string> {
    this.apiUrlPromise ??= fetch('config.json', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}))
      .then((cfg: { apiUrl?: string }) => (cfg.apiUrl || DEFAULT_API_URL).replace(/\/+$/, ''));
    return this.apiUrlPromise;
  }

  async health(signal?: AbortSignal): Promise<boolean> {
    try {
      const res = await fetch(`${await this.apiUrl()}/health`, { cache: 'no-store', signal });
      return res.ok;
    } catch {
      return false;
    }
  }

  async createJob(image: File, opts: GenerateOptions, signal?: AbortSignal): Promise<string> {
    const form = new FormData();
    form.append('image', image);
    form.append('remove_background', String(opts.removeBackground));
    if (opts.resolution) form.append('mc_resolution', String(opts.resolution));
    const res = await fetch(`${await this.apiUrl()}/generate`, { method: 'POST', body: form, signal });
    if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
    const body = await res.json();
    const id = body.job_id ?? body.id;
    if (!id) throw new Error('The server did not return a job id.');
    return String(id);
  }

  async getJob(id: string, signal?: AbortSignal): Promise<Job> {
    const res = await fetch(`${await this.apiUrl()}/jobs/${encodeURIComponent(id)}`, {
      cache: 'no-store',
      signal,
    });
    if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
    const raw = await res.json();
    return {
      jobId: String(raw.job_id ?? raw.id ?? id),
      status: normalizeStatus(raw.status),
      progress: typeof raw.progress === 'number' ? raw.progress : null,
      stage: typeof raw.stage === 'string' ? raw.stage.toLowerCase() : null,
      resultUrl: raw.result_url ?? null,
      viewsUrl: raw.views_url ? await this.absolute(raw.views_url) : null,
      error: raw.error ?? null,
      createdAt: parseTime(raw.created_at),
    };
  }

  private async absolute(path: string): Promise<string> {
    if (/^https?:\/\//.test(path)) return path;
    return `${await this.apiUrl()}${path.startsWith('/') ? '' : '/'}${path}`;
  }

  /** Downloads the finished mesh as a Blob. */
  async fetchResult(job: Job, signal?: AbortSignal): Promise<Blob> {
    const url = await this.absolute(job.resultUrl ?? `/jobs/${encodeURIComponent(job.jobId)}/result`);
    const res = await fetch(url, { signal });
    if (!res.ok) throw new Error(`Could not load the mesh (${res.status}).`);
    return new Blob([await res.blob()], { type: 'model/gltf-binary' });
  }
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    // FastAPI puts errors in "detail" (a string, or a list of validation errors).
    if (typeof body?.detail === 'string') return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join('; ');
  } catch {
    // not JSON
  }
  if (res.status === 413) return 'That image is too large.';
  return `Request failed (${res.status} ${res.statusText})`;
}

function normalizeStatus(s: unknown): JobStatus {
  switch (String(s).toLowerCase()) {
    case 'succeeded':
    case 'success':
    case 'done':
    case 'completed':
    case 'complete':
      return 'succeeded';
    case 'failed':
    case 'error':
      return 'failed';
    case 'running':
    case 'processing':
    case 'started':
      return 'running';
    default:
      return 'queued';
  }
}

function parseTime(v: unknown): number | null {
  if (typeof v === 'number') return v < 1e12 ? v * 1000 : v;
  if (typeof v !== 'string') return null;
  // FastAPI may omit the timezone on naive datetimes; treat those as UTC.
  const t = Date.parse(/(?:[zZ]|[+-]\d\d:?\d\d)$/.test(v) ? v : v + 'Z');
  return Number.isNaN(t) ? null : t;
}
