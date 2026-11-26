import { Injectable, signal } from '@angular/core';

// Client for the FastAPI backend (backend/ in this repo).
// Agreed contract:
//   POST /generate              multipart "image" (+ optional remove_background) -> 202 {job_id}
//   GET  /jobs/{job_id}         -> {job_id, status, progress, result_url, error, ...}
//   GET  /jobs/{job_id}/result  -> .glb (model/gltf-binary)
//
// The backend URL is chosen at runtime, so one static build (e.g. on GitHub
// Pages) can talk to any backend, including a tunnel URL that changes every
// session. First match wins:
//   1. ?api=https://... in the page address (saved, then removed from the address)
//   2. the URL saved in this browser (set from the "Server" dialog)
//   3. "apiUrl" in /config.json, deployment-wide default
//   4. http://localhost:8000

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

/** How the generated views become a mesh. */
export type Reconstructor = 'instantmesh' | 'neus';

export interface GenerateOptions {
  removeBackground: boolean;
  reconstructor?: Reconstructor;
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
const STORAGE_KEY = 'facet.apiUrl';
const QUERY_PARAM = 'api';
const HEALTH_TIMEOUT_MS = 8000;

/** Trims and validates a backend URL; returns null if it is not http(s). */
export function normalizeApiUrl(raw: string | null | undefined): string | null {
  const value = raw?.trim();
  if (!value) return null;
  try {
    const url = new URL(/^https?:\/\//i.test(value) ? value : `https://${value}`);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    return `${url.origin}${url.pathname}`.replace(/\/+$/, '');
  } catch {
    return null;
  }
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  /** The backend in use, once resolved. */
  readonly baseUrl = signal<string | null>(null);
  /** Where baseUrl came from: the page address or this browser ("custom"), or the deployment default. */
  readonly source = signal<'custom' | 'default'>('default');
  private defaultUrlPromise?: Promise<string>;

  constructor() {
    const fromQuery = normalizeApiUrl(new URLSearchParams(location.search).get(QUERY_PARAM));
    if (fromQuery) {
      store(fromQuery);
      const url = new URL(location.href);
      url.searchParams.delete(QUERY_PARAM);
      history.replaceState(history.state, '', url);
    }
    const saved = fromQuery ?? normalizeApiUrl(load());
    if (saved) {
      this.baseUrl.set(saved);
      this.source.set('custom');
    }
  }

  /** The deployment default from config.json (or localhost). */
  defaultUrl(): Promise<string> {
    this.defaultUrlPromise ??= fetch('config.json', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}))
      .then((cfg: { apiUrl?: string }) => normalizeApiUrl(cfg.apiUrl) ?? DEFAULT_API_URL);
    return this.defaultUrlPromise;
  }

  /** Use `url` for every request from now on and remember it in this browser; null goes back to the default. */
  async setBaseUrl(url: string | null): Promise<void> {
    const clean = normalizeApiUrl(url);
    store(clean);
    this.source.set(clean ? 'custom' : 'default');
    this.baseUrl.set(clean ?? (await this.defaultUrl()));
  }

  private async apiUrl(): Promise<string> {
    const current = this.baseUrl();
    if (current) return current;
    const fallback = await this.defaultUrl();
    // A custom URL may have been set while config.json was loading.
    if (!this.baseUrl()) this.baseUrl.set(fallback);
    return this.baseUrl()!;
  }

  /** True if the backend (the current one, or `base`) answers /health within a few seconds. */
  async health(base?: string): Promise<boolean> {
    try {
      const res = await fetch(`${base ?? (await this.apiUrl())}/health`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async createJob(image: File, opts: GenerateOptions, signal?: AbortSignal): Promise<string> {
    const form = new FormData();
    form.append('image', image);
    form.append('remove_background', String(opts.removeBackground));
    if (opts.reconstructor) form.append('reconstructor', opts.reconstructor);
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

function load(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function store(url: string | null): void {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, url);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage blocked (private mode): the choice lasts for this page load only
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
