import { Component, computed, ElementRef, inject, OnDestroy, OnInit, signal, viewChild } from '@angular/core';
import { ApiError, ApiService, JobStatus } from './api.service';
import { ModelViewer } from './model-viewer';

const POLL_MS = 2500;
// Consecutive failed status checks tolerated before giving up (network blips
// and 5xx are common over a multi-minute job).
const MAX_POLL_FAILURES = 5;
const HEALTH_MS = 30000;
const ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];
// Client-side guard only; the backend enforces its own limit (413).
const MAX_BYTES = 10 * 1024 * 1024;

type Phase =
  | { kind: 'idle' }
  | { kind: 'uploading' }
  | { kind: 'processing'; jobId: string; status: JobStatus; progress: number | null; stage: string | null }
  | { kind: 'fetching'; jobId: string }
  | { kind: 'done'; jobId: string; modelUrl: string; size: number; seconds: number }
  | { kind: 'error'; message: string; jobId?: string };

// Pipeline steps shown while a job runs. The middle four match the backend's
// optional "stage" field on GET /jobs/{id}.
const PIPELINE = [
  { key: 'upload', label: 'Upload', status: 'Uploading' },
  { key: 'queue', label: 'Queue', status: 'Queued' },
  { key: 'segmenting', label: 'Segment', status: 'Removing background' },
  { key: 'multiview', label: 'Multi-view', status: 'Generating views' },
  { key: 'reconstructing', label: 'Reconstruct', status: 'Reconstructing' },
  { key: 'meshing', label: 'Mesh', status: 'Extracting mesh' },
  { key: 'export', label: 'Export', status: 'Exporting' },
] as const;
type StepKey = (typeof PIPELINE)[number]['key'];

@Component({
  selector: 'app-root',
  imports: [ModelViewer],
  templateUrl: './app.html',
})
export class App implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly fileInput = viewChild<ElementRef<HTMLInputElement>>('fileInput');
  private readonly stage = viewChild<ElementRef<HTMLElement>>('stage');
  private abort?: AbortController;
  private timer?: ReturnType<typeof setInterval>;
  private healthTimer?: ReturnType<typeof setInterval>;
  private jobId?: string;

  /** Whether the running job removes the background (adds the Segment step). */
  private readonly segmentStep = signal(true);
  protected readonly steps = computed(() =>
    PIPELINE.filter((s) => s.key !== 'segmenting' || this.segmentStep()),
  );
  protected readonly resolutions = [
    { value: 128, label: 'Draft' },
    { value: 256, label: 'Standard' },
    { value: 320, label: 'High' },
  ];
  protected readonly accept = ACCEPTED.join(',');
  protected readonly file = signal<File | null>(null);
  protected readonly preview = signal<string | null>(null);
  protected readonly removeBg = signal(true);
  protected readonly resolution = signal(256);
  protected readonly phase = signal<Phase>({ kind: 'idle' });
  protected readonly dragging = signal(false);
  protected readonly elapsed = signal(0);
  protected readonly apiOnline = signal<boolean | null>(null);

  protected readonly busy = computed(() => {
    const k = this.phase().kind;
    return k === 'uploading' || k === 'processing' || k === 'fetching';
  });
  protected readonly baseName = computed(() => this.file()?.name.replace(/\.[^.]+$/, '') || 'mesh');

  private readonly stepKey = computed<StepKey | null>(() => {
    const p = this.phase();
    if (p.kind === 'uploading') return 'upload';
    if (p.kind === 'fetching') return 'export';
    if (p.kind !== 'processing') return null;
    if (p.status === 'queued') return 'queue';
    const known = this.steps().find((s) => s.key === p.stage);
    return known ? known.key : this.segmentStep() ? 'segmenting' : 'multiview';
  });
  /** Index into steps() of the step currently running. */
  protected readonly stepIndex = computed(() => this.steps().findIndex((s) => s.key === this.stepKey()));
  protected readonly percent = computed(() => {
    const p = this.phase();
    if (p.kind !== 'processing' || p.status !== 'running' || p.progress == null) return null;
    return Math.round(Math.min(1, Math.max(0, p.progress)) * 100);
  });
  protected readonly statusText = computed(() => {
    const p = this.phase();
    switch (p.kind) {
      case 'idle':
        return this.file() ? 'Ready' : 'Waiting for image';
      case 'uploading':
      case 'processing':
      case 'fetching':
        return PIPELINE.find((s) => s.key === this.stepKey())?.status ?? 'Working';
      case 'done':
        return 'Complete';
      case 'error':
        return 'Failed';
    }
  });
  protected readonly currentJobId = computed(() => {
    const p = this.phase();
    return 'jobId' in p ? p.jobId : undefined;
  });

  ngOnInit() {
    const check = () => this.api.health().then((ok) => this.apiOnline.set(ok));
    check();
    this.healthTimer = setInterval(check, HEALTH_MS);

    // A job id in the URL means a reload during a long job: pick it back up.
    const jobId = new URLSearchParams(location.search).get('job');
    if (jobId) this.run(null, jobId);
  }

  ngOnDestroy() {
    this.cancel();
    clearInterval(this.healthTimer);
    this.setPreview(null);
    this.setPhase({ kind: 'idle' });
  }

  protected onFileChange(e: Event) {
    this.pickFile((e.target as HTMLInputElement).files?.[0]);
  }

  protected onDragOver(e: DragEvent) {
    e.preventDefault();
    if (!this.busy()) this.dragging.set(true);
  }

  protected onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragging.set(false);
    if (!this.busy()) this.pickFile(e.dataTransfer?.files?.[0]);
  }

  protected browse() {
    this.fileInput()?.nativeElement.click();
  }

  private pickFile(f: File | undefined) {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      this.setPhase({ kind: 'error', message: 'Unsupported file type. Use a PNG, JPEG or WebP image.' });
      return;
    }
    if (f.size > MAX_BYTES) {
      this.setPhase({ kind: 'error', message: 'This image is larger than 10 MB. Resize it and try again.' });
      return;
    }
    this.cancel();
    this.file.set(f);
    this.setPreview(URL.createObjectURL(f));
    this.setPhase({ kind: 'idle' });
  }

  protected generate() {
    const file = this.file();
    if (file) this.run(file);
  }

  /** Uploads `file` as a new job, or resumes polling `resumeId`, until the mesh is loaded. */
  private async run(file: File | null, resumeId?: string) {
    this.cancel();
    const ctrl = new AbortController();
    this.abort = ctrl;
    const { signal } = ctrl;
    this.jobId = resumeId;
    let start = this.startTimer();

    try {
      let jobId: string;
      if (file) {
        this.segmentStep.set(this.removeBg());
        this.setPhase({ kind: 'uploading' });
        jobId = await this.api.createJob(
          file,
          { removeBackground: this.removeBg(), resolution: this.resolution() },
          signal,
        );
      } else {
        jobId = resumeId!;
      }
      this.jobId = jobId;
      setJobInUrl(jobId);
      this.setPhase({ kind: 'processing', jobId, status: 'queued', progress: null, stage: null });

      let failures = 0;
      for (;;) {
        let job;
        try {
          job = await this.api.getJob(jobId, signal);
          failures = 0;
        } catch (err) {
          const transient = err instanceof TypeError || (err instanceof ApiError && err.status >= 500);
          if (!transient || ++failures >= MAX_POLL_FAILURES) throw err;
          await sleep(POLL_MS * failures, signal);
          continue;
        }
        if (!file && job.createdAt && job.createdAt < start) start = this.startTimer(job.createdAt);
        if (job.status === 'failed') throw new Error(job.error || 'The reconstruction failed on the server.');
        if (job.status === 'succeeded') {
          // Fetch the .glb once and reuse it for both the viewer and the download
          // link (an <a download> pointing at another origin is ignored by browsers).
          this.setPhase({ kind: 'fetching', jobId });
          const glb = await this.api.fetchResult(job, signal);
          this.setPhase({
            kind: 'done',
            jobId,
            modelUrl: URL.createObjectURL(glb),
            size: glb.size,
            seconds: Math.round((Date.now() - start) / 1000),
          });
          // On narrow screens the viewer sits below the controls; bring it into view.
          const stage = this.stage()?.nativeElement;
          if (stage && stage.getBoundingClientRect().top > window.innerHeight * 0.5) {
            stage.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
          return;
        }
        this.setPhase({ kind: 'processing', jobId, status: job.status, progress: job.progress, stage: job.stage });
        await sleep(POLL_MS, signal);
      }
    } catch (err) {
      if (signal.aborted) return;
      let message =
        err instanceof TypeError
          ? 'Could not reach the generation server. Check that the backend is running.'
          : err instanceof Error
            ? err.message
            : String(err);
      if (!file && err instanceof ApiError && err.status === 404) {
        message = 'That job no longer exists on the server. Upload the image again to start a new one.';
        setJobInUrl(null);
      }
      this.setPhase({ kind: 'error', message, jobId: this.jobId });
    } finally {
      if (this.abort === ctrl) this.stopTimer();
    }
  }

  protected cancelJob() {
    this.cancel();
    setJobInUrl(null);
    this.setPhase({ kind: 'idle' });
  }

  protected reset() {
    this.cancel();
    setJobInUrl(null);
    this.file.set(null);
    this.setPreview(null);
    this.setPhase({ kind: 'idle' });
    const input = this.fileInput()?.nativeElement;
    if (input) input.value = '';
  }

  protected formatBytes(n: number) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  protected formatTime(s: number) {
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, '0')}`;
  }

  private cancel() {
    this.abort?.abort();
    this.abort = undefined;
    this.stopTimer();
  }

  // Object URLs are released whenever they are replaced.
  private setPreview(url: string | null) {
    const old = this.preview();
    if (old) URL.revokeObjectURL(old);
    this.preview.set(url);
  }

  private setPhase(p: Phase) {
    const old = this.phase();
    if (old.kind === 'done' && !(p.kind === 'done' && p.modelUrl === old.modelUrl)) {
      URL.revokeObjectURL(old.modelUrl);
    }
    this.phase.set(p);
  }

  private startTimer(start = Date.now()) {
    this.stopTimer();
    this.elapsed.set(Math.max(0, Math.round((Date.now() - start) / 1000)));
    this.timer = setInterval(() => this.elapsed.set(Math.round((Date.now() - start) / 1000)), 500);
    return start;
  }

  private stopTimer() {
    clearInterval(this.timer);
    this.timer = undefined;
  }
}

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

/** Keeps the running job's id in the address bar so a reload can resume it. */
function setJobInUrl(jobId: string | null) {
  const url = new URL(location.href);
  if (jobId) url.searchParams.set('job', jobId);
  else url.searchParams.delete('job');
  history.replaceState(history.state, '', url);
}
