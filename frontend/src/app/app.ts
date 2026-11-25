import { Component, computed, ElementRef, OnDestroy, signal, viewChild, inject } from '@angular/core';
import { ApiService, JobStatus } from './api.service';
import { ModelViewer } from './model-viewer';

const POLL_MS = 1500;
const ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];
// Client-side guard only; the backend enforces its own limit (413).
const MAX_BYTES = 10 * 1024 * 1024;

type Phase =
  | { kind: 'idle' }
  | { kind: 'uploading' }
  | { kind: 'processing'; jobId: string; status: JobStatus; progress: number | null }
  | { kind: 'fetching'; jobId: string }
  | { kind: 'done'; jobId: string; modelUrl: string; size: number }
  | { kind: 'error'; message: string };

@Component({
  selector: 'app-root',
  imports: [ModelViewer],
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  private readonly api = inject(ApiService);
  private readonly fileInput = viewChild<ElementRef<HTMLInputElement>>('fileInput');
  private abort?: AbortController;
  private timer?: ReturnType<typeof setInterval>;

  protected readonly accept = ACCEPTED.join(',');
  protected readonly file = signal<File | null>(null);
  protected readonly preview = signal<string | null>(null);
  protected readonly removeBg = signal(true);
  protected readonly phase = signal<Phase>({ kind: 'idle' });
  protected readonly dragging = signal(false);
  protected readonly elapsed = signal(0);

  protected readonly busy = computed(() => {
    const k = this.phase().kind;
    return k === 'uploading' || k === 'processing' || k === 'fetching';
  });
  protected readonly baseName = computed(() => this.file()?.name.replace(/\.[^.]+$/, '') || 'mesh');
  protected readonly progressLabel = computed(() => {
    const p = this.phase();
    if (p.kind === 'uploading') return 'Uploading image…';
    if (p.kind === 'fetching') return 'Downloading mesh…';
    if (p.kind === 'processing') return p.status === 'queued' ? 'Waiting in queue…' : 'Generating mesh…';
    return '';
  });
  protected readonly percent = computed(() => {
    const p = this.phase();
    if (p.kind !== 'processing' || p.progress == null) return null;
    return Math.round(Math.min(1, Math.max(0, p.progress)) * 100);
  });

  ngOnDestroy() {
    this.cancel();
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

  private pickFile(f: File | undefined) {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      this.setPhase({ kind: 'error', message: 'Please choose a PNG, JPEG or WebP image.' });
      return;
    }
    if (f.size > MAX_BYTES) {
      this.setPhase({ kind: 'error', message: 'That image is over 10 MB. Please use a smaller one.' });
      return;
    }
    this.cancel();
    this.file.set(f);
    this.setPreview(URL.createObjectURL(f));
    this.setPhase({ kind: 'idle' });
  }

  protected async generate() {
    const file = this.file();
    if (!file) return;
    this.cancel();
    const ctrl = new AbortController();
    this.abort = ctrl;
    const { signal } = ctrl;
    this.startTimer();

    try {
      this.setPhase({ kind: 'uploading' });
      const jobId = await this.api.createJob(file, this.removeBg(), signal);
      this.setPhase({ kind: 'processing', jobId, status: 'queued', progress: null });

      let job = await this.api.getJob(jobId, signal);
      while (job.status === 'queued' || job.status === 'running') {
        this.setPhase({ kind: 'processing', jobId, status: job.status, progress: job.progress });
        await sleep(POLL_MS, signal);
        job = await this.api.getJob(jobId, signal);
      }
      if (job.status === 'failed') throw new Error(job.error || 'Generation failed on the server.');

      // Fetch the .glb once and reuse it for both the viewer and the download
      // link (an <a download> pointing at another origin is ignored by browsers).
      this.setPhase({ kind: 'fetching', jobId });
      const glb = await this.api.fetchResult(job, signal);
      this.setPhase({ kind: 'done', jobId, modelUrl: URL.createObjectURL(glb), size: glb.size });
    } catch (err) {
      if (signal.aborted) return;
      const message =
        err instanceof TypeError
          ? 'Could not reach the server. Is the backend running?'
          : err instanceof Error
            ? err.message
            : String(err);
      this.setPhase({ kind: 'error', message });
    } finally {
      if (this.abort === ctrl) this.stopTimer();
    }
  }

  protected reset() {
    this.cancel();
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

  private startTimer() {
    this.stopTimer();
    const start = Date.now();
    this.elapsed.set(0);
    this.timer = setInterval(() => this.elapsed.set(Math.round((Date.now() - start) / 1000)), 500);
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
