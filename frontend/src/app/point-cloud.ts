import { afterNextRender, Component, ElementRef, inject, OnDestroy, viewChild } from '@angular/core';

// A slowly turning point cloud of a vase-like surface of revolution, drawn on
// a 2D canvas. It is decoration for the landing screen: a hint of what the
// app produces. Static when the user prefers reduced motion.
@Component({
  selector: 'app-point-cloud',
  host: { class: 'point-cloud', 'aria-hidden': 'true' },
  template: `<canvas #canvas></canvas>`,
})
export class PointCloud implements OnDestroy {
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private readonly host = inject(ElementRef<HTMLElement>);
  private frame = 0;
  private observer?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.start());
  }

  ngOnDestroy() {
    cancelAnimationFrame(this.frame);
    this.observer?.disconnect();
  }

  private start() {
    const canvas = this.canvas().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const points = vasePoints(2600);
    const accent = getComputedStyle(this.host.nativeElement).getPropertyValue('--accent').trim() || '#c8f560';
    let w = 0;
    let h = 0;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    this.observer = new ResizeObserver(resize);
    this.observer.observe(canvas);

    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const tilt = 0.32;
    const draw = (t: number) => {
      const a = still ? 0.6 : t / 7000;
      const ca = Math.cos(a), sa = Math.sin(a), ct = Math.cos(tilt), st = Math.sin(tilt);
      const scale = Math.min(w, h) * 0.36;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = accent;
      for (const [x, y, z] of points) {
        const rx = x * ca + z * sa;
        const rz = -x * sa + z * ca;
        const ry = y * ct - rz * st;
        const depth = y * st + rz * ct; // -1 (back) .. 1 (front)
        const persp = 1 / (1.9 - depth * 0.35);
        const px = w / 2 + rx * scale * persp * 1.6;
        const py = h / 2 + ry * scale * persp * 1.6;
        ctx.globalAlpha = 0.18 + 0.7 * ((depth + 1) / 2);
        const r = 0.6 + 0.9 * ((depth + 1) / 2);
        ctx.fillRect(px - r / 2, py - r / 2, r, r);
      }
      if (!still) this.frame = requestAnimationFrame(draw);
    };
    this.frame = requestAnimationFrame(draw);
  }
}

/** Points on a vase profile, jittered like a photogrammetry scan. */
function vasePoints(n: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const v = i / n; // 0 bottom .. 1 top
    const y = 1 - 2 * v;
    // Body, waist and lip of a vase.
    const radius = 0.34 + 0.3 * Math.sin(Math.PI * Math.min(1, v * 1.25)) - 0.16 * Math.exp(-((v - 0.82) ** 2) / 0.004) + 0.05 * Math.exp(-((v - 0.97) ** 2) / 0.0006);
    const theta = i * golden;
    const j = 1 + (Math.random() - 0.5) * 0.03;
    out.push([Math.cos(theta) * radius * j, -y * 0.95, Math.sin(theta) * radius * j]);
  }
  return out;
}
