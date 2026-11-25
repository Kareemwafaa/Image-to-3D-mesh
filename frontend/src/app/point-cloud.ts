import { afterNextRender, Component, ElementRef, inject, OnDestroy, viewChild } from '@angular/core';

// A slowly turning point cloud of a vase-like surface of revolution, drawn on
// a 2D canvas: a hint of what the app produces. It is decoration, but it can
// be played with:
//   * drag (mouse or touch) to spin it; it keeps spinning with inertia and
//     eases back into the idle turntable;
//   * on devices with a fine pointer, it leans slightly toward the cursor and
//     the points under the cursor brighten and part, like a scanner probe.
// With prefers-reduced-motion there is no idle spin, inertia or hover effect;
// dragging still turns it, and it only redraws while being dragged.

const IDLE_SPIN = 1 / 7000; // rad per ms, the turntable speed
const BASE_PITCH = 0.32; // camera tilt, rad
const DRAG_YAW = 0.009; // rad per px dragged
const DRAG_PITCH = 0.004;
const PITCH_RANGE = [0.02, 0.75] as const;
const SPIN_SETTLE_MS = 900; // how fast inertia decays back to the idle spin
const PITCH_SETTLE_MS = 1400; // how fast a dragged tilt drifts back
const LEAN_YAW = 0.22; // max hover lean, rad
const LEAN_PITCH = 0.12;
const LEAN_SETTLE_MS = 260;
const PROBE_RADIUS = 70; // px, hover highlight radius
const PROBE_PUSH = 5; // px, how far points part under the cursor
const MAX_FLING = 0.012; // rad per ms, caps a hard fling

@Component({
  selector: 'app-point-cloud',
  host: { class: 'point-cloud', 'aria-hidden': 'true' },
  template: `<canvas #canvas></canvas>`,
})
export class PointCloud implements OnDestroy {
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private readonly host = inject(ElementRef<HTMLElement>);
  private frame = 0;
  private cleanup: (() => void)[] = [];

  constructor() {
    afterNextRender(() => this.start());
  }

  ngOnDestroy() {
    cancelAnimationFrame(this.frame);
    this.cleanup.forEach((fn) => fn());
  }

  private start() {
    const canvas = this.canvas().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const points = vasePoints(2600);
    const accent = getComputedStyle(this.host.nativeElement).getPropertyValue('--accent').trim() || '#c8f560';
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;

    let w = 0;
    let h = 0;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      requestDraw();
    };

    // --- view state -------------------------------------------------------
    let yaw = 0.6;
    let pitch = BASE_PITCH;
    let spin = still ? 0 : IDLE_SPIN; // current yaw velocity
    let lean = { yaw: 0, pitch: 0 }; // smoothed hover lean
    let leanTarget = { yaw: 0, pitch: 0 };
    let probe: { x: number; y: number } | null = null; // hover point, canvas px
    let probeStrength = 0;
    let drag: { id: number; x: number; y: number; t: number; v: number } | null = null;
    let visible = true;
    let last = 0;

    // --- drawing ------------------------------------------------------------
    const draw = () => {
      const cy = Math.cos(yaw + lean.yaw), sy = Math.sin(yaw + lean.yaw);
      const p = pitch + lean.pitch;
      const cp = Math.cos(p), sp = Math.sin(p);
      const scale = Math.min(w, h) * 0.36 * 1.6;
      const r2 = PROBE_RADIUS * PROBE_RADIUS;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = accent;
      for (const [x, y, z] of points) {
        const rx = x * cy + z * sy;
        const rz = -x * sy + z * cy;
        const ry = y * cp - rz * sp;
        const depth = y * sp + rz * cp; // -1 (back) .. 1 (front)
        const persp = 1 / (1.9 - depth * 0.35);
        let px = w / 2 + rx * scale * persp;
        let py = h / 2 + ry * scale * persp;
        const near = (depth + 1) / 2;
        let alpha = 0.18 + 0.7 * near;
        let r = 0.6 + 0.9 * near;
        if (probeStrength > 0.01 && probe) {
          const dx = px - probe.x;
          const dy = py - probe.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < r2) {
            // Smooth bump: 1 at the cursor, 0 at the edge of the probe.
            const k = (1 - d2 / r2) ** 2 * probeStrength * (0.4 + 0.6 * near);
            const d = Math.sqrt(d2) || 1;
            px += (dx / d) * PROBE_PUSH * k;
            py += (dy / d) * PROBE_PUSH * k;
            alpha = Math.min(1, alpha + 0.5 * k);
            r += 0.9 * k;
          }
        }
        ctx.globalAlpha = alpha;
        ctx.fillRect(px - r / 2, py - r / 2, r, r);
      }
    };

    // --- animation ----------------------------------------------------------
    const ease = (dt: number, ms: number) => 1 - Math.exp(-dt / ms);
    const tick = (t: number) => {
      this.frame = 0;
      const dt = last ? Math.min(t - last, 64) : 16;
      last = t;
      if (!drag) {
        yaw += spin * dt;
        if (!still) {
          spin += (IDLE_SPIN - spin) * ease(dt, SPIN_SETTLE_MS);
          pitch += (BASE_PITCH - pitch) * ease(dt, PITCH_SETTLE_MS);
        }
      }
      if (!still) {
        const k = ease(dt, LEAN_SETTLE_MS);
        lean = { yaw: lean.yaw + (leanTarget.yaw - lean.yaw) * k, pitch: lean.pitch + (leanTarget.pitch - lean.pitch) * k };
        probeStrength += ((probe && !drag ? 1 : 0) - probeStrength) * k;
      }
      draw();
      // Reduced motion: draw only when something changed.
      if (!still && visible) this.frame = requestAnimationFrame(tick);
    };
    const requestDraw = () => {
      if (!this.frame && visible) {
        last = 0;
        this.frame = requestAnimationFrame(tick);
      }
    };

    // --- input ----------------------------------------------------------------
    const local = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 || drag) return;
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp, v: 0 };
      spin = 0;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('dragging');
    };

    const onMove = (e: PointerEvent) => {
      if (drag && e.pointerId === drag.id) {
        const dx = e.clientX - drag.x;
        const dy = e.clientY - drag.y;
        const dt = Math.max(e.timeStamp - drag.t, 1);
        yaw += dx * DRAG_YAW;
        pitch = clamp(pitch + dy * DRAG_PITCH, ...PITCH_RANGE);
        // Exponentially smoothed velocity, for the fling on release.
        drag.v = 0.7 * ((dx * DRAG_YAW) / dt) + 0.3 * drag.v;
        drag.x = e.clientX;
        drag.y = e.clientY;
        drag.t = e.timeStamp;
        requestDraw();
        return;
      }
      if (finePointer && !still && e.pointerType === 'mouse') {
        const { x, y } = local(e);
        probe = { x, y };
        // Lean toward the cursor: -1..1 across the canvas.
        leanTarget = { yaw: ((x / w) * 2 - 1) * LEAN_YAW, pitch: ((y / h) * 2 - 1) * -LEAN_PITCH };
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.id) return;
      // A pause before letting go means no fling.
      const fresh = e.timeStamp - drag.t < 80;
      spin = still ? 0 : clamp(fresh ? drag.v : 0, -MAX_FLING, MAX_FLING);
      drag = null;
      canvas.classList.remove('dragging');
      requestDraw();
    };

    const onLeave = () => {
      probe = null;
      leanTarget = { yaw: 0, pitch: 0 };
    };

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('pointerleave', onLeave);

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    // Stop animating while the hero is scrolled out of view.
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) requestDraw();
    });
    io.observe(canvas);

    this.cleanup.push(
      () => resizeObserver.disconnect(),
      () => io.disconnect(),
      () => canvas.removeEventListener('pointerdown', onDown),
      () => canvas.removeEventListener('pointermove', onMove),
      () => canvas.removeEventListener('pointerup', onUp),
      () => canvas.removeEventListener('pointercancel', onUp),
      () => canvas.removeEventListener('pointerleave', onLeave),
    );
    resize();
  }
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
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
