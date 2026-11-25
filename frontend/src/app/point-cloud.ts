import { afterNextRender, Component, ElementRef, inject, OnDestroy, viewChild } from '@angular/core';

// A slowly turning point cloud of a vase-like surface of revolution, drawn on
// a 2D canvas: a hint of what the app produces. It is decoration, but it can
// be played with:
//   * drag (mouse or touch) in any direction to turn it all the way round,
//     like a trackball; let go mid-drag and it keeps spinning, slows down,
//     then rights itself and returns to the idle turntable;
//   * on devices with a fine pointer, it leans slightly toward the cursor and
//     the points under the cursor brighten and part, like a scanner probe.
// With prefers-reduced-motion there is no idle spin, inertia, righting or
// hover effect; dragging still turns it, and it only redraws while dragged.
//
// Orientation is a unit quaternion mapping object space to view space
// (x right, y down the screen, z toward the viewer). Drags rotate about view
// axes, the idle turntable rotates about the object's own vertical axis.

type Quat = [number, number, number, number]; // w, x, y, z
type Vec3 = [number, number, number];

const IDLE_SPIN = 1 / 7000; // rad per ms, the turntable speed
const BASE_PITCH = 0.32; // resting camera tilt, rad
const START_YAW = 0.6;
const DRAG_RAD_PER_PX = 0.009;
const MAX_FLING = 0.012; // rad per ms, caps a hard fling
const FLING_SETTLE_MS = 900; // how fast a fling slows down
const UPRIGHT_DELAY_MS = 1200; // idle time before it rights itself
const UPRIGHT_SETTLE_MS = 900;
const LEAN_YAW = 0.22; // max hover lean, rad
const LEAN_PITCH = 0.12;
const LEAN_SETTLE_MS = 260;
const PROBE_RADIUS = 70; // px, hover highlight radius
const PROBE_PUSH = 5; // px, how far points part under the cursor

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
    // Resting pose: turned START_YAW about the vertical, tilted BASE_PITCH.
    let orient = mul(axisAngle([1, 0, 0], BASE_PITCH), axisAngle([0, 1, 0], START_YAW));
    const uprightAxis = rotate(axisAngle([1, 0, 0], BASE_PITCH), [0, 1, 0]);
    let fling: Vec3 = [0, 0, 0]; // view-space angular velocity, rad per ms
    let idleSpin = still ? 0 : IDLE_SPIN; // ramps back up after a drag
    let sinceRelease = Infinity;
    let lean = { yaw: 0, pitch: 0 }; // smoothed hover lean
    let leanTarget = { yaw: 0, pitch: 0 };
    let probe: { x: number; y: number } | null = null; // hover point, canvas px
    let probeStrength = 0;
    let drag: { id: number; x: number; y: number; t: number; v: Vec3 } | null = null;
    let visible = true;
    let last = 0;

    // --- drawing ------------------------------------------------------------
    const draw = () => {
      const view = mul(mul(axisAngle([1, 0, 0], lean.pitch), axisAngle([0, 1, 0], lean.yaw)), orient);
      const m = matrix(view);
      const scale = Math.min(w, h) * 0.36 * 1.6;
      const r2 = PROBE_RADIUS * PROBE_RADIUS;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = accent;
      for (const [x, y, z] of points) {
        const rx = m[0] * x + m[1] * y + m[2] * z;
        const ry = m[3] * x + m[4] * y + m[5] * z;
        const depth = m[6] * x + m[7] * y + m[8] * z; // -1 (back) .. 1 (front)
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
      if (!drag && !still) {
        sinceRelease += dt;
        // Fling: keep turning about the view axis it was thrown around, slowing down.
        const speed = Math.hypot(...fling);
        if (speed > 1e-6) {
          orient = mul(axisAngle(scaleVec(fling, 1 / speed), speed * dt), orient);
          fling = scaleVec(fling, 1 - ease(dt, FLING_SETTLE_MS));
        }
        // Turntable: about the object's own vertical axis, so it spins in place.
        idleSpin += (IDLE_SPIN - idleSpin) * ease(dt, FLING_SETTLE_MS);
        orient = mul(orient, axisAngle([0, 1, 0], idleSpin * dt));
        // After a pause, swing the vase's axis back to upright (shortest way).
        if (sinceRelease > UPRIGHT_DELAY_MS) {
          const up = rotate(orient, [0, 1, 0]);
          const axis = cross(up, uprightAxis);
          const sin = Math.hypot(...axis);
          const cos = dot(up, uprightAxis);
          const angle = Math.atan2(sin, cos);
          if (angle > 1e-4) {
            const k = ease(dt, UPRIGHT_SETTLE_MS);
            const a: Vec3 = sin > 1e-6 ? scaleVec(axis, 1 / sin) : [1, 0, 0]; // upside down: any horizontal axis
            orient = mul(axisAngle(a, angle * k), orient);
          }
        }
        orient = normalize(orient);
        const k = ease(dt, LEAN_SETTLE_MS);
        lean = { yaw: lean.yaw + (leanTarget.yaw - lean.yaw) * k, pitch: lean.pitch + (leanTarget.pitch - lean.pitch) * k };
        probeStrength += ((probe ? 1 : 0) - probeStrength) * k;
      } else if (drag && !still) {
        probeStrength += (0 - probeStrength) * ease(dt, LEAN_SETTLE_MS);
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
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 || drag) return;
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp, v: [0, 0, 0] };
      fling = [0, 0, 0];
      idleSpin = 0;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('dragging');
    };

    const onMove = (e: PointerEvent) => {
      if (drag && e.pointerId === drag.id) {
        const dx = e.clientX - drag.x;
        const dy = e.clientY - drag.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 0) {
          // Trackball: the surface follows the pointer. Moving along (dx, dy)
          // turns the vase about the screen axis perpendicular to it.
          const axis: Vec3 = [-dy / dist, dx / dist, 0];
          const angle = dist * DRAG_RAD_PER_PX;
          orient = normalize(mul(axisAngle(axis, angle), orient));
          const dt = Math.max(e.timeStamp - drag.t, 1);
          drag.v = addVec(scaleVec(axis, (0.7 * angle) / dt), scaleVec(drag.v, 0.3));
        }
        drag.x = e.clientX;
        drag.y = e.clientY;
        drag.t = e.timeStamp;
        requestDraw();
        return;
      }
      if (finePointer && !still && e.pointerType === 'mouse') {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        probe = { x, y };
        // Lean toward the cursor: -1..1 across the canvas.
        leanTarget = { yaw: ((x / w) * 2 - 1) * LEAN_YAW, pitch: ((y / h) * 2 - 1) * -LEAN_PITCH };
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.id) return;
      // A pause before letting go means no fling.
      const fresh = e.timeStamp - drag.t < 80;
      const speed = Math.hypot(...drag.v);
      fling = still || !fresh || speed === 0 ? [0, 0, 0] : scaleVec(drag.v, Math.min(1, MAX_FLING / speed));
      sinceRelease = 0;
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

// --- small quaternion helpers ------------------------------------------------

function axisAngle([x, y, z]: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  return [Math.cos(angle / 2), x * s, y * s, z * s];
}

/** a * b: apply b, then a. */
function mul([aw, ax, ay, az]: Quat, [bw, bx, by, bz]: Quat): Quat {
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function normalize(q: Quat): Quat {
  const n = Math.hypot(...q) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Row-major 3x3 rotation matrix of a unit quaternion. */
function matrix([w, x, y, z]: Quat): number[] {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

function rotate(q: Quat, [x, y, z]: Vec3): Vec3 {
  const m = matrix(q);
  return [m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z];
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const scaleVec = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const addVec = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

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
