import { afterNextRender, Component, ElementRef, inject, OnDestroy, viewChild } from '@angular/core';

// The landing page's hero: the product in one loop. A flat grid of points
// shaped like a photo of a mug (with viewfinder corners) lifts off the page,
// row by row, and folds into a 3D point cloud of that mug; the background
// points fade away as the object is "segmented". The mug turns for a while,
// then flattens back into the photo, and the loop repeats.
//
// It can be played with:
//   * drag (mouse or touch) in any direction to turn it all the way round,
//     like a trackball. Grabbing the photo folds it into 3D straight away;
//     let go mid-drag and it keeps spinning, slows down and rights itself;
//   * with a mouse, it leans toward the cursor and the points under the
//     cursor brighten and part, like a scanner probe.
// With prefers-reduced-motion it shows the finished mesh, still, and only
// moves while being dragged. It stops animating while scrolled out of view.
//
// Every point has a "photo" position (on the plane z = 0, snapped to the grid)
// and a "mesh" position. The photo is literally the mesh seen head-on: each
// surface point starts at its own orthographic projection, so folding is
// just depth coming back. Orientation is a unit quaternion from object space
// to view space (x right, y down the screen, z toward the viewer).

type Quat = [number, number, number, number]; // w, x, y, z
type Vec3 = [number, number, number];

// Timing (ms)
const PHOTO_HOLD = 2400;
const MESH_HOLD = 5600;
const MORPH_MS = 1900;
const RESUME_AFTER_DRAG = 1200;
// Motion
const IDLE_SPIN = 1 / 5200; // rad per ms
const REST: Quat = mul(axisAngle([1, 0, 0], -0.36), axisAngle([0, 1, 0], -0.55)); // looking down into the mug, handle toward us
const DRAG_RAD_PER_PX = 0.009;
const MAX_FLING = 0.012;
const FLING_SETTLE_MS = 900;
const UPRIGHT_SETTLE_MS = 900;
const LEAN_YAW = 0.22;
const LEAN_PITCH = 0.12;
const LEAN_SETTLE_MS = 260;
const PROBE_RADIUS = 70; // px
const PROBE_PUSH = 5; // px
// Layout (object units)
const FRAME = { x: 0.95, y: 1.1 }; // half-size of the photo
const GRID = 0.05; // photo grid spacing

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

    const scene = buildScene();
    const style = getComputedStyle(this.host.nativeElement);
    const accent = style.getPropertyValue('--accent').trim() || '#c8f560';
    const frameColor = style.getPropertyValue('--text-3').trim() || '#637080';
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

    // --- state ----------------------------------------------------------------
    // morph: 0 = flat photo, 1 = 3D mesh. It moves linearly toward its target;
    // each point eases through it with a delay by row, so the photo peels up.
    let morph = still ? 1 : 0;
    let morphTarget = morph;
    let hold = PHOTO_HOLD * 0.6; // time spent in the current resting state
    let orient: Quat = still ? REST : [1, 0, 0, 0];
    const uprightAxis = rotate(REST, [0, 1, 0]);
    let fling: Vec3 = [0, 0, 0];
    let idleSpin = 0;
    let sinceRelease = Infinity;
    let lean = { yaw: 0, pitch: 0 };
    let leanTarget = { yaw: 0, pitch: 0 };
    let probe: { x: number; y: number } | null = null;
    let probeStrength = 0;
    let drag: { id: number; x: number; y: number; t: number; v: Vec3 } | null = null;
    let visible = true;
    let last = 0;

    // --- drawing ----------------------------------------------------------------
    const draw = () => {
      const view = mul(mul(axisAngle([1, 0, 0], lean.pitch), axisAngle([0, 1, 0], lean.yaw)), orient);
      const m = matrix(view);
      const scale = Math.min(w, h) * 0.6;
      const project = (x: number, y: number, z: number) => {
        const depth = m[6] * x + m[7] * y + m[8] * z;
        const persp = 1 / (1.9 - depth * 0.35);
        return {
          px: w / 2 + (m[0] * x + m[1] * y + m[2] * z) * scale * persp,
          py: h / 2 + (m[3] * x + m[4] * y + m[5] * z) * scale * persp,
          depth,
        };
      };
      ctx.clearRect(0, 0, w, h);

      // Viewfinder corners around the photo, tilting with it.
      const frameAlpha = 0.7 * (1 - smooth(Math.min(1, morph * 2.2)));
      if (frameAlpha > 0.01) {
        ctx.globalAlpha = frameAlpha;
        ctx.strokeStyle = frameColor;
        ctx.lineWidth = 1;
        const arm = 0.16;
        for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          const cx = sx * FRAME.x, cy = sy * FRAME.y;
          const a = project(cx - sx * arm, cy, 0), c = project(cx, cy, 0), b = project(cx, cy - sy * arm, 0);
          ctx.beginPath();
          ctx.moveTo(a.px, a.py);
          ctx.lineTo(c.px, c.py);
          ctx.lineTo(b.px, b.py);
          ctx.stroke();
        }
      }

      const r2 = PROBE_RADIUS * PROBE_RADIUS;
      const { count, photo, mesh, photoAlpha, flatOnly, delay } = scene;
      ctx.fillStyle = accent;
      for (let i = 0; i < count; i++) {
        // Per-point progress: rows lower in the photo start later.
        const t = smooth(clamp((morph - delay[i] * 0.35) / 0.65, 0, 1));
        let x: number, y: number, z: number, alphaFlat: number;
        const i2 = i * 2, i3 = i * 3;
        if (flatOnly[i]) {
          // Background and filler points: drift outward and fade as the object is cut out.
          if (t > 0.98) continue;
          const spread = 1 + 0.12 * t;
          x = photo[i2] * spread;
          y = photo[i2 + 1] * spread;
          z = 0;
          alphaFlat = photoAlpha[i] * (1 - t);
        } else {
          x = photo[i2] + (mesh[i3] - photo[i2]) * t;
          y = photo[i2 + 1] + (mesh[i3 + 1] - photo[i2 + 1]) * t;
          z = mesh[i3 + 2] * t;
          alphaFlat = photoAlpha[i];
        }
        const p = project(x, y, z);
        let { px, py } = p;
        const near = clamp((p.depth + 1) / 2, 0, 1);
        const meshAlpha = 0.18 + 0.7 * near;
        let alpha = flatOnly[i] ? alphaFlat : alphaFlat + (meshAlpha - alphaFlat) * t;
        let r = 1.5 + (0.6 + 0.9 * near - 1.5) * t;
        if (alpha < 0.01) continue;
        if (probeStrength > 0.01 && probe) {
          const dx = px - probe.x;
          const dy = py - probe.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < r2) {
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

    // --- animation --------------------------------------------------------------
    const ease = (dt: number, ms: number) => 1 - Math.exp(-dt / ms);
    const tick = (now: number) => {
      this.frame = 0;
      const dt = last ? Math.min(now - last, 64) : 16;
      last = now;
      if (!still) {
        sinceRelease += dt;
        const interacting = !!drag || sinceRelease < RESUME_AFTER_DRAG;

        // The loop: rest as a photo, fold up, rest as a mesh, flatten.
        if (morph === morphTarget && !interacting) {
          hold += dt;
          if (morph === 0 && hold > PHOTO_HOLD) (morphTarget = 1), (hold = 0);
          else if (morph === 1 && hold > MESH_HOLD) (morphTarget = 0), (hold = 0);
        }
        const step = dt / MORPH_MS;
        morph = morphTarget > morph ? Math.min(morphTarget, morph + step) : Math.max(morphTarget, morph - step);

        if (!drag) {
          if (morphTarget === 0) {
            // Flattening: face the camera squarely so the photo lands flat.
            fling = [0, 0, 0];
            idleSpin = 0;
            orient = slerp(orient, [1, 0, 0, 0], ease(dt, 320));
            if (morph === 0) sinceRelease = Infinity; // the next fold settles into the resting view again
          } else if (morph < 1 && sinceRelease === Infinity) {
            // Folding on its own: settle into the resting 3/4 view.
            orient = slerp(orient, REST, ease(dt, 420));
          } else {
            const speed = Math.hypot(...fling);
            if (speed > 1e-6) {
              orient = mul(axisAngle(scaleVec(fling, 1 / speed), speed * dt), orient);
              fling = scaleVec(fling, 1 - ease(dt, FLING_SETTLE_MS));
            }
            idleSpin += (IDLE_SPIN - idleSpin) * ease(dt, FLING_SETTLE_MS);
            orient = mul(orient, axisAngle([0, 1, 0], idleSpin * dt));
            if (!interacting) orient = uprightToward(orient, uprightAxis, ease(dt, UPRIGHT_SETTLE_MS));
          }
          orient = normalize(orient);
        }

        const k = ease(dt, LEAN_SETTLE_MS);
        lean = { yaw: lean.yaw + (leanTarget.yaw - lean.yaw) * k, pitch: lean.pitch + (leanTarget.pitch - lean.pitch) * k };
        probeStrength += ((probe && !drag ? 1 : 0) - probeStrength) * k;
      }
      draw();
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
      // Grabbing the photo turns it into the 3D object.
      morphTarget = 1;
      hold = 0;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('dragging');
    };

    const onMove = (e: PointerEvent) => {
      if (drag && e.pointerId === drag.id) {
        const dx = e.clientX - drag.x;
        const dy = e.clientY - drag.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 0) {
          // Trackball: moving along (dx, dy) turns about the screen axis perpendicular to it.
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
        leanTarget = { yaw: ((x / w) * 2 - 1) * LEAN_YAW, pitch: ((y / h) * 2 - 1) * -LEAN_PITCH };
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.id) return;
      const fresh = e.timeStamp - drag.t < 80; // a pause before letting go means no fling
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

// --- the scene ------------------------------------------------------------------

interface Scene {
  count: number;
  /** Photo position per point, (x, y) on the z = 0 plane. */
  photo: Float32Array;
  /** Mesh position per point, (x, y, z). Unused for flat-only points. */
  mesh: Float32Array;
  /** Brightness in the photo (0 = not visible in the photo). */
  photoAlpha: Float32Array;
  /** 1 for points that exist only in the photo (background, gap fillers). */
  flatOnly: Uint8Array;
  /** 0 (top row) .. 1 (bottom row): when the point starts to fold. */
  delay: Float32Array;
}

/** Mug proportions, object units; y points down the screen, so the rim is at -H. */
const MUG = { h: 0.72, rTop: 0.56, rBottom: 0.5, handleX: 0.56, handleR: 0.3, handleTube: 0.075, handleArc: (100 * Math.PI) / 180 };
const MUG_SHIFT = -0.2; // centres mug + handle in the frame

function mugRadius(y: number) {
  return MUG.rTop + (MUG.rBottom - MUG.rTop) * ((y + MUG.h) / (2 * MUG.h));
}

/** Is (x, y) inside the mug's head-on silhouette? */
function inSilhouette(x: number, y: number) {
  const bx = x - MUG_SHIFT;
  if (Math.abs(y) <= MUG.h && Math.abs(bx) <= mugRadius(y)) return true;
  const hx = bx - MUG.handleX;
  const d = Math.hypot(hx, y);
  return (
    Math.abs(d - MUG.handleR) <= MUG.handleTube && Math.abs(Math.atan2(y, hx)) <= MUG.handleArc
  );
}

/** Surface points of a mug (body, inner wall, base, handle) with normals. */
function mugSurface(): { p: Vec3; n: Vec3 }[] {
  const out: { p: Vec3; n: Vec3 }[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  const jit = () => 1 + (Math.random() - 0.5) * 0.02;
  const counts = { side: 1500, inner: 320, base: 280, handle: 500 }; // 2,600, as the hero's label says
  for (let i = 0; i < counts.side; i++) {
    const y = -MUG.h + (2 * MUG.h * (i + 0.5)) / counts.side;
    const r = mugRadius(y) * jit();
    const a = i * golden;
    out.push({ p: [Math.cos(a) * r, y, Math.sin(a) * r], n: [Math.cos(a), 0, Math.sin(a)] });
  }
  for (let i = 0; i < counts.inner; i++) {
    // The inside wall near the rim gives the opening its thickness.
    const y = -MUG.h + 0.5 * MUG.h * ((i + 0.5) / counts.inner);
    const r = (mugRadius(y) - 0.045) * jit();
    const a = i * golden;
    out.push({ p: [Math.cos(a) * r, y, Math.sin(a) * r], n: [-Math.cos(a), 0, -Math.sin(a)] });
  }
  for (let i = 0; i < counts.base; i++) {
    const r = MUG.rBottom * Math.sqrt((i + 0.5) / counts.base);
    const a = i * golden;
    out.push({ p: [Math.cos(a) * r, MUG.h, Math.sin(a) * r], n: [0, 1, 0] });
  }
  for (let i = 0; i < counts.handle; i++) {
    const u = -MUG.handleArc + (2 * MUG.handleArc * (i + 0.5)) / counts.handle;
    const v = i * golden * 3;
    const ring = MUG.handleR + MUG.handleTube * Math.cos(v);
    out.push({
      p: [MUG.handleX + ring * Math.cos(u), ring * Math.sin(u), MUG.handleTube * Math.sin(v)],
      n: [Math.cos(v) * Math.cos(u), Math.cos(v) * Math.sin(u), Math.sin(v)],
    });
  }
  return out.map(({ p, n }) => ({ p: [p[0] + MUG_SHIFT, p[1], p[2]], n }));
}

function buildScene(): Scene {
  const surface = mugSurface();
  const cols = Math.round((2 * FRAME.x) / GRID) + 1;
  const rows = Math.round((2 * FRAME.y) / GRID) + 1;
  const cell = (x: number, y: number) => [Math.round((x + FRAME.x) / GRID), Math.round((y + FRAME.y) / GRID)];
  const taken = new Set<number>();
  const light = normalizeVec([-0.35, -0.4, 0.85]); // from the upper left, in front
  const shade = (n: Vec3) => 0.45 + 0.55 * Math.max(0, dot(n, light));

  const photo: number[] = [];
  const mesh: number[] = [];
  const photoAlpha: number[] = [];
  const flatOnly: number[] = [];
  const delay: number[] = [];
  const push = (px: number, py: number, m: Vec3, a: number, flat: boolean) => {
    photo.push(px, py);
    mesh.push(...m);
    photoAlpha.push(a);
    flatOnly.push(flat ? 1 : 0);
    delay.push((py + FRAME.y) / (2 * FRAME.y));
  };

  // Surface points start at their own head-on projection, snapped to the grid.
  // One point per grid cell shows in the photo: the frontmost, facing us.
  const order = surface.map((_, i) => i).sort((a, b) => surface[b].p[2] - surface[a].p[2]);
  for (const i of order) {
    const { p, n } = surface[i];
    const [c, r] = cell(p[0], p[1]);
    const key = r * cols + c;
    const visible = n[2] > 0.05 && !taken.has(key);
    if (visible) taken.add(key);
    push(c * GRID - FRAME.x, r * GRID - FRAME.y, p, visible ? shade(n) : 0, false);
  }
  // The rest of the photo: grid points over the background, and fillers for
  // any gaps inside the object's silhouette.
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (taken.has(r * cols + c)) continue;
      const x = c * GRID - FRAME.x;
      const y = r * GRID - FRAME.y;
      const inside = inSilhouette(x, y);
      push(x, y, [x, y, 0], inside ? 0.6 : 0.2, true);
    }
  }
  return {
    count: photoAlpha.length,
    photo: new Float32Array(photo),
    mesh: new Float32Array(mesh),
    photoAlpha: new Float32Array(photoAlpha),
    flatOnly: new Uint8Array(flatOnly),
    delay: new Float32Array(delay),
  };
}

// --- maths ------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function smooth(t: number) {
  return t * t * (3 - 2 * t);
}

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

/** Spherical interpolation from a toward b by t, the short way round. */
function slerp(a: Quat, b: Quat, t: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s: Quat = d < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
  d = Math.abs(d);
  if (d > 0.9995) return normalize([a[0] + (s[0] - a[0]) * t, a[1] + (s[1] - a[1]) * t, a[2] + (s[2] - a[2]) * t, a[3] + (s[3] - a[3]) * t]);
  const th = Math.acos(d);
  const wa = Math.sin((1 - t) * th) / Math.sin(th);
  const wb = Math.sin(t * th) / Math.sin(th);
  return [a[0] * wa + s[0] * wb, a[1] * wa + s[1] * wb, a[2] * wa + s[2] * wb, a[3] * wa + s[3] * wb];
}

/** Turn q a fraction k of the way so the object's vertical axis points along `target` (shortest arc). */
function uprightToward(q: Quat, target: Vec3, k: number): Quat {
  const up = rotate(q, [0, 1, 0]);
  const axis = cross(up, target);
  const sin = Math.hypot(...axis);
  const angle = Math.atan2(sin, dot(up, target));
  if (angle < 1e-4) return q;
  const a: Vec3 = sin > 1e-6 ? scaleVec(axis, 1 / sin) : [1, 0, 0];
  return mul(axisAngle(a, angle * k), q);
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
const normalizeVec = (a: Vec3): Vec3 => scaleVec(a, 1 / (Math.hypot(...a) || 1));
