"""Turn results.jsonl into a Markdown table, a CSV and an SVG chart.

Statistics:
* Each cell is the mean over the objects a method reconstructed, with a 95%
  bootstrap confidence interval of that mean (2,000 resamples of the objects,
  percentile method). With ~30 objects the intervals are wide; overlapping
  intervals mean "not clearly different", not "the same".
* Because every method sees the same objects and the same generated views,
  methods are also compared per object (a paired design, much more sensitive
  than comparing two means): the win rate, and a two-sided Wilcoxon
  signed-rank test on the per-object differences (no normality assumption).
* Failed runs are counted in the "ok" column and left out of the means. A
  method that fails on hard objects would look better than it is, so read the
  means together with that column.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METHOD_LABELS = {"instantmesh": "InstantMesh", "neus": "NeuS (classic)"}
VIEWS_LABELS = {"generated": "Zero123++ views", "oracle": "ground-truth views"}
# Categorical slots 1 and 2 of the reference palette (validated for colour-vision deficiency).
METHOD_COLORS = {"instantmesh": "#257de0", "neus": "#f47230"}
FALLBACK_COLORS = ("#00b894", "#6a4ce2", "#ec4647")

# (key in the row, column title, higher is better, format)
COLUMNS = (
    ("aligned.chamfer", "Chamfer ↓", False, "{:.4f}"),
    ("aligned.fscore@0.05", "F@0.05 ↑", True, "{:.3f}"),
    ("aligned.fscore@0.1", "F@0.1 ↑", True, "{:.3f}"),
    ("aligned.fscore@0.2", "F@0.2 ↑", True, "{:.3f}"),
    ("aligned.normal_consistency", "Normal cons. ↑", True, "{:.3f}"),
    ("render.mask_iou", "Mask IoU ↑", True, "{:.3f}"),
    ("render.psnr", "PSNR (dB) ↑", True, "{:.2f}"),
)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # A resumed run may have retried a failure: keep the last row per key.
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[(row["object"], row["method"], row["views"])] = row
    return list(latest.values())


def get(row: dict, dotted: str):
    cur = row
    for part in dotted.split(".", 1):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class Group:
    method: str
    views: str
    rows: list[dict]
    total: int

    @property
    def ok(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") == "ok"]

    @property
    def label(self) -> str:
        name = METHOD_LABELS.get(self.method, self.method)
        return name if self.views == "generated" else f"{name}, {VIEWS_LABELS.get(self.views, self.views)}"

    def values(self, key: str) -> np.ndarray:
        vals = [get(r, key) for r in self.ok]
        return np.array([v for v in vals if v is not None and math.isfinite(v)], dtype=np.float64)

    def summary(self, key: str) -> tuple[float, float, float] | None:
        v = self.values(key)
        if len(v) == 0:
            return None
        lo, hi = bootstrap_ci(v)
        return float(v.mean()), lo, hi


def groups(rows: list[dict]) -> list[Group]:
    order = {"instantmesh": 0, "neus": 1}
    objects = {r["object"] for r in rows}
    keys = sorted({(r["method"], r["views"]) for r in rows}, key=lambda k: (k[1] != "generated", order.get(k[0], 9), k))
    return [Group(m, v, [r for r in rows if r["method"] == m and r["views"] == v], len(objects)) for m, v in keys]


def paired(rows: list[dict], a: str, b: str, views: str, key: str, higher_better: bool) -> dict | None:
    """Per-object comparison of methods a and b on the same objects."""
    by = {}
    for r in rows:
        if r["views"] == views and r.get("status") == "ok" and get(r, key) is not None:
            by.setdefault(r["object"], {})[r["method"]] = get(r, key)
    pairs = [(d[a], d[b]) for d in by.values() if a in d and b in d]
    if len(pairs) < 5:
        return None
    x = np.array(pairs, dtype=np.float64)
    diff = x[:, 0] - x[:, 1]
    wins = int((diff > 0).sum() if higher_better else (diff < 0).sum())
    try:
        from scipy.stats import wilcoxon

        p = float(wilcoxon(x[:, 0], x[:, 1]).pvalue) if np.any(diff != 0) else 1.0
    except Exception:  # pragma: no cover - scipy missing or degenerate input
        p = math.nan
    return {"n": len(pairs), "wins": wins, "p": p, "mean_diff": float(diff.mean())}


def _fmt(summary, fmt: str) -> str:
    if summary is None:
        return "–"
    mean, lo, hi = summary
    if math.isnan(lo):
        return fmt.format(mean)
    return f"{fmt.format(mean)} <sub>[{fmt.format(lo)}, {fmt.format(hi)}]</sub>"


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def markdown(rows: list[dict], config: dict | None = None) -> str:
    gs = groups(rows)
    head = ["Method", "Ok"] + [c[1] for c in COLUMNS] + ["Recon. time (s)", "Peak GPU (GiB)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for g in gs:
        cells = [g.label, f"{len(g.ok)}/{g.total}"]
        cells += [_fmt(g.summary(key), fmt) for key, _, _, fmt in COLUMNS]
        t = _median(g.values("time_recon_s"))
        mem = g.values("peak_gpu_recon_mib")
        cells.append("–" if t is None else f"{t:.1f}")
        cells.append("–" if len(mem) == 0 else f"{mem.max() / 1024:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    notes = [
        "",
        "Mean over objects, 95% bootstrap interval in brackets. Distances are in units of the ground "
        "truth's half-size (it fits [-1, 1]³) after similarity alignment; F@τ counts points within τ. "
        "Time is the median reconstruction + cleanup time per object; memory is the largest peak "
        "PyTorch allocation during reconstruction.",
    ]
    gen = [g for g in gs if g.views == "generated"]
    view_t = _median(gen[0].values("time_views_s")) if gen else None
    if view_t is not None:
        notes.append(f"Generating the six Zero123++ views (shared by both methods) took a median {view_t:.1f} s.")
    methods = [g.method for g in gen]
    if "instantmesh" in methods and "neus" in methods:
        for key, title, hb, _ in COLUMNS[:5]:
            pr = paired(rows, "instantmesh", "neus", "generated", key, hb)
            if pr:
                notes.append(
                    f"- {title.split(' ')[0]}: InstantMesh better on {pr['wins']}/{pr['n']} objects "
                    f"(Wilcoxon signed-rank p = {pr['p']:.2g})."
                )
    if config:
        notes.append("")
        notes.append(
            f"Setup: {config.get('n_objects')} GSO objects, input rendered at {config.get('input_elevation')}° "
            f"elevation, {config.get('diffusion_steps')} diffusion steps, NeuS {config.get('recon_iters')} "
            f"iterations, {config.get('n_points'):,} surface samples per mesh."
        )
    return "\n".join(lines + notes) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["object", "method", "views", "status", "faces", "time_views_s", "time_recon_s", "peak_gpu_recon_mib"]
    keys = [c[0] for c in COLUMNS] + ["aligned.accuracy", "aligned.completeness", "unaligned.chamfer"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields + keys + ["error"])
        for r in sorted(rows, key=lambda r: (r["object"], r["method"], r["views"])):
            w.writerow([r.get(k, "") for k in fields] + [get(r, k) for k in keys] + [r.get("error", "")])


# ---------------------------------------------------------------------------- chart


def svg_chart(rows: list[dict]) -> str:
    """Three small multiples (F@0.1, Chamfer, time), one horizontal bar per method, with CIs."""
    gs = [g for g in groups(rows) if g.ok]
    panels = [
        ("F-score @ 0.1", "higher is better", "aligned.fscore@0.1", "{:.2f}", "mean"),
        ("Chamfer distance", "lower is better", "aligned.chamfer", "{:.3f}", "mean"),
        ("Reconstruction time (s)", "median, lower is better", "time_recon_s", "{:.0f}", "median"),
    ]
    label_w, panel_w, gap, row_h, top, pad = 210, 220, 28, 34, 58, 16
    width = pad + label_w + len(panels) * panel_w + (len(panels) - 1) * gap + pad
    height = top + len(gs) * row_h + 34
    ink, muted, grid = "#191919", "#666666", "#e6e6e6"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>',
    ]
    fallback = iter(FALLBACK_COLORS)
    colors = {}
    for g in gs:
        if g.method not in colors:
            colors[g.method] = METHOD_COLORS.get(g.method) or next(fallback, "#888888")
    for i, g in enumerate(gs):
        y = top + i * row_h + row_h / 2
        out.append(
            f'<text x="{pad + label_w - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{ink}">'
            f"{_esc(g.label)}</text>"
        )
    for p, (title, sub, key, fmt, stat) in enumerate(panels):
        x0 = pad + label_w + p * (panel_w + gap)
        out.append(f'<text x="{x0}" y="22" font-size="13" font-weight="600" fill="{ink}">{_esc(title)}</text>')
        out.append(f'<text x="{x0}" y="38" font-size="11" fill="{muted}">{_esc(sub)}</text>')
        stats = []
        for g in gs:
            v = g.values(key)
            if len(v) == 0:
                stats.append(None)
            elif stat == "median":
                stats.append((float(np.median(v)), math.nan, math.nan))
            else:
                stats.append((float(v.mean()), *bootstrap_ci(v)))
        top_val = max([s[2] if not math.isnan(s[2]) else s[0] for s in stats if s] or [1.0])
        scale = (panel_w - 44) / (top_val * 1.05 or 1.0)
        out.append(
            f'<line x1="{x0}" x2="{x0}" y1="{top - 6}" y2="{top + len(gs) * row_h}" stroke="{grid}" stroke-width="1"/>'
        )
        for i, (g, s) in enumerate(zip(gs, stats, strict=True)):
            if s is None:
                continue
            mean, lo, hi = s
            y = top + i * row_h + 8
            h = row_h - 16
            w = max(mean * scale, 2)
            opacity = "1" if g.views == "generated" else "0.45"
            out.append(
                f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{colors[g.method]}" '
                f'fill-opacity="{opacity}"><title>{_esc(g.label)}: {fmt.format(mean)}</title></rect>'
            )
            end = x0 + w
            if not math.isnan(lo):
                cy = y + h / 2
                out.append(
                    f'<line x1="{x0 + lo * scale:.1f}" x2="{x0 + hi * scale:.1f}" y1="{cy}" y2="{cy}" '
                    f'stroke="{ink}" stroke-width="1.5"/>'
                )
                end = max(end, x0 + hi * scale)
            out.append(
                f'<text x="{end + 6:.1f}" y="{y + h / 2 + 4:.1f}" font-size="11" fill="{ink}">{fmt.format(mean)}</text>'
            )
    n = max((len(g.ok) for g in gs), default=0)
    out.append(
        f'<text x="{pad}" y="{height - 12}" font-size="10.5" fill="{muted}">Google Scanned Objects, {n} objects. '
        "Bars: mean (median for time); black lines: 95% bootstrap interval.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_report(out_dir: Path) -> dict[str, Path]:
    rows = load_rows(out_dir / "results.jsonl")
    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    paths = {"markdown": out_dir / "results.md", "csv": out_dir / "results.csv", "chart": out_dir / "chart.svg"}
    paths["markdown"].write_text(markdown(rows, config))
    write_csv(rows, paths["csv"])
    paths["chart"].write_text(svg_chart(rows))
    return paths
