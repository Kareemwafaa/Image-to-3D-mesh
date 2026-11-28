"""Command line: `python -m app.bench {run,report,subset} ...`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import gso
from .runner import BenchConfig


def _csv(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.bench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run (or resume) the benchmark")
    run.add_argument("--out", type=Path, required=True, help="results folder (resumes if it has results)")
    run.add_argument("--data", type=Path, default=Path.home() / ".cache" / "facet" / "gso", help="GSO download cache")
    run.add_argument("--methods", type=_csv, default=("instantmesh", "neus"))
    run.add_argument("--objects", type=int, default=30, help="size of the hashed GSO subset")
    run.add_argument("--names", type=_csv, default=None, help="explicit GSO model names instead of the subset")
    run.add_argument("--oracle", action="store_true", help="also reconstruct from ground-truth renders of the views")
    run.add_argument("--oracle-only", action="store_true", help="only the oracle run (no diffusion model needed)")
    run.add_argument("--recon-iters", type=int, default=2500, help="NeuS iterations")
    run.add_argument("--diffusion-steps", type=int, default=75)
    run.add_argument("--elevation", type=float, default=10.0, help="input view elevation in degrees")
    run.add_argument("--points", type=int, default=100_000, help="surface samples per mesh")
    run.add_argument("--eval-views", type=int, default=4, help="novel views for mask IoU/PSNR (0 = skip)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--no-meshes", action="store_true", help="do not save the aligned meshes")
    run.add_argument("--time-limit", type=float, default=None, help="stop starting new objects after N minutes")
    run.add_argument("--instantmesh-variant", choices=("large", "base"), default="large")
    run.add_argument("--grid-res", type=int, default=None, help="InstantMesh FlexiCubes grid (default 128)")
    run.add_argument("--device", default="auto")

    rep = sub.add_parser("report", help="write results.md, results.csv and chart.svg")
    rep.add_argument("--out", type=Path, required=True)

    sel = sub.add_parser("subset", help="print the hashed GSO subset (needs internet)")
    sel.add_argument("--objects", type=int, default=30)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "subset":
        names = gso.select_subset(gso.list_models(), args.objects)
        print("\n".join(names))
        return 0

    if args.cmd == "report":
        from .report import write_report

        for kind, path in write_report(args.out).items():
            print(f"{kind}: {path}")
        print()
        print((args.out / "results.md").read_text())
        return 0

    cfg = BenchConfig(
        methods=args.methods,
        n_objects=args.objects,
        names=args.names,
        oracle=args.oracle or args.oracle_only,
        generated=not args.oracle_only,
        recon_iters=args.recon_iters,
        diffusion_steps=args.diffusion_steps,
        input_elevation=args.elevation,
        n_points=args.points,
        eval_views=args.eval_views,
        seed=args.seed,
        retry_failed=args.retry_failed,
        save_meshes=not args.no_meshes,
    )
    backend = _backend(args, need_diffusion=cfg.generated)
    from .report import write_report
    from .runner import run as run_bench

    run_bench(cfg, backend, args.out, args.data, limit_minutes=args.time_limit)
    write_report(args.out)
    print((args.out / "results.md").read_text())
    return 0


def _backend(args, need_diffusion: bool):
    from ..backends.multiview import MultiViewBackend
    from ..core.config import Settings

    s = Settings()
    backend = MultiViewBackend(
        s.zero123pp_model,
        s.zero123pp_pipeline,
        args.device,
        unet=s.zero123pp_unet,
        instantmesh_variant=args.instantmesh_variant,
        preload_instantmesh="instantmesh" in args.methods,
        low_vram=s.low_vram,
        instantmesh_grid_res=args.grid_res or s.instantmesh_grid_res,
    )
    if need_diffusion:
        backend.load()
    elif "instantmesh" in args.methods:
        backend.lrm.load()
    return backend


if __name__ == "__main__":
    sys.exit(main())
