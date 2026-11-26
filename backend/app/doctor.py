"""Check that a GPU environment can run the whole pipeline, before loading any model.

    cd backend && python -m app.doctor            # check imports, versions, GPU
    cd backend && python -m app.doctor --fetch    # ...and download the InstantMesh source

A missing package otherwise shows up only after several GB of weights have
downloaded and the first job fails (that is how a missing torchvision was
found). This imports every module each stage needs and reports all problems
at once. The notebooks run it right after installing.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass

# (import name, pip package, what needs it)
REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("torch", "torch", "every neural stage"),
    ("torchvision", "torchvision", "the Zero123++ custom pipeline"),
    ("diffusers", "diffusers", "Zero123++"),
    ("transformers", "transformers", "Zero123++ (CLIP) and InstantMesh (DINO)"),
    ("huggingface_hub", "huggingface-hub", "model downloads"),
    ("accelerate", "accelerate", "Zero123++ loading"),
    ("rembg", "rembg", "background removal"),
    ("onnxruntime", "onnxruntime", "background removal"),
    ("einops", "einops", "InstantMesh"),
    ("cv2", "opencv-python-headless", "InstantMesh (imported by its mesh utils)"),
    ("skimage", "scikit-image", "NeuS marching cubes"),
    ("scipy", "scipy", "mesh cleanup"),
    ("trimesh", "trimesh", "mesh cleanup and export"),
    ("fast_simplification", "fast-simplification", "mesh decimation"),
    ("xatlas", "xatlas", "texture baking (UV unwrapping)"),
)

# GUESS (not measured yet): Zero123++ (fp16) plus InstantMesh-large should fit in
# a free T4 (15 GB usable). Below this we only warn, since NeuS needs far less.
MIN_GPU_GB = 14


@dataclass
class Check:
    ok: bool
    name: str
    detail: str
    warn: bool = False


def check_imports(required=REQUIRED) -> list[Check]:
    out = []
    for module, package, used_by in required:
        try:
            mod = importlib.import_module(module)
            out.append(Check(True, module, getattr(mod, "__version__", "")))
        except Exception as exc:  # ImportError, or a broken binary wheel
            out.append(Check(False, module, f"{type(exc).__name__}: {exc}. pip install {package} (for {used_by})"))
    return out


def check_torch_pair() -> list[Check]:
    try:
        import torch
        import torchvision
    except Exception:
        return []  # already reported by check_imports
    t_minor = int(torch.__version__.split(".")[1])
    v_minor = int(torchvision.__version__.split(".")[1])
    ok = v_minor == t_minor + 15
    pair = f"torch {torch.__version__}, torchvision {torchvision.__version__}"
    return [Check(ok, "torch/torchvision", pair if ok else f"{pair} are not built for each other")]


def check_gpu(min_gb: float = MIN_GPU_GB) -> list[Check]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return [Check(False, "gpu", "CUDA is not available; the pipeline needs an NVIDIA GPU")]
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 2**30
    detail = f"{props.name}, {gb:.1f} GB"
    if gb < min_gb:
        detail += (
            f". InstantMesh needs about {min_gb} GB with Zero123++ loaded; "
            "set RECONSTRUCTOR=neus PRELOAD_INSTANTMESH=false to use the classic mode"
        )
    return [Check(True, "gpu", detail, warn=gb < min_gb)]


def check_instantmesh_source() -> list[Check]:
    try:
        from .pipeline.instantmesh import ensure_source, import_upstream

        root = ensure_source()
        import_upstream()
        return [Check(True, "instantmesh source", str(root))]
    except Exception as exc:
        return [Check(False, "instantmesh source", f"{type(exc).__name__}: {exc}")]


def run(fetch: bool = False, gpu: bool = True) -> list[Check]:
    checks = check_imports() + check_torch_pair()
    if gpu:
        checks += check_gpu()
    if fetch and all(c.ok for c in checks if c.name in ("torch", "transformers", "einops", "cv2")):
        checks += check_instantmesh_source()
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="also download and import the InstantMesh source")
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU check (for CPU-only machines)")
    args = parser.parse_args(argv)
    checks = run(fetch=args.fetch, gpu=not args.no_gpu)
    width = max(len(c.name) for c in checks)
    for c in checks:
        status = "FAIL" if not c.ok else ("warn" if c.warn else "ok  ")
        print(f"  {status}  {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"==> {len(failed)} problem(s) found; fix them before starting the server.")
        return 1
    print("==> Environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
