# Image to 3D mesh: backend

A service that turns one photo of an object into a textured 3D mesh (`.glb`).
It is a two-stage generative reconstruction pipeline: a diffusion model
imagines the unseen sides of the object, then a reconstructor turns those
views into a surface. The default reconstructor is InstantMesh, a feed-forward
sparse-view model (seconds); the classic one fits a neural signed distance
field (NeuS) to the views (minutes). Either way the mesh is cleaned up and
textured by projecting the views back onto it.

```mermaid
flowchart LR
    A[Photo] --> B["1. Segmentation<br/>U²-Net"]
    B --> C["2. Multi-view diffusion<br/>Zero123++ v1.2"]
    C -->|6 views| R{"4-5. Reconstruction"}
    R -->|default| IM["InstantMesh<br/>DINO + triplane transformer<br/>+ FlexiCubes"]
    R -->|classic| NS["NeuS fit<br/>+ marching cubes"]
    IM --> CL["6. Cleanup<br/>floaters, holes, Taubin,<br/>QEM decimation"]
    NS --> CL
    CL --> TX["7. Texture baking<br/>xatlas UVs + projection<br/>of the 6 views"]
    TX --> G[.glb]
```

| Stage | Module | Method |
| --- | --- | --- |
| 1 | `app/pipeline/segmentation.py` | U²-Net salient object segmentation (rembg), crop and centre |
| 2 | `app/pipeline/multiview.py` | Zero123++ v1.2 (InstantMesh's white-background UNet): six consistent novel views |
| 3 | `app/pipeline/cameras.py` | The six fixed camera poses, pixel rays |
| 4-5 | `app/pipeline/instantmesh.py` | **Default.** InstantMesh LRM: views + cameras to triplane to FlexiCubes mesh, one forward pass |
| 4 | `app/pipeline/neus.py` | Classic: NeuS neural SDF, multi-resolution grid encoding, unbiased volume rendering |
| 5 | `app/pipeline/reconstruct.py`, `meshing.py` | Classic: per-object optimisation (RGB, silhouette, Eikonal), marching cubes |
| 6 | `app/pipeline/postprocess.py` | Floater removal, hole filling, Taubin smoothing, quadric decimation |
| 7 | `app/pipeline/texturing.py`, `raster.py` | UV unwrap, visibility-aware projective texturing, seam padding, glTF export |

Why InstantMesh is the default, with its theory, is in
[ADR 0010](../docs/architecture/adr/0010-instantmesh-texture-baking-cleanup.md).

Around the pipeline sits a small distributed system: a FastAPI gateway, a
Redis Streams job queue, GPU workers, Postgres and S3 storage, with retries,
dead letters, idempotency, rate limits, live progress (SSE), metrics and
traces. It also runs as one process on a free notebook GPU. The design, the
failure handling and the decision records (ADRs) are in
[docs/architecture](../docs/architecture/README.md).

## The theory, stage by stage

### 1. Segmentation

The photo's background would otherwise be reconstructed as geometry, so we
segment the object with **U²-Net** (Qin et al., 2020), a nested U-Net whose
residual U-blocks capture multi-scale context for salient object detection.
The result is cropped to the object's bounding box, padded to a square and
scaled so the object fills 85% of the frame, which matches what the diffusion
model saw in training. The same segmenter later gives a silhouette mask for
each generated view.

### 2. Multi-view diffusion (Zero123++)

A single image does not determine the back of an object: the problem is
ill-posed. We resolve the ambiguity with a learned prior. **Zero123++**
(Shi et al., 2023) is Stable Diffusion fine-tuned on renders of Objaverse
objects to produce six views of the input object at fixed poses.

- **One canvas, six views.** The six 320×320 views are tiled into one 960×640
  image and denoised *jointly*. Self-attention spans the whole canvas, so the
  views exchange information at every denoising step, which is what makes them
  geometrically consistent. (Its predecessor, Zero-1-to-3, sampled each view
  independently, and the views disagreed with each other.)
- **Two kinds of conditioning.** Global: CLIP image embedding through
  cross-attention. Local: *reference attention*, where the UNet also processes
  the (noised) input image and its self-attention keys and values are
  concatenated into those of the canvas, so fine appearance details transfer.
- **Fixed poses.** v1.2 outputs azimuths 30°, 90°, …, 330° relative to the
  input, at alternating absolute elevations +20° and −10°, with a 30° field of
  view and a normalised object scale. Because these poses are known, we need
  no structure-from-motion or pose estimation.
- Sampling uses the Euler-ancestral sampler with "trailing" timestep spacing,
  75 steps by default (the authors note 28 suffice for coarse results).

### 3. Cameras

Each view gets a pinhole camera on a sphere of radius 4 looking at the origin
(OpenGL convention, z-up world). Every pixel then defines a ray
`r(t) = o + t·d`. With six views at 320² this gives 614,400 supervised rays.

### 4. Neural SDF and unbiased volume rendering (NeuS)

The shape is the zero level set of a **signed distance function**
`f: ℝ³ → ℝ` (negative inside, positive outside), represented by a neural
field. Optimising it from images needs a differentiable renderer. Plain NeRF
learns a density, which gives no well-defined surface; naively converting an
SDF to a density gives a rendering weight that peaks *in front of* the
surface (biased). **NeuS** (Wang et al., 2021) derives a density that is
unbiased:

- Let `Φ_s(x) = sigmoid(s·x)` be the logistic CDF with sharpness `s`. Define the
  opaque density along a ray as
  `ρ(t) = max(−(dΦ_s(f(r(t)))/dt) / Φ_s(f(r(t))), 0)`.
  Then the weight `w(t) = T(t)·ρ(t)` (with transmittance `T`) has its maximum
  exactly where the ray meets the surface, and it accounts for occlusion.
- Discretised over a ray section `[t_i, t_{i+1}]`:
  `α_i = max((Φ_s(f_i) − Φ_s(f_{i+1})) / Φ_s(f_i), 0)`, and the pixel colour is
  `Σ_i T_i α_i c_i` with `T_i = Π_{j<i}(1 − α_j)`.
- `s` is learned (`s = exp(10·v)`). `1/s` acts as the surface "thickness":
  large at first so gradients reach far, it shrinks as the fit improves.
- **Hierarchical sampling.** 64 uniform samples per ray, then 64 more placed by
  inverse-transform sampling where a sharpened copy of the current SDF (fixed
  `s = 64·2^k` over 4 rounds) predicts the surface.

**Field architecture.** Following Instant-NGP (Müller et al., 2022), the input
point is encoded by trilinear interpolation into six dense 3D feature grids at
resolutions 16 to 128 (geometric progression), concatenated with the raw
coordinates and passed to a 2-layer MLP (64 units, Softplus). The MLP predicts
an SDF *residual* on top of a sphere of radius 0.5, so optimisation starts from
a valid SDF. A second small MLP predicts colour from position, normal, view
direction and a 15-dimensional geometry feature.

**Normals by finite differences.** The normal is `∇f`. As in Neuralangelo
(Li et al., 2023) we take it by central differences with a step tied to the
finest active grid cell. With grid encodings, analytic gradients are only
locally (per-cell) supported; numerical gradients spread the Eikonal signal
across neighbouring cells and give smoother surfaces. (They also avoid
second-order derivatives of `grid_sample`, which PyTorch does not implement.)

### 5. Optimisation

For each request we fit a fresh field (test-time optimisation, about 2,500
steps of 1,024 random rays, Adam with warm-up and cosine decay):

`L = L_rgb + 0.5·L_mask + 0.1·L_eik`

- `L_rgb`: L1 between the rendered colour and the generated view, composited on
  white with its mask.
- `L_mask`: binary cross-entropy between rendered opacity `Σ_i w_i` and the
  silhouette. With only six views, silhouettes are the strongest geometric
  signal; this is a soft, differentiable form of visual-hull carving.
- `L_eik = E[(‖∇f‖ − 1)²]` at ray samples and random points: the Eikonal
  equation holds for true distance functions (Gropp et al., 2020), so this keeps
  `f` an SDF and its zero set a clean surface.

Two schedules make this robust to sparse, slightly inconsistent views:
**coarse-to-fine** (Neuralangelo) unlocks the fine grid levels gradually over
the first half of training, so the overall shape forms before detail; and
**cosine annealing** (NeuS) moves the section-SDF estimate from a smoothed to
the exact form over the first 20%.

### 6. Mesh extraction

**Marching cubes** (Lorensen & Cline, 1987) evaluates `f` on an N³ grid
(`mc_resolution`, 256 by default). In each cell whose corners change sign it
emits triangles from a case table (256 sign patterns, 15 up to symmetry),
placing vertices on edges by linear interpolation of `f`. Because `f` is a
distance, interpolation is accurate and the output is watertight. We then
drop disconnected pieces under 5% of the largest one's area (floaters), make
the faces point along `∇f`, colour each vertex by querying the colour field
head-on (view direction `−n`), rotate to glTF's y-up frame and export `.glb`.

### Why NeuS and not Gaussian splatting?

3D Gaussian splatting (Kerbl et al., 2023) optimises much faster, but it
represents a *radiance* field with no explicit surface. Getting a mesh needs
extra machinery (SuGaR, 2D Gaussian splatting) and is fragile with only six
views. An SDF has the surface built in, the Eikonal prior regularises the
sparse-view problem, and marching cubes gives a watertight mesh directly.

### 7. InstantMesh (the default reconstructor)

A Large Reconstruction Model maps the six views straight to a shape. A DINO
ViT-B/16 encodes each view into patch tokens, with the view's camera injected
through adaptive LayerNorm. A 16-layer transformer turns learned queries into
a **triplane** (three 64×64 feature planes, xy/xz/yz): each layer
cross-attends from the plane tokens to all 2400 view tokens. A point's feature
is the sum of three bilinear plane lookups, so the 3D field costs O(N²)
memory. Small MLPs read SDF, vertex offsets and colour from it, and
**FlexiCubes** extracts the mesh on a 128³ grid with learned per-cube
interpolation weights. That extraction is differentiable, so the model was
trained on rendered depth, normals and silhouettes of its own mesh: geometry
supervision that NeuS never gets.

InstantMesh's code is not on PyPI, so `ensure_source()` downloads upstream's
`src/` at a pinned commit into `~/.cache/facet/` (as package
`instantmesh_src`). Upstream imports nvdiffrast, which needs a CUDA compiler;
inference does not use it, so a stub stands in when it is missing.

Upstream predicts the FlexiCubes weights from the features of every cube's 8
corners gathered for all 128³ cubes at once: one 2.1M × 8 × 240 float tensor,
15 GiB. That ran out of memory on the first real T4 run, so
`chunked_geometry_prediction` replaces that step: it works through the grid in
chunks and re-samples each chunk's corner features from the triplane instead
of gathering them from a stored copy. The outputs are identical (tested
against upstream). On GPUs under 20 GiB, Zero123++ is also parked in CPU RAM
while InstantMesh runs (`LOW_VRAM`).

InstantMesh was trained on views over a pure white background, and anything
else is read as solid material. Its white-background UNet must therefore be
conditioned on an input composited onto white (sudo-ai's pipeline uses gray;
InstantMesh ships a copy with that one line changed), and before
reconstruction we force every view's background to exactly 1.0 using the
U²-Net masks. If the SDF still has no surface, upstream silently returns a
cube the size of the whole grid; `looks_like_the_grid_box` turns that into a
clear `empty_mesh` error instead of a fake result.

### 8. Cleanup and texture baking

`postprocess.py`: drop floaters, close holes up to 200 edges (fan around the
centroid, wound to match its neighbours), **Taubin smoothing** (λ = 0.5,
μ = −0.53: a shrink step and an inflate step together form a low-pass filter,
so bumps disappear but the volume does not), then **quadric error metric
decimation** to 40k faces (each edge collapse is chosen to minimise the summed
squared distance to the original faces' planes).

`texturing.py`: xatlas cuts the mesh into charts and packs them in a 1024²
atlas. Rasterising in UV space gives each texel a 3D point and normal. Each
texel is projected into the six views; a z-buffer (the same NumPy rasteriser,
`raster.py`) rejects views where it is hidden, the view's mask rejects
background pixels, and the weight `max(0, n·d)⁴` prefers the camera that
sees it most head-on (k = 12, so each texel comes almost entirely from its
best view: the six views are separate diffusion samples, and averaging their
small colour disagreements produced purple and red blotches). Masks are
eroded by 2 px so silhouette-edge pixels, which mix in background, are not
used. Texels no camera sees take the model's own colour. Finally the texture
is colour-matched to the input photo (Reinhard-style statistics transfer in
YCbCr): luma mean and spread are matched, chroma is pulled towards the
photo's and its spread can only shrink. This fixes exposure drift (a matte
black mug came out light grey) and mutes leftover blotches.

### Why keep NeuS?

It is the "classic" baseline: per-object optimisation, fully inspectable,
with no learned prior. Running both on the same six views shows what the
learned prior buys (`reconstructor=neus`, or *Reconstruction: NeuS* in the
web app).

## What has been verified, and what are guesses

Verified here (CPU, see `tests/`): the API and the whole job system end to end
with a mock model, in lite mode and against real Postgres, Redis and S3; the
camera geometry; the NeuS renderer, optimiser and mesher, which recover
synthetic scenes rendered from the six Zero123++ cameras (an off-centre sphere
in the test suite; two coloured spheres at 64 px and 400 steps reached a mean
surface error of 0.018 with correct colours).

Verified for the quality stages (CPU, `tests/test_quality.py`): cleanup on a
noisy, holey sphere with a floater (watertight, volume within 5%, decimated);
texture baking of a two-tone sphere rendered analytically from the six
cameras (over 99% of texels in the right colour at 320 px and 1024²; a
330k-face mesh cleaned to 40k faces and baked in about 13 s on 4 CPU cores,
1.0 MB `.glb` instead of 5.9 MB); the multiview backend end to end with
stand-ins for Zero123++ and InstantMesh; and upstream InstantMesh's real code
(tiny random weights, CPU) running through our wrapper without nvdiffrast.

Measured on a GPU (2026-11-26, Kaggle T4, NeuS mode): a mug in 5:57, right
shape, lumpy surface, 9.4 MB. That run is what motivated ADR 0010.

Not run yet, because this environment has no GPU and cannot reach Hugging Face:

- **InstantMesh with its real weights**, the white-background UNet, and the
  speed on a T4. The first T4 run (2026-11-26) failed with out-of-memory in
  FlexiCubes; fixed by chunking (see stage 7). GUESS: the job now fits in 15 GB
  and takes under a minute. If memory runs out, set
  `INSTANTMESH_VARIANT=base`, or `RECONSTRUCTOR=neus PRELOAD_INSTANTMESH=false`.

- **Zero123++ inference and the Modal deployment** have not been run. The code
  follows the authors' README and pipeline source (diffusers 0.20.2).
- **GUESS: camera radius 4.0** (from InstantMesh). Zero123++ only states a
  "normalised object size". A wrong radius rescales the object; it does not
  distort it.
- **GUESS: azimuth direction.** Counter-clockwise seen from +z, as in InstantMesh.
  If generated views come out mirrored relative to the cameras, the reconstruction
  will be poor; flipping the sign of `ZERO123PP_AZIMUTHS` fixes it.
- **GUESS: timings.** Roughly 10-30 s for diffusion and about 10 s for
  InstantMesh plus cleanup and baking on a T4. NeuS mode took 5:57 in total on
  a T4. Lower `recon_iters` or `diffusion_steps` for speed.
- **GUESS: hyperparameters** (`ReconConfig`) were chosen by reasoning and
  synthetic tests, not tuned on real generated views.

## API

Interactive docs are served at `/docs`. The full route list is in
[docs/architecture](../docs/architecture/README.md#api).

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/v1/jobs` (alias `/generate`) | Multipart upload. `202` with the job id and `Location`. Honours `Idempotency-Key` |
| `GET` | `/v1/jobs/{id}` (alias `/jobs/{id}`) | Job status (see below) |
| `GET` | `/v1/jobs/{id}/events` | Live progress as Server-Sent Events |
| `GET` | `/v1/jobs/{id}/result` | The mesh, `model/gltf-binary`. 404 until ready, 409 if the job failed |
| `GET` | `/v1/jobs/{id}/views` | The six generated views as one 3×2 PNG |
| `GET` | `/health`, `/readyz`, `/metrics` | Status, readiness, Prometheus metrics |

`POST /v1/jobs` form fields (all optional except `image`):

| Field | Default | Range | Notes |
| --- | --- | --- | --- |
| `image` | | | PNG, JPEG or WebP, up to 10 MB |
| `remove_background` | `true` | | `false` if the image already has transparency |
| `foreground_ratio` | `0.85` | 0.5 to 1.0 | How much of the frame the object fills |
| `mc_resolution` | `256` | 32 to 512 | Marching cubes grid (NeuS only). Higher means finer mesh |
| `diffusion_steps` | `75` | 10 to 150 | Zero123++ denoising steps |
| `reconstructor` | `instantmesh` | `instantmesh`, `neus` | Feed-forward model (fast) or the classic per-object fit |
| `recon_iters` | `2500` | 100 to 20000 | NeuS optimisation steps (NeuS only) |
| `texture` | `true` | | Bake a texture from the views; `false` keeps vertex colours |
| `texture_size` | `1024` | 256 to 2048 | Texture atlas size in pixels |
| `color_match` | `1.0` | 0 to 1 | Match the texture's colours and brightness to the photo |
| `target_faces` | `40000` | 0 to 500000 | Decimation target; `0` keeps every triangle |
| `seed` | random | | Fix it for reproducible views |

`GET /v1/jobs/{id}`:

```json
{
  "job_id": "3f2c...",
  "status": "running",
  "stage": "reconstructing",
  "progress": 0.62,
  "attempts": 1,
  "result_url": null,
  "views_url": "/v1/jobs/3f2c.../views",
  "events_url": "/v1/jobs/3f2c.../events",
  "error": null,
  "error_code": null,
  "dead_lettered": false,
  "created_at": "2026-11-25T19:00:00Z",
  "updated_at": "2026-11-25T19:01:10Z"
}
```

`status` is `queued`, `running`, `succeeded` or `failed`. `stage` is
`segmenting`, `multiview`, `segmenting_views`, `reconstructing`, `meshing`,
then `done`. A failed job has a stable `error_code` (for example
`no_foreground`, `timeout`, `gpu_oom`). Errors are `application/problem+json`.

```bash
JOB=$(curl -s -H "Idempotency-Key: chair-1" -F image=@chair.jpg $API/v1/jobs | jq -r .job_id)
curl -N $API/v1/jobs/$JOB/events          # live progress
curl -o chair.glb $API/v1/jobs/$JOB/result
```

## Run it

### Development, no GPU

Lite mode runs everything in one process (SQLite, local files, in-memory
queue, in-process worker). The `mock` model returns a coloured cube, so the
API and the frontend can be built on any laptop.

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/dev.txt
MODEL_BACKEND=mock python -m app.lite      # http://localhost:8000/docs
pytest -m "not integration"                # ~1-2 min; one test trains a tiny NeuS on CPU
```

### Free GPU: Colab or Kaggle notebook

The easiest way to run the real pipeline. The notebook in
[`notebooks/facet_backend.ipynb`](../notebooks/facet_backend.ipynb) installs a pinned environment
(`scripts/setup_gpu_env.sh`), starts the API in lite mode on the free T4 and exposes it through a
Cloudflare quick tunnel (`scripts/serve_public.py`).

### The full stack (Docker Compose)

From the repository root:

```bash
docker compose --profile cpu up --build                 # mock model, any machine
docker compose --profile gpu up --build                 # real pipeline, NVIDIA GPU (~12 GB)
docker compose --profile cpu up --scale worker-cpu=3    # more workers
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318 \
  docker compose --profile cpu --profile observability up --build
```

This starts Postgres, Redis, an S3 server (SeaweedFS), a migration job, the API
gateway on port 8000 and the workers. The `observability` profile adds
Prometheus (9090), Grafana (3001, dashboard provisioned) and Jaeger (16686).
`bash deploy/smoke_test.sh` submits a job end to end. The default admin key is
`i3d_00000000_dev-admin-key-change-me`; set `ADMIN_API_KEY` for anything public.

Integration tests run against the same services:

```bash
docker compose up -d postgres redis s3
I3D_IT_DATABASE_URL=postgresql+asyncpg://i3d:i3d@localhost:5432/i3d \
I3D_IT_REDIS_URL=redis://localhost:6379/15 I3D_IT_S3_ENDPOINT=http://localhost:8333 \
  pytest -m integration
```

### Deploy on Modal (optional)

```bash
pip install modal && modal setup
cd backend
CORS_ORIGINS=https://your-frontend.example modal deploy modal_app.py
```

One GPU container (A10G by default; set `MODAL_GPU` to change it) runs lite
mode with the weights baked into its image, and scales to zero after 5 idle
minutes. Job history lives in the container, so it is lost when it scales
down. Use `modal serve modal_app.py` for a live-reloading dev deployment.

## Configuration

All settings are environment variables (`app/core/config.py` has the full list).

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `full` | `full` (separate workers) or `lite` (one process; `python -m app.lite` sets it) |
| `MODEL_BACKEND` | `multiview` | `multiview` or `mock` |
| `DATABASE_URL` | Postgres on localhost | SQLAlchemy URL; lite mode uses SQLite |
| `REDIS_URL` | `redis://localhost:6379/0` | Queue, events and rate limits (full mode) |
| `STORAGE_BACKEND`, `S3_*` | `s3` | `s3` (any S3-compatible store) or `local` |
| `AUTH_MODE` | `optional` | `optional` (anonymous allowed, per-IP limits) or `required` |
| `ADMIN_API_KEY` | none | Bootstrap admin key, `i3d_<8 hex>_<secret>` |
| `CORS_ORIGINS`, `CORS_ORIGIN_REGEX` | localhost | Allowed origins, comma-separated, or `*` |
| `MAX_ATTEMPTS`, `JOB_TIMEOUT_S` | `3`, `900` | Retries for transient errors, per-job deadline |
| `VISIBILITY_TIMEOUT_S` | `60` | Lease after which a dead worker's job is reclaimed |
| `ANON_SUBMIT_PER_MINUTE`, `ANON_MAX_INFLIGHT` | `5`, `2` | Limits for anonymous callers (API keys get higher ones) |
| `JOB_TTL_HOURS` | `0` | Delete job outputs after this; `0` keeps them |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none | Turns tracing on |
| `DIFFUSION_STEPS`, `RECON_ITERS`, `MC_RESOLUTION`, `FOREGROUND_RATIO` | | Defaults for the form fields above |
| `RECONSTRUCTOR`, `TEXTURE_SIZE`, `TARGET_FACES` | `instantmesh`, `1024`, `40000` | Defaults for the quality form fields |
| `INSTANTMESH_VARIANT` | `large` | `large` or `base` (smaller, less GPU memory) |
| `PRELOAD_INSTANTMESH` | `true` | Load InstantMesh at start-up; `false` if you only use NeuS |
| `INSTANTMESH_GRID_RES` | `128` | FlexiCubes grid; try `192` for cavities and thin walls (more time and memory, quality unmeasured) |
| `COLOR_MATCH` | `1.0` | 0 to 1: how strongly the texture is matched to the input photo's colours |
| `LOW_VRAM` | automatic | Move Zero123++ to CPU RAM while reconstructing; on by default for GPUs under 20 GiB |
| `ZERO123PP_UNET` | `instantmesh` | `instantmesh` (white background) or `original` (sudo-ai's gray) |
| `INSTANTMESH_SRC` | download | Use an existing checkout of InstantMesh's `src/` (renamed `instantmesh_src`) |

## Next steps worth doing

- **Evaluate quantitatively.** Chamfer distance and volume IoU against ground
  truth on Google Scanned Objects, comparing this pipeline with a feed-forward
  baseline such as TripoSR. This turns the project from a demo into a result.
- Estimate the input view's elevation and add it as a seventh supervised view.
- Use Zero123++'s normal-generation ControlNet for normal supervision and
  cleaner masks.
- Compare InstantMesh and NeuS side by side on the same views (time, Chamfer
  distance, file size) and publish the table.

## References

- Shi et al., *Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model*, 2023.
- Liu et al., *Zero-1-to-3: Zero-shot One Image to 3D Object*, ICCV 2023.
- Wang et al., *NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction*, NeurIPS 2021.
- Müller et al., *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*, SIGGRAPH 2022.
- Li et al., *Neuralangelo: High-Fidelity Neural Surface Reconstruction*, CVPR 2023.
- Gropp et al., *Implicit Geometric Regularization for Learning Shapes*, ICML 2020.
- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020.
- Lorensen & Cline, *Marching Cubes: A High Resolution 3D Surface Construction Algorithm*, SIGGRAPH 1987.
- Hong et al., *LRM: Large Reconstruction Model for Single Image to 3D*, ICLR 2024.
- Shen et al., *Flexible Isosurface Extraction for Gradient-Based Mesh Optimization* (FlexiCubes), SIGGRAPH 2023.
- Taubin, *A Signal Processing Approach to Fair Surface Design*, SIGGRAPH 1995.
- Garland & Heckbert, *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH 1997.
- Qin et al., *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*, Pattern Recognition 2020.
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Xu et al., *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models*, 2024.
- Tochilkin et al., *TripoSR: Fast 3D Object Reconstruction from a Single Image*, 2024.
