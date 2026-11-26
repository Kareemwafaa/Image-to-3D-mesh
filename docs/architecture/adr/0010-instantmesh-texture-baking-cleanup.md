# ADR 0010: InstantMesh by default, texture baking and mesh cleanup

- Status: accepted
- Date: 2026-11-26

## Context

The first real GPU run (Kaggle T4, "Fine" detail) turned a photo of a mug
into a mesh with the right shape, but:

- **Slow:** 5 min 57 s, almost all of it the per-object NeuS fit.
- **Lumpy:** the surface shows low-frequency bumps and folds. Zero123++'s six
  views are not perfectly consistent with each other (each is a diffusion
  sample), and an optimiser that must explain all six pixel-exactly bends the
  surface to fit their disagreements.
- **Dull colour:** per-vertex colour from a smooth colour field, blurred
  further by averaging inconsistent views.
- **Large:** 9.4 MB of `.glb` for a mug (marching cubes on 320³ emits
  hundreds of thousands of near-uniform triangles).

## Options for reconstruction

1. **Tune NeuS** (more steps, stronger mask and smoothness priors). Cheap, but
   it does not fix the root cause: optimisation from six inconsistent views
   is under-constrained, and it stays slow.
2. **Gaussian splatting** (faster optimisation). No explicit surface; meshing
   it reliably from six views is an open problem.
3. **A feed-forward Large Reconstruction Model trained on Zero123++ outputs:
   InstantMesh** (Xu et al., 2024; Apache-2.0 code and weights).
4. **Other LRMs** (TripoSR, LGM, CRM). TripoSR reconstructs from the single
   input image (no multi-view stage, which this project is built around); LGM
   outputs Gaussians; CRM uses its own multi-view model.

## Decision

Option 3 as the default, with NeuS kept as a selectable **"classic"** mode
(`reconstructor=neus`) so the two approaches can be compared on the same views.
Every mesh, from either reconstructor, then goes through the same cleanup and
texture baking stages.

### Why InstantMesh works better here: the theory

**Learned prior instead of per-object fitting.** NeuS solves
`min_f Σ_views ‖render(f) − view‖` from scratch for every object. With six
views the problem has many solutions, and the optimiser has no notion of what
real objects look like. InstantMesh *amortises* that optimisation: a network
`g_θ(views, cameras) → shape` was trained on ~270k Objaverse objects, rendered
from the same six Zero123++ poses, so θ already encodes a prior over plausible
shapes. Inference is one forward pass (seconds instead of minutes), and
inconsistencies in the views are averaged out by the prior instead of being
carved into the surface.

**Architecture** (the "LRM" family, Hong et al., 2024):

1. *Encoder*: a DINO ViT-B/16 turns each 320×320 view into patch tokens. Each
   view's camera (a flattened 4×4 pose plus intrinsics) modulates the ViT's
   LayerNorms (adaptive LayerNorm), so the tokens know where they were seen
   from.
2. *Triplane decoder*: a 16-layer transformer whose queries are learned
   tokens for three axis-aligned 64×64 feature planes (xy, xz, yz). Each layer
   cross-attends from the plane tokens to all 6 × 400 view tokens, then
   self-attends among the plane tokens. A triplane is a factorised 3D feature
   volume: the feature at a point is the sum of three bilinear lookups, O(N²)
   memory instead of O(N³).
3. *Field heads*: small MLPs map a point's triplane feature to an SDF value,
   a vertex displacement, FlexiCubes weights, and a colour.
4. *FlexiCubes* (Shen et al., 2023) extracts the surface from the SDF on a
   128³ grid. Marching cubes places each vertex at a fixed linear
   interpolation along a cube edge; FlexiCubes adds learned per-cube weights
   (where on each edge, and how to split each cube into triangles) and
   per-vertex offsets. Because the extraction is differentiable, InstantMesh
   was trained end to end on **rendered depth, normals and silhouettes of the
   mesh itself**, not only on a volume-rendered image. That geometric
   supervision is why its surfaces are smooth and sharp-edged where NeuS's
   are lumpy.

InstantMesh also ships a version of the Zero123++ UNet fine-tuned to render
the views on a **white background**, which is what its reconstructor saw in
training. We load those weights by default (`ZERO123PP_UNET=instantmesh`);
NeuS composites the views on white anyway, so it is unaffected.

### Texture baking

Both reconstructors colour vertices from a smooth neural colour field, so
colour detail is limited by mesh density and the field's smoothness. The six
views are sharper and their cameras are known exactly, so we project them
back (projective texturing):

1. UV-unwrap the cleaned mesh with xatlas (charts, conformal flattening,
   packing into a 1024² atlas).
2. Rasterise the mesh in UV space: every texel gets a 3D point `p` and normal
   `n` (barycentric interpolation).
3. For each view `i`: project `p` with the pinhole model, test visibility
   against a z-buffer of the mesh rendered from camera `i`, check that the
   pixel is inside the view's silhouette, and weight by
   `w_i = max(0, n·d_i)^4`, `d_i` the direction to the camera. Head-on views
   win because there one pixel covers the least surface (least resampling
   blur); the power 4 makes blending close to "pick the best view" while
   still feathering seams between views.
4. Texels no camera sees (the underside: all cameras are within 20° of the
   equator) take the reconstructor's own colour.
5. Dilate colours outwards into empty texels so mip-mapping does not pull
   black into chart seams.

The rasteriser (`app/pipeline/raster.py`) is ~100 lines of vectorised NumPy,
not nvdiffrast or OpenGL, so it runs anywhere, including CI.

### Cleanup

In order: drop floaters (components under 5% of the largest one's area);
close holes up to 200 boundary edges with a fan around the centroid, wound
to match the neighbours; Taubin smoothing (λ = 0.5, μ = −0.53), a low-pass
filter that removes bumps of a few triangles without shrinking the object
like plain Laplacian smoothing would (3 iterations for InstantMesh, 10 for
NeuS); quadric-error decimation (Garland & Heckbert, 1997) to 40k faces,
which removes triangles where the surface is flat and keeps them where it
curves; then consistent outward winding.

## Consequences

- Expected GPU time per object drops from ~6 minutes to well under a minute,
  most of it Zero123++ diffusion. **Not measured yet** (no GPU here).
- The `.glb` shrinks: on a synthetic 330k-face sphere, cleanup and baking
  gave 40k faces and a 1.0 MB file (vs 5.9 MB raw), in 3 s + 10 s on a CPU.
- New downloads on first start: InstantMesh-large (~1.7 GB), the white-
  background UNet (~3.5 GB) and DINO (~350 MB). Model weights are Apache-2.0.
- InstantMesh's code is not on PyPI. We download upstream's `src/` at a
  pinned commit into `~/.cache/facet` (renamed to `instantmesh_src` so it
  cannot clash with other packages) rather than vendoring ~3k lines. It
  imports nvdiffrast, which needs a CUDA compiler to install; inference only
  needs FlexiCubes (pure PyTorch), so a stub module stands in for nvdiffrast
  and raises a clear error if anything ever calls it. CI downloads the source
  and runs a tiny, randomly initialised InstantMesh on CPU, so a mismatch with
  upstream's code fails the build.
- **GUESS: memory.** Zero123++ (fp16) plus InstantMesh-large (fp32) should fit
  a free 16 GB T4. If it does not, `INSTANTMESH_VARIANT=base` is smaller, and
  `RECONSTRUCTOR=neus PRELOAD_INSTANTMESH=false` needs neither.
- NeuS stays fully supported and tested; choose it per request with
  `reconstructor=neus` or in the web app's settings.
- `python -m app.doctor` checks every import, the torch/torchvision pairing,
  the GPU, and the InstantMesh source before the server starts, so a missing
  package fails in seconds instead of after a 5 GB download.

## References

- Xu et al., *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models*, 2024.
- Hong et al., *LRM: Large Reconstruction Model for Single Image to 3D*, ICLR 2024.
- Shen et al., *Flexible Isosurface Extraction for Gradient-Based Mesh Optimization* (FlexiCubes), SIGGRAPH 2023.
- Chan et al., *Efficient Geometry-aware 3D Generative Adversarial Networks* (triplanes, EG3D), CVPR 2022.
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021.
- Taubin, *A Signal Processing Approach to Fair Surface Design*, SIGGRAPH 1995.
- Garland & Heckbert, *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH 1997.
- Young et al., *xatlas* (UV atlas generation), github.com/jpcy/xatlas.
