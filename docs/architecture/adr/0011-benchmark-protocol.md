# ADR 0011: An accuracy benchmark on Google Scanned Objects

- Status: accepted
- Date: 2026-11-26

## Context

ADR 0010 made InstantMesh the default and kept NeuS as a "classic" mode, judged by eye on one
mug. That says nothing about how either method does on objects in general, or whether the gap is
real. We want a measurement that is standard (comparable in spirit to the papers), reproducible
by anyone on a free GPU, and honest about its uncertainty.

## Decisions

1. **Dataset: Google Scanned Objects**, CC BY 4.0, downloaded at run time from Gazebo Fuel, never
   committed. Alternatives: OmniObject3D (larger, but a license agreement is needed per user),
   Objaverse (the models are trained on it, so results would be inflated), our own photos (no
   ground truth).
2. **Subset by hash, not by hand**: the 30 names with the smallest SHA-256(salt:name). Uniform,
   reproducible, order-independent and nested (N=50 contains N=30).
3. **Rendered input, alpha instead of U²-Net.** We measure reconstruction, not segmentation or
   photo realism. Renders use the repo's NumPy rasteriser, so no OpenGL/EGL is needed on Kaggle
   or in CI.
4. **Generate views once per object and share them** between methods (paired design). This
   needed `MultiViewBackend.generate` split into `make_views` and `reconstruct_views`; the job
   path calls the same two steps, so the benchmark measures exactly what the app runs.
5. **Oracle-view run** as an ablation that separates diffusion error from reconstruction error.
6. **Alignment**: known orientation (input camera on +x), bounding-box normalisation, then
   multi-start similarity ICP with two-way, trimmed correspondences; the ground truth never moves.
   Unaligned scores are stored too.
7. **Metrics**: Chamfer (L1, plus squared), F-score at 0.05/0.1/0.2, normal consistency, mask IoU
   and PSNR in novel views, time and peak GPU memory. Bootstrap CIs and a Wilcoxon signed-rank test
   instead of bare means.
8. **Resumable, one JSON line per (object, method, views)**, so Kaggle's 12-hour limit or an
   out-of-memory error on one object costs at most that object.

## Consequences

- One Kaggle T4 run (about 5 to 6 hours with the oracle run) produces the README table and chart.
- The benchmark code is tested on CPU in CI with synthetic objects and a fake backend whose
  "reconstructions" are known transforms of the ground truth, so a broken metric or alignment
  fails CI; the real models are only exercised on the GPU run.
- Numbers depend on our alignment and normalisation, so they are comparable between our methods,
  and only loosely with published tables.
