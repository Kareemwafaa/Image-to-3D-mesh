# Benchmark: how accurate are the reconstructions?

The app can turn a photo into a mesh in two ways: **InstantMesh**, a feed-forward network that
predicts the shape in one pass, and **NeuS**, which fits a neural surface to the views by
optimisation. This benchmark measures how close each one gets to the real object, how long it
takes and how much GPU memory it needs, on a standard dataset, in a way anyone can rerun.

- Code: [`backend/app/bench/`](../backend/app/bench/) (`python -m app.bench run|report|subset`)
- Notebook: [`notebooks/facet_benchmark.ipynb`](../notebooks/facet_benchmark.ipynb) (free Kaggle T4)
- Tests: [`backend/tests/test_bench.py`](../backend/tests/test_bench.py) (CPU, synthetic objects)
- Decision record: [ADR 0011](architecture/adr/0011-benchmark-protocol.md)

## Dataset

**Google Scanned Objects (GSO)** (Downs et al., 2022): 1,030 household objects scanned into
textured meshes. It is the usual test set for single-image 3D (Zero123, SyncDreamer, Wonder3D and
InstantMesh all report on it): real, everyday objects that are not in Objaverse, the synthetic
dataset these models are trained on.

**License: CC BY 4.0.** The meshes are not stored in this repository; the benchmark downloads them
from [Gazebo Fuel](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research),
where Google Research publishes them. Credit: *"Google Scanned Objects" by Google Research,
https://goo.gle/scanned-objects, CC BY 4.0.* (The Zero123++ weights the pipeline uses are
CC BY-NC 4.0: fine for research and a portfolio, not for commercial use.)

**Subset.** 30 objects, chosen by hashing rather than by hand: each model name is hashed with
SHA-256 together with a fixed salt (`facet-gso-v1`), and the 30 smallest hashes win. That is a
uniform random sample nobody picked, anyone can reproduce it from the catalogue, it does not depend
on the order the server lists models in, and a larger run contains the smaller one. The names are
written to `subset.json` with the results.

## Protocol

For each object:

1. **Normalise the scan.** Centre its bounding box at the origin and scale it so it fits the cube
   [-1, 1]³ and touches it. Every distance below is in these units (fractions of the object's
   half-size), so a big object and a small one count the same.
2. **Make the input photo.** Render the scan from a camera on the +x axis, 10° above the horizon,
   30° field of view, 512×512, with its scanned texture and a soft headlight. Crop and centre it
   exactly as the app does with an upload. The rendered alpha replaces background removal, because
   segmentation is not what this benchmark measures.
3. **Generate the six views once** with Zero123++ (fixed seed per object) and give **the same views
   to both methods**. Each object is then its own control, so the comparison isolates the
   reconstruction step.
4. **Reconstruct and clean up** with the app's own code (`MultiViewBackend.reconstruct_views`,
   including floater removal, hole filling, Taubin smoothing and decimation to 40k faces). Texture
   baking is skipped: it does not change the geometry.
5. **Align.** A single photo does not fix an object's size or distance, so a prediction is only
   defined up to a similarity transform (rotation, translation, scale). Both meshes are normalised
   as in step 1; the frames already agree on orientation, because the pipeline puts the input
   camera on +x with z up, the same as the render in step 2. Then similarity ICP (Besl & McKay,
   1992, with Umeyama's closed-form fit at each step) refines the fit, from 9 starting rotations
   within ±15° of the identity, keeping the best. Correspondences go both ways, so shrinking the
   prediction is not rewarded, and the worst 10% of pairs are ignored so a floater cannot pull the
   fit. The scan never moves after step 1. Scores before ICP are recorded too (`unaligned`).
6. **Score** on 100,000 points sampled uniformly over each surface (by area).

**Oracle views (optional, `--oracle`).** The same reconstruction from six views rendered from the
scan itself at Zero123++'s camera poses, i.e. perfect views. The gap between the normal run and
this one is the error the diffusion model adds; what is left is the reconstructor's own error.

## Metrics

With P sampled on the prediction, G on the scan, and d(x, S) the distance from x to the nearest
point of S:

| Metric | Definition | Reads as |
| --- | --- | --- |
| Accuracy | mean over P of d(p, G) | Is what we built really there? |
| Completeness | mean over G of d(g, P) | Did we build everything that is there? |
| **Chamfer distance** ↓ | (accuracy + completeness) / 2, plain (not squared) distances | Average surface error |
| **F-score@τ** ↑ | harmonic mean of precision (share of P within τ of G) and recall (share of G within τ of P), τ = 0.05, 0.1, 0.2 | Share of the surface that is right to within τ; bounded, robust to outliers (Knapitsch et al., 2017) |
| **Normal consistency** ↑ | mean of \|n_p · n_nearest\|, both directions | Orientation agreement: catches lumps and stair-steps that distances barely see (Mescheder et al., 2019) |
| Mask IoU ↑ | silhouette intersection over union in 4 novel views | Does the outline match from new angles? |
| PSNR ↑ | 4 novel views, whole image, white background | Rough appearance check (see caveats) |
| Time | wall clock of reconstruction + cleanup; view generation reported separately | |
| Peak GPU | largest PyTorch allocation during reconstruction | Will it fit a smaller GPU? |

F@0.2 matches the threshold InstantMesh reports on GSO (objects in [-1, 1]); 0.1 and 0.05 are
stricter. Squared Chamfer, per-object precision and recall, and the fitted alignment (scale,
rotation angle, translation) are also in `results.jsonl`.

## Statistics

- Each table cell is the **mean over objects with a 95% bootstrap confidence interval** (2,000
  resamples of the objects). With 30 objects the intervals are wide on purpose: overlapping
  intervals mean "not clearly different".
- Because both methods see the same objects and views, they are also compared **per object**:
  win rate, and a two-sided **Wilcoxon signed-rank test** on the per-object differences (no
  assumption that errors are normally distributed).
- A failed run (for example out of memory) is counted in the *Ok* column and left out of the means.
  Read the two together: a method that fails on the hard objects looks better than it is.

## Sanity checks

- The CPU tests feed the loop "reconstructions" that are known transforms of the ground truth:
  alignment brings them back to Chamfer < 0.03 and F@0.05 > 0.97, while a sphere stand-in scores
  far worse, so a broken metric or alignment fails CI.
- Frame conventions: a small NeuS fit on CPU to oracle views of a synthetic object lands in the
  scan's frame with Chamfer 0.051 and F@0.1 0.89 **before any alignment** (the same mesh turned
  90° scores 0.226 and 0.34). ICP then only moves it by 1.5° and 3% in scale. So the renderer,
  the cameras and the reconstruction agree on which way is up and which way the photo faces.

## Caveats

- **Rendered, not photographed, inputs.** Real photos add lighting, noise and segmentation errors.
  These numbers are an upper bound on real-world accuracy, the same as in the papers.
- **ICP helps every method.** It is applied equally, and the unaligned scores are kept, but
  absolute numbers are not directly comparable with papers that align differently.
- **PSNR is only indicative.** The prediction's colours come from the generated views (lighting
  baked in), the scan's from its texture, and the hidden underside is guessed.
- **30 objects** is enough to tell methods apart when the gap is large, not to rank small
  differences. `--objects 100` is the next step if Kaggle time allows.
- **One seed** per object. Diffusion is stochastic; repeating with `--seed` gives a sense of the
  variance.

## Run it

On Kaggle: open [`notebooks/facet_benchmark.ipynb`](../notebooks/facet_benchmark.ipynb), turn on
**GPU T4 x2** and **Internet**, then *Save Version → Save & Run All*. About 5 to 6 hours for 30
objects, both methods, with the oracle run. Results land in `/kaggle/working/bench/`.

On your own GPU:

```bash
bash scripts/setup_gpu_env.sh
cd backend
/tmp/facet-env/bin/python -m app.bench run --out ../bench-results --oracle   # resumable
/tmp/facet-env/bin/python -m app.bench report --out ../bench-results         # table + chart
```

Useful options: `--objects N`, `--methods instantmesh`, `--names A,B`, `--oracle-only` (no diffusion
model needed), `--recon-iters`, `--time-limit MINUTES`, `--retry-failed`.

## Results

*Pending the first Kaggle run.* The table (`results.md`) and chart (`chart.svg`) go here and in the
main README.

## References

- Downs et al., *Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items*, ICRA 2022.
- Knapitsch et al., *Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction*, SIGGRAPH 2017.
- Mescheder et al., *Occupancy Networks: Learning 3D Reconstruction in Function Space*, CVPR 2019.
- Besl & McKay, *A Method for Registration of 3-D Shapes*, TPAMI 1992.
- Umeyama, *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*, TPAMI 1991.
- Wilcoxon, *Individual Comparisons by Ranking Methods*, Biometrics Bulletin 1945.
- Efron, *Bootstrap Methods: Another Look at the Jackknife*, Annals of Statistics 1979.
