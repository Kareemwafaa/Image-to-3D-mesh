# Record a demo for LinkedIn (and the README)

A 30 to 60 second screen recording does more for the project than any description. Here is a
plan that fits in one free-GPU session.

## 1. Prepare (before you start the GPU)

- **Pick 3 to 4 photos** of single objects on plain backgrounds: a mug, a shoe, a toy, a plant pot.
  Matte objects with clear shapes reconstruct best. Avoid glass, mirrors, thin wires and hair.
  Take them yourself, or use photos you have the rights to.
- **Test them first.** Start the notebook ([guide](run-on-a-free-gpu.md)), run every photo once,
  and keep the best 2 or 3. Download each `.glb` and the six-view image as a backup.
- Close other tabs, hide bookmarks, and set the browser zoom so the app fills the window at
  1920×1080 (or 1280×720).

## 2. Record

Tools: **Windows**: Xbox Game Bar (`Win`+`G`) or [OBS Studio](https://obsproject.com) (free).
**macOS**: `Cmd`+`Shift`+`5`. Record only the browser window.

Shot list (about 45 seconds):

| Time | On screen | Why |
| --- | --- | --- |
| 0–3 s | The app's landing page, **Server ready** in the corner | Shows it's live, not a mock-up |
| 3–8 s | Drop a photo onto the page, click **Generate 3D model** | The one-step input |
| 8–20 s | The progress screen; the **six generated views** appear | The diffusion stage, the "wow" moment |
| 20–35 s | The finished mesh, rotating; drag to turn it | The result |
| 35–45 s | **Download GLB**, open it in Blender or Windows 3D Viewer | It's a real, usable asset |

A real job takes minutes, so record it once and **cut or speed up** the waiting part (2× to 8×) in
an editor (Clipchamp on Windows, iMovie on macOS, or [Kdenlive](https://kdenlive.org), all free).
Say in the caption that it is sped up.

## 3. Export

- **LinkedIn video:** MP4, 1080p, under 1 minute, with captions burned in (most people watch muted).
- **README GIF:** 8 to 12 seconds of the best part, 960 px wide, 15 fps, under 10 MB. Convert with
  [ffmpeg](https://ffmpeg.org):

  ```bash
  ffmpeg -i demo.mp4 -ss 20 -t 10 -vf "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" docs/media/demo.gif
  ```

  Then in [README.md](../README.md), replace `docs/media/facet-landing.png` with
  `docs/media/demo.gif`. (Or drag the MP4 into any GitHub issue comment box, copy the link it
  makes, and paste that link on its own line in the README: GitHub shows it as a video player.)

## 4. Measure (for the results table)

While you record, note for each object: the time the app shows when it finishes, and the six-view
image (right-click → save). The notebook's log prints per-stage timings for every job. Put 2 or 3
rows in the **Results** table of the README: photo, six views, a screenshot or GIF of the mesh,
and the time. Real numbers from a free T4 are a strong signal on a CV.

## 5. Write the post

Keep it short and concrete. A structure that works:

1. One line on what it does: *"I built Facet: upload one photo, get a textured 3D model."*
2. The video.
3. Two or three lines on how: multi-view diffusion (Zero123++) imagines the unseen sides, a neural
   signed distance field (NeuS) is fitted to those views by differentiable volume rendering, and
   marching cubes extracts the mesh. Served by a queue-based API with GPU workers.
4. One honest line on limits (minutes per object, struggles with transparent objects).
5. The repo link, and that anyone can run it free on Colab or Kaggle.
