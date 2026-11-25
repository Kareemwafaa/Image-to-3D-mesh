# Facet (Image to 3D): frontend

Angular app for the Image-to-3D platform: upload an image, watch the job's
progress, view the generated mesh in 3D and download it as `.glb`.

## Run locally

```bash
npm install
npm start            # http://localhost:4200
```

### Which backend it talks to

One static build works with any backend; nothing is baked in at build time. The URL is chosen at
runtime, first match wins:

1. `?api=<url>` in the page address. It is saved in the browser and removed from the address. The
   free-GPU notebook prints a link like this, so visitors connect with one click.
2. The URL saved in this browser, set from the **Server** button in the top-right corner (it checks
   `GET /health` before switching).
3. `apiUrl` in `public/config.json` (`dist/frontend/browser/config.json` after a build): the
   deployment's default. Editing it needs no rebuild.
4. `http://localhost:8000`.

When no backend answers, the landing page offers to connect one. The backend must allow
this app's origin in `CORS_ORIGINS` (`*` is fine: the API uses no cookies).

## Build and test

```bash
npm run build        # static files in dist/frontend/browser, host on any static host
npx ng test --watch=false
```

## How it talks to the backend

| Call | Used for |
| --- | --- |
| `POST /generate` (multipart `image`, `remove_background`, `mc_resolution`) | starts a job, returns `{job_id}` |
| `GET /jobs/{job_id}` every 2.5 s | status, `progress` 0..1, optional `stage` (segmenting, multiview, reconstructing, meshing), `error`; up to 5 transient failures are retried |
| `GET /health` every 30 s | server status dot in the top bar |
| `GET /jobs/{job_id}/result` | the `.glb`, fetched once and reused for the viewer and the download |

The landing page's hero (`point-cloud.ts`, a small 2D-canvas renderer) shows the product in one loop:
a grid of points shaped like a photo of a mug peels up and folds into a 3D point cloud of the mug,
turns, and flattens back. Drag in any direction to turn it (trackball, with inertia); grabbing the
photo folds it straight away. With a mouse it leans toward the cursor and points near it light up.
Reduced-motion users get the finished mesh, still, which moves only while dragged.

The 3D viewer is Google's [`<model-viewer>`](https://modelviewer.dev), loaded lazily.
Client-side limits (PNG/JPEG/WebP, 10 MB) are a guess; the backend has the final say.

Jobs take a few minutes, so the running job's id is kept in the address bar
(`?job=...`). Reloading the page resumes it instead of losing it.

## Deploy

Pushed to GitHub Pages by `.github/workflows/pages.yml` on every change to `frontend/` on `main`.
One-time setup: repo Settings → Pages → Source: "GitHub Actions".
