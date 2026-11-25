# Facet (Image to 3D): frontend

Angular app for the Image-to-3D platform: upload an image, watch the job's
progress, view the generated mesh in 3D and download it as `.glb`.

## Run locally

```bash
npm install
npm start            # http://localhost:4200
```

The backend URL is read at runtime from `public/config.json`
(`dist/frontend/browser/config.json` after a build), default `http://localhost:8000`.
To point a deployment at another backend, edit that file; no rebuild needed.
The backend must allow this app's origin in `CORS_ORIGINS`.

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

The 3D viewer is Google's [`<model-viewer>`](https://modelviewer.dev), loaded lazily.
Client-side limits (PNG/JPEG/WebP, 10 MB) are a guess; the backend has the final say.

Jobs take a few minutes, so the running job's id is kept in the address bar
(`?job=...`). Reloading the page resumes it instead of losing it.
