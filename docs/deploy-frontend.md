# Host the web app (GitHub Pages, or Vercel)

The web app in [`frontend/`](../frontend/) is a static site: after a build it is just HTML, CSS and
JavaScript files, so any static host can serve it for free. It does not contain the backend's
address; each visitor picks a backend at runtime (the **Server** button, or `?api=` in the address).
So you build and deploy it **once**, and it keeps working when your backend's address changes.

## GitHub Pages (what this repo uses)

The workflow [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) tests, builds and
publishes the app every time `frontend/` changes on `main`.

**One-time setup** (in the GitHub website):

1. Open the repository → **Settings** → **Pages** (left sidebar).
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Go to the **Actions** tab → **Deploy frontend to GitHub Pages** → **Run workflow** (or push any
   change under `frontend/`).
4. When the run is green, the app is live at `https://<your-username>.github.io/<repo-name>/`, for
   this repo **https://kareemwafaa.github.io/Image-to-3D-mesh/**.

Notes:

- The repository must be **public** for Pages on a free GitHub account.
- Forks work too: the workflow uses the fork's own repository name for the site path.
- If you fork, also change `FRONTEND_URL` in the notebook and `DEFAULT_FRONTEND` in
  `scripts/serve_public.py` to your own Pages address, so the notebook links to your copy.

## Vercel (optional alternative)

1. Sign in at [vercel.com](https://vercel.com) with GitHub, click **Add New → Project** and import
   the repository.
2. Set **Root Directory** to `frontend`. Vercel detects Angular. Set **Output Directory** to
   `dist/frontend/browser`.
3. Click **Deploy**. You get an address like `https://image-to-3d-mesh.vercel.app`.

Vercel redeploys on every push to `main`.

## Point the app at a default backend (optional)

[`frontend/public/config.json`](../frontend/public/config.json) holds the address the app uses when
the visitor hasn't chosen one:

```json
{ "apiUrl": "http://localhost:8000" }
```

Leave it as `localhost` for a public demo (visitors are guided to start their own backend), or set it
to a backend that is always on. On a host, you can edit `config.json` in the built files without
rebuilding.

## The backend must allow the app's address (CORS)

Browsers only let a page call a backend on another domain if the backend says so. The launcher
`scripts/serve_public.py` sets `CORS_ORIGINS=*` (any site), which is safe because the API uses no
cookies or logins. If you run the backend another way, set `CORS_ORIGINS` to your app's address,
for example `CORS_ORIGINS=https://kareemwafaa.github.io,http://localhost:4200`.
