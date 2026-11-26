#!/usr/bin/env python3
"""Run the backend on this machine and give it a public HTTPS URL.

    python scripts/serve_public.py
    python scripts/serve_public.py --frontend https://you.github.io/Image-to-3D-mesh/

Made for free notebook GPUs (Colab, Kaggle), which have no public address:

  1. starts the API (lite mode: API and GPU worker in one process) on localhost;
  2. opens a Cloudflare quick tunnel to it, which needs no account and gives a
     random https://<words>.trycloudflare.com address that lasts until this
     script stops;
  3. waits until the models are loaded, then prints the public URL and a link
     that opens the web app already connected to it.

Stop it with Ctrl+C (or the notebook's stop button); both processes are shut
down. Uses only the standard library, so it runs under any Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DEFAULT_FRONTEND = "https://kareemwafaa.github.io/Image-to-3D-mesh/"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "facet"
CLOUDFLARED_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
# Quick-tunnel hostnames are random hyphenated words. Requiring a hyphen keeps
# api.trycloudflare.com, which appears in cloudflared's error messages, out.
TUNNEL_URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")


def log(msg: str) -> None:
    print(f"[facet] {msg}", flush=True)


# --------------------------------------------------------------------------
# Child processes
# --------------------------------------------------------------------------


def pipe_output(proc: subprocess.Popen, prefix: str, on_line=None) -> threading.Thread:
    """Copy a child's output to ours, line by line, with a prefix."""

    def run() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if on_line:
                on_line(line)
            print(f"{prefix} {line}", flush=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def server_command(python: str, port: int) -> list[str]:
    # Lite mode (app/lite.py) runs the API and the GPU worker in one process.
    # Older checkouts only have the plain FastAPI app; serve that instead.
    if (BACKEND / "app" / "lite.py").exists():
        return [python, "-m", "app.lite"]
    return [python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]


def start_server(python: str, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(HOST="127.0.0.1", PORT=str(port), PYTHONUNBUFFERED="1")
    # The API sets no cookies and has no login, so any web page may call it.
    # This lets the GitHub Pages app (or anyone's copy of it) connect.
    env.setdefault("CORS_ORIGINS", "*")
    env.setdefault("PRELOAD_MODEL", "true")
    # Lets PyTorch grow memory segments instead of fragmenting the T4's 15 GB.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = server_command(python, port)
    log(f"Starting the API: {' '.join(cmd[1:])}")
    return subprocess.Popen(
        cmd, cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )


def cloudflared_binary() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if system != "linux" or arch is None:
        sys.exit(
            "cloudflared is not installed. Get it from "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ "
            "(macOS: brew install cloudflared), or run with --no-tunnel."
        )
    path = CACHE / f"cloudflared-linux-{arch}"
    if not path.exists():
        log("Downloading cloudflared")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(CLOUDFLARED_RELEASE + path.name, tmp)
        tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR)
        tmp.rename(path)
    return str(path)


def start_tunnel(port: int) -> tuple[subprocess.Popen, threading.Event, list[str]]:
    """Start a Cloudflare quick tunnel. The returned event fires once its URL is known."""
    found: list[str] = []
    ready = threading.Event()
    proc = subprocess.Popen(
        [cloudflared_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def run() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            match = TUNNEL_URL.search(line)
            if match and not found:
                found.append(match.group(0))
                ready.set()
            if " ERR " in line or "failed" in line.lower():  # cloudflared is chatty; show only problems
                print(f"[tunnel] {line}", flush=True)

    threading.Thread(target=run, daemon=True).start()
    return proc, ready, found


# --------------------------------------------------------------------------
# Waiting for the API
# --------------------------------------------------------------------------


def get_health(base: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as res:
            return json.load(res)
    except Exception:
        return None


def wait_until_ready(base: str, server: subprocess.Popen, timeout_s: float) -> dict:
    """Block until /health reports the model loaded (first start downloads ~5 GB of weights)."""
    started = time.monotonic()
    last_note = 0.0
    while True:
        if server.poll() is not None:
            sys.exit(f"The API exited with code {server.returncode}; see its log above.")
        health = get_health(base)
        if health and health.get("model_error"):
            sys.exit(f"The model failed to load: {health['model_error']}")
        if health and health.get("model_loaded"):
            return health
        waited = time.monotonic() - started
        if waited > timeout_s:
            sys.exit(f"The model was not ready after {timeout_s / 60:.0f} minutes.")
        if waited - last_note >= 30:
            state = "API up, loading models" if health else "starting the API"
            log(f"Waiting: {state} ({waited / 60:.1f} min)")
            last_note = waited
        time.sleep(2)


def banner(public_url: str | None, local_url: str, frontend: str | None) -> str:
    lines = ["", "=" * 72, "  Facet backend is ready", ""]
    if public_url:
        lines += [f"  Public API URL   {public_url}"]
        if frontend:
            sep = "&" if "?" in frontend else "?"
            app_link = f"{frontend}{sep}api={urllib.parse.quote(public_url, safe='')}"
            lines += ["", "  Open the app, already connected:", f"  {app_link}"]
        lines += ["", "  Or test from a terminal:", f"  curl -F image=@photo.jpg {public_url}/generate"]
    else:
        lines += [f"  Local API URL    {local_url}"]
    lines += [
        "",
        "  Keep this cell running. The URL changes every time you restart it.",
        "  Stop with Ctrl+C or the stop button to free the GPU.",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--python", default=sys.executable, help="Python that has the backend's requirements")
    parser.add_argument(
        "--frontend", default=DEFAULT_FRONTEND, help="Web app URL to link to; empty string for none"
    )
    parser.add_argument("--no-tunnel", action="store_true", help="Serve on localhost only")
    parser.add_argument("--ready-timeout", type=float, default=30, help="Minutes to wait for the models")
    args = parser.parse_args()

    local_url = f"http://127.0.0.1:{args.port}"
    procs: list[subprocess.Popen] = []

    def stop_all() -> None:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        log("Stopped.")

    # A notebook's stop button sends SIGINT (KeyboardInterrupt); `kill` sends SIGTERM.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        server = start_server(args.python, args.port)
        procs.append(server)
        pipe_output(server, "[api]")

        public_url = None
        if not args.no_tunnel:
            tunnel, tunnel_ready, found = start_tunnel(args.port)
            procs.append(tunnel)
            deadline = time.monotonic() + 60
            while not tunnel_ready.wait(timeout=1):
                if tunnel.poll() is not None or time.monotonic() > deadline:
                    sys.exit(
                        "Could not open a Cloudflare tunnel (see [tunnel] lines above). "
                        "Check that this notebook has internet access, then run the cell again."
                    )
            public_url = found[0]
            log(f"Tunnel open: {public_url} (the API is still starting)")

        health = wait_until_ready(local_url, server, args.ready_timeout * 60)
        log(f"Model ready (backend: {health.get('backend')})")
        print(banner(public_url, local_url, args.frontend or None), flush=True)

        while all(p.poll() is None for p in procs):
            time.sleep(5)
        dead = next(p for p in procs if p.poll() is not None)
        sys.exit(f"The {'API' if dead is server else 'tunnel'} exited with code {dead.returncode}.")
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == "__main__":
    main()
