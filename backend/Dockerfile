# One Dockerfile, three images (build with --target):
#   api         FastAPI gateway, no ML dependencies (~500 MB vs several GB)
#   worker-cpu  worker with the mock model, for laptops and CI
#   worker-gpu  worker with Zero123++ + NeuS; weights baked in (BAKE_WEIGHTS=1)
#
# The PyPI torch wheel bundles CUDA 12.1, so a slim Python base works on any
# host with an NVIDIA driver and the NVIDIA container toolkit.

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app

# --- gateway ------------------------------------------------------------------
FROM base AS api
COPY requirements/api.txt requirements/
RUN pip install -r requirements/api.txt
COPY alembic.ini ./
COPY app ./app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD curl -fs http://127.0.0.1:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

# --- CPU worker (mock model) ---------------------------------------------------------
FROM base AS worker-cpu
COPY requirements/api.txt requirements/worker-cpu.txt requirements/
RUN pip install -r requirements/worker-cpu.txt
COPY alembic.ini ./
COPY app ./app
ENV MODEL_BACKEND=mock U2NET_HOME=/home/app/.u2net
USER app
EXPOSE 9100
CMD ["python", "-m", "app.worker"]

# --- GPU worker ----------------------------------------------------------------------
FROM base AS worker-gpu
ARG BAKE_WEIGHTS=1
COPY requirements/api.txt requirements/worker-cpu.txt requirements/worker-gpu.txt requirements/
RUN pip install -r requirements/worker-gpu.txt
ENV HF_HOME=/opt/models/hf U2NET_HOME=/opt/models/u2net MODEL_BACKEND=multiview
RUN mkdir -p /opt/models && if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -c "import rembg; rembg.new_session('u2net')" \
        && python -c "from huggingface_hub import snapshot_download as d; d('sudo-ai/zero123plus-v1.2'); d('sudo-ai/zero123plus-pipeline')" ; \
    fi && chown -R app /opt/models
COPY alembic.ini ./
COPY app ./app
USER app
EXPOSE 9100
CMD ["python", "-m", "app.worker"]
