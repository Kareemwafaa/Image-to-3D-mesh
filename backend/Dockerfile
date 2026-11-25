# Self-hosted GPU image (any machine or cloud with an NVIDIA GPU).
# The main deployment target is Modal (see modal_app.py); this is the
# alternative for running the same service on your own GPU box.
#
#   docker build -t image-to-3d-api backend/
#   docker run --gpus all -p 8000:8000 -v $PWD/data:/data image-to-3d-api
#
# Options:
#   --build-arg WITH_TRIPOSR=1   also install the TripoSR feed-forward baseline
#   --build-arg BAKE_WEIGHTS=0   download weights at first start instead of build time
#   -e MODEL_BACKEND=mock        no GPU needed (placeholder meshes)

# GUESS: torch 2.4.1 / CUDA 12.1 matches requirements-gpu.txt. *devel* is only
# needed for WITH_TRIPOSR=1 (it compiles torchmcubes); runtime is enough otherwise.
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
FROM ${BASE_IMAGE}

ARG WITH_TRIPOSR=0
# GUESS: upstream has no releases; this is main as of 2026-06-04.
ARG TRIPOSR_REF=107cefdc244c39106fa830359024f6a2f1c78871
ARG BAKE_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    U2NET_HOME=/opt/u2net \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1 \
    PYTHONPATH=/opt/TripoSR

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential cmake libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt requirements-triposr.txt ./
RUN pip install -r requirements-gpu.txt uvicorn[standard]

RUN if [ "$WITH_TRIPOSR" = "1" ]; then \
        git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR \
        && git -C /opt/TripoSR checkout "$TRIPOSR_REF" \
        && pip install scikit-build-core pybind11 \
        && pip install --no-build-isolation -r requirements-triposr.txt ; \
    fi

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -c "import rembg; rembg.new_session('u2net')" \
        && python -c "from huggingface_hub import snapshot_download as d; d('sudo-ai/zero123plus-v1.2'); d('sudo-ai/zero123plus-pipeline')" \
        && if [ "$WITH_TRIPOSR" = "1" ]; then \
            python -c "from huggingface_hub import hf_hub_download as d; [d('stabilityai/TripoSR', f) for f in ('config.yaml','model.ckpt')]" ; \
        fi ; \
    fi

COPY app ./app

ENV MODEL_BACKEND=multiview \
    DATA_DIR=/data \
    PORT=8000
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fs http://localhost:${PORT}/health || exit 1

# One worker process: the models live in memory and jobs share one GPU.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
