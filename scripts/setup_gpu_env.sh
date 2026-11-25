#!/usr/bin/env bash
# Builds an isolated Python environment for the GPU backend.
#
#   bash scripts/setup_gpu_env.sh            # creates /tmp/facet-env
#   FACET_ENV=~/facet-env bash scripts/setup_gpu_env.sh
#
# Hosted notebooks (Colab, Kaggle) come with hundreds of preinstalled packages
# whose versions change without notice. Installing our pinned stack into its
# own virtual environment, with its own Python 3.11, makes a run reproducible
# no matter what the host image ships. uv does the install in a minute or two.
set -euo pipefail

ENV_DIR="${FACET_ENV:-/tmp/facet-env}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REQ="$ROOT/backend/requirements/lite-gpu.txt"
OVERRIDES=()
if [[ ! -f "$REQ" ]]; then
  # Older layout. Its transformers pin conflicts with its huggingface-hub pin,
  # so pin a patch release that accepts both.
  REQ="$ROOT/backend/requirements-gpu.txt"
  printf 'transformers==4.35.2\n' > /tmp/facet-overrides.txt
  OVERRIDES=(--override /tmp/facet-overrides.txt)
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  python3 -m pip install --quiet uv
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "==> Creating Python 3.11 environment in $ENV_DIR"
  uv venv --quiet --python 3.11 "$ENV_DIR"
fi

echo "==> Installing $(realpath --relative-to="$ROOT" "$REQ")"
uv pip install --quiet --python "$ENV_DIR/bin/python" -r "$REQ" "uvicorn[standard]>=0.30" "${OVERRIDES[@]}"

"$ENV_DIR/bin/python" - <<'PY'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"==> Ready: torch {torch.__version__}, GPU: {gpu}")
PY
