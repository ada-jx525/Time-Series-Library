#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-branchworld}
CUDA_TAG=${2:-cu121}

conda create -y -n "${ENV_NAME}" python=3.11
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

if [[ "${CUDA_TAG}" == "cpu" ]]; then
  conda run -n "${ENV_NAME}" pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
else
  conda run -n "${ENV_NAME}" pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

conda run -n "${ENV_NAME}" pip install -r requirements_branchworld.txt

echo "Environment ${ENV_NAME} is ready. Activate with: conda activate ${ENV_NAME}"
