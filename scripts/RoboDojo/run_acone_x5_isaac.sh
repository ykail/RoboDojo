#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CONDA_SH="/home/acone/miniconda3/etc/profile.d/conda.sh"
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_GPU_MB:-12000}"

[[ -f "${CONDA_SH}" ]] || {
    echo "[acOne Isaac][ERROR] Conda activation script is missing: ${CONDA_SH}" >&2
    exit 2
}

source "${CONDA_SH}"
conda activate RoboDojo

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
[[ "${free_gpu_mb}" =~ ^[0-9]+$ ]] || {
    echo "[acOne Isaac][ERROR] Could not read free memory on GPU 0" >&2
    exit 2
}
if (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "[acOne Isaac][ERROR] GPU 0 has ${free_gpu_mb} MiB free; at least ${MIN_FREE_GPU_MB} MiB is required." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
    exit 2
fi

export X5_LEROBOT_PYTHON="/home/acone/micromamba/envs/arx-py310/bin/python"
export ROBODOJO_LEROBOT_ROOT="/home/acone/data/lerobot"
export ROBODOJO_LEROBOT_REPO_ID="robodojo_make_toast_x5_online_dagger_v1"
export ROBODOJO_POLICY_PORT="18080"
export ROBODOJO_X5_SOURCE_HOST="127.0.0.1"
export ROBODOJO_X5_SOURCE_PORT="8770"
export ROBODOJO_ENV_GPU="0"
export DISPLAY="${DISPLAY:-:1}"

cd "${ROBODOJO_ROOT}"
exec "${SCRIPT_DIR}/run_x5_dagger_isaac.sh"
