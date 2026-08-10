#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CONDA_SH="/home/acone/miniconda3/etc/profile.d/conda.sh"
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_GPU_MB:-12000}"
DOCUMENTS_DIR="/home/acone/Documents"
REQUIRED_DRIVER_MAJOR="580"

[[ -f "${CONDA_SH}" ]] || {
    echo "[acOne Isaac][ERROR] Conda activation script is missing: ${CONDA_SH}" >&2
    exit 2
}

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | head -n 1 | tr -d '[:space:]')"
driver_major="${driver_version%%.*}"
[[ "${driver_major}" =~ ^[0-9]+$ && "${driver_major}" == "${REQUIRED_DRIVER_MAJOR}" ]] || {
    echo "[acOne Isaac][ERROR] Isaac Sim 5.1 requires R580 on acOne; found ${driver_version:-unknown}." >&2
    echo "[acOne Isaac][ERROR] R595 crashes in RTX SceneDB before RoboDojo starts." >&2
    exit 2
}
if [[ -e "${DOCUMENTS_DIR}" && ! -w "${DOCUMENTS_DIR}" ]]; then
    echo "[acOne Isaac][ERROR] ${DOCUMENTS_DIR} is not writable." >&2
    echo "[acOne Isaac][ERROR] Run: sudo chown acone:acone ${DOCUMENTS_DIR}" >&2
    exit 2
fi
mkdir -p \
    "${DOCUMENTS_DIR}/Kit/shared/screenshots" \
    "${DOCUMENTS_DIR}/Kit/apps/Isaac-Sim/scripts/new_stage"

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
export ROBODOJO_TASK="${ROBODOJO_TASK:-make_toast}"
export ROBODOJO_LEROBOT_ROOT="${ROBODOJO_LEROBOT_ROOT:-/home/acone/data/lerobot}"
export ROBODOJO_LEROBOT_REPO_ID="${ROBODOJO_LEROBOT_REPO_ID:-robodojo_${ROBODOJO_TASK}_x5_online_dagger_v1}"
export ROBODOJO_CHECKPOINT_ID="${ROBODOJO_CHECKPOINT_ID:-RoboDojo-sim-arx_x5-joint-0/59999}"
export ROBODOJO_EXPECTED_KAI0_COMMIT="${ROBODOJO_EXPECTED_KAI0_COMMIT-ecc1a7451c3156b1e5f7533851dbb0222896206f}"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="${ROBODOJO_EXPECTED_CHECKPOINT_DIGEST-sha256:70bb68139ba717553d9a9c3055bb322b85046d729377ee46eaaf997c1eaac4}"
export ROBODOJO_POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
export ROBODOJO_X5_SOURCE_HOST="127.0.0.1"
export ROBODOJO_X5_SOURCE_PORT="8770"
export ROBODOJO_ENV_GPU="0"
export DISPLAY="${DISPLAY:-:1}"

cd "${ROBODOJO_ROOT}"
exec "${SCRIPT_DIR}/run_x5_dagger_isaac.sh"
