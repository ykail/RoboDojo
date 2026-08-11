#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CONDA_SH="/home/acone/miniconda3/etc/profile.d/conda.sh"
LEROBOT_PYTHON="/home/acone/micromamba/envs/arx-py310/bin/python"

RAW_ROOT=""
DATASET_ROOT="/home/acone/data/lerobot"
DATASET_ID=""
TASK="fill_pen_holder"
FPS="25"
ENV_GPU="0"
MAX_BUNDLES="0"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_acone_x5_batch_replay.sh \
    --raw-root PATH \
    --dataset-id ID \
    [--dataset-root /home/acone/data/lerobot] \
    [--task fill_pen_holder] \
    [--fps 25] \
    [--env-gpu 0] \
    [--max-bundles N]

This command needs neither the Hoo policy server nor the physical X5 source.
It resumes both the raw queue and an existing compatible LeRobot dataset.
EOF
}

die() {
    echo "[X5 raw replay][ERROR] $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-root) need_value "$@"; RAW_ROOT="$2"; shift 2 ;;
        --raw-root=*) RAW_ROOT="${1#*=}"; shift ;;
        --dataset-root) need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
        --dataset-root=*) DATASET_ROOT="${1#*=}"; shift ;;
        --dataset-id) need_value "$@"; DATASET_ID="$2"; shift 2 ;;
        --dataset-id=*) DATASET_ID="${1#*=}"; shift ;;
        --task) need_value "$@"; TASK="$2"; shift 2 ;;
        --task=*) TASK="${1#*=}"; shift ;;
        --fps) need_value "$@"; FPS="$2"; shift 2 ;;
        --fps=*) FPS="${1#*=}"; shift ;;
        --env-gpu) need_value "$@"; ENV_GPU="$2"; shift 2 ;;
        --env-gpu=*) ENV_GPU="${1#*=}"; shift ;;
        --max-bundles) need_value "$@"; MAX_BUNDLES="$2"; shift 2 ;;
        --max-bundles=*) MAX_BUNDLES="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ -n "${RAW_ROOT}" ]] || die "--raw-root is required"
[[ -d "${RAW_ROOT}" ]] || die "raw collection does not exist: ${RAW_ROOT}"
[[ -n "${DATASET_ID}" ]] || die "--dataset-id is required"
[[ "${DATASET_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "dataset ID contains unsupported characters"
[[ "${TASK}" =~ ^[a-z0-9_]+$ ]] || die "invalid task name: ${TASK}"
[[ "${FPS}" == "25" ]] || die "the canonical X5 replay rate is fixed at 25 Hz"
[[ "${ENV_GPU}" =~ ^[0-9]+$ ]] || die "--env-gpu must be a non-negative integer"
[[ "${MAX_BUNDLES}" =~ ^[0-9]+$ ]] || die "--max-bundles must be a non-negative integer"
[[ -f "${CONDA_SH}" ]] || die "Conda activation script is missing: ${CONDA_SH}"
[[ -x "${LEROBOT_PYTHON}" ]] || die "LeRobot Python is unavailable: ${LEROBOT_PYTHON}"

mkdir -p "${DATASET_ROOT}"
source "${CONDA_SH}"
conda activate RoboDojo

export ROBODOJO_CONTROL_MODE="x5_raw_replay_25hz"
export ROBODOJO_X5_RAW_ROOT="${RAW_ROOT}"
export ROBODOJO_X5_RAW_REPLAY_FPS="${FPS}"
export ROBODOJO_X5_RAW_REPLAY_MAX_BUNDLES="${MAX_BUNDLES}"
export ROBODOJO_LEROBOT_PYTHON="${LEROBOT_PYTHON}"
export ROBODOJO_LEROBOT_ROOT="${DATASET_ROOT}"
export ROBODOJO_LEROBOT_REPO_ID="${DATASET_ID}"
export ROBODOJO_LEROBOT_RESUME=0
if [[ -d "${DATASET_ROOT%/}/${DATASET_ID}" ]]; then
    export ROBODOJO_LEROBOT_RESUME=1
fi
export ROBODOJO_LEROBOT_VCODEC="${ROBODOJO_LEROBOT_VCODEC:-h264}"
export ROBODOJO_LEROBOT_ENCODER_THREADS="${ROBODOJO_LEROBOT_ENCODER_THREADS:-2}"
export ROBODOJO_LEROBOT_STREAMING_ENCODING=1
export ROBODOJO_RENDERING_MODE=quality
export ROBODOJO_X5_CUDA_PIPELINE=0
export ROBODOJO_MAIN_RATE_LIMIT_HZ="${ROBODOJO_MAIN_RATE_LIMIT_HZ:-250}"
export ROBODOJO_HEADLESS=1
export HEADLESS=1
export LIVESTREAM=0
export EVAL_NUM=native
export ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-3}"
export ROBODOJO_RUN_ID="x5_raw_replay_$(date -u +%Y%m%dT%H%M%S%NZ)_$$_${RANDOM}"

echo "[X5 raw replay] raw=${RAW_ROOT}"
echo "[X5 raw replay] output=${DATASET_ROOT%/}/${DATASET_ID}"
echo "[X5 raw replay] task=${TASK} fps=${FPS} GPU=${ENV_GPU}"
echo "[X5 raw replay] policy, tunnel and physical X5 are not used"

cd "${ROBODOJO_ROOT}"
exec bash scripts/eval_policy.sh \
    --root_dir "${ROBODOJO_ROOT}" \
    --task_name "${TASK}" \
    --env_cfg_type arx_x5 \
    --device_id "${ENV_GPU}" \
    --policy_name Pi_05 \
    --port 1 \
    --protocol ws \
    --policy_runtime xpolicy_ws_v0 \
    --action_type joint \
    --additional_info x5_raw_replay_25hz \
    --seed 0 \
    --host 127.0.0.1
