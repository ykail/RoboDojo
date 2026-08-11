#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CONDA_SH="/home/acone/miniconda3/etc/profile.d/conda.sh"
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_GPU_MB:-12000}"
DOCUMENTS_DIR="/home/acone/Documents"
REQUIRED_DRIVER_MAJOR="580"

POLICY_DIR=""
TASK=""
CHECKPOINT_ID=""
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
DATASET_ID="${ROBODOJO_LEROBOT_REPO_ID:-}"
DATASET_ROOT="${ROBODOJO_LEROBOT_ROOT:-/home/acone/data/lerobot}"
PREFLIGHT_ONLY=0
CAPTURE_MODE="online"
RAW_ROOT=""
TARGET_EPISODES="50"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_acone_x5_isaac.sh \
    --policy-dir HOO_CHECKPOINT_PATH \
    --task TASK \
    [--port 18081]

Required:
  --policy-dir PATH          The same checkpoint path passed to run_hoo_policy.sh.
                             It is used as a policy identity; it need not exist on acOne.
  --task TASK                RoboDojo task, for example fill_pen_holder or make_toast.

Optional:
  --checkpoint-id ID         Default: TASK/POLICY_DIR_BASENAME.
  --port N                   Local SSH-tunnel port (default: 18081).
  --dataset-id ID            Default includes task, checkpoint name, digest, and Kai0 commit.
  --dataset-root PATH        LeRobot root (default: /home/acone/data/lerobot).
  --capture-mode MODE        online or raw-deferred (default: online).
  --raw-root PATH            Required for raw-deferred collection.
  --target-episodes N        Durable raw bundles to collect (default: 50).
  --preflight-only           Verify and print the live policy identity; do not start Isaac.
  -h, --help                 Show this help.

Both --option value and --option=value forms are accepted.
The launcher discovers the live HELLO, then pins its exact commit and checkpoint digest
before Isaac starts.
EOF
}

die() {
    echo "[acOne Isaac][ERROR] $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-dir)
            need_value "$1" "${2:-}"
            POLICY_DIR="$2"
            shift 2
            ;;
        --policy-dir=*) POLICY_DIR="${1#*=}"; shift ;;
        --task)
            need_value "$1" "${2:-}"
            TASK="$2"
            shift 2
            ;;
        --task=*) TASK="${1#*=}"; shift ;;
        --checkpoint-id)
            need_value "$1" "${2:-}"
            CHECKPOINT_ID="$2"
            shift 2
            ;;
        --checkpoint-id=*) CHECKPOINT_ID="${1#*=}"; shift ;;
        --port|--policy-port)
            need_value "$1" "${2:-}"
            POLICY_PORT="$2"
            shift 2
            ;;
        --port=*|--policy-port=*) POLICY_PORT="${1#*=}"; shift ;;
        --dataset-id)
            need_value "$1" "${2:-}"
            DATASET_ID="$2"
            shift 2
            ;;
        --dataset-id=*) DATASET_ID="${1#*=}"; shift ;;
        --dataset-root)
            need_value "$1" "${2:-}"
            DATASET_ROOT="$2"
            shift 2
            ;;
        --dataset-root=*) DATASET_ROOT="${1#*=}"; shift ;;
        --capture-mode)
            need_value "$1" "${2:-}"
            CAPTURE_MODE="$2"
            shift 2
            ;;
        --capture-mode=*) CAPTURE_MODE="${1#*=}"; shift ;;
        --raw-root)
            need_value "$1" "${2:-}"
            RAW_ROOT="$2"
            shift 2
            ;;
        --raw-root=*) RAW_ROOT="${1#*=}"; shift ;;
        --target-episodes)
            need_value "$1" "${2:-}"
            TARGET_EPISODES="$2"
            shift 2
            ;;
        --target-episodes=*) TARGET_EPISODES="${1#*=}"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ -n "${POLICY_DIR}" ]] || die "--policy-dir is required"
[[ -n "${TASK}" ]] || die "--task is required"
[[ "${TASK}" =~ ^[a-z0-9_]+$ ]] \
    || die "task must contain only lowercase letters, digits, and underscores"
[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || die "port must be an integer"
(( POLICY_PORT >= 1 && POLICY_PORT <= 65535 )) || die "port must be in [1, 65535]"
[[ "${CAPTURE_MODE}" == "online" || "${CAPTURE_MODE}" == "raw-deferred" ]] \
    || die "--capture-mode must be online or raw-deferred"
[[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]] \
    || die "--target-episodes must be a positive integer"
if [[ "${CAPTURE_MODE}" == "raw-deferred" ]]; then
    [[ -n "${RAW_ROOT}" ]] || die "--raw-root is required for raw-deferred"
    mkdir -p "${RAW_ROOT}"
    RAW_ROOT="$(cd "${RAW_ROOT}" && pwd -P)"
fi

POLICY_DIR="${POLICY_DIR%/}"
POLICY_NAME="$(basename -- "${POLICY_DIR}")"
[[ "${POLICY_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "policy directory basename is not a safe identifier: ${POLICY_NAME}"
if [[ -z "${CHECKPOINT_ID}" ]]; then
    CHECKPOINT_ID="${TASK}/${POLICY_NAME}"
fi
[[ "${CHECKPOINT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] \
    || die "checkpoint ID contains unsupported characters: ${CHECKPOINT_ID}"
if [[ -n "${DATASET_ID}" ]]; then
    [[ "${DATASET_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || die "dataset ID contains unsupported characters: ${DATASET_ID}"
fi

[[ -f "${CONDA_SH}" ]] || die "Conda activation script is missing: ${CONDA_SH}"

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

if ! (exec 3<>"/dev/tcp/127.0.0.1/${POLICY_PORT}") >/dev/null 2>&1; then
    die "Hoo policy tunnel is not reachable at 127.0.0.1:${POLICY_PORT}"
fi

echo "[acOne Isaac] discovering policy HELLO at ws://127.0.0.1:${POLICY_PORT}"
PROVENANCE_JSON="$(
    PYTHONPATH="${ROBODOJO_ROOT}" python \
        "${SCRIPT_DIR}/preflight_policy_v1.py" \
        --url "ws://127.0.0.1:${POLICY_PORT}" \
        --expected-checkpoint-id "${CHECKPOINT_ID}" \
        --require-clean
)" || die "policy HELLO discovery failed"

PROVENANCE_FIELDS="$(
    ROBODOJO_POLICY_PROVENANCE_JSON="${PROVENANCE_JSON}" python -c '
import json
import os
import re

payload = json.loads(os.environ["ROBODOJO_POLICY_PROVENANCE_JSON"])
revision = payload.get("code_revision", "")
digest = payload.get("checkpoint_digest", "")
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise SystemExit("HELLO code_revision is not a full lowercase git commit")
if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
    raise SystemExit("HELLO checkpoint_digest is not sha256:<64 lowercase hex>")
print(revision + "\t" + digest)
'
)" || die "policy HELLO provenance is invalid"
IFS=$'\t' read -r EXPECTED_KAI0_COMMIT EXPECTED_CHECKPOINT_DIGEST <<<"${PROVENANCE_FIELDS}"

if [[ -z "${DATASET_ID}" ]]; then
    DIGEST_HEX="${EXPECTED_CHECKPOINT_DIGEST#sha256:}"
    DATASET_ID="robodojo_${TASK}_x5_online_dagger_${POLICY_NAME}_${DIGEST_HEX:0:12}_${EXPECTED_KAI0_COMMIT:0:8}_timing25_v2"
fi

echo "[acOne Isaac] task=${TASK}"
echo "[acOne Isaac] policy_dir_on_hoo=${POLICY_DIR}"
echo "[acOne Isaac] checkpoint_id=${CHECKPOINT_ID}"
echo "[acOne Isaac] pinned Kai0=${EXPECTED_KAI0_COMMIT} clean=true"
echo "[acOne Isaac] pinned digest=${EXPECTED_CHECKPOINT_DIGEST}"
if [[ "${CAPTURE_MODE}" == "raw-deferred" ]]; then
    echo "[acOne Isaac] capture=raw-deferred target=${TARGET_EPISODES} root=${RAW_ROOT}"
    echo "[acOne Isaac] Right atomically commits raw only; RGB is generated by batch replay"
else
    echo "[acOne Isaac] dataset=${DATASET_ROOT%/}/${DATASET_ID}"
    echo "[acOne Isaac] manual timing=wall-time zero-order hold resampled to 25Hz"
fi

if (( PREFLIGHT_ONLY )); then
    echo "[acOne Isaac] PREFLIGHT OK; Isaac was not started"
    exit 0
fi

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
[[ "${free_gpu_mb}" =~ ^[0-9]+$ ]] || die "could not read free memory on GPU 0"
if (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "[acOne Isaac][ERROR] GPU 0 has ${free_gpu_mb} MiB free; at least ${MIN_FREE_GPU_MB} MiB is required." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
    exit 2
fi

export X5_LEROBOT_PYTHON="/home/acone/micromamba/envs/arx-py310/bin/python"
export ROBODOJO_TASK="${TASK}"
export ROBODOJO_LEROBOT_ROOT="${DATASET_ROOT}"
export ROBODOJO_LEROBOT_REPO_ID="${DATASET_ID}"
export ROBODOJO_CHECKPOINT_ID="${CHECKPOINT_ID}"
export ROBODOJO_EXPECTED_KAI0_COMMIT="${EXPECTED_KAI0_COMMIT}"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="${EXPECTED_CHECKPOINT_DIGEST}"
export ROBODOJO_POLICY_PORT="${POLICY_PORT}"
export ROBODOJO_X5_SOURCE_HOST="127.0.0.1"
export ROBODOJO_X5_SOURCE_PORT="8770"
export ROBODOJO_ENV_GPU="0"
export ROBODOJO_X5_CAPTURE_MODE="${CAPTURE_MODE}"
export ROBODOJO_X5_RAW_ROOT="${RAW_ROOT}"
export ROBODOJO_X5_TARGET_EPISODES="${TARGET_EPISODES}"
export DISPLAY="${DISPLAY:-:1}"

cd "${ROBODOJO_ROOT}"
exec "${SCRIPT_DIR}/run_x5_dagger_isaac.sh"
