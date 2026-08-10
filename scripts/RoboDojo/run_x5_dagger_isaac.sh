#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
SOURCE_HOST="${ROBODOJO_X5_SOURCE_HOST:-127.0.0.1}"
SOURCE_PORT="${ROBODOJO_X5_SOURCE_PORT:-8770}"
DATASET_ROOT="${ROBODOJO_LEROBOT_ROOT:-${HOME}/data/lerobot}"
DATASET_ID="${ROBODOJO_LEROBOT_REPO_ID:-robodojo_make_toast_x5_online_dagger_v1}"
DATASET_PATH="${DATASET_ROOT%/}/${DATASET_ID}"
LEROBOT_PYTHON="${X5_LEROBOT_PYTHON:-${ROBODOJO_LEROBOT_PYTHON:-${HOME}/vibe_code/kai0-robodojo-policy-v1/.venv/bin/python}}"
EVAL_NUM="${ROBODOJO_EVAL_NUM:-1}"
ENV_GPU="${ROBODOJO_ENV_GPU:-0}"
SEED="${ROBODOJO_SEED:-0}"
POLICY_SEED="${ROBODOJO_POLICY_SEED:-${SEED}}"
ENCODER_THREADS="${ROBODOJO_LEROBOT_ENCODER_THREADS:-2}"

CHECKPOINT_ID="RoboDojo-sim-arx_x5-joint-0/59999"
EXPECTED_KAI0_COMMIT="ecc1a7451c3156b1e5f7533851dbb0222896206f"
EXPECTED_CHECKPOINT_DIGEST="sha256:70bb68139ba717553d9a9d9c3055bb322b85046d729377ee46eaaf997c1eaac4"
MIRROR_PROTOCOL="robodojo_dual_joint_mirror_v1"
MIRROR_PROFILE="arx_x5_identity_joint_v1"

die() {
    echo "[X5 Isaac][ERROR] $*" >&2
    exit 2
}

for port in "${POLICY_PORT}" "${SOURCE_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] || die "policy/source ports must be integers"
    (( port >= 1 && port <= 65535 )) || die "policy/source ports must be in [1, 65535]"
done
[[ "${SOURCE_HOST}" == "127.0.0.1" || "${SOURCE_HOST}" == "localhost" ]] \
    || die "the X5 hardware source must remain on loopback"
[[ -f "${ROBODOJO_ROOT}/scripts/RoboDojo/eval_kai0_pi05.sh" ]] \
    || die "RoboDojo evaluator launcher is missing under ${ROBODOJO_ROOT}"
[[ -x "${LEROBOT_PYTHON}" ]] || die "LeRobot writer Python is not executable: ${LEROBOT_PYTHON}"
[[ "${ROBODOJO_DUAL_MIRROR_PROFILE:-${MIRROR_PROFILE}}" == "${MIRROR_PROFILE}" ]] \
    || die "X5 collection requires profile ${MIRROR_PROFILE}"
[[ "${ROBODOJO_DUAL_MIRROR_PROTOCOL:-${MIRROR_PROTOCOL}}" == "${MIRROR_PROTOCOL}" ]] \
    || die "X5 collection requires protocol ${MIRROR_PROTOCOL}"

if ! (exec 3<>"/dev/tcp/127.0.0.1/${POLICY_PORT}") >/dev/null 2>&1; then
    die "Hoo policy tunnel is not reachable at 127.0.0.1:${POLICY_PORT}"
fi
if ! (exec 3<>"/dev/tcp/${SOURCE_HOST}/${SOURCE_PORT}") >/dev/null 2>&1; then
    die "X5 hardware source is not reachable at ${SOURCE_HOST}:${SOURCE_PORT}"
fi

args=(
    --task make_toast
    --checkpoint-id "${CHECKPOINT_ID}"
    --external-policy-server-url "ws://127.0.0.1:${POLICY_PORT}"
    --expected-kai0-commit "${EXPECTED_KAI0_COMMIT}"
    --expected-checkpoint-digest "${EXPECTED_CHECKPOINT_DIGEST}"
    --lerobot-python "${LEROBOT_PYTHON}"
    --lerobot-root "${DATASET_ROOT}"
    --lerobot-repo-id "${DATASET_ID}"
    --lerobot-vcodec "${ROBODOJO_LEROBOT_VCODEC:-h264}"
    --encoder-threads "${ENCODER_THREADS}"
    --eval-num "${EVAL_NUM}"
    --env-gpu "${ENV_GPU}"
    --seed "${SEED}"
    --policy-seed "${POLICY_SEED}"
    --control-mode x5_policy_joint_intervention
)

if [[ -e "${DATASET_PATH}" || -L "${DATASET_PATH}" ]]; then
    [[ -f "${DATASET_PATH}/meta/info.json" ]] \
        || die "existing output is not a LeRobot v3 dataset: ${DATASET_PATH}"
    args+=(--resume)
fi

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJO_REALTIME=1
# Keep the official visual domain. During intervention the three data-camera
# render products are paused and their quality frames are rendered afterwards
# from exact simulator snapshots.
export ROBODOJO_RENDERING_MODE=quality
# Keep the stock CPU tensor pipeline: RoboDojo's reward and scene logic passes
# poses to NumPy/Shapely.  The CUDA/Fabric path is experimental and can be
# enabled explicitly with ROBODOJO_X5_CUDA_PIPELINE=1 after those boundaries
# are ported.
export ROBODOJO_X5_CUDA_PIPELINE="${ROBODOJO_X5_CUDA_PIPELINE:-0}"
# make_toast uses dt=4 ms and ten physics steps per 25 Hz action.  The old
# 125 Hz Kit cap alone could consume roughly 80 ms before camera rendering.
export ROBODOJO_MAIN_RATE_LIMIT_HZ="${ROBODOJO_MAIN_RATE_LIMIT_HZ:-250}"
export ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-1}"
export ROBODOJO_DUAL_MIRROR_HOST="${SOURCE_HOST}"
export ROBODOJO_DUAL_MIRROR_PORT="${SOURCE_PORT}"
export ROBODOJO_DUAL_MIRROR_PROTOCOL="${MIRROR_PROTOCOL}"
export ROBODOJO_DUAL_MIRROR_PROFILE="${MIRROR_PROFILE}"
export ROBODOJO_DUAL_MIRROR_RECORD=1
export ROBODOJO_X5_CODE_ROOT="${ROBODOJO_ROOT}"
export ROBODOJO_RUN_ID="x5_online_dagger_$(date -u +%Y%m%dT%H%M%S%NZ)_$$_${RANDOM}"

echo "[X5 Isaac] Hoo policy=ws://127.0.0.1:${POLICY_PORT}"
echo "[X5 Isaac] hardware=${SOURCE_HOST}:${SOURCE_PORT} protocol=${MIRROR_PROTOCOL}"
echo "[X5 Isaac] profile=${MIRROR_PROFILE} signs=[+1,+1,+1,+1,+1,+1]"
echo "[X5 Isaac] dataset=${DATASET_PATH}"
echo "[X5 Isaac] rendering=${ROBODOJO_RENDERING_MODE} (official quality is the default)"
echo "[X5 Isaac] CUDA tensor/Fabric pipeline=${ROBODOJO_X5_CUDA_PIPELINE} (0 is the supported default)"
echo "[X5 Isaac] Kit main-loop cap=${ROBODOJO_MAIN_RATE_LIMIT_HZ}Hz"
echo "[X5 Isaac] use the global i key to enter/leave intervention; terminal focus is not required"
echo "[X5 Isaac] an episode is committed automatically at its terminal outcome"

cd "${ROBODOJO_ROOT}"
exec bash scripts/RoboDojo/eval_kai0_pi05.sh "${args[@]}"
