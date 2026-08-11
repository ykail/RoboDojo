#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
SOURCE_HOST="${ROBODOJO_X5_SOURCE_HOST:-127.0.0.1}"
SOURCE_PORT="${ROBODOJO_X5_SOURCE_PORT:-8770}"
DATASET_ROOT="${ROBODOJO_LEROBOT_ROOT:-${HOME}/data/lerobot}"
TASK="${ROBODOJO_TASK:-make_toast}"
DATASET_ID="${ROBODOJO_LEROBOT_REPO_ID:-robodojo_${TASK}_x5_online_dagger_simstep25_v3}"
DATASET_PATH="${DATASET_ROOT%/}/${DATASET_ID}"
LEROBOT_PYTHON="${X5_LEROBOT_PYTHON:-${ROBODOJO_LEROBOT_PYTHON:-${HOME}/vibe_code/kai0-robodojo-policy-v1/.venv/bin/python}}"
EVAL_NUM="${ROBODOJO_EVAL_NUM:-1}"
ENV_GPU="${ROBODOJO_ENV_GPU:-0}"
SEED="${ROBODOJO_SEED:-0}"
POLICY_SEED="${ROBODOJO_POLICY_SEED:-${SEED}}"
ENCODER_THREADS="${ROBODOJO_LEROBOT_ENCODER_THREADS:-2}"
CAPTURE_MODE="${ROBODOJO_X5_CAPTURE_MODE:-online}"
RAW_ROOT="${ROBODOJO_X5_RAW_ROOT:-}"
TARGET_EPISODES="${ROBODOJO_X5_TARGET_EPISODES:-50}"

CHECKPOINT_ID="${ROBODOJO_CHECKPOINT_ID:-}"
EXPECTED_KAI0_COMMIT="${ROBODOJO_EXPECTED_KAI0_COMMIT:-}"
EXPECTED_CHECKPOINT_DIGEST="${ROBODOJO_EXPECTED_CHECKPOINT_DIGEST:-}"
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
[[ -n "${CHECKPOINT_ID}" ]] \
    || die "ROBODOJO_CHECKPOINT_ID is unset; start through run_acone_x5_isaac.sh"
[[ "${EXPECTED_KAI0_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "ROBODOJO_EXPECTED_KAI0_COMMIT must be the commit discovered from policy HELLO"
[[ "${EXPECTED_CHECKPOINT_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "ROBODOJO_EXPECTED_CHECKPOINT_DIGEST must be the digest discovered from policy HELLO"
[[ "${CAPTURE_MODE}" == "online" || "${CAPTURE_MODE}" == "raw-deferred" ]] \
    || die "ROBODOJO_X5_CAPTURE_MODE must be online or raw-deferred"
[[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]] \
    || die "ROBODOJO_X5_TARGET_EPISODES must be a positive integer"
if [[ "${CAPTURE_MODE}" == "raw-deferred" ]]; then
    [[ -n "${RAW_ROOT}" && -d "${RAW_ROOT}" ]] \
        || die "ROBODOJO_X5_RAW_ROOT must be an existing directory in raw-deferred mode"
fi

if ! (exec 3<>"/dev/tcp/127.0.0.1/${POLICY_PORT}") >/dev/null 2>&1; then
    die "Hoo policy tunnel is not reachable at 127.0.0.1:${POLICY_PORT}"
fi
if ! (exec 3<>"/dev/tcp/${SOURCE_HOST}/${SOURCE_PORT}") >/dev/null 2>&1; then
    die "X5 hardware source is not reachable at ${SOURCE_HOST}:${SOURCE_PORT}"
fi

args=(
    --task "${TASK}"
    --checkpoint-id "${CHECKPOINT_ID}"
    --external-policy-server-url "ws://127.0.0.1:${POLICY_PORT}"
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

args+=(
    --expected-kai0-commit "${EXPECTED_KAI0_COMMIT}"
    --expected-checkpoint-digest "${EXPECTED_CHECKPOINT_DIGEST}"
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
# A dataset row is one completed 25 Hz simulator control transition.  Never
# stretch slow wall-clock collection by cloning observation/action rows.
export ROBODOJO_LEROBOT_TIMING_CONTRACT="sim_step_exact_25hz_v1"
# Keep the official visual domain.  Policy and intervention frames are both
# captured online so Right only drains the encoder and commits the episode.
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
export ROBODOJO_DUAL_MIRROR_TIMEOUT_S="${ROBODOJO_DUAL_MIRROR_TIMEOUT_S:-15}"
export ROBODOJO_DUAL_MIRROR_PROTOCOL="${MIRROR_PROTOCOL}"
export ROBODOJO_DUAL_MIRROR_PROFILE="${MIRROR_PROFILE}"
if [[ "${CAPTURE_MODE}" == "raw-deferred" ]]; then
    export ROBODOJO_DUAL_MIRROR_RECORD=0
    export ROBODOJO_X5_RAW_CAPTURE=1
else
    export ROBODOJO_DUAL_MIRROR_RECORD=1
    export ROBODOJO_X5_RAW_CAPTURE=0
fi
export ROBODOJO_X5_CODE_ROOT="${ROBODOJO_ROOT}"
export ROBODOJO_RUN_ID="x5_${TASK}_online_dagger_$(date -u +%Y%m%dT%H%M%S%NZ)_$$_${RANDOM}"

echo "[X5 Isaac] Hoo policy=ws://127.0.0.1:${POLICY_PORT}"
echo "[X5 Isaac] task=${TASK} checkpoint=${CHECKPOINT_ID}"
echo "[X5 Isaac] hardware=${SOURCE_HOST}:${SOURCE_PORT} protocol=${MIRROR_PROTOCOL}"
echo "[X5 Isaac] profile=${MIRROR_PROFILE} signs=[+1,+1,+1,+1,+1,+1]"
echo "[X5 Isaac] rendering=${ROBODOJO_RENDERING_MODE} (official quality is the default)"
if [[ "${CAPTURE_MODE}" == "raw-deferred" ]]; then
    echo "[X5 Isaac] recording=atomic raw bundles; online LeRobot/video writer disabled"
    echo "[X5 Isaac] raw=${RAW_ROOT} target=${TARGET_EPISODES}; completed bundles auto-resume"
else
    echo "[X5 Isaac] dataset=${DATASET_PATH}"
    echo "[X5 Isaac] recording=one complete policy+human LeRobot episode per Right"
    echo "[X5 Isaac] timing=one real simulator transition per 25Hz row; no fill frames"
    echo "[X5 Isaac] target=${TARGET_EPISODES}; completed episodes auto-resume"
fi
echo "[X5 Isaac] CUDA tensor/Fabric pipeline=${ROBODOJO_X5_CUDA_PIPELINE} (0 is the supported default)"
echo "[X5 Isaac] Kit main-loop cap=${ROBODOJO_MAIN_RATE_LIMIT_HZ}Hz"
echo "[X5 Isaac] global keys: i=intervene, Left=discard/retry, Right=save/next"
echo "[X5 Isaac] operator collection has no task step limit"

cd "${ROBODOJO_ROOT}"
exec bash scripts/RoboDojo/eval_kai0_pi05.sh "${args[@]}"
