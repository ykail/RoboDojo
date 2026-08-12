#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
SOURCE_HOST="${ROBODOJO_PIPERX_SOURCE_HOST:-127.0.0.1}"
SOURCE_PORT="${ROBODOJO_PIPERX_SOURCE_PORT:-8770}"
DATASET_ROOT="${ROBODOJO_LEROBOT_ROOT:-/home/hoo/data/lerobot}"
TASK="${ROBODOJO_TASK:-make_toast}"
DATASET_ID="${ROBODOJO_LEROBOT_REPO_ID:-robodojo_${TASK}_piperx_online_dagger_simstep25_v1}"
DATASET_PATH="${DATASET_ROOT%/}/${DATASET_ID}"
LEROBOT_PYTHON="${ROBODOJO_LEROBOT_PYTHON:-/home/hoo/RoboDojo/third_party/kai0/.venv/bin/python}"
PIPERX_CODE_ROOT="${PIPERX_BRIDGE_ROOT:-/home/hoo/piper_x/lerobot_sealab-piperx-online-dagger-v2}"
EVAL_NUM="${ROBODOJO_EVAL_NUM:-1}"
ENV_GPU="${ROBODOJO_ENV_GPU:-0}"
SEED="${ROBODOJO_SEED:-0}"
POLICY_SEED="${ROBODOJO_POLICY_SEED:-${SEED}}"
ENCODER_THREADS="${ROBODOJO_LEROBOT_ENCODER_THREADS:-2}"
TARGET_EPISODES="${ROBODOJO_DUAL_MIRROR_TARGET_EPISODES:-${ROBODOJO_PIPERX_TARGET_EPISODES:-50}}"

CHECKPOINT_ID="${ROBODOJO_CHECKPOINT_ID:-}"
EXPECTED_KAI0_COMMIT="${ROBODOJO_EXPECTED_KAI0_COMMIT:-}"
EXPECTED_CHECKPOINT_DIGEST="${ROBODOJO_EXPECTED_CHECKPOINT_DIGEST:-}"
MIRROR_PROTOCOL="robodojo_dual_joint_mirror_v1"
MIRROR_PROFILE="arx_x5_piperx_relative_joint_v1"

die() {
    echo "[PiPER-X Isaac][ERROR] $*" >&2
    exit 2
}

for port in "${POLICY_PORT}" "${SOURCE_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] || die "policy/source ports must be integers"
    (( port >= 1 && port <= 65535 )) || die "policy/source ports must be in [1, 65535]"
done
[[ "${POLICY_PORT}" != "${SOURCE_PORT}" ]] || die "policy and hardware source ports must differ"
[[ "${SOURCE_HOST}" == "127.0.0.1" || "${SOURCE_HOST}" == "localhost" ]] \
    || die "the PiPER-X hardware source must remain on loopback"
[[ -f "${ROBODOJO_ROOT}/scripts/RoboDojo/eval_kai0_pi05.sh" ]] \
    || die "RoboDojo evaluator launcher is missing under ${ROBODOJO_ROOT}"
[[ -x "${LEROBOT_PYTHON}" ]] || die "LeRobot writer Python is not executable: ${LEROBOT_PYTHON}"
[[ -d "${PIPERX_CODE_ROOT}" ]] || die "PiPER-X code root is missing: ${PIPERX_CODE_ROOT}"
[[ -d "${PIPERX_CODE_ROOT}/.git" || -f "${PIPERX_CODE_ROOT}/.git" ]] \
    || die "PiPER-X code root is not a Git checkout: ${PIPERX_CODE_ROOT}"
PIPERX_COMMIT="$(git -C "${PIPERX_CODE_ROOT}" rev-parse --verify HEAD)" \
    || die "cannot read PiPER-X code revision"
[[ "${PIPERX_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "PiPER-X revision is not a full commit"
[[ -z "$(git -C "${PIPERX_CODE_ROOT}" status --porcelain=v1 --untracked-files=normal)" ]] \
    || die "PiPER-X code checkout is dirty; refusing unverifiable collection"
[[ "${ROBODOJO_DUAL_MIRROR_PROFILE:-${MIRROR_PROFILE}}" == "${MIRROR_PROFILE}" ]] \
    || die "PiPER-X collection requires profile ${MIRROR_PROFILE}"
[[ "${ROBODOJO_DUAL_MIRROR_PROTOCOL:-${MIRROR_PROTOCOL}}" == "${MIRROR_PROTOCOL}" ]] \
    || die "PiPER-X collection requires protocol ${MIRROR_PROTOCOL}"
[[ -n "${CHECKPOINT_ID}" ]] || die "ROBODOJO_CHECKPOINT_ID is unset; use run_hoo_piperx_isaac.sh"
[[ "${EXPECTED_KAI0_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "ROBODOJO_EXPECTED_KAI0_COMMIT must be the commit discovered from policy HELLO"
[[ "${EXPECTED_CHECKPOINT_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "ROBODOJO_EXPECTED_CHECKPOINT_DIGEST must be the digest discovered from policy HELLO"
[[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]] \
    || die "target episodes must be a positive integer"

if ! (exec 3<>"/dev/tcp/127.0.0.1/${POLICY_PORT}") >/dev/null 2>&1; then
    die "external policy tunnel is not reachable at 127.0.0.1:${POLICY_PORT}"
fi
if ! (exec 3<>"/dev/tcp/${SOURCE_HOST}/${SOURCE_PORT}") >/dev/null 2>&1; then
    die "PiPER-X hardware source is not reachable at ${SOURCE_HOST}:${SOURCE_PORT}"
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
    --control-mode piperx_policy_joint_intervention
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
export ROBODOJO_LEROBOT_TIMING_CONTRACT="sim_step_exact_25hz_v1"
export ROBODOJO_RENDERING_MODE=quality
export ROBODOJO_MAIN_RATE_LIMIT_HZ="${ROBODOJO_MAIN_RATE_LIMIT_HZ:-250}"
export ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-1}"
export ROBODOJO_DUAL_MIRROR_HOST="${SOURCE_HOST}"
export ROBODOJO_DUAL_MIRROR_PORT="${SOURCE_PORT}"
export ROBODOJO_DUAL_MIRROR_TIMEOUT_S="${ROBODOJO_DUAL_MIRROR_TIMEOUT_S:-15}"
export ROBODOJO_DUAL_MIRROR_PROTOCOL="${MIRROR_PROTOCOL}"
export ROBODOJO_DUAL_MIRROR_PROFILE="${MIRROR_PROFILE}"
export ROBODOJO_DUAL_MIRROR_RECORD=1
export ROBODOJO_DUAL_MIRROR_TARGET_EPISODES="${TARGET_EPISODES}"
export ROBODOJO_PIPERX_TARGET_EPISODES="${TARGET_EPISODES}"
# A stale X5 shell must never turn a PiPER-X run into raw-deferred mode.
export ROBODOJO_X5_RAW_CAPTURE=0
export ROBODOJO_PIPERX_CODE_ROOT="${PIPERX_CODE_ROOT}"
export ROBODOJO_RUN_ID="piperx_${TASK}_online_dagger_$(date -u +%Y%m%dT%H%M%S%NZ)_$$_${RANDOM}"

echo "[PiPER-X Isaac] external policy=ws://127.0.0.1:${POLICY_PORT}"
echo "[PiPER-X Isaac] task=${TASK} checkpoint=${CHECKPOINT_ID}"
echo "[PiPER-X Isaac] hardware=${SOURCE_HOST}:${SOURCE_PORT} protocol=${MIRROR_PROTOCOL}"
echo "[PiPER-X Isaac] hardware_code=${PIPERX_CODE_ROOT}@${PIPERX_COMMIT} clean=true"
echo "[PiPER-X Isaac] profile=${MIRROR_PROFILE} signs=[+1,+1,-1,-1,+1,+1]"
echo "[PiPER-X Isaac] dataset=${DATASET_PATH}"
echo "[PiPER-X Isaac] one Right commits one complete policy+human episode; Left retries the same layout"
echo "[PiPER-X Isaac] timing=one real simulator transition per 25Hz row; no fill frames"
echo "[PiPER-X Isaac] target=${TARGET_EPISODES}; committed episodes auto-resume"
echo "[PiPER-X Isaac] global keys: i=toggle intervention, Left=discard/retry, Right=save/next"
echo "[PiPER-X Isaac] operator collection has no task step limit"

cd "${ROBODOJO_ROOT}"
exec bash scripts/RoboDojo/eval_kai0_pi05.sh "${args[@]}"
