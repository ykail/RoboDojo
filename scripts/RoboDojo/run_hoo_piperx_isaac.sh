#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

POLICY_DIR=""
TASK=""
CHECKPOINT_ID=""
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
SOURCE_PORT="${ROBODOJO_PIPERX_SOURCE_PORT:-8770}"
DATASET_ID="${ROBODOJO_LEROBOT_REPO_ID:-}"
DATASET_ROOT="${ROBODOJO_LEROBOT_ROOT:-/home/hoo/data/lerobot}"
TARGET_EPISODES="50"
LEROBOT_PYTHON="${ROBODOJO_LEROBOT_PYTHON:-/home/hoo/RoboDojo/third_party/kai0/.venv/bin/python}"
PREFLIGHT_PYTHON="${ROBODOJO_PREFLIGHT_PYTHON:-/opt/anaconda3/envs/RoboDojo/bin/python}"
ISAAC_PYTHON="${ROBODOJO_ISAAC_PYTHON:-/opt/anaconda3/envs/RoboDojo/bin/python}"
ASSETS_PATH="${ROBODOJO_ASSETS_PATH:-/home/hoo/RoboDojo-piperx-dagger-v2/.cache/robodojo_assets_repo/Assets}"
PIPERX_CODE_ROOT="${PIPERX_BRIDGE_ROOT:-/home/hoo/piper_x/lerobot_sealab-piperx-online-dagger-v2}"
PREFLIGHT_ONLY=0
REQUIRED_KAI0_COMMIT=""
REQUIRED_CHECKPOINT_DIGEST=""

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_hoo_piperx_isaac.sh \
    --policy-dir REMOTE_POLICY_PATH \
    --task TASK \
    [--checkpoint-id ID] [--port 18080]

The policy directory is an identity string and need not exist on Hoo.  Start
the yikai policy server and SSH tunnel first, then start the two-PiPER hardware
owner before this launcher.

Optional:
  --source-port N       PiPER-X source port (default: 8770)
  --dataset-id ID       Auto-generated from task and live policy provenance
  --dataset-root PATH   Default: /home/hoo/data/lerobot
  --target-episodes N   Right-accepted durable episodes (default: 50)
  --lerobot-python PATH LeRobot writer environment
  --isaac-python PATH   Isaac Sim/RoboDojo Python
                        (default: /opt/anaconda3/envs/RoboDojo/bin/python)
  --assets-path PATH    Existing RoboDojo Assets root
  --expected-kai0-commit COMMIT
                        Optionally require this exact policy-server commit
  --expected-checkpoint-digest sha256:HEX
                        Optionally require this exact checkpoint digest
  --preflight-only      Verify policy HELLO without starting Isaac
EOF
}

die() {
    echo "[Hoo PiPER-X Isaac][ERROR] $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-dir) need_value "$1" "${2:-}"; POLICY_DIR="$2"; shift 2 ;;
        --policy-dir=*) POLICY_DIR="${1#*=}"; shift ;;
        --task) need_value "$1" "${2:-}"; TASK="$2"; shift 2 ;;
        --task=*) TASK="${1#*=}"; shift ;;
        --checkpoint-id) need_value "$1" "${2:-}"; CHECKPOINT_ID="$2"; shift 2 ;;
        --checkpoint-id=*) CHECKPOINT_ID="${1#*=}"; shift ;;
        --port|--policy-port) need_value "$1" "${2:-}"; POLICY_PORT="$2"; shift 2 ;;
        --port=*|--policy-port=*) POLICY_PORT="${1#*=}"; shift ;;
        --source-port) need_value "$1" "${2:-}"; SOURCE_PORT="$2"; shift 2 ;;
        --source-port=*) SOURCE_PORT="${1#*=}"; shift ;;
        --dataset-id) need_value "$1" "${2:-}"; DATASET_ID="$2"; shift 2 ;;
        --dataset-id=*) DATASET_ID="${1#*=}"; shift ;;
        --dataset-root) need_value "$1" "${2:-}"; DATASET_ROOT="$2"; shift 2 ;;
        --dataset-root=*) DATASET_ROOT="${1#*=}"; shift ;;
        --target-episodes) need_value "$1" "${2:-}"; TARGET_EPISODES="$2"; shift 2 ;;
        --target-episodes=*) TARGET_EPISODES="${1#*=}"; shift ;;
        --lerobot-python) need_value "$1" "${2:-}"; LEROBOT_PYTHON="$2"; shift 2 ;;
        --lerobot-python=*) LEROBOT_PYTHON="${1#*=}"; shift ;;
        --isaac-python) need_value "$1" "${2:-}"; ISAAC_PYTHON="$2"; shift 2 ;;
        --isaac-python=*) ISAAC_PYTHON="${1#*=}"; shift ;;
        --assets-path) need_value "$1" "${2:-}"; ASSETS_PATH="$2"; shift 2 ;;
        --assets-path=*) ASSETS_PATH="${1#*=}"; shift ;;
        --expected-kai0-commit) need_value "$1" "${2:-}"; REQUIRED_KAI0_COMMIT="$2"; shift 2 ;;
        --expected-kai0-commit=*) REQUIRED_KAI0_COMMIT="${1#*=}"; shift ;;
        --expected-checkpoint-digest) need_value "$1" "${2:-}"; REQUIRED_CHECKPOINT_DIGEST="$2"; shift 2 ;;
        --expected-checkpoint-digest=*) REQUIRED_CHECKPOINT_DIGEST="${1#*=}"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ -n "${POLICY_DIR}" ]] || die "--policy-dir is required"
[[ -n "${TASK}" ]] || die "--task is required"
[[ "${TASK}" =~ ^[a-z0-9_]+$ ]] || die "task contains unsupported characters"
for port in "${POLICY_PORT}" "${SOURCE_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] || die "ports must be integers"
    (( port >= 1 && port <= 65535 )) || die "ports must be in [1, 65535]"
done
[[ "${POLICY_PORT}" != "${SOURCE_PORT}" ]] || die "policy and source ports must differ"
[[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "--target-episodes must be positive"
[[ -z "${REQUIRED_KAI0_COMMIT}" || "${REQUIRED_KAI0_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "--expected-kai0-commit must be a full lowercase commit"
[[ -z "${REQUIRED_CHECKPOINT_DIGEST}" || "${REQUIRED_CHECKPOINT_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "--expected-checkpoint-digest must be sha256:<64 lowercase hex>"
[[ -x "${PREFLIGHT_PYTHON}" ]] || die "RoboDojo preflight Python is not executable: ${PREFLIGHT_PYTHON}"
[[ -x "${ISAAC_PYTHON}" ]] || die "Isaac Sim Python is not executable: ${ISAAC_PYTHON}"
[[ -x "${LEROBOT_PYTHON}" ]] || die "LeRobot writer Python is not executable: ${LEROBOT_PYTHON}"
[[ -d "${PIPERX_CODE_ROOT}" ]] || die "PiPER-X code root is missing: ${PIPERX_CODE_ROOT}"
[[ -f "${ASSETS_PATH}/Robots/x5/robot_config.yml" ]] \
    || die "RoboDojo X5 assets are missing: ${ASSETS_PATH}"
for asset_subdir in Object Material Eval_Layout; do
    [[ -d "${ASSETS_PATH}/${asset_subdir}" ]] \
        || die "RoboDojo asset subdirectory is missing: ${ASSETS_PATH}/${asset_subdir}"
done
if ! "${ISAAC_PYTHON}" -c 'import isaacsim' >/dev/null 2>&1; then
    die "Isaac Sim Python cannot import isaacsim: ${ISAAC_PYTHON}"
fi

POLICY_DIR="${POLICY_DIR%/}"
POLICY_NAME="$(basename -- "${POLICY_DIR}")"
[[ "${POLICY_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "policy basename is not a safe identifier: ${POLICY_NAME}"
if [[ -z "${CHECKPOINT_ID}" ]]; then
    CHECKPOINT_ID="${TASK}/${POLICY_NAME}"
fi
[[ "${CHECKPOINT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] \
    || die "checkpoint ID contains unsupported characters"

if ! (exec 3<>"/dev/tcp/127.0.0.1/${POLICY_PORT}") >/dev/null 2>&1; then
    die "yikai policy tunnel is not reachable at 127.0.0.1:${POLICY_PORT}"
fi

PREFLIGHT_ARGS=(
    --url "ws://127.0.0.1:${POLICY_PORT}"
    --expected-checkpoint-id "${CHECKPOINT_ID}"
    --require-clean
)
if [[ -n "${REQUIRED_KAI0_COMMIT}" ]]; then
    PREFLIGHT_ARGS+=(--expected-code-revision "${REQUIRED_KAI0_COMMIT}")
fi
if [[ -n "${REQUIRED_CHECKPOINT_DIGEST}" ]]; then
    PREFLIGHT_ARGS+=(--expected-checkpoint-digest "${REQUIRED_CHECKPOINT_DIGEST}")
fi

echo "[Hoo PiPER-X Isaac] discovering policy HELLO at ws://127.0.0.1:${POLICY_PORT}"
PROVENANCE_JSON="$(
    PYTHONPATH="${ROBODOJO_ROOT}" "${PREFLIGHT_PYTHON}" \
        "${SCRIPT_DIR}/preflight_policy_v1.py" \
        "${PREFLIGHT_ARGS[@]}"
)" || die "policy HELLO discovery failed"

PROVENANCE_FIELDS="$(
    ROBODOJO_POLICY_PROVENANCE_JSON="${PROVENANCE_JSON}" "${PREFLIGHT_PYTHON}" -c '
import json, os, re
payload = json.loads(os.environ["ROBODOJO_POLICY_PROVENANCE_JSON"])
revision = payload.get("code_revision", "")
digest = payload.get("checkpoint_digest", "")
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise SystemExit("HELLO code_revision is not a full lowercase commit")
if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
    raise SystemExit("HELLO checkpoint_digest is not canonical sha256")
print(revision + "\t" + digest)
'
)" || die "policy HELLO provenance is invalid"
IFS=$'\t' read -r EXPECTED_KAI0_COMMIT EXPECTED_CHECKPOINT_DIGEST <<<"${PROVENANCE_FIELDS}"

if [[ -z "${DATASET_ID}" ]]; then
    DIGEST_HEX="${EXPECTED_CHECKPOINT_DIGEST#sha256:}"
    DATASET_ID="robodojo_${TASK}_piperx_online_dagger_${POLICY_NAME}_${DIGEST_HEX:0:12}_${EXPECTED_KAI0_COMMIT:0:8}_simstep25_v1"
fi
[[ "${DATASET_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "dataset ID contains unsupported characters"

echo "[Hoo PiPER-X Isaac] policy_dir_on_yikai=${POLICY_DIR}"
echo "[Hoo PiPER-X Isaac] checkpoint_id=${CHECKPOINT_ID}"
echo "[Hoo PiPER-X Isaac] pinned Kai0=${EXPECTED_KAI0_COMMIT} clean=true"
echo "[Hoo PiPER-X Isaac] pinned digest=${EXPECTED_CHECKPOINT_DIGEST}"
echo "[Hoo PiPER-X Isaac] isaac_python=${ISAAC_PYTHON}"
echo "[Hoo PiPER-X Isaac] assets=${ASSETS_PATH}"
echo "[Hoo PiPER-X Isaac] dataset=${DATASET_ROOT%/}/${DATASET_ID} target=${TARGET_EPISODES}"

if (( PREFLIGHT_ONLY )); then
    echo "[Hoo PiPER-X Isaac] PREFLIGHT OK; Isaac was not started"
    exit 0
fi

export ROBODOJO_TASK="${TASK}"
export ROBODOJO_LEROBOT_ROOT="${DATASET_ROOT}"
export ROBODOJO_LEROBOT_REPO_ID="${DATASET_ID}"
export ROBODOJO_LEROBOT_PYTHON="${LEROBOT_PYTHON}"
export ROBODOJO_EVAL_PYTHON="${ISAAC_PYTHON}"
export ROBODOJO_PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON}"
export ROBODOJO_ASSETS_PATH="${ASSETS_PATH}"
export ROBODOJO_CHECKPOINT_ID="${CHECKPOINT_ID}"
export ROBODOJO_EXPECTED_KAI0_COMMIT="${EXPECTED_KAI0_COMMIT}"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="${EXPECTED_CHECKPOINT_DIGEST}"
export ROBODOJO_POLICY_PORT="${POLICY_PORT}"
export ROBODOJO_PIPERX_SOURCE_HOST="127.0.0.1"
export ROBODOJO_PIPERX_SOURCE_PORT="${SOURCE_PORT}"
export ROBODOJO_DUAL_MIRROR_TARGET_EPISODES="${TARGET_EPISODES}"
export ROBODOJO_PIPERX_TARGET_EPISODES="${TARGET_EPISODES}"
export PIPERX_BRIDGE_ROOT="${PIPERX_CODE_ROOT}"
export ROBODOJO_PIPERX_CODE_ROOT="${PIPERX_CODE_ROOT}"

cd "${ROBODOJO_ROOT}"
exec "${SCRIPT_DIR}/run_piperx_dagger_isaac.sh"
