#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/observe_pi05_keyboard.sh \
    --task TASK --ckpt CHECKPOINT [options]

Required:
  --task TASK                 Simulation task name
  --ckpt CHECKPOINT           Pi0.5 checkpoint name or absolute step directory

Options:
  --layouts NUM               Number of layouts to inspect (default: 10)
  --seed NUM                  Saved-layout set seed (default: 0)
  --env-cfg NAME              Robot config (default: arx_x5)
  --policy-gpu ID             Pi0.5 server GPU (default: 0)
  --env-gpu ID                Isaac Sim GPU (default: 0)
  --policy-env PATH           Pi0.5 uv environment or 'uv' (default: uv)
  --rendering-mode MODE       quality/balanced/performance (default: quality)
  -h, --help                  Show this help

The wrapper starts one visible Isaac Sim environment and Pi0.5 policy server.
LEFT ARROW counts the current layout and immediately advances to the next one.
After NUM layouts the program exits automatically. ESCAPE or BACKSPACE exits
early. No LeRobot data, benchmark result, or evaluation video is saved.
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[observe_pi05_keyboard] Missing value for $1" >&2
    exit 2
  fi
}

task=""
ckpt=""
layouts="10"
seed="0"
env_cfg="arx_x5"
policy_gpu="0"
env_gpu="0"
policy_env="uv"
rendering_mode="quality"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
    --layouts|--episodes) need_value "$@"; layouts="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
    --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --rendering-mode) need_value "$@"; rendering_mode="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[observe_pi05_keyboard] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${task}" || -z "${ckpt}" ]]; then
  echo "[observe_pi05_keyboard] --task and --ckpt are required" >&2
  usage >&2
  exit 2
fi
if [[ ! "${layouts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[observe_pi05_keyboard] --layouts must be a positive integer" >&2
  exit 2
fi
if [[ "${env_cfg}" != "arx_x5" ]]; then
  echo "[observe_pi05_keyboard] keyboard observation currently supports only --env-cfg arx_x5" >&2
  exit 2
fi
case "${rendering_mode}" in
  quality|balanced|performance) ;;
  *)
    echo "[observe_pi05_keyboard] --rendering-mode must be quality, balanced, or performance" >&2
    exit 2
    ;;
esac

export ROBODOJO_CONTROL_MODE="keyboard_observe"
export ROBODOJO_HEADLESS="0"
export HEADLESS="0"
export LIVESTREAM="0"
export ROBODOJO_REALTIME="1"
export ROBODOJO_RENDERING_MODE="${rendering_mode}"
export ROBODOJO_HIDE_ISAACLAB_WINDOW="1"

echo "[observe_pi05_keyboard] task=${task} ckpt=${ckpt} layouts=${layouts} seed=${seed}"
echo "[observe_pi05_keyboard] LEFT ARROW=next layout ESCAPE/BACKSPACE=exit; nothing is recorded"

bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
  --policy-dir "${ROOT_DIR}/XPolicyLab/policy/Pi_05" \
  --task "${task}" \
  --ckpt "${ckpt}" \
  --env-cfg "${env_cfg}" \
  --action-type joint \
  --seed "${seed}" \
  --policy-gpu "${policy_gpu}" \
  --env-gpu "${env_gpu}" \
  --policy-env "${policy_env}" \
  --eval-env RoboDojo \
  --eval-num "${layouts}"
