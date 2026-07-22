#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/collect_pi05_keyboard.sh \
    --task TASK --ckpt CHECKPOINT [options]

Required:
  --task TASK             Simulation task name
  --ckpt CHECKPOINT       Pi0.5 checkpoint directory name

Options:
  --record-dir PATH       HDF5 root (default: /home/piper/data/RoboDojo_interventions)
  --episodes NUM          Counted rollouts before normal exit (success/fail), task-capped (default: 1)
  --lerobot-repo-id ID    Also rebuild a Kai0-compatible LeRobot v3 dataset after collection
  --lerobot-root PATH     LeRobot base directory (default: /home/piper/data/lerobot)
  --lerobot-max NUM       Maximum HDF5 episodes to export (default: 1000000)
  --lerobot-vcodec NAME   CPU video codec: h264/hevc/libsvtav1 (default: h264)
  --seed NUM              Layout seed (default: 0)
  --env-cfg NAME          Robot config (default: arx_x5)
  --policy-gpu ID         Pi0.5 server GPU (default: 0)
  --env-gpu ID            Isaac Sim GPU (default: 0)
  --policy-env PATH       Pi0.5 uv environment or 'uv' (default: uv)
  --pos-step METERS       Translation per 25 Hz tick (default: 0.005)
  --rot-step RADIANS      Rotation per 25 Hz tick (default: 0.02)
  -h, --help              Show this help

The Isaac Sim window must be visible and focused for keyboard events.
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[collect_pi05_keyboard] Missing value for $1" >&2
    exit 2
  fi
}

task=""
ckpt=""
record_dir="/home/piper/data/RoboDojo_interventions"
episodes="1"
seed="0"
env_cfg="arx_x5"
policy_gpu="0"
env_gpu="0"
policy_env="uv"
pos_step="0.005"
rot_step="0.02"
lerobot_repo_id=""
lerobot_root="/home/piper/data/lerobot"
lerobot_max="1000000"
lerobot_vcodec="h264"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
    --record-dir) need_value "$@"; record_dir="$2"; shift 2 ;;
    --episodes) need_value "$@"; episodes="$2"; shift 2 ;;
    --lerobot-repo-id) need_value "$@"; lerobot_repo_id="$2"; shift 2 ;;
    --lerobot-root) need_value "$@"; lerobot_root="$2"; shift 2 ;;
    --lerobot-max) need_value "$@"; lerobot_max="$2"; shift 2 ;;
    --lerobot-vcodec) need_value "$@"; lerobot_vcodec="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
    --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --pos-step) need_value "$@"; pos_step="$2"; shift 2 ;;
    --rot-step) need_value "$@"; rot_step="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[collect_pi05_keyboard] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${task}" || -z "${ckpt}" ]]; then
  echo "[collect_pi05_keyboard] --task and --ckpt are required" >&2
  usage >&2
  exit 2
fi
if [[ "${record_dir}" != /* ]]; then
  record_dir="${ROOT_DIR}/${record_dir}"
fi
mkdir -p "${record_dir}" "${ROOT_DIR}/data"
record_dir="$(cd "${record_dir}" && pwd)"
dataset_name="$(basename "${record_dir}")"
dataset_link="${ROOT_DIR}/data/${dataset_name}"
if [[ "${dataset_link}" != "${record_dir}" ]]; then
  if [[ -e "${dataset_link}" || -L "${dataset_link}" ]]; then
    if [[ ! -L "${dataset_link}" || "$(readlink -f "${dataset_link}")" != "${record_dir}" ]]; then
      echo "[collect_pi05_keyboard] Refusing to replace existing data path: ${dataset_link}" >&2
      exit 1
    fi
  else
    ln -s "${record_dir}" "${dataset_link}"
  fi
fi

export ROBODOJO_CONTROL_MODE="keyboard_intervention"
export ROBODOJO_HEADLESS="0"
export HEADLESS="0"
export LIVESTREAM="0"
export ROBODOJO_RECORD_DIR="${record_dir}"
export ROBODOJO_REALTIME="1"
export ROBODOJO_TELEOP_POS_STEP="${pos_step}"
export ROBODOJO_TELEOP_ROT_STEP="${rot_step}"

echo "[collect_pi05_keyboard] task=${task} ckpt=${ckpt} episodes=${episodes}"
echo "[collect_pi05_keyboard] record_dir=${record_dir}"
echo "[collect_pi05_keyboard] On normal completion, Isaac Sim exits at the requested task-capped rollout count;"
echo "[collect_pi05_keyboard] R saves/retries the same layout without consuming that count."

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
  --eval-num "${episodes}"

if [[ -n "${lerobot_repo_id}" ]]; then
  lerobot_python="${ROOT_DIR}/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
  if [[ ! -x "${lerobot_python}" ]]; then
    echo "[collect_pi05_keyboard] LeRobot exporter environment not found: ${lerobot_python}" >&2
    echo "[collect_pi05_keyboard] HDF5 trajectories are safe; run Pi_05/install.sh before exporting." >&2
    exit 1
  fi
  if [[ "${lerobot_root}" != /* ]]; then
    lerobot_root="${ROOT_DIR}/${lerobot_root}"
  fi
  mkdir -p "${lerobot_root}"
  lerobot_root="$(cd "${lerobot_root}" && pwd)"
  echo "[collect_pi05_keyboard] exporting LeRobot v3 (CPU ${lerobot_vcodec})"
  echo "[collect_pi05_keyboard] lerobot_root=${lerobot_root}/${lerobot_repo_id}"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" "${lerobot_python}" \
    "${ROOT_DIR}/scripts/RoboDojo/export_interventions_lerobot_v30.py" \
    "${dataset_name}.${task}.${env_cfg}" \
    --repo-id "${lerobot_repo_id}" \
    --root "${lerobot_root}" \
    --max-episodes "${lerobot_max}" \
    --vcodec "${lerobot_vcodec}" \
    --overwrite
fi
