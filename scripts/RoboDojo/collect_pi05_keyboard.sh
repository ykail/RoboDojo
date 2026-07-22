#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/collect_pi05_keyboard.sh \
    --task TASK --ckpt CHECKPOINT [options]

Required:
  --task TASK                 Simulation task name
  --ckpt CHECKPOINT           Pi0.5 checkpoint directory name

LeRobot v3 output:
  --lerobot-repo-id ID        Dataset id (default: robodojo_interventions_TASK)
  --lerobot-root PATH         Dataset base directory (default: $HOME/data/lerobot)
  --resume                    Append after the last finalized episode in an existing dataset
  --lerobot-vcodec NAME       CPU video codec (default: h264)
  --encoder-threads NUM       CPU streaming-video encoder threads (default: 16)

Simulation / policy:
  --seed NUM                  Layout set seed (default: 0)
  --env-cfg NAME              Robot config (currently only arx_x5; default: arx_x5)
  --policy-gpu ID             Pi0.5 server GPU (default: 0)
  --env-gpu ID                Isaac Sim GPU (default: 0)
  --policy-env PATH           Pi0.5 uv environment or 'uv' (default: uv)
  --pos-step METERS           Translation per 25 Hz tick (default: 0.005)
  --rot-step RADIANS          Rotation per 25 Hz tick (default: 0.02)
  --rendering-mode MODE       quality/balanced/performance (default: quality)
  -h, --help                  Show this help

Compatibility options:
  --episodes NUM              Accepted but ignored; only ESCAPE/BACKSPACE exits
  --record-dir PATH           Deprecated alias for --lerobot-root (no HDF5 is written)

The wrapper starts both the Pi0.5 policy server and visible Isaac Sim.  Keep
the Isaac Sim window focused.  Candidate frames are staged from reset onward:
RIGHT accepts and advances, LEFT discards and retries, ESCAPE accepts and exits,
and BACKSPACE discards and exits.
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
seed="0"
env_cfg="arx_x5"
policy_gpu="0"
env_gpu="0"
policy_env="uv"
pos_step="0.005"
rot_step="0.02"
rendering_mode="quality"
lerobot_repo_id=""
lerobot_root="${HOME:?HOME must be set}/data/lerobot"
lerobot_vcodec="h264"
encoder_threads="16"
resume="0"
deprecated_episodes=""
deprecated_record_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
    --lerobot-repo-id) need_value "$@"; lerobot_repo_id="$2"; shift 2 ;;
    --lerobot-root) need_value "$@"; lerobot_root="$2"; shift 2 ;;
    --resume) resume="1"; shift ;;
    --lerobot-vcodec) need_value "$@"; lerobot_vcodec="$2"; shift 2 ;;
    --encoder-threads) need_value "$@"; encoder_threads="$2"; shift 2 ;;
    --episodes) need_value "$@"; deprecated_episodes="$2"; shift 2 ;;
    --record-dir)
      need_value "$@"
      deprecated_record_dir="$2"
      lerobot_root="$2"
      shift 2
      ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
    --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --pos-step) need_value "$@"; pos_step="$2"; shift 2 ;;
    --rot-step) need_value "$@"; rot_step="$2"; shift 2 ;;
    --rendering-mode) need_value "$@"; rendering_mode="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[collect_pi05_keyboard] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${task}" || -z "${ckpt}" ]]; then
  echo "[collect_pi05_keyboard] --task and --ckpt are required" >&2
  usage >&2
  exit 2
fi
case "${rendering_mode}" in
  quality|balanced|performance) ;;
  *)
    echo "[collect_pi05_keyboard] --rendering-mode must be quality, balanced, or performance" >&2
    exit 2
    ;;
esac
if [[ "${env_cfg}" != "arx_x5" ]]; then
  echo "[collect_pi05_keyboard] direct intervention collection currently supports only --env-cfg arx_x5" >&2
  exit 2
fi
if [[ ! "${encoder_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[collect_pi05_keyboard] --encoder-threads must be a positive integer" >&2
  exit 2
fi
case "${lerobot_vcodec}" in
  h264|hevc|libsvtav1) ;;
  *)
    echo "[collect_pi05_keyboard] --lerobot-vcodec must be h264, hevc, or libsvtav1" >&2
    exit 2
    ;;
esac

if [[ -z "${lerobot_repo_id}" ]]; then
  lerobot_repo_id="robodojo_interventions_${task}"
fi
if [[ "${lerobot_repo_id}" == /* \
  || "${lerobot_repo_id}" == */ \
  || "${lerobot_repo_id}" == *"//"* \
  || "/${lerobot_repo_id}/" == *"/../"* \
  || "/${lerobot_repo_id}/" == *"/./"* ]]; then
  echo "[collect_pi05_keyboard] unsafe --lerobot-repo-id: ${lerobot_repo_id}" >&2
  exit 2
fi
if [[ "${lerobot_root}" != /* ]]; then
  lerobot_root="${ROOT_DIR}/${lerobot_root}"
fi
mkdir -p "${lerobot_root}"
lerobot_root="$(cd "${lerobot_root}" && pwd)"
dataset_path="${lerobot_root}/${lerobot_repo_id}"

if [[ ( -e "${dataset_path}" || -L "${dataset_path}" ) && "${resume}" != "1" ]]; then
  echo "[collect_pi05_keyboard] Dataset already exists: ${dataset_path}" >&2
  echo "[collect_pi05_keyboard] Pass --resume to append; existing data is never overwritten." >&2
  exit 1
fi
if [[ ( -e "${dataset_path}" || -L "${dataset_path}" ) \
  && "${resume}" == "1" \
  && ! -f "${dataset_path}/meta/info.json" ]]; then
  echo "[collect_pi05_keyboard] Existing path is not a LeRobot v3 dataset: ${dataset_path}" >&2
  echo "[collect_pi05_keyboard] Missing: ${dataset_path}/meta/info.json" >&2
  exit 1
fi

lerobot_python="${ROOT_DIR}/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
if [[ ! -x "${lerobot_python}" ]]; then
  echo "[collect_pi05_keyboard] Pi0.5/LeRobot environment not found: ${lerobot_python}" >&2
  echo "[collect_pi05_keyboard] Run: bash ${ROOT_DIR}/XPolicyLab/policy/Pi_05/install.sh" >&2
  exit 1
fi

if [[ -n "${deprecated_episodes}" ]]; then
  echo "[collect_pi05_keyboard] NOTE: --episodes=${deprecated_episodes} is ignored; use ESCAPE/BACKSPACE to exit."
fi
if [[ -n "${deprecated_record_dir}" ]]; then
  echo "[collect_pi05_keyboard] NOTE: --record-dir is now a --lerobot-root alias; HDF5 output was removed."
fi

export ROBODOJO_CONTROL_MODE="keyboard_intervention"
export ROBODOJO_OPERATOR_DRIVEN="1"
export ROBODOJO_HEADLESS="0"
export HEADLESS="0"
export LIVESTREAM="0"
export ROBODOJO_REALTIME="1"
export ROBODOJO_TELEOP_POS_STEP="${pos_step}"
export ROBODOJO_TELEOP_ROT_STEP="${rot_step}"
export ROBODOJO_RENDERING_MODE="${rendering_mode}"
export ROBODOJO_HIDE_ISAACLAB_WINDOW="1"

# The simulator process stays in the RoboDojo Conda environment.  It launches
# a CPU-only child with this separate Pi/LeRobot interpreter, avoiding imports
# of LeRobot/Torch into Isaac Sim and avoiding any extra GPU allocation.
export ROBODOJO_LEROBOT_PYTHON="${lerobot_python}"
export ROBODOJO_LEROBOT_ROOT="${lerobot_root}"
export ROBODOJO_LEROBOT_REPO_ID="${lerobot_repo_id}"
export ROBODOJO_LEROBOT_RESUME="${resume}"
export ROBODOJO_LEROBOT_VCODEC="${lerobot_vcodec}"
export ROBODOJO_LEROBOT_ENCODER_THREADS="${encoder_threads}"
export ROBODOJO_LEROBOT_STREAMING_ENCODING="1"
export ROBODOJO_TASK_NAME="${task}"
export ROBODOJO_ENV_CFG="${env_cfg}"
export ROBODOJO_CHECKPOINT="${ckpt}"
unset EVAL_NUM

echo "[collect_pi05_keyboard] task=${task} ckpt=${ckpt} layout_seed=${seed}"
echo "[collect_pi05_keyboard] dataset=${dataset_path} resume=${resume}"
echo "[collect_pi05_keyboard] LeRobot video encoder: CPU ${lerobot_vcodec}, threads=${encoder_threads}"
echo "[collect_pi05_keyboard] No natural-success or step timeout; only operator decisions end an attempt."
echo "[collect_pi05_keyboard] RIGHT=accept/next LEFT=discard/retry ESCAPE=accept/exit BACKSPACE=discard/exit"

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
  --eval-env RoboDojo
