#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/restore_rollout_state.sh \
    --task make_toast \
    --dataset-root /path/to/lerobot/dataset \
    --episode 12 \
    (--frame 460 | --time-s 18.4) \
    [--env-gpu 0] [--env-cfg arx_x5]

The command launches one visible, policy-free Isaac environment, recreates the
episode's saved layout, writes the selected simulator snapshot back, and holds
the restored frame until the window is closed or Ctrl+C is pressed.
EOF
}

die() {
  echo "[restore_rollout_state][ERROR] $*" >&2
  exit 2
}

task=""
dataset_root=""
episode=""
frame=""
time_s=""
env_gpu="0"
env_cfg="arx_x5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task|--dataset-root|--episode|--frame|--time-s|--env-gpu|--env-cfg)
      [[ $# -ge 2 && "$2" != --* ]] || die "Missing value for $1"
      case "$1" in
        --task) task="$2" ;;
        --dataset-root) dataset_root="$2" ;;
        --episode) episode="$2" ;;
        --frame) frame="$2" ;;
        --time-s) time_s="$2" ;;
        --env-gpu) env_gpu="$2" ;;
        --env-cfg) env_cfg="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$task" ]] || die "--task is required"
[[ -n "$dataset_root" ]] || die "--dataset-root is required"
[[ -d "$dataset_root" ]] || die "dataset does not exist: $dataset_root"
[[ -n "$episode" ]] || die "--episode is required"
if [[ -n "$frame" && -n "$time_s" ]] || [[ -z "$frame" && -z "$time_s" ]]; then
  die "select exactly one of --frame or --time-s"
fi

selection_args=()
if [[ -n "$frame" ]]; then
  selection_args+=(--restore_frame "$frame")
else
  selection_args+=(--restore_time_s "$time_s")
fi

export ROBODOJO_CONTROL_MODE=state_restore
export ROBODOJO_HEADLESS=0
export EVAL_NUM=1
export ROBODOJO_MAX_BASH_RETRIES=1

exec bash "${ROOT_DIR}/scripts/eval_policy.sh" \
  --root_dir "$ROOT_DIR" \
  --task_name "$task" \
  --env_cfg_type "$env_cfg" \
  --device_id "$env_gpu" \
  --policy_name ReplayRestore \
  --port 1 \
  --additional_info replay_restore \
  --seed 0 \
  --protocol ws \
  --policy_runtime xpolicy_ws_v0 \
  --restore_dataset_root "$dataset_root" \
  --restore_episode "$episode" \
  "${selection_args[@]}"
