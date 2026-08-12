#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LAUNCH_DIR="$(pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/eval_kai0_pi05.sh \
    --task TASK \
    --checkpoint-id ID \
    [--checkpoint-dir PATH | --external-policy-server-url URL] \
    [options]

Required:
  --task TASK                 RoboDojo simulation task name
  --checkpoint-id ID          Stable checkpoint identity recorded by policy-v1

Kai0 policy server:
  --checkpoint-dir PATH       Local-mode Kai0 JAX checkpoint step directory
  --external-policy-server-url URL
                              Existing loopback policy-v1 URL, normally an SSH
                              tunnel such as ws://127.0.0.1:18080
  --expected-kai0-commit REV  Required full Kai0 commit in external mode
  --expected-checkpoint-digest DIGEST
                              Optional sha256:<64 hex> HELLO provenance check
  --kai0-root PATH            Kai0 checkout (default: third_party/kai0)
  --kai0-python PATH          Kai0 Python executable
                              (default: KAI0_ROOT/.venv/bin/python)
  --policy-gpu ID             GPU visible to Kai0 (default: 0)
  --port NUM                  Local policy WebSocket port (default: 8000)
  --policy-seed NUM           Policy episode seed (default: --seed)
  --lerobot-python PATH       Policy-client/recorder Python; required in
                              external mode

RoboDojo simulation:
  --eval-num NUM              Number of evaluation episodes (default: 10)
  --env-gpu ID                GPU visible to Isaac Sim (default: 0)
  --seed NUM                  RoboDojo layout-set seed (default: 0)
  --control-mode MODE         policy, keyboard_intervention,
                              keyboard_observe, piperx_sim_dagger,
                              piperx_policy_leader_mirror, or
                              piperx_policy_joint_intervention, or
                              x5_policy_joint_intervention
                              (default: policy)
  --headless                  Disable the Isaac Sim window (GUI is the default)

Intervention recording:
  --lerobot-root PATH         Dataset parent (default: $HOME/data/lerobot)
  --lerobot-repo-id ID        Dataset name
                              (default: robodojo_interventions_TASK)
  --resume                    Append to an existing compatible dataset
  --lerobot-vcodec CODEC      h264, hevc, or libsvtav1 (default: h264)
  --encoder-threads NUM       CPU video encoder threads (default: 2)

PiPER-X simulator DAgger (uses the separately supervised hardware owner):
  --piperx-bridge-host HOST   Must be loopback (default: 127.0.0.1)
  --piperx-bridge-port NUM    Existing LeRobot bridge port (default: 8765)
  --piperx-connect-timeout S  Initial TCP timeout (default: 5.0)
  --piperx-response-timeout S Per-request/deadline timeout (default: 0.2)
  --piperx-arm-timeout S      First supervised four-arm bring-up (default: 60.0)
  --piperx-transition-timeout S
                              Leader role transition timeout (default: 10.0)
  --piperx-heartbeat-interval S
                              Session heartbeat period (default: 0.25)
  --piperx-max-ik-failures N  Consecutive rejected samples before abort
                              (default: 25)

Other:
  --dry-run                   Validate inputs and print both commands only
  -h, --help                  Show this help

Local mode keeps the Kai0 server in the Kai0 checkout. External mode never
loads a checkpoint/JAX on this machine and never starts or stops a Kai0
process. Both modes reject a dirty server through HELLO provenance.
EOF
}

die() {
  echo "[eval_kai0_pi05][ERROR] $*" >&2
  exit 2
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    die "Missing value for $1"
  fi
}

print_command() {
  local label="$1"
  shift
  printf '[dry-run] %s:' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

task=""
checkpoint_dir=""
checkpoint_id=""
external_policy_server_url=""
expected_checkpoint_digest=""
expected_kai0_commit=""
kai0_root="${ROOT_DIR}/third_party/kai0"
kai0_python=""
lerobot_python=""
preflight_python="${ROBODOJO_PREFLIGHT_PYTHON:-}"
eval_num="10"
port="8000"
env_gpu="0"
policy_gpu="0"
seed="0"
policy_seed=""
control_mode="policy"
headless="0"
lerobot_root="${ROBODOJO_LEROBOT_ROOT:-${HOME}/data/lerobot}"
lerobot_repo_id="${ROBODOJO_LEROBOT_REPO_ID:-}"
resume="0"
lerobot_vcodec="h264"
encoder_threads="2"
piperx_record="${ROBODOJO_PIPERX_RECORD:-1}"
dual_mirror_record="${ROBODOJO_DUAL_MIRROR_RECORD:-0}"
piperx_bridge_host="${ROBODOJO_PIPERX_BRIDGE_HOST:-127.0.0.1}"
piperx_bridge_port="${ROBODOJO_PIPERX_BRIDGE_PORT:-8765}"
piperx_connect_timeout="${ROBODOJO_PIPERX_CONNECT_TIMEOUT_S:-5.0}"
piperx_response_timeout="${ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S:-0.2}"
piperx_arm_timeout="${ROBODOJO_PIPERX_ARM_TIMEOUT_S:-60.0}"
piperx_transition_timeout="${ROBODOJO_PIPERX_TRANSITION_TIMEOUT_S:-10.0}"
piperx_heartbeat_interval="${ROBODOJO_PIPERX_HEARTBEAT_INTERVAL_S:-0.25}"
piperx_max_ik_failures="${ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES:-25}"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      need_value "$@"
      task="$2"
      shift 2
      ;;
    --checkpoint-dir)
      need_value "$@"
      checkpoint_dir="$2"
      shift 2
      ;;
    --checkpoint-id)
      need_value "$@"
      checkpoint_id="$2"
      shift 2
      ;;
    --external-policy-server-url)
      need_value "$@"
      external_policy_server_url="$2"
      shift 2
      ;;
    --expected-checkpoint-digest)
      need_value "$@"
      expected_checkpoint_digest="$2"
      shift 2
      ;;
    --expected-kai0-commit)
      need_value "$@"
      expected_kai0_commit="$2"
      shift 2
      ;;
    --kai0-root)
      need_value "$@"
      kai0_root="$2"
      shift 2
      ;;
    --kai0-python)
      need_value "$@"
      kai0_python="$2"
      shift 2
      ;;
    --lerobot-python)
      need_value "$@"
      lerobot_python="$2"
      shift 2
      ;;
    --eval-num)
      need_value "$@"
      eval_num="$2"
      shift 2
      ;;
    --port)
      need_value "$@"
      port="$2"
      shift 2
      ;;
    --env-gpu)
      need_value "$@"
      env_gpu="$2"
      shift 2
      ;;
    --policy-gpu)
      need_value "$@"
      policy_gpu="$2"
      shift 2
      ;;
    --seed)
      need_value "$@"
      seed="$2"
      shift 2
      ;;
    --policy-seed)
      need_value "$@"
      policy_seed="$2"
      shift 2
      ;;
    --control-mode)
      need_value "$@"
      control_mode="$2"
      shift 2
      ;;
    --headless)
      headless="1"
      shift
      ;;
    --lerobot-root)
      need_value "$@"
      lerobot_root="$2"
      shift 2
      ;;
    --lerobot-repo-id)
      need_value "$@"
      lerobot_repo_id="$2"
      shift 2
      ;;
    --resume)
      resume="1"
      shift
      ;;
    --lerobot-vcodec)
      need_value "$@"
      lerobot_vcodec="$2"
      shift 2
      ;;
    --encoder-threads)
      need_value "$@"
      encoder_threads="$2"
      shift 2
      ;;
    --piperx-bridge-host)
      need_value "$@"
      piperx_bridge_host="$2"
      shift 2
      ;;
    --piperx-bridge-port)
      need_value "$@"
      piperx_bridge_port="$2"
      shift 2
      ;;
    --piperx-connect-timeout)
      need_value "$@"
      piperx_connect_timeout="$2"
      shift 2
      ;;
    --piperx-response-timeout)
      need_value "$@"
      piperx_response_timeout="$2"
      shift 2
      ;;
    --piperx-arm-timeout)
      need_value "$@"
      piperx_arm_timeout="$2"
      shift 2
      ;;
    --piperx-transition-timeout)
      need_value "$@"
      piperx_transition_timeout="$2"
      shift 2
      ;;
    --piperx-heartbeat-interval)
      need_value "$@"
      piperx_heartbeat_interval="$2"
      shift 2
      ;;
    --piperx-max-ik-failures)
      need_value "$@"
      piperx_max_ik_failures="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${task}" ]] || die "--task is required"
[[ -n "${checkpoint_id}" ]] || die "--checkpoint-id is required"
if [[ -z "${external_policy_server_url}" ]]; then
  [[ -n "${checkpoint_dir}" ]] || die "--checkpoint-dir is required in local policy mode"
else
  [[ -z "${checkpoint_dir}" ]] \
    || die "--checkpoint-dir cannot be combined with --external-policy-server-url"
  [[ -n "${expected_kai0_commit}" ]] \
    || die "--expected-kai0-commit is required in external policy mode"
fi
if [[ -n "${expected_checkpoint_digest}" \
  && ! "${expected_checkpoint_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  die "--expected-checkpoint-digest must be sha256:<64 lowercase hex>"
fi
if [[ -n "${expected_kai0_commit}" && ! "${expected_kai0_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  die "--expected-kai0-commit must be a full lowercase Git commit"
fi
checkpoint_id_normalized="${checkpoint_id//\\//}"
if [[ "${checkpoint_id_normalized}" == /* \
  || "${checkpoint_id_normalized}" == "~/"* \
  || "${checkpoint_id_normalized}" == [Ff][Ii][Ll][Ee]:* \
  || "${checkpoint_id_normalized}" =~ ^[[:alpha:]]: \
  || "/${checkpoint_id_normalized}/" == *"/../"* \
  || "/${checkpoint_id_normalized}/" == *"/./"* ]]; then
  die "--checkpoint-id must be a portable identity, not a local path"
fi
[[ "${eval_num}" =~ ^[1-9][0-9]*$ ]] || die "--eval-num must be a positive integer"
[[ "${port}" =~ ^[0-9]+$ ]] || die "--port must be an integer"
(( port >= 1 && port <= 65535 )) || die "--port must be in [1, 65535]"
[[ "${env_gpu}" =~ ^[0-9]+$ ]] || die "--env-gpu must be a non-negative integer"
[[ "${policy_gpu}" =~ ^[0-9]+$ ]] || die "--policy-gpu must be a non-negative integer"
[[ "${seed}" =~ ^-?[0-9]+$ ]] || die "--seed must be an integer"
[[ "${encoder_threads}" =~ ^[1-9][0-9]*$ ]] \
  || die "--encoder-threads must be a positive integer"
case "${lerobot_vcodec}" in
  h264|hevc|libsvtav1) ;;
  *) die "--lerobot-vcodec must be h264, hevc, or libsvtav1" ;;
esac

if [[ -z "${policy_seed}" ]]; then
  policy_seed="${seed}"
fi
[[ "${policy_seed}" =~ ^[0-9]+$ ]] || die "--policy-seed must be a non-negative integer"
(( policy_seed <= 4294967295 )) || die "--policy-seed must be in [0, 2^32 - 1]"

external_policy_port=""
if [[ -n "${external_policy_server_url}" ]]; then
  if [[ "${external_policy_server_url}" =~ ^ws://(127[.]0[.]0[.]1|localhost):([0-9]+)(/[^[:space:]]*)?$ ]]; then
    external_policy_port="${BASH_REMATCH[2]}"
  else
    die "--external-policy-server-url must be a loopback ws:// URL with an explicit port"
  fi
  (( external_policy_port >= 1 && external_policy_port <= 65535 )) \
    || die "external policy server port must be in [1, 65535]"
  port="${external_policy_port}"
fi

case "${control_mode}" in
  policy|keyboard_intervention|keyboard_observe|piperx_sim_dagger|piperx_policy_leader_mirror|piperx_policy_joint_intervention|x5_policy_joint_intervention) ;;
  *)
    die "unsupported --control-mode: ${control_mode}"
    ;;
esac
case "${piperx_record}" in
  1|true|TRUE|yes|YES|on|ON) piperx_record="1" ;;
  0|false|FALSE|no|NO|off|OFF) piperx_record="0" ;;
  *) die "ROBODOJO_PIPERX_RECORD must be a boolean" ;;
esac
case "${dual_mirror_record}" in
  1|true|TRUE|yes|YES|on|ON) dual_mirror_record="1" ;;
  0|false|FALSE|no|NO|off|OFF) dual_mirror_record="0" ;;
  *) die "ROBODOJO_DUAL_MIRROR_RECORD must be a boolean" ;;
esac
if [[ "${dual_mirror_record}" == "1" \
  && "${control_mode}" != "piperx_policy_joint_intervention" \
  && "${control_mode}" != "x5_policy_joint_intervention" ]]; then
  die "ROBODOJO_DUAL_MIRROR_RECORD=1 requires a dual-joint intervention mode"
fi
if [[ "${headless}" == "1" && "${control_mode}" != "policy" ]]; then
  die "--headless cannot be combined with an interactive/observation control mode"
fi

if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  case "${piperx_bridge_host}" in
    127.0.0.1|localhost|::1) ;;
    *) die "--piperx-bridge-host must be loopback" ;;
  esac
  [[ "${piperx_bridge_port}" =~ ^[0-9]+$ ]] || die "--piperx-bridge-port must be an integer"
  (( piperx_bridge_port >= 1 && piperx_bridge_port <= 65535 )) \
    || die "--piperx-bridge-port must be in [1, 65535]"
  [[ "${piperx_bridge_port}" != "${port}" ]] \
    || die "policy and PiPER-X bridge ports must be different"
  [[ "${piperx_max_ik_failures}" =~ ^[1-9][0-9]*$ ]] \
    || die "--piperx-max-ik-failures must be a positive integer"
  for timeout_value in \
    "${piperx_connect_timeout}" \
    "${piperx_response_timeout}" \
    "${piperx_arm_timeout}" \
    "${piperx_transition_timeout}" \
    "${piperx_heartbeat_interval}"; do
    [[ "${timeout_value}" =~ ^[0-9]+([.][0-9]+)?$ && "${timeout_value}" =~ [1-9] ]] \
      || die "PiPER-X timeout/heartbeat values must be positive numbers"
  done
fi

server_script=""
if [[ -z "${external_policy_server_url}" ]]; then
  if [[ "${kai0_root}" != /* ]]; then
    kai0_root="${LAUNCH_DIR}/${kai0_root}"
  fi
  [[ -d "${kai0_root}" ]] || die "Kai0 root does not exist: ${kai0_root}"
  kai0_root="$(cd "${kai0_root}" && pwd -P)"

  server_script="${kai0_root}/scripts/serve_robodojo_policy.py"
  [[ -f "${server_script}" ]] || die "Kai0 policy server not found: ${server_script}"

  if [[ -z "${kai0_python}" ]]; then
    kai0_python="${kai0_root}/.venv/bin/python"
  elif [[ "${kai0_python}" == */* ]]; then
    if [[ "${kai0_python}" != /* ]]; then
      kai0_python="${LAUNCH_DIR}/${kai0_python}"
    fi
  else
    kai0_python="$(command -v "${kai0_python}")" \
      || die "Kai0 Python is not on PATH: ${kai0_python}"
  fi
  [[ -x "${kai0_python}" ]] || die "Kai0 Python is not executable: ${kai0_python}"
  kai0_python="$(cd "$(dirname "${kai0_python}")" && pwd -P)/$(basename "${kai0_python}")"
fi

if [[ -n "${external_policy_server_url}" && -z "${lerobot_python}" ]]; then
  die "--lerobot-python is required in external policy mode"
fi
if [[ -z "${lerobot_python}" \
  && ( "${control_mode}" == "keyboard_intervention" \
    || "${control_mode}" == "piperx_sim_dagger" ) ]]; then
  lerobot_python="${kai0_python}"
fi
if [[ -n "${lerobot_python}" ]]; then
  if [[ "${lerobot_python}" == */* ]]; then
    if [[ "${lerobot_python}" != /* ]]; then
      lerobot_python="${LAUNCH_DIR}/${lerobot_python}"
    fi
  else
    lerobot_python="$(command -v "${lerobot_python}")" \
      || die "LeRobot Python is not on PATH: ${lerobot_python}"
  fi
  [[ -x "${lerobot_python}" ]] || die "LeRobot Python is not executable: ${lerobot_python}"
  lerobot_python="$(cd "$(dirname "${lerobot_python}")" && pwd -P)/$(basename "${lerobot_python}")"
fi
if [[ -n "${external_policy_server_url}" ]]; then
  if [[ -z "${preflight_python}" ]]; then
    preflight_python="$(command -v python)" \
      || die "Active RoboDojo Python is not on PATH"
  elif [[ "${preflight_python}" != /* ]]; then
    preflight_python="$(command -v "${preflight_python}")" \
      || die "Policy preflight Python is not on PATH: ${preflight_python}"
  fi
  [[ -x "${preflight_python}" ]] \
    || die "Policy preflight Python is not executable: ${preflight_python}"
  preflight_python="$(cd "$(dirname "${preflight_python}")" && pwd -P)/$(basename "${preflight_python}")"
fi

needs_lerobot_dataset="0"
if [[ "${control_mode}" == "keyboard_intervention" \
  || ( "${control_mode}" == "piperx_sim_dagger" && "${piperx_record}" == "1" ) \
  || ( "${control_mode}" == "piperx_policy_joint_intervention" \
    && "${dual_mirror_record}" == "1" ) \
  || ( "${control_mode}" == "x5_policy_joint_intervention" \
    && "${dual_mirror_record}" == "1" ) ]]; then
  needs_lerobot_dataset="1"
fi

if [[ "${needs_lerobot_dataset}" == "1" ]]; then
  if [[ -z "${lerobot_repo_id}" ]]; then
    lerobot_repo_id="robodojo_interventions_${task}"
  fi
  if [[ "${lerobot_repo_id}" == /* \
    || "${lerobot_repo_id}" == */ \
    || "${lerobot_repo_id}" == *"//"* \
    || "/${lerobot_repo_id}/" == *"/../"* \
    || "/${lerobot_repo_id}/" == *"/./"* ]]; then
    die "unsafe --lerobot-repo-id: ${lerobot_repo_id}"
  fi
  if [[ "${lerobot_root}" != /* ]]; then
    lerobot_root="${LAUNCH_DIR}/${lerobot_root}"
  fi
  dataset_path="${lerobot_root%/}/${lerobot_repo_id}"
  if [[ ( -e "${dataset_path}" || -L "${dataset_path}" ) && "${resume}" != "1" ]]; then
    die "Dataset already exists: ${dataset_path}; pass --resume to append"
  fi
  if [[ ( -e "${dataset_path}" || -L "${dataset_path}" ) \
    && "${resume}" == "1" \
    && ! -f "${dataset_path}/meta/info.json" ]]; then
    die "Existing path is not a LeRobot v3 dataset: ${dataset_path}"
  fi
  if [[ "${dry_run}" != "1" ]]; then
    mkdir -p "${lerobot_root}"
    lerobot_root="$(cd "${lerobot_root}" && pwd -P)"
    if ! env \
      -u PYTHONHOME \
      -u VIRTUAL_ENV \
      -u CONDA_PREFIX \
      -u CONDA_DEFAULT_ENV \
      CUDA_VISIBLE_DEVICES="" \
      PYTHONNOUSERSITE="1" \
      "${lerobot_python}" -c \
        'import numpy; from lerobot.datasets.lerobot_dataset import LeRobotDataset' \
        >/dev/null; then
      die "Recorder environment cannot import LeRobotDataset and its dependencies: ${lerobot_python}"
    fi
  fi
fi

if [[ -z "${external_policy_server_url}" ]]; then
  if [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${LAUNCH_DIR}/${checkpoint_dir}"
  fi
  [[ -d "${checkpoint_dir}" ]] || die "Checkpoint directory does not exist: ${checkpoint_dir}"
  checkpoint_dir="$(cd "${checkpoint_dir}" && pwd -P)"
fi

eval_script="${ROOT_DIR}/scripts/eval_policy.sh"
[[ -f "${eval_script}" ]] || die "RoboDojo evaluator not found: ${eval_script}"

ready_timeout_s="${ROBODOJO_POLICY_READY_TIMEOUT_S:-600}"
[[ "${ready_timeout_s}" =~ ^[1-9][0-9]*$ ]] \
  || die "ROBODOJO_POLICY_READY_TIMEOUT_S must be a positive integer"

server_host="127.0.0.1"
policy_mem_fraction="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
policy_url="${external_policy_server_url:-ws://${server_host}:${port}}"

server_cmd=()
if [[ -z "${external_policy_server_url}" ]]; then
  server_cmd=(
    env
    -u PYTHONHOME
    -u VIRTUAL_ENV
    -u CONDA_PREFIX
    -u CONDA_DEFAULT_ENV
    "CUDA_VISIBLE_DEVICES=${policy_gpu}"
    "XLA_PYTHON_CLIENT_MEM_FRACTION=${policy_mem_fraction}"
    "PYTHONNOUSERSITE=1"
    "PYTHONPATH=${kai0_root}/src"
    "${kai0_python}"
    "${server_script}"
    --checkpoint-dir "${checkpoint_dir}"
    --checkpoint-id "${checkpoint_id}"
    --host "${server_host}"
    --port "${port}"
  )
fi

eval_environment=(
  "EVAL_NUM=${eval_num}"
  "ROBODOJO_CONTROL_MODE=${control_mode}"
  "ROBODOJO_HEADLESS=${headless}"
  "HEADLESS=${headless}"
  "LIVESTREAM=0"
)
if [[ "${control_mode}" == "keyboard_intervention" \
  || "${control_mode}" == "piperx_sim_dagger" ]]; then
  eval_environment+=(
    "ROBODOJO_OPERATOR_DRIVEN=1"
    "ROBODOJO_REALTIME=1"
    "ROBODOJO_HIDE_ISAACLAB_WINDOW=1"
    "ROBODOJO_TASK_NAME=${task}"
    "ROBODOJO_ENV_CFG=arx_x5"
    "ROBODOJO_CHECKPOINT=${checkpoint_id}"
  )
fi
if [[ ( "${control_mode}" == "piperx_policy_joint_intervention" \
    || "${control_mode}" == "x5_policy_joint_intervention" ) \
  && "${dual_mirror_record}" == "1" ]]; then
  eval_environment+=(
    "ROBODOJO_TASK_NAME=${task}"
    "ROBODOJO_ENV_CFG=arx_x5"
    "ROBODOJO_CHECKPOINT=${checkpoint_id}"
  )
fi
if [[ "${control_mode}" == "piperx_policy_joint_intervention" \
  || "${control_mode}" == "x5_policy_joint_intervention" ]]; then
  eval_environment+=(
    "ROBODOJO_DUAL_MIRROR_RECORD=${dual_mirror_record}"
    "ROBODOJO_DUAL_MIRROR_PROTOCOL=robodojo_dual_joint_mirror_v1"
  )
  if [[ "${control_mode}" == "x5_policy_joint_intervention" ]]; then
    eval_environment+=(
      "ROBODOJO_DUAL_MIRROR_PROFILE=arx_x5_identity_joint_v1"
    )
  else
    eval_environment+=(
      "ROBODOJO_DUAL_MIRROR_PROFILE=arx_x5_piperx_relative_joint_v1"
    )
  fi
fi
if [[ "${needs_lerobot_dataset}" == "1" ]]; then
  eval_environment+=(
    "ROBODOJO_LEROBOT_PYTHON=${lerobot_python}"
    "ROBODOJO_LEROBOT_ROOT=${lerobot_root}"
    "ROBODOJO_LEROBOT_REPO_ID=${lerobot_repo_id}"
    "ROBODOJO_LEROBOT_RESUME=${resume}"
    "ROBODOJO_LEROBOT_VCODEC=${lerobot_vcodec}"
    "ROBODOJO_LEROBOT_ENCODER_THREADS=${encoder_threads}"
    "ROBODOJO_LEROBOT_STREAMING_ENCODING=1"
  )
fi
if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  eval_environment+=(
    "ROBODOJO_PIPERX_RECORD=${piperx_record}"
    "ROBODOJO_PIPERX_BRIDGE_HOST=${piperx_bridge_host}"
    "ROBODOJO_PIPERX_BRIDGE_PORT=${piperx_bridge_port}"
    "ROBODOJO_PIPERX_CONNECT_TIMEOUT_S=${piperx_connect_timeout}"
    "ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S=${piperx_response_timeout}"
    "ROBODOJO_PIPERX_ARM_TIMEOUT_S=${piperx_arm_timeout}"
    "ROBODOJO_PIPERX_TRANSITION_TIMEOUT_S=${piperx_transition_timeout}"
    "ROBODOJO_PIPERX_HEARTBEAT_INTERVAL_S=${piperx_heartbeat_interval}"
    "ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES=${piperx_max_ik_failures}"
  )
fi

provenance_args=(
  --expected_policy_checkpoint_id "${checkpoint_id}"
  --require_policy_clean
)
if [[ -n "${expected_checkpoint_digest}" ]]; then
  provenance_args+=(--expected_policy_checkpoint_digest "${expected_checkpoint_digest}")
fi
if [[ -n "${expected_kai0_commit}" ]]; then
  provenance_args+=(--expected_policy_code_revision "${expected_kai0_commit}")
fi

preflight_cmd=()
if [[ -n "${external_policy_server_url}" ]]; then
  preflight_script="${ROOT_DIR}/scripts/RoboDojo/preflight_policy_v1.py"
  [[ -f "${preflight_script}" ]] || die "Policy-v1 preflight script not found: ${preflight_script}"
  preflight_cmd=(
    env
    -u PYTHONHOME
    -u VIRTUAL_ENV
    -u CONDA_PREFIX
    -u CONDA_DEFAULT_ENV
    "CUDA_VISIBLE_DEVICES="
    "PYTHONNOUSERSITE=1"
    "PYTHONPATH=${ROOT_DIR}"
    "${preflight_python}"
    "${preflight_script}"
    --url "${policy_url}"
    --expected-checkpoint-id "${checkpoint_id}"
    --expected-code-revision "${expected_kai0_commit}"
    --require-clean
  )
  if [[ -n "${expected_checkpoint_digest}" ]]; then
    preflight_cmd+=(--expected-checkpoint-digest "${expected_checkpoint_digest}")
  fi
fi

eval_cmd=(
  env
  "${eval_environment[@]}"
  bash "${eval_script}"
  --root_dir "${ROOT_DIR}"
  --task_name "${task}"
  --env_cfg_type arx_x5
  --device_id "${env_gpu}"
  --policy_name Kai0_Pi05
  --port "${port}"
  --additional_info kai0_pi05_strict_v1
  --seed "${seed}"
  --host "${server_host}"
  --protocol ws
  --policy_server_url "${policy_url}"
  --policy_runtime robodojo_policy_v1
  --policy_seed "${policy_seed}"
  --action_type joint
  "${provenance_args[@]}"
)

echo "[eval_kai0_pi05] task=${task} eval_num=${eval_num} control_mode=${control_mode}"
if [[ -n "${external_policy_server_url}" ]]; then
  echo "[eval_kai0_pi05] policy_mode=external checkpoint_id=${checkpoint_id}"
  echo "[eval_kai0_pi05] expected_kai0_commit=${expected_kai0_commit} clean=required"
else
  echo "[eval_kai0_pi05] checkpoint=${checkpoint_dir} checkpoint_id=${checkpoint_id}"
  echo "[eval_kai0_pi05] Kai0=${kai0_root} policy_gpu=${policy_gpu} env_gpu=${env_gpu}"
fi
if [[ -n "${expected_checkpoint_digest}" ]]; then
  echo "[eval_kai0_pi05] expected_checkpoint_digest=${expected_checkpoint_digest}"
fi
echo "[eval_kai0_pi05] policy_url=${policy_url} headless=${headless}"
if [[ "${needs_lerobot_dataset}" == "1" ]]; then
  echo "[eval_kai0_pi05] LeRobot=${lerobot_root%/}/${lerobot_repo_id} resume=${resume}"
elif [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  echo "[eval_kai0_pi05] LeRobot recording disabled"
fi
if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  echo "[eval_kai0_pi05] PiPER-X bridge=${piperx_bridge_host}:${piperx_bridge_port}"
  echo "[eval_kai0_pi05] PiPER-X profile=arx_x5_piperx_relative_joint_v1 (automatic relative anchors)"
  echo "[eval_kai0_pi05] NOTE: the separate supervised bridge owns CAN; the first episode authorizes its arm stage"
fi

if [[ "${dry_run}" == "1" ]]; then
  if [[ -z "${external_policy_server_url}" ]]; then
    echo "[dry-run] Kai0 working directory: ${kai0_root}"
    print_command "Kai0 server" "${server_cmd[@]}"
  else
    echo "[dry-run] external policy server is never started or stopped by this launcher"
    print_command "Policy HELLO preflight" "${preflight_cmd[@]}"
  fi
  print_command "RoboDojo evaluator" "${eval_cmd[@]}"
  exit 0
fi

tcp_endpoint_is_open() {
  local endpoint_host="$1"
  local endpoint_port="$2"
  (exec 3<>"/dev/tcp/${endpoint_host}/${endpoint_port}") >/dev/null 2>&1
}

tcp_is_open() {
  tcp_endpoint_is_open "${server_host}" "${port}"
}

server_is_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl --fail --silent --show-error --max-time 1 \
      "http://${server_host}:${port}/healthz" >/dev/null 2>&1
    return
  fi
  tcp_is_open
}

if [[ -n "${external_policy_server_url}" ]]; then
  if ! tcp_is_open; then
    die "external policy tunnel is not reachable at ${server_host}:${port}"
  fi
  echo "[eval_kai0_pi05] external tunnel is reachable; verifying HELLO before Isaac startup"
  "${preflight_cmd[@]}"
  echo "[eval_kai0_pi05] HELLO provenance accepted; starting RoboDojo"
  exec "${eval_cmd[@]}"
fi

if tcp_is_open; then
  die "TCP port ${server_host}:${port} is already in use"
fi

server_pid=""

cleanup() {
  local status=$?
  local attempt
  trap - EXIT INT TERM

  if [[ -n "${server_pid}" ]]; then
    if kill -0 "${server_pid}" 2>/dev/null; then
      echo "[eval_kai0_pi05] stopping Kai0 server pid=${server_pid}"
      kill -TERM "${server_pid}" 2>/dev/null || true
      for ((attempt = 0; attempt < 50; attempt++)); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
      if kill -0 "${server_pid}" 2>/dev/null; then
        echo "[eval_kai0_pi05] Kai0 server did not stop; killing pid=${server_pid}" >&2
        kill -KILL "${server_pid}" 2>/dev/null || true
      fi
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi

  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[eval_kai0_pi05] starting Kai0 server on ${policy_url}"
(
  cd "${kai0_root}"
  exec "${server_cmd[@]}"
) &
server_pid=$!

echo "[eval_kai0_pi05] waiting up to ${ready_timeout_s}s for TCP readiness (pid=${server_pid})"
ready="0"
start_seconds="${SECONDS}"
while (( SECONDS - start_seconds < ready_timeout_s )); do
  if server_is_ready; then
    ready="1"
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    if wait "${server_pid}"; then
      server_rc="0"
    else
      server_rc="$?"
    fi
    server_pid=""
    echo "[eval_kai0_pi05][ERROR] Kai0 server exited before readiness (rc=${server_rc})" >&2
    exit 1
  fi
  sleep 0.25
done

if [[ "${ready}" != "1" ]]; then
  echo "[eval_kai0_pi05][ERROR] Timed out waiting for ${policy_url}" >&2
  exit 1
fi

echo "[eval_kai0_pi05] Kai0 server is ready; starting RoboDojo"
"${eval_cmd[@]}"
