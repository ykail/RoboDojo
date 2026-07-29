#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LAUNCH_DIR="$(pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/eval_kai0_pi05.sh \
    --task TASK \
    --checkpoint-dir PATH \
    --checkpoint-id ID \
    [options]

Required:
  --task TASK                 RoboDojo simulation task name
  --checkpoint-dir PATH       Kai0 JAX checkpoint step directory
  --checkpoint-id ID          Stable checkpoint identity recorded by policy-v1

Kai0 policy server:
  --kai0-root PATH            Kai0 checkout (default: third_party/kai0)
  --kai0-python PATH          Kai0 Python executable
                              (default: KAI0_ROOT/.venv/bin/python)
  --policy-gpu ID             GPU visible to Kai0 (default: 0)
  --port NUM                  Local policy WebSocket port (default: 8000)
  --policy-seed NUM           Policy episode seed (default: --seed)

RoboDojo simulation:
  --eval-num NUM              Number of evaluation episodes (default: 10)
  --env-gpu ID                GPU visible to Isaac Sim (default: 0)
  --seed NUM                  RoboDojo layout-set seed (default: 0)
  --control-mode MODE         policy, keyboard_intervention,
                              keyboard_observe, or piperx_sim_dagger
                              (default: policy)
  --headless                  Disable the Isaac Sim window (GUI is the default)

Intervention recording:
  --lerobot-root PATH         Dataset parent (default: $HOME/data/lerobot)
  --lerobot-repo-id ID        Dataset name
                              (default: robodojo_interventions_TASK)
  --resume                    Append to an existing compatible dataset
  --lerobot-vcodec CODEC      h264, hevc, or libsvtav1 (default: h264)
  --encoder-threads NUM       CPU video encoder threads (default: 2)

PiPER-X simulator DAgger (does not start or enable hardware):
  --piperx-calibration PATH   Required retarget JSON with calibrated=true
  --piperx-bridge-host HOST   Must be loopback (default: 127.0.0.1)
  --piperx-bridge-port NUM    Existing LeRobot bridge port (default: 8765)
  --piperx-connect-timeout S  Initial TCP timeout (default: 5.0)
  --piperx-response-timeout S Per-request/deadline timeout (default: 0.2)
  --piperx-heartbeat-interval S
                              Session heartbeat period (default: 0.25)
  --piperx-max-ik-failures N  Consecutive rejected samples before abort
                              (default: 25)

Other:
  --dry-run                   Validate inputs and print both commands only
  -h, --help                  Show this help

The Kai0 server stays in the Kai0 submodule and uses its own Python
environment. This launcher does not use an XPolicyLab policy environment.
XLA_PYTHON_CLIENT_MEM_FRACTION defaults to 0.3 and may be overridden in the
calling environment.
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
kai0_root="${ROOT_DIR}/third_party/kai0"
kai0_python=""
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
piperx_calibration="${ROBODOJO_PIPERX_CALIBRATION:-}"
piperx_bridge_host="${ROBODOJO_PIPERX_BRIDGE_HOST:-127.0.0.1}"
piperx_bridge_port="${ROBODOJO_PIPERX_BRIDGE_PORT:-8765}"
piperx_connect_timeout="${ROBODOJO_PIPERX_CONNECT_TIMEOUT_S:-5.0}"
piperx_response_timeout="${ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S:-0.2}"
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
    --piperx-calibration)
      need_value "$@"
      piperx_calibration="$2"
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
[[ -n "${checkpoint_dir}" ]] || die "--checkpoint-dir is required"
[[ -n "${checkpoint_id}" ]] || die "--checkpoint-id is required"
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

case "${control_mode}" in
  policy|keyboard_intervention|keyboard_observe|piperx_sim_dagger) ;;
  *)
    die "--control-mode must be policy, keyboard_intervention, keyboard_observe, or piperx_sim_dagger"
    ;;
esac
if [[ "${headless}" == "1" && "${control_mode}" != "policy" ]]; then
  die "--headless cannot be combined with an interactive/observation control mode"
fi

if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  [[ -n "${piperx_calibration}" ]] || die "--piperx-calibration is required for piperx_sim_dagger"
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
    "${piperx_heartbeat_interval}"; do
    [[ "${timeout_value}" =~ ^[0-9]+([.][0-9]+)?$ && "${timeout_value}" =~ [1-9] ]] \
      || die "PiPER-X timeout/heartbeat values must be positive numbers"
  done
  if [[ "${piperx_calibration}" != /* ]]; then
    piperx_calibration="${LAUNCH_DIR}/${piperx_calibration}"
  fi
  [[ -f "${piperx_calibration}" ]] || die "PiPER-X calibration does not exist: ${piperx_calibration}"
  piperx_calibration="$(cd "$(dirname "${piperx_calibration}")" && pwd -P)/$(basename "${piperx_calibration}")"
fi

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

if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  if ! env \
    -u PYTHONHOME \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    "PYTHONPATH=${ROOT_DIR}" \
    "${kai0_python}" -c \
    'import sys; from src.eval_client.piperx_retarget import RetargetConfig; RetargetConfig.from_file(sys.argv[1])' \
    "${piperx_calibration}"; then
    die "PiPER-X calibration is incomplete or unsafe; verify the full schema and calibrated=true"
  fi
fi

if [[ "${control_mode}" == "keyboard_intervention" \
  || "${control_mode}" == "piperx_sim_dagger" ]]; then
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
      "${kai0_python}" -c 'import numpy; import lerobot' >/dev/null; then
      die "Kai0 environment cannot import numpy/lerobot: ${kai0_python}"
    fi
  fi
fi

if [[ "${checkpoint_dir}" != /* ]]; then
  checkpoint_dir="${LAUNCH_DIR}/${checkpoint_dir}"
fi
[[ -d "${checkpoint_dir}" ]] || die "Checkpoint directory does not exist: ${checkpoint_dir}"
checkpoint_dir="$(cd "${checkpoint_dir}" && pwd -P)"

eval_script="${ROOT_DIR}/scripts/eval_policy.sh"
[[ -f "${eval_script}" ]] || die "RoboDojo evaluator not found: ${eval_script}"

ready_timeout_s="${ROBODOJO_POLICY_READY_TIMEOUT_S:-600}"
[[ "${ready_timeout_s}" =~ ^[1-9][0-9]*$ ]] \
  || die "ROBODOJO_POLICY_READY_TIMEOUT_S must be a positive integer"

server_host="127.0.0.1"
policy_mem_fraction="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
policy_url="ws://${server_host}:${port}"

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
    "ROBODOJO_LEROBOT_PYTHON=${kai0_python}"
    "ROBODOJO_LEROBOT_ROOT=${lerobot_root}"
    "ROBODOJO_LEROBOT_REPO_ID=${lerobot_repo_id}"
    "ROBODOJO_LEROBOT_RESUME=${resume}"
    "ROBODOJO_LEROBOT_VCODEC=${lerobot_vcodec}"
    "ROBODOJO_LEROBOT_ENCODER_THREADS=${encoder_threads}"
    "ROBODOJO_LEROBOT_STREAMING_ENCODING=1"
    "ROBODOJO_TASK_NAME=${task}"
    "ROBODOJO_ENV_CFG=arx_x5"
    "ROBODOJO_CHECKPOINT=${checkpoint_id}"
  )
fi
if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  eval_environment+=(
    "ROBODOJO_PIPERX_CALIBRATION=${piperx_calibration}"
    "ROBODOJO_PIPERX_BRIDGE_HOST=${piperx_bridge_host}"
    "ROBODOJO_PIPERX_BRIDGE_PORT=${piperx_bridge_port}"
    "ROBODOJO_PIPERX_CONNECT_TIMEOUT_S=${piperx_connect_timeout}"
    "ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S=${piperx_response_timeout}"
    "ROBODOJO_PIPERX_HEARTBEAT_INTERVAL_S=${piperx_heartbeat_interval}"
    "ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES=${piperx_max_ik_failures}"
  )
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
)

echo "[eval_kai0_pi05] task=${task} eval_num=${eval_num} control_mode=${control_mode}"
echo "[eval_kai0_pi05] checkpoint=${checkpoint_dir} checkpoint_id=${checkpoint_id}"
echo "[eval_kai0_pi05] Kai0=${kai0_root} policy_gpu=${policy_gpu} env_gpu=${env_gpu}"
echo "[eval_kai0_pi05] policy_url=${policy_url} headless=${headless}"
if [[ "${control_mode}" == "keyboard_intervention" \
  || "${control_mode}" == "piperx_sim_dagger" ]]; then
  echo "[eval_kai0_pi05] LeRobot=${lerobot_root%/}/${lerobot_repo_id} resume=${resume}"
fi
if [[ "${control_mode}" == "piperx_sim_dagger" ]]; then
  echo "[eval_kai0_pi05] PiPER-X bridge=${piperx_bridge_host}:${piperx_bridge_port}"
  echo "[eval_kai0_pi05] PiPER-X calibration=${piperx_calibration}"
  echo "[eval_kai0_pi05] NOTE: this command never starts or enables PiPER-X hardware"
fi

if [[ "${dry_run}" == "1" ]]; then
  echo "[dry-run] Kai0 working directory: ${kai0_root}"
  print_command "Kai0 server" "${server_cmd[@]}"
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
