#!/usr/bin/env bash

set -euo pipefail

STRICT_KAI0_ROOT="${ROBODOJO_KAI0_STRICT_ROOT:-/home/hoo/RoboDojo/third_party/kai0}"
OUTPUT_ENGINE_KAI0_ROOT="${ROBODOJO_KAI0_OUTPUT_ENGINE_ROOT:-/home/hoo/kai0-output-engine-policy-v1}"
KAI0_PYTHON="${ROBODOJO_KAI0_PYTHON:-${STRICT_KAI0_ROOT}/.venv/bin/python}"
STRICT_KAI0_COMMIT="ecc1a7451c3156b1e5f7533851dbb0222896206f"
OUTPUT_ENGINE_KAI0_COMMIT="76d26714c276c9a4812066854d248111382fe591"

POLICY_DIR=""
TASK=""
CHECKPOINT_ID=""
CHECKPOINT_STEP=""
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
POLICY_GPU="${ROBODOJO_POLICY_GPU:-0}"
POLICY_HOST="127.0.0.1"
POLICY_RUN_DIR=""
KAI0_MODE="auto"
DRY_RUN=0
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_POLICY_GPU_MB:-14000}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_hoo_policy.sh \
    --policy-dir PATH \
    --task TASK \
    [--port 18081]

Required:
  --policy-dir PATH          Checkpoint directory containing params/ and assets/.
  --task TASK                RoboDojo task, for example fill_pen_holder or make_toast.

Optional:
  --checkpoint-id ID         Default: TASK/POLICY_DIR_BASENAME.
  --checkpoint-step N        Inferred from names such as 59999 or 9999_my.
  --port N                   Loopback WebSocket port (default: 18081).
  --gpu N                    CUDA device on Hoo (default: 0).
  --run-dir PATH             Policy runtime/provenance directory.
  --kai0-mode MODE           auto, strict, or output-engine (default: auto).
  --dry-run                  Resolve and validate without starting the server.
  -h, --help                 Show this help.

Both --option value and --option=value forms are accepted.
EOF
}

die() {
    echo "[Hoo policy][ERROR] $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-dir)
            need_value "$1" "${2:-}"
            POLICY_DIR="$2"
            shift 2
            ;;
        --policy-dir=*) POLICY_DIR="${1#*=}"; shift ;;
        --task)
            need_value "$1" "${2:-}"
            TASK="$2"
            shift 2
            ;;
        --task=*) TASK="${1#*=}"; shift ;;
        --checkpoint-id)
            need_value "$1" "${2:-}"
            CHECKPOINT_ID="$2"
            shift 2
            ;;
        --checkpoint-id=*) CHECKPOINT_ID="${1#*=}"; shift ;;
        --checkpoint-step)
            need_value "$1" "${2:-}"
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --checkpoint-step=*) CHECKPOINT_STEP="${1#*=}"; shift ;;
        --port|--policy-port)
            need_value "$1" "${2:-}"
            POLICY_PORT="$2"
            shift 2
            ;;
        --port=*|--policy-port=*) POLICY_PORT="${1#*=}"; shift ;;
        --gpu)
            need_value "$1" "${2:-}"
            POLICY_GPU="$2"
            shift 2
            ;;
        --gpu=*) POLICY_GPU="${1#*=}"; shift ;;
        --run-dir)
            need_value "$1" "${2:-}"
            POLICY_RUN_DIR="$2"
            shift 2
            ;;
        --run-dir=*) POLICY_RUN_DIR="${1#*=}"; shift ;;
        --kai0-mode)
            need_value "$1" "${2:-}"
            KAI0_MODE="$2"
            shift 2
            ;;
        --kai0-mode=*) KAI0_MODE="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ -n "${POLICY_DIR}" ]] || die "--policy-dir is required"
[[ -n "${TASK}" ]] || die "--task is required"
[[ -n "${POLICY_DIR}" ]] || die "--policy-dir cannot be empty"
[[ -n "${TASK}" ]] || die "--task cannot be empty"
[[ "${TASK}" =~ ^[a-z0-9_]+$ ]] \
    || die "task must contain only lowercase letters, digits, and underscores"
[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || die "port must be an integer"
(( POLICY_PORT >= 1 && POLICY_PORT <= 65535 )) || die "port must be in [1, 65535]"
[[ "${POLICY_GPU}" =~ ^[0-9]+$ ]] || die "gpu must be a non-negative integer"
[[ "${KAI0_MODE}" == "auto" || "${KAI0_MODE}" == "strict" || "${KAI0_MODE}" == "output-engine" ]] \
    || die "--kai0-mode must be auto, strict, or output-engine"

if [[ "${POLICY_DIR}" == '~/'* ]]; then
    POLICY_DIR="${HOME}/${POLICY_DIR#\~/}"
fi
POLICY_DIR="${POLICY_DIR%/}"
POLICY_NAME="$(basename -- "${POLICY_DIR}")"
[[ "${POLICY_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "policy directory basename is not a safe identifier: ${POLICY_NAME}"

if [[ -z "${CHECKPOINT_ID}" ]]; then
    CHECKPOINT_ID="${TASK}/${POLICY_NAME}"
fi
[[ "${CHECKPOINT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] \
    || die "checkpoint ID contains unsupported characters: ${CHECKPOINT_ID}"

if [[ -z "${CHECKPOINT_STEP}" ]]; then
    if [[ "${POLICY_NAME}" =~ ^([0-9]+)([_-].+)?$ ]]; then
        CHECKPOINT_STEP="${BASH_REMATCH[1]}"
    elif [[ "${POLICY_NAME}" =~ [_-]([0-9]+)$ ]]; then
        CHECKPOINT_STEP="${BASH_REMATCH[1]}"
    else
        die "cannot infer checkpoint step from ${POLICY_NAME}; pass --checkpoint-step"
    fi
fi
[[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]] || die "checkpoint step must be a non-negative integer"

METADATA_PATH="${POLICY_DIR}/params/_METADATA"
NORM_STATS_PATH="${POLICY_DIR}/assets/arx_x5_sim/norm_stats.json"
[[ -d "${POLICY_DIR}/params" ]] || die "checkpoint params are missing: ${POLICY_DIR}/params"
[[ -f "${METADATA_PATH}" ]] || die "checkpoint metadata is missing: ${METADATA_PATH}"
[[ -f "${NORM_STATS_PATH}" ]] || die "checkpoint norm stats are missing: ${NORM_STATS_PATH}"
[[ -x "${KAI0_PYTHON}" ]] || die "Kai0 Python is not executable: ${KAI0_PYTHON}"

DETECTED_SCHEMA="$("${KAI0_PYTHON}" -c '
import ast
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
raw_keys = metadata.get("tree_metadata", {})
if not isinstance(raw_keys, dict):
    raise SystemExit("tree_metadata is not an object")
keys = set()
for raw in raw_keys:
    try:
        key = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        continue
    if isinstance(key, tuple):
        keys.add("/".join(str(part) for part in key))

strict = {
    f"params/{name}/{field}/value"
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    for field in ("bias", "kernel")
}
output_engine = {
    f"params/output_engine/{name}/{field}/value"
    for name in (
        "flow_decoders/action",
        "flow_input_projs/action",
        "time_mlp_in",
        "time_mlp_out",
    )
    for field in ("bias", "kernel")
}
strict_seen = bool(strict & keys)
output_seen = bool(output_engine & keys)
if strict_seen and not strict <= keys:
    raise SystemExit("incomplete strict action-head parameters")
if output_seen and not output_engine <= keys:
    raise SystemExit("incomplete output-engine action-head parameters")
if (strict <= keys) == (output_engine <= keys):
    raise SystemExit("checkpoint schema is ambiguous or unsupported")
print("strict" if strict <= keys else "output-engine")
' "${METADATA_PATH}")" || die "could not resolve checkpoint schema from ${METADATA_PATH}"

if [[ "${KAI0_MODE}" != "auto" && "${KAI0_MODE}" != "${DETECTED_SCHEMA}" ]]; then
    die "requested Kai0 mode ${KAI0_MODE}, but checkpoint schema is ${DETECTED_SCHEMA}"
fi

case "${DETECTED_SCHEMA}" in
    strict)
        KAI0_ROOT="${STRICT_KAI0_ROOT}"
        EXPECTED_KAI0_COMMIT="${STRICT_KAI0_COMMIT}"
        ;;
    output-engine)
        KAI0_ROOT="${OUTPUT_ENGINE_KAI0_ROOT}"
        EXPECTED_KAI0_COMMIT="${OUTPUT_ENGINE_KAI0_COMMIT}"
        ;;
    *) die "internal schema routing error: ${DETECTED_SCHEMA}" ;;
esac

SERVER_SCRIPT="${KAI0_ROOT}/scripts/serve_robodojo_policy.py"
[[ -d "${KAI0_ROOT}/.git" || -f "${KAI0_ROOT}/.git" ]] \
    || die "Kai0 checkout is missing: ${KAI0_ROOT}"
[[ -f "${SERVER_SCRIPT}" ]] || die "policy-v1 server is missing: ${SERVER_SCRIPT}"
[[ -f "${STRICT_KAI0_ROOT}/uv.lock" && -f "${KAI0_ROOT}/uv.lock" ]] \
    || die "Kai0 uv.lock is missing"
cmp -s "${STRICT_KAI0_ROOT}/uv.lock" "${KAI0_ROOT}/uv.lock" \
    || die "selected Kai0 checkout does not match the shared Python environment lock"

ACTUAL_KAI0_COMMIT="$(git -C "${KAI0_ROOT}" rev-parse --verify HEAD)"
[[ "${ACTUAL_KAI0_COMMIT}" == "${EXPECTED_KAI0_COMMIT}" ]] \
    || die "Kai0 commit mismatch: expected ${EXPECTED_KAI0_COMMIT}, got ${ACTUAL_KAI0_COMMIT}"
[[ -z "$(git -C "${KAI0_ROOT}" status --porcelain=v1 --untracked-files=normal)" ]] \
    || die "Kai0 worktree is dirty; refusing provenance launch"

if [[ -z "${POLICY_RUN_DIR}" ]]; then
    POLICY_RUN_DIR="${HOME}/.local/state/kai0/robodojo_policy_v1/${TASK}_${POLICY_NAME}"
fi

server_args=(
    "${KAI0_PYTHON}"
    "${SERVER_SCRIPT}"
    --checkpoint-dir "${POLICY_DIR}"
    --checkpoint-id "${CHECKPOINT_ID}"
    --checkpoint-step "${CHECKPOINT_STEP}"
    --host "${POLICY_HOST}"
    --port "${POLICY_PORT}"
    --run-dir "${POLICY_RUN_DIR}"
)

echo "[Hoo policy] task=${TASK}"
echo "[Hoo policy] checkpoint_dir=${POLICY_DIR}"
echo "[Hoo policy] checkpoint_id=${CHECKPOINT_ID} step=${CHECKPOINT_STEP}"
echo "[Hoo policy] schema=${DETECTED_SCHEMA}"
echo "[Hoo policy] Kai0=${KAI0_ROOT}@${EXPECTED_KAI0_COMMIT} clean=true"
echo "[Hoo policy] endpoint=ws://${POLICY_HOST}:${POLICY_PORT} gpu=${POLICY_GPU}"
echo "[Hoo policy] canonical checkpoint digest will be printed by the server and sent in HELLO"

if (( DRY_RUN )); then
    printf '[Hoo policy] dry-run command:'
    printf ' %q' "${server_args[@]}"
    printf '\n'
    exit 0
fi

if (exec 3<>"/dev/tcp/${POLICY_HOST}/${POLICY_PORT}") >/dev/null 2>&1; then
    die "policy port is already in use: ${POLICY_HOST}:${POLICY_PORT}"
fi
FREE_GPU_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${POLICY_GPU}" | tr -d '[:space:]')"
[[ "${FREE_GPU_MB}" =~ ^[0-9]+$ ]] || die "could not read free memory for GPU ${POLICY_GPU}"
(( FREE_GPU_MB >= MIN_FREE_GPU_MB )) \
    || die "GPU ${POLICY_GPU} has ${FREE_GPU_MB} MiB free; need at least ${MIN_FREE_GPU_MB} MiB"

echo "[Hoo policy] loading checkpoint before opening the WebSocket"
echo "[Hoo policy] keep this terminal open; Ctrl-C stops only this server"

cd "${KAI0_ROOT}"
exec env \
    -u PYTHONHOME \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${KAI0_ROOT}/src" \
    "${server_args[@]}"
