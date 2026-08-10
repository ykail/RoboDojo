#!/usr/bin/env bash

set -euo pipefail

KAI0_ROOT="/home/hoo/kai0-output-engine-policy-v1"
KAI0_PYTHON="/home/hoo/jiran/project/codebases/kai0/.venv/bin/python"
CHECKPOINT_DIR="/home/hoo/checkpoints/9999_my"
CHECKPOINT_ID="fill_pen_holder/9999_my"
EXPECTED_KAI0_COMMIT="76d26714c276c9a4812066854d248111382fe591"
EXPECTED_CHECKPOINT_DIGEST="sha256:7e3cbf579a37640a13c0d152cd5913b9142437627db1e5c33b4266602c2c46ab"
POLICY_HOST="127.0.0.1"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
POLICY_RUN_DIR="${ROBODOJO_POLICY_RUN_DIR:-/home/hoo/.local/state/kai0/robodojo_policy_v1}"
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_POLICY_GPU_MB:-14000}"

die() {
    echo "[Hoo pen-holder policy][ERROR] $*" >&2
    exit 2
}

[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || die "policy port must be an integer"
(( POLICY_PORT >= 1 && POLICY_PORT <= 65535 )) || die "policy port must be in [1, 65535]"
[[ -x "${KAI0_PYTHON}" ]] || die "Kai0 Python is unavailable: ${KAI0_PYTHON}"
[[ -f "${KAI0_ROOT}/scripts/serve_robodojo_policy.py" ]] \
    || die "strict policy server is missing"
[[ -d "${CHECKPOINT_DIR}/params" ]] || die "checkpoint params are missing"
[[ -f "${CHECKPOINT_DIR}/assets/arx_x5_sim/norm_stats.json" ]] \
    || die "checkpoint arx_x5_sim norm stats are missing"

actual_commit="$(git -C "${KAI0_ROOT}" rev-parse --verify HEAD)"
[[ "${actual_commit}" == "${EXPECTED_KAI0_COMMIT}" ]] \
    || die "Kai0 commit mismatch: ${actual_commit}"
[[ -z "$(git -C "${KAI0_ROOT}" status --porcelain=v1 --untracked-files=normal)" ]] \
    || die "Kai0 compatibility worktree is dirty"
conflicting_policy="$(pgrep -af 'serve_robodojo_policy.py' || true)"
[[ -z "${conflicting_policy}" ]] || die \
    "another policy server is using the GPU; stop it first: ${conflicting_policy}"
free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
[[ "${free_gpu_mb}" =~ ^[0-9]+$ ]] || die "could not read free GPU memory"
(( free_gpu_mb >= MIN_FREE_GPU_MB )) || die \
    "GPU 0 has only ${free_gpu_mb} MiB free; need at least ${MIN_FREE_GPU_MB} MiB"
if (exec 3<>"/dev/tcp/${POLICY_HOST}/${POLICY_PORT}") >/dev/null 2>&1; then
    die "policy port is already in use: ${POLICY_HOST}:${POLICY_PORT}"
fi

echo "[Hoo pen-holder policy] Kai0=${EXPECTED_KAI0_COMMIT} clean=true"
echo "[Hoo pen-holder policy] checkpoint=${CHECKPOINT_ID}"
echo "[Hoo pen-holder policy] digest=${EXPECTED_CHECKPOINT_DIGEST}"
echo "[Hoo pen-holder policy] loading before opening ws://${POLICY_HOST}:${POLICY_PORT}"
echo "[Hoo pen-holder policy] keep this terminal open; Ctrl-C stops only this server"

cd "${KAI0_ROOT}"
exec env \
    -u PYTHONHOME \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${KAI0_ROOT}/src" \
    "${KAI0_PYTHON}" scripts/serve_robodojo_policy.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-id "${CHECKPOINT_ID}" \
    --checkpoint-step 9999 \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --run-dir "${POLICY_RUN_DIR}"
