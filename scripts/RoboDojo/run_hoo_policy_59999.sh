#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
KAI0_ROOT="${KAI0_ROOT:-${HOME}/RoboDojo/third_party/kai0}"
CHECKPOINT_DIR="${ROBODOJO_CHECKPOINT_DIR:-${HOME}/RoboDojo/.cache/robodojo_ckpt_huggingface_repo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999}"
KAI0_PYTHON="${KAI0_PYTHON:-${KAI0_ROOT}/.venv/bin/python}"
SERVER_SCRIPT="${ROBODOJO_POLICY_SERVER_SCRIPT:-${KAI0_ROOT}/scripts/serve_robodojo_policy.py}"
POLICY_HOST="${ROBODOJO_POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
POLICY_RUN_DIR="${ROBODOJO_POLICY_RUN_DIR:-${HOME}/.local/state/kai0/robodojo_policy_v1}"

CHECKPOINT_ID="RoboDojo-sim-arx_x5-joint-0/59999"
EXPECTED_KAI0_COMMIT="ecc1a7451c3156b1e5f7533851dbb0222896206f"
EXPECTED_CHECKPOINT_DIGEST="sha256:70bb68139ba717553d9a9d9c3055bb322b85046d729377ee46eaaf997c1eaac4"

die() {
    echo "[Hoo policy][ERROR] $*" >&2
    exit 2
}

case "${POLICY_HOST}" in
    127.0.0.1|localhost) ;;
    *) die "ROBODOJO_POLICY_HOST must stay on loopback; use the SSH tunnel from the X5 PC" ;;
esac
[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || die "ROBODOJO_POLICY_PORT must be an integer"
(( POLICY_PORT >= 1 && POLICY_PORT <= 65535 )) || die "ROBODOJO_POLICY_PORT must be in [1, 65535]"

command -v git >/dev/null || die "git is not available"
[[ -d "${KAI0_ROOT}/.git" || -f "${KAI0_ROOT}/.git" ]] || die "Kai0 checkout not found: ${KAI0_ROOT}"
[[ -x "${KAI0_PYTHON}" ]] || die "Kai0 Python is not executable: ${KAI0_PYTHON}"
[[ -f "${SERVER_SCRIPT}" ]] || die "strict policy-v1 server is missing: ${SERVER_SCRIPT}"
[[ -d "${CHECKPOINT_DIR}/params" ]] || die "checkpoint params are missing: ${CHECKPOINT_DIR}/params"
[[ -f "${CHECKPOINT_DIR}/assets/arx_x5_sim/norm_stats.json" ]] \
    || die "checkpoint norm stats are missing: ${CHECKPOINT_DIR}/assets/arx_x5_sim/norm_stats.json"

actual_commit="$(git -C "${KAI0_ROOT}" rev-parse --verify HEAD)"
[[ "${actual_commit}" == "${EXPECTED_KAI0_COMMIT}" ]] \
    || die "Kai0 commit mismatch: expected ${EXPECTED_KAI0_COMMIT}, got ${actual_commit}"
[[ -z "$(git -C "${KAI0_ROOT}" status --porcelain=v1 --untracked-files=normal)" ]] \
    || die "Kai0 worktree is dirty; refusing provenance launch"

echo "[Hoo policy] Kai0=${EXPECTED_KAI0_COMMIT} clean=true"
echo "[Hoo policy] checkpoint=${CHECKPOINT_ID}"
echo "[Hoo policy] expected_digest=${EXPECTED_CHECKPOINT_DIGEST}"
echo "[Hoo policy] loading checkpoint before opening ws://${POLICY_HOST}:${POLICY_PORT}"
echo "[Hoo policy] keep this terminal open; Ctrl-C stops only the policy server"

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
    "${KAI0_PYTHON}" "${SERVER_SCRIPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-id "${CHECKPOINT_ID}" \
    --checkpoint-step 59999 \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --run-dir "${POLICY_RUN_DIR}"
