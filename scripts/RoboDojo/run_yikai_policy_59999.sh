#!/usr/bin/env bash

set -euo pipefail

YIKAI_TARGET="${YIKAI_SSH_TARGET:-yikai}"
KAI0_ROOT="${YIKAI_KAI0_ROOT:-/home/ykail/vibe_code/RoboDojo/third_party/kai0}"
CHECKPOINT_DIR="${YIKAI_CHECKPOINT_DIR:-/home/ykail/data/RoboDojo_hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999}"
POLICY_PORT="${YIKAI_POLICY_PORT:-18080}"
GPU="${YIKAI_POLICY_GPU:-0}"
EXPECTED_KAI0_COMMIT="ecc1a7451c3156b1e5f7533851dbb0222896206f"
CHECKPOINT_ID="RoboDojo-sim-arx_x5-joint-0/59999"

[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || {
    echo "[yikai policy][ERROR] YIKAI_POLICY_PORT must be an integer" >&2
    exit 2
}

echo "[yikai policy] starting ${CHECKPOINT_ID} on ${YIKAI_TARGET}:127.0.0.1:${POLICY_PORT}"
echo "[yikai policy] this terminal owns the remote server; Ctrl-C stops it"

exec ssh -tt "${YIKAI_TARGET}" \
    env \
    KAI0_ROOT="${KAI0_ROOT}" \
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    POLICY_PORT="${POLICY_PORT}" \
    POLICY_GPU="${GPU}" \
    EXPECTED_KAI0_COMMIT="${EXPECTED_KAI0_COMMIT}" \
    CHECKPOINT_ID="${CHECKPOINT_ID}" \
    bash -s <<'REMOTE'
set -euo pipefail

KAI0_PYTHON="${KAI0_ROOT}/.venv/bin/python"
[[ -x "${KAI0_PYTHON}" ]] || {
    echo "[yikai policy][ERROR] missing Kai0 Python: ${KAI0_PYTHON}" >&2
    exit 2
}
[[ -d "${CHECKPOINT_DIR}/params" ]] || {
    echo "[yikai policy][ERROR] missing checkpoint params: ${CHECKPOINT_DIR}/params" >&2
    exit 2
}
[[ -f "${CHECKPOINT_DIR}/assets/arx_x5_sim/norm_stats.json" ]] || {
    echo "[yikai policy][ERROR] missing norm stats" >&2
    exit 2
}

ACTUAL_COMMIT="$(git -C "${KAI0_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_KAI0_COMMIT}" ]] || {
    echo "[yikai policy][ERROR] Kai0 commit ${ACTUAL_COMMIT} != ${EXPECTED_KAI0_COMMIT}" >&2
    exit 2
}
[[ -z "$(git -C "${KAI0_ROOT}" status --porcelain --untracked-files=normal)" ]] || {
    echo "[yikai policy][ERROR] Kai0 worktree is dirty" >&2
    exit 2
}

echo "[yikai policy] Kai0=${ACTUAL_COMMIT} clean=true"
echo "[yikai policy] checkpoint=${CHECKPOINT_ID}"
cd "${KAI0_ROOT}"
exec env \
    -u PYTHONHOME \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${KAI0_ROOT}/src" \
    "${KAI0_PYTHON}" scripts/serve_robodojo_policy.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-id "${CHECKPOINT_ID}" \
    --checkpoint-step 59999 \
    --host 127.0.0.1 \
    --port "${POLICY_PORT}"
REMOTE
