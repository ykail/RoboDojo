#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${SCRIPT_DIR}/run_hoo_policy.sh" \
    --policy-dir "${ROBODOJO_CHECKPOINT_DIR:-/home/hoo/RoboDojo/.cache/robodojo_ckpt_huggingface_repo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999}" \
    --task make_toast \
    --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
    --checkpoint-step 59999 \
    --port "${ROBODOJO_POLICY_PORT:-18080}" \
    "$@"
