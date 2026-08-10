#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${SCRIPT_DIR}/run_acone_x5_isaac.sh" \
    --policy-dir /home/hoo/checkpoints/9999 \
    --task fill_pen_holder \
    --checkpoint-id pi05_robodojo_three_task_base/fill_pen_kong_toast_300_base_official_norm_v1/9999 \
    --port "${ROBODOJO_POLICY_PORT:-18081}" \
    --dataset-id robodojo_fill_pen_holder_x5_online_dagger_9999_timing25_v2 \
    "$@"
