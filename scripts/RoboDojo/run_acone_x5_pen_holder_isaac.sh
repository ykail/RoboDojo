#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${SCRIPT_DIR}/run_acone_x5_isaac.sh" \
    --policy-dir /home/hoo/checkpoints/9999_my \
    --task fill_pen_holder \
    --checkpoint-id fill_pen_holder/9999_my \
    --port "${ROBODOJO_POLICY_PORT:-18081}" \
    --dataset-id robodojo_fill_pen_holder_x5_online_dagger_9999_my_timing25_v2 \
    "$@"
