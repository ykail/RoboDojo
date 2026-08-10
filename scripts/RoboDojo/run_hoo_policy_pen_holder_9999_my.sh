#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${SCRIPT_DIR}/run_hoo_policy.sh" \
    --policy-dir /home/hoo/checkpoints/9999_my \
    --task fill_pen_holder \
    --checkpoint-id fill_pen_holder/9999_my \
    --checkpoint-step 9999 \
    --port "${ROBODOJO_POLICY_PORT:-18081}" \
    "$@"
