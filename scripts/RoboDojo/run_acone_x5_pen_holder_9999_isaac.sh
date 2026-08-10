#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export ROBODOJO_TASK="fill_pen_holder"
export ROBODOJO_POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
export ROBODOJO_LEROBOT_REPO_ID="robodojo_fill_pen_holder_x5_online_dagger_9999_v1"
export ROBODOJO_CHECKPOINT_ID="pi05_robodojo_three_task_base/fill_pen_kong_toast_300_base_official_norm_v1/9999"
export ROBODOJO_EXPECTED_KAI0_COMMIT="ecc1a7451c3156b1e5f7533851dbb0222896206f"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="sha256:2b906f8e1d4932d7f7cb57aa2d8113f83efcc9f5fb35a0fc17c2934346c4eec2"

exec "${SCRIPT_DIR}/run_acone_x5_isaac.sh"
