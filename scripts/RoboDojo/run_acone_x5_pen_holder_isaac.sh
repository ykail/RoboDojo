#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export ROBODOJO_TASK="fill_pen_holder"
export ROBODOJO_POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
export ROBODOJO_LEROBOT_REPO_ID="robodojo_fill_pen_holder_x5_online_dagger_9999_my_v1"
export ROBODOJO_CHECKPOINT_ID="fill_pen_holder/9999_my"
export ROBODOJO_EXPECTED_KAI0_COMMIT="76d26714c276c9a4812066854d248111382fe591"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="sha256:7e3cbf579a37640a13c0d152cd5913b9142437627db1e5c33b4266602c2c46ab"

exec "${SCRIPT_DIR}/run_acone_x5_isaac.sh"
