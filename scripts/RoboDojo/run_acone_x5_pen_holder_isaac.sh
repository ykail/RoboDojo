#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export ROBODOJO_TASK="fill_pen_holder"
export ROBODOJO_LEROBOT_REPO_ID="robodojo_fill_pen_holder_x5_online_dagger_9999_my_v1"
export ROBODOJO_CHECKPOINT_ID="fill_pen_holder/9999_my"
# The compatible clean Kai0 revision and immutable checkpoint digest are set
# by the matching Hoo launcher.  Empty values still require HELLO provenance
# and a clean server checkout, but do not pretend this custom checkpoint is
# the official 59999 artifact.
export ROBODOJO_EXPECTED_KAI0_COMMIT="${ROBODOJO_EXPECTED_KAI0_COMMIT:-}"
export ROBODOJO_EXPECTED_CHECKPOINT_DIGEST="${ROBODOJO_EXPECTED_CHECKPOINT_DIGEST:-}"

exec "${SCRIPT_DIR}/run_acone_x5_isaac.sh"
