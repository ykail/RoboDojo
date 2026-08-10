#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export HOO_SSH_TARGET="${HOO_SSH_TARGET:-hoo@10.19.127.58}"
export ROBODOJO_POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
export HOO_POLICY_PORT="${HOO_POLICY_PORT:-18081}"

exec "${SCRIPT_DIR}/run_hoo_policy_tunnel_from_x5.sh"
