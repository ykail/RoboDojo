#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${SCRIPT_DIR}/run_acone_x5_tunnel.sh" \
    --port "${ROBODOJO_POLICY_PORT:-18081}" \
    "$@"
