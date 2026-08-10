#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
POLICY_PORT="${ROBODOJO_POLICY_PORT:-18081}"
HOO_TARGET="${HOO_SSH_TARGET:-hoo@10.19.127.58}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_acone_x5_tunnel.sh [--port 18081] [--hoo TARGET]

Both --option value and --option=value forms are accepted.
EOF
}

die() {
    echo "[acOne tunnel][ERROR] $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|--policy-port)
            [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
            POLICY_PORT="$2"
            shift 2
            ;;
        --port=*|--policy-port=*) POLICY_PORT="${1#*=}"; shift ;;
        --hoo)
            [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
            HOO_TARGET="$2"
            shift 2
            ;;
        --hoo=*) HOO_TARGET="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ "${POLICY_PORT}" =~ ^[0-9]+$ ]] || die "port must be an integer"
(( POLICY_PORT >= 1 && POLICY_PORT <= 65535 )) || die "port must be in [1, 65535]"

export HOO_SSH_TARGET="${HOO_TARGET}"
export ROBODOJO_POLICY_PORT="${POLICY_PORT}"
export HOO_POLICY_PORT="${POLICY_PORT}"

exec "${SCRIPT_DIR}/run_hoo_policy_tunnel_from_x5.sh"
