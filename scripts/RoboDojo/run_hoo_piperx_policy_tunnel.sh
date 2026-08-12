#!/usr/bin/env bash

set -euo pipefail

LOCAL_PORT="${ROBODOJO_POLICY_PORT:-18080}"
REMOTE_PORT="${YIKAI_POLICY_PORT:-${LOCAL_PORT}}"
YIKAI_TARGET="${YIKAI_SSH_TARGET:-yikai}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_hoo_piperx_policy_tunnel.sh \
    [--local-port 18080] [--remote-port 18080] [--yikai yikai]

Runs in the foreground.  Ctrl-C stops only the tunnel.
EOF
}

die() {
    echo "[Hoo PiPER-X tunnel][ERROR] $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-port) [[ $# -ge 2 ]] || die "$1 requires a value"; LOCAL_PORT="$2"; shift 2 ;;
        --local-port=*) LOCAL_PORT="${1#*=}"; shift ;;
        --remote-port) [[ $# -ge 2 ]] || die "$1 requires a value"; REMOTE_PORT="$2"; shift 2 ;;
        --remote-port=*) REMOTE_PORT="${1#*=}"; shift ;;
        --yikai) [[ $# -ge 2 ]] || die "$1 requires a value"; YIKAI_TARGET="$2"; shift 2 ;;
        --yikai=*) YIKAI_TARGET="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

for port in "${LOCAL_PORT}" "${REMOTE_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] || die "ports must be integers"
    (( port >= 1 && port <= 65535 )) || die "ports must be in [1, 65535]"
done
if (exec 3<>"/dev/tcp/127.0.0.1/${LOCAL_PORT}") >/dev/null 2>&1; then
    die "local port is already in use: 127.0.0.1:${LOCAL_PORT}"
fi

echo "[Hoo PiPER-X tunnel] 127.0.0.1:${LOCAL_PORT} -> ${YIKAI_TARGET}:127.0.0.1:${REMOTE_PORT}"
echo "[Hoo PiPER-X tunnel] keep this terminal open; Ctrl-C stops only the tunnel"
exec ssh \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=3 \
    -N \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${YIKAI_TARGET}"
