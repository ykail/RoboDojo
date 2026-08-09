#!/usr/bin/env bash

set -euo pipefail

SSH_BIN="${SSH_BIN:-ssh}"
HOO_SSH_TARGET="${HOO_SSH_TARGET:-hoo@10.19.127.58}"
HOO_SSH_IDENTITY_FILE="${HOO_SSH_IDENTITY_FILE:-}"
LOCAL_POLICY_PORT="${ROBODOJO_POLICY_PORT:-18080}"
HOO_POLICY_PORT="${HOO_POLICY_PORT:-18080}"

die() {
    echo "[Hoo tunnel][ERROR] $*" >&2
    exit 2
}

for port in "${LOCAL_POLICY_PORT}" "${HOO_POLICY_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] || die "policy ports must be integers"
    (( port >= 1 && port <= 65535 )) || die "policy ports must be in [1, 65535]"
done
command -v "${SSH_BIN}" >/dev/null || die "SSH executable not found: ${SSH_BIN}"

ssh_args=(
    -N
    -T
    -o BatchMode=yes
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
)
if [[ -n "${HOO_SSH_IDENTITY_FILE}" ]]; then
    [[ -r "${HOO_SSH_IDENTITY_FILE}" ]] \
        || die "SSH identity is not readable: ${HOO_SSH_IDENTITY_FILE}"
    ssh_args+=(-o IdentitiesOnly=yes -i "${HOO_SSH_IDENTITY_FILE}")
fi

echo "[Hoo tunnel] X5 127.0.0.1:${LOCAL_POLICY_PORT} -> Hoo 127.0.0.1:${HOO_POLICY_PORT}"
echo "[Hoo tunnel] keep this terminal open; Ctrl-C closes only the tunnel"
exec "${SSH_BIN}" "${ssh_args[@]}" \
    -L "127.0.0.1:${LOCAL_POLICY_PORT}:127.0.0.1:${HOO_POLICY_PORT}" \
    "${HOO_SSH_TARGET}"
