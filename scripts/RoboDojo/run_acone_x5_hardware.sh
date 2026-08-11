#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RAW_ROOT="${ROBODOJO_X5_RAW_ROOT:-}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/RoboDojo/run_acone_x5_hardware.sh [--raw-root PATH]

--raw-root enables the serialized 100 Hz hardware trace used by deferred
25 Hz replay. Omit it to keep the legacy online-only behavior.
EOF
}

die() {
    echo "[acOne hardware][ERROR] $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-root)
            [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
            RAW_ROOT="$2"
            shift 2
            ;;
        --raw-root=*) RAW_ROOT="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

if [[ -n "${RAW_ROOT}" ]]; then
    mkdir -p "${RAW_ROOT}"
    RAW_ROOT="$(cd "${RAW_ROOT}" && pwd -P)"
fi

conflicting_owners="$(pgrep -af 'lerobot-record-ui|lerobot_record|inference_pi0_arx_acone' || true)"
if [[ -n "${conflicting_owners}" ]]; then
    echo "[acOne hardware][ERROR] Another ARX process may own the arms/CAN:" >&2
    echo "${conflicting_owners}" >&2
    exit 2
fi

export X5_PYTHON="/home/acone/Robot_Lab/.venv/bin/python"
export X5_LEFT_CAN="can1"
export X5_RIGHT_CAN="can3"
export X5_LEFT_MODEL="X5"
export X5_RIGHT_MODEL="X5"
export X5_HOME_RAD="0 0 0 0 0 0"
export X5_HOME_GRIPPER_FRACTION="1.0"
export X5_HOME_DURATION_S="5"
export X5_FREQUENCY_HZ="100"
export X5_FOLLOW_PREVIEW_S="0.04"
export X5_HOTKEY_DISPLAY="${DISPLAY:-:1}"
export ROBODOJO_X5_SOURCE_HOST="127.0.0.1"
export ROBODOJO_X5_SOURCE_PORT="8770"
export ROBODOJO_X5_RAW_ROOT="${RAW_ROOT}"

exec "${SCRIPT_DIR}/run_x5_dagger_hardware.sh"
