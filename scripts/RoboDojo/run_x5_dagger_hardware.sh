#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
ROBOT_LAB_ROOT="${ROBOT_LAB_ROOT:-${HOME}/Robot_Lab}"
SOURCE_SCRIPT="${X5_SOURCE_SCRIPT:-${ROBODOJO_ROOT}/scripts/RoboDojo/x5_dual_joint_mirror_source.py}"
SOURCE_HOST="${ROBODOJO_X5_SOURCE_HOST:-127.0.0.1}"
SOURCE_PORT="${ROBODOJO_X5_SOURCE_PORT:-8770}"
LEFT_CAN="${X5_LEFT_CAN:-can1}"
RIGHT_CAN="${X5_RIGHT_CAN:-can3}"
LEFT_MODEL="${X5_LEFT_MODEL:-X5}"
RIGHT_MODEL="${X5_RIGHT_MODEL:-X5}"
HOME_RAD_TEXT="${X5_HOME_RAD:-0 0 0 0 0 0}"
HOME_GRIPPER_FRACTION="${X5_HOME_GRIPPER_FRACTION:-1.0}"
HOME_DURATION_S="${X5_HOME_DURATION_S:-5}"
FREQUENCY_HZ="${X5_FREQUENCY_HZ:-100}"
FOLLOW_PREVIEW_S="${X5_FOLLOW_PREVIEW_S:-0.04}"
HOTKEY_DISPLAY="${X5_HOTKEY_DISPLAY:-${DISPLAY:-:1}}"
SDK_MODULE="${X5_SDK_MODULE:-arx5_interface}"
IP_BIN="${IP_BIN:-ip}"

die() {
    echo "[X5 hardware][ERROR] $*" >&2
    exit 2
}

is_true() {
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        0|false|FALSE|no|NO|off|OFF) return 1 ;;
        *) die "expected boolean, got: $1" ;;
    esac
}

if [[ -n "${X5_PYTHON:-}" ]]; then
    x5_python="${X5_PYTHON}"
elif [[ -x "${ROBOT_LAB_ROOT}/.venv/bin/python" ]]; then
    x5_python="${ROBOT_LAB_ROOT}/.venv/bin/python"
elif [[ -x "${ROBODOJO_ROOT}/.venv/bin/python" ]]; then
    x5_python="${ROBODOJO_ROOT}/.venv/bin/python"
else
    x5_python="$(command -v python3 || true)"
fi

[[ -n "${x5_python}" && -x "${x5_python}" ]] \
    || die "X5 Python is unavailable; set X5_PYTHON to the environment containing ${SDK_MODULE}"
[[ -f "${SOURCE_SCRIPT}" ]] || die "X5 source is missing: ${SOURCE_SCRIPT}"
[[ "${SOURCE_HOST}" == "127.0.0.1" || "${SOURCE_HOST}" == "localhost" ]] \
    || die "the hardware source must bind only to loopback"
[[ "${SOURCE_PORT}" =~ ^[0-9]+$ ]] || die "ROBODOJO_X5_SOURCE_PORT must be an integer"
(( SOURCE_PORT >= 1 && SOURCE_PORT <= 65535 )) \
    || die "ROBODOJO_X5_SOURCE_PORT must be in [1, 65535]"
[[ "${LEFT_CAN}" != "${RIGHT_CAN}" ]] || die "left and right X5 cannot share one CAN interface"

if ! env PYTHONPATH="${ROBODOJO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${x5_python}" -c \
    'import importlib, sys; importlib.import_module(sys.argv[1])' "${SDK_MODULE}"; then
    die "${SDK_MODULE} cannot be imported by ${x5_python}; select the verified ARX SDK environment with X5_PYTHON"
fi

command -v "${IP_BIN}" >/dev/null || die "ip utility is unavailable: ${IP_BIN}"
for can_name in "${LEFT_CAN}" "${RIGHT_CAN}"; do
    can_line="$("${IP_BIN}" -o link show dev "${can_name}" 2>/dev/null)" \
        || die "CAN interface does not exist: ${can_name}"
    if ! grep -Eq '(<|,)UP(,|>)' <<<"${can_line}"; then
        die "CAN interface is not UP: ${can_name}"
    fi
done

if (exec 3<>"/dev/tcp/127.0.0.1/${SOURCE_PORT}") >/dev/null 2>&1; then
    die "source port is already in use: 127.0.0.1:${SOURCE_PORT}"
fi

read -r -a home_rad <<<"${HOME_RAD_TEXT}"
[[ "${#home_rad[@]}" -eq 6 ]] || die "X5_HOME_RAD must contain exactly six values"

source_args=(
    --host "${SOURCE_HOST}"
    --port "${SOURCE_PORT}"
    --left-can "${LEFT_CAN}"
    --right-can "${RIGHT_CAN}"
    --left-model "${LEFT_MODEL}"
    --right-model "${RIGHT_MODEL}"
    --home-rad "${home_rad[@]}"
    --home-gripper-fraction "${HOME_GRIPPER_FRACTION}"
    --home-duration-s "${HOME_DURATION_S}"
    --frequency-hz "${FREQUENCY_HZ}"
    --follow-preview-s "${FOLLOW_PREVIEW_S}"
    --display "${HOTKEY_DISPLAY}"
)
if is_true "${X5_SKIP_HOME:-0}"; then
    source_args+=(--skip-home)
fi
if ! is_true "${X5_GLOBAL_HOTKEYS:-1}"; then
    source_args+=(--no-global-hotkeys)
fi

export DISPLAY="${HOTKEY_DISPLAY}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export PYTHONPATH="${ROBODOJO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[X5 hardware] SDK and CAN preflight passed: left=${LEFT_CAN} right=${RIGHT_CAN}"
echo "[X5 hardware] source=${SOURCE_HOST}:${SOURCE_PORT} protocol=robodojo_dual_joint_mirror_v1"
echo "[X5 hardware] policy follow uses ${FOLLOW_PREVIEW_S}s ARX SDK interpolation"
echo "[X5 hardware] only one global operator key is used: i toggles manual intervention ON/OFF"
echo "[X5 hardware] terminal focus is not required for global i"
echo "[X5 hardware] there is no s/r recording key; episode commit is owned by Isaac"
echo "[X5 hardware] keep this terminal open; Ctrl-C stops the hardware owner"

cd "${ROBODOJO_ROOT}"
exec "${x5_python}" "${SOURCE_SCRIPT}" "${source_args[@]}"
