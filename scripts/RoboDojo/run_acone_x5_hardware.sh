#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export X5_PYTHON="/home/acone/Robot_Lab/.venv/bin/python"
export X5_LEFT_CAN="can1"
export X5_RIGHT_CAN="can3"
export X5_LEFT_MODEL="X5"
export X5_RIGHT_MODEL="X5"
export X5_HOME_RAD="0 0 0 0 0 0"
export X5_HOME_GRIPPER_FRACTION="1.0"
export X5_HOME_DURATION_S="5"
export X5_FREQUENCY_HZ="100"
export X5_HOTKEY_DISPLAY="${DISPLAY:-:1}"
export ROBODOJO_X5_SOURCE_HOST="127.0.0.1"
export ROBODOJO_X5_SOURCE_PORT="8770"

exec "${SCRIPT_DIR}/run_x5_dagger_hardware.sh"
