#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# This launcher intentionally uses the active RoboDojo Python only for the
# dependency-free orchestration layer. Kai0 and the LeRobot writer continue to
# run in Kai0's explicitly selected virtual environment.
exec python3 "${ROOT_DIR}/scripts/RoboDojo/collect_kai0_rollouts.py" "$@"
