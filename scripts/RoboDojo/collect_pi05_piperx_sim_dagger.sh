#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/collect_pi05_piperx_sim_dagger.sh \
    --task TASK \
    --checkpoint-dir PATH \
    --checkpoint-id ID \
    --piperx-calibration PATH \
    [eval_kai0_pi05.sh options]

This starts Kai0 and RoboDojo only. It never starts, enables, or configures
PiPER-X hardware. Start the supervised LeRobot hardware bridge separately,
verify its safety checks, then run this command.

The dedicated launcher fixes --control-mode=piperx_sim_dagger. Use
eval_kai0_pi05.sh --help for the complete option list.
EOF
  exit 0
fi

for argument in "$@"; do
  if [[ "${argument}" == "--control-mode" || "${argument}" == --control-mode=* ]]; then
    echo "[collect_pi05_piperx_sim_dagger][ERROR] --control-mode is fixed by this launcher" >&2
    exit 2
  fi
done

exec bash "${ROOT_DIR}/scripts/RoboDojo/eval_kai0_pi05.sh" \
  "$@" \
  --control-mode piperx_sim_dagger
