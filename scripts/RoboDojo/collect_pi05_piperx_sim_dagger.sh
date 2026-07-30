#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/RoboDojo/collect_pi05_piperx_sim_dagger.sh \
    --task TASK \
    --checkpoint-id ID \
    [--checkpoint-dir PATH | --external-policy-server-url URL] \
    [eval_kai0_pi05.sh options]

This starts Kai0 and RoboDojo. The separately supervised LeRobot process owns
all CAN devices; after policy/simulator preflight, the first episode explicitly
authorizes that bridge to perform its bounded four-arm bring-up. No manual
frame calibration file is used: takeover is anchored automatically.

External policy mode requires a loopback SSH tunnel, the expected full Kai0
commit, and a recorder Python. It never starts or stops Kai0 on this machine.

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
