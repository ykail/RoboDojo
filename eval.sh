#!/usr/bin/env bash
set -euo pipefail

cd /home/asus/codes/RoboDojo
source /home/asus/miniconda/etc/profile.d/conda.sh

# Port 6000 is kai0's native inference protocol.  Kai0Relay translates it to
# the XPolicyLab protocol expected by the RoboDojo simulator, so let the
# generic evaluator launch a local relay and clean it up when evaluation ends.
export KAI0_URL="${KAI0_URL:-ws://10.19.125.53:6000}"
# Optional inclusive layout range, for example: EVAL_LAYOUT_RANGE=100-149.
# Leave empty to use layouts in their normal order.
export EVAL_LAYOUT_RANGE="${EVAL_LAYOUT_RANGE:-0:49}"


conda activate RoboDojo
bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Kai0Relay \
    --task make_kong \
    --env-cfg arx_x5 \
    --env-gpu 0 \
    --policy-gpu 0 \
    --action-type joint \
    --ckpt external \
    --seed 0 \
    --eval-num "${EVAL_NUM:-native}" \
    --policy-env RoboDojo \
    --eval-env RoboDojo
