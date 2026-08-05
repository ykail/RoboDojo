#!/bin/bash

# In one terminal, run
ssh -N \
  -L 8000:127.0.0.1:8888 \
  -p 30725 \
  root@10.15.171.204

# In another terminal, run
bash scripts/robodojo.sh client \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task make_kong \
  --ckpt sim \
  --env-cfg arx_x5 \
  --action-type joint \
  --seed 0 \
  --env-gpu 0 \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --protocol openpi \
  --eval-num native
