#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "python3 scripts/m1_transport_distill.py" >/dev/null; do sleep 30; done
python3 scripts/m2_budget_policy.py qwen3
python3 scripts/m2_budget_policy.py qwen2
echo "M2 done"
