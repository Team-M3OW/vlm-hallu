#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "python3 scripts/m2_budget_policy.py" >/dev/null; do sleep 60; done
python3 scripts/m1_transport_distill.py --ceonly
python3 scripts/m1_transport_distill.py --split1
echo "M1 controls done"
