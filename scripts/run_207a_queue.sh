#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_205_queue" >/dev/null; do sleep 60; done
python3 scripts/phase176_readout_sensitivity.py qwen2
echo "207a done"
