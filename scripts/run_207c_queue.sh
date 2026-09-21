#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_final_queue" >/dev/null; do sleep 60; done
python3 scripts/phase176c_transport.py qwen3_8b
echo "207c done"
