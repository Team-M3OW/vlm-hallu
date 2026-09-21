#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_post_queue" >/dev/null; do sleep 60; done
python3 scripts/phase199_layersweep.py qwen2
echo "207b done"
