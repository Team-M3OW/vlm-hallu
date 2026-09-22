#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_everything.sh|phase21[3-6]_" >/dev/null; do sleep 60; done
for i in $(seq 1 240); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "$f" -ge 20000 ] && break; sleep 30
done
echo "=== DWA-STEERED WARP ==="
python3 scripts/phase217_dwa_warp.py 2>&1 | grep -vE "it/s|%\|" | tail -4
echo "WARP DONE"
