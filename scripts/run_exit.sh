#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_everything.sh|run_warp.sh|phase21[3-7]_" >/dev/null; do sleep 60; done
for m in qwen3 qwen2; do
  for i in $(seq 1 240); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    [ "$f" -ge 20000 ] && break; sleep 30
  done
  echo "=== EXIT SWEEP $m ==="
  python3 scripts/phase219_exit_depth.py $m 2>&1 | grep -vE "it/s|%\|" | tail -4
done
echo "EXIT SWEEP DONE"
