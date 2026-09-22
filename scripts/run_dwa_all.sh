#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "dl_gemma|dl_ib" >/dev/null; do sleep 60; done
echo "=== downloads finished; waiting for the phase213 matrix ==="
while pgrep -f "phase213_multi|run_213_queue" >/dev/null; do sleep 60; done
for m in llava_ov llava_next gemma3_4b; do
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    [ "$free" -ge 22000 ] && break; sleep 30
  done
  echo "=== DWA MAPS $m ==="
  python3 scripts/phase214_dwa_multi.py $m 200 2>&1 | grep -vE "it/s|%\|" | tail -4
  echo "=== DWA EVAL $m ==="
  python3 scripts/phase215_dwa_eval.py $m 2>&1 | grep -vE "it/s|%\|" | tail -6
done
echo "DWA ALL DONE"
