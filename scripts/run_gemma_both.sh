#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for m in stock adaptive; do
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    [ "$free" -ge 20000 ] && break; sleep 30
  done
  echo "=== GEMMA $m ==="
  python3 scripts/phase216_gemma_pool.py $m vstar 200 2>&1 | grep -vE "it/s|%\|" | tail -4
done
echo "GEMMA BOTH DONE"
