#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 22000 ]; then
    echo "$(date) free=${free}MiB -> launching qwen2 in-domain" >> logs/250_qwen2.log
    python3 scripts/phase250_indomain_dwa.py qwen2_7b 200 >> logs/250_qwen2.log 2>&1
    echo "$(date) qwen2 run finished" >> logs/250_qwen2.log
    break
  fi
  sleep 120
done
