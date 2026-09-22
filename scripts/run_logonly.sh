#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for m in qwen3 qwen2; do
  for i in $(seq 1 480); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    [ "$f" -ge 20000 ] && break; sleep 30
  done
  echo "=== LOG-ONLY $m ==="
  python3 scripts/phase223_logonly.py $m 2>&1 | grep -vE "it/s|%\|" | tail -3
done
echo "LOGONLY DONE"
