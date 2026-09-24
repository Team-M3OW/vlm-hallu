#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for spec in "qwen3_2b vstar 191" "qwen3_2b docvqa 600" "qwen2_7b vstar 191" "qwen2_7b docvqa 600"; do
  set -- $spec
  while :; do
    ram=$(free -g | awk '/^Mem:/{print $7}')
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$ram" -ge 20 ] && [ "$gpu" -ge 20000 ] && break
    echo "229 waiting: ram=${ram}G gpu=${gpu}M $(date +%H:%M)" >> logs/seq.log; sleep 60
  done
  echo "=== START 229 $1 $2 $3 $(date +%H:%M) ===" >> logs/seq.log
  python3 scripts/phase229_dwa_dynW.py $1 $2 $3 > logs/seq_229_$1_$2.log 2>&1
  echo "=== DONE 229 $1 $2 rc=$? $(date +%H:%M) ===" >> logs/seq.log
done
echo "229 DONE $(date +%H:%M)" > logs/229_done.flag
