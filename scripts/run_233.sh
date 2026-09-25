#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for spec in "qwen3_2b vstar 191" "qwen3_2b textvqa 500" "qwen3_2b docvqa 500" "qwen2_7b vstar 191" "qwen2_7b textvqa 500" "qwen2_7b docvqa 500"; do
  set -- $spec
  while :; do
    ram=$(free -g | awk '/^Mem:/{print $7}')
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$ram" -ge 20 ] && [ "$gpu" -ge 8000 ] && break
    echo "233 waiting: ram=${ram}G gpu=${gpu}M $(date +%H:%M)" >> logs/seq.log; sleep 60
  done
  echo "=== START 233 $1 $2 $3 $(date +%H:%M) ===" >> logs/seq.log
  python3 scripts/phase233_endtask_head.py $1 $2 $3 > logs/seq_233_$1_$2.log 2>&1
  echo "=== DONE 233 $1 $2 rc=$? $(date +%H:%M) ===" >> logs/seq.log
done
echo "233 DONE $(date +%H:%M)" > logs/233_done.flag
